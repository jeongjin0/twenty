"""
Projection Test - Visualize Input/Output Projection Behavior

Tests:
1. Load dataset sample → VAE encode → Input Projection → VAE decode → Visualize
2. Check if output projection produces layers in correct order
3. Verify projection weights are loaded correctly
"""

import argparse
import os
import sys
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusion.model.nets.PixArt_layer_inpainting import PixArtLayerInpainting
from diffusion.model.nets.PixArt import PixArt_XL_2
from diffusion.data.multilayer_builder import build_mulan_dataloader
from scipy.optimize import linear_sum_assignment


@torch.no_grad()
def test_projection(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("="*60)
    print("Projection Test")
    print("="*60)

    # ========================================
    # Load Model
    # ========================================
    print("\n[1/4] Loading model...")

    pretrained_pixart = PixArt_XL_2(
        input_size=args.image_size // 8,
        in_channels=4,
        caption_channels=4096,
        model_max_length=120,
        pred_sigma=True,
    )

    model = PixArtLayerInpainting(
        pretrained_pixart=pretrained_pixart,
        max_layers=args.max_layers,
        input_size=args.image_size // 8,
        pred_sigma=True,
    ).to(device).eval()

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location='cpu')

    # Debug: Check checkpoint structure
    print(f"\n  Checkpoint keys: {list(ckpt.keys())}")

    state_dict = ckpt.get('state_dict_ema', ckpt.get('state_dict', ckpt))
    print(f"  State dict keys (first 10): {list(state_dict.keys())[:10]}")
    print(f"  Total keys: {len(state_dict)}")

    # Check if projection keys exist
    input_proj_keys = [k for k in state_dict.keys() if 'input_proj' in k]
    output_proj_keys = [k for k in state_dict.keys() if 'output_proj' in k]
    print(f"  Input projection keys: {len(input_proj_keys)}")
    print(f"  Output projection keys: {len(output_proj_keys)}")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"  ✓ Model loaded from {args.checkpoint}")
    if missing:
        print(f"  ⚠ Missing keys: {len(missing)}")
        for k in missing[:3]:
            print(f"      {k}")
    if unexpected:
        print(f"  ⚠ Unexpected keys: {len(unexpected)}")
        for k in unexpected[:3]:
            print(f"      {k}")

    # Check projection weights
    print("\n  Projection weight stats:")
    print(f"    input_proj.enc_conv1: mean={model.input_proj.enc_conv1.weight.mean():.6f}, std={model.input_proj.enc_conv1.weight.std():.6f}")
    print(f"    output_proj.dec_conv1: mean={model.output_proj.dec_conv1.weight.mean():.6f}, std={model.output_proj.dec_conv1.weight.std():.6f}")

    # ========================================
    # Load VAE
    # ========================================
    print("\n[2/4] Loading VAE...")
    vae = AutoencoderKL.from_pretrained(args.vae_path).to(device).eval()
    print(f"  ✓ VAE loaded")

    # ========================================
    # Load Dataset
    # ========================================
    print("\n[3/4] Loading dataset...")
    dataloader = build_mulan_dataloader(
        data_roots=args.data_roots,
        batch_size=1,
        resolution=args.image_size,
        max_layers=args.max_layers,
        num_workers=0,
        shuffle=False,
        caption_type='blip2',
        max_samples=args.max_samples,
    )
    print(f"  ✓ Dataset loaded")

    # ========================================
    # Test Projections
    # ========================================
    print("\n[4/4] Testing projections...")

    os.makedirs(args.output_dir, exist_ok=True)
    scale_factor = 0.18215

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= args.num_samples:
            break

        layers, captions, num_layers, image_ids = batch
        layers = layers.to(device)  # (1, N, 4, H, W)

        B, N, C, H, W = layers.shape
        n_valid = num_layers[0].item()
        image_id = image_ids[0]
        h, w = H // 8, W // 8

        print(f"\n  Sample {batch_idx}: {image_id} ({n_valid} valid layers)")

        sample_dir = os.path.join(args.output_dir, f"sample_{batch_idx:02d}_{image_id}")
        os.makedirs(sample_dir, exist_ok=True)

        # ========================================
        # 1. Encode layers to VAE latent
        # ========================================
        layers_rgb = layers[:, :, :3, :, :]  # (1, N, 3, H, W)
        layers_flat = layers_rgb.reshape(N, 3, H, W)
        z_layers = vae.encode(layers_flat).latent_dist.mode() * scale_factor  # (N, 4, h, w)
        z_layers = z_layers.unsqueeze(0)  # (1, N, 4, h, w)

        print(f"    Layer latents: {z_layers.shape}")

        # ========================================
        # 2. Test Input Projection (layers → merged)
        # ========================================
        # Create dummy mask (no mask)
        layer_mask = torch.zeros(1, N, device=device)

        # Flatten for input projection
        z_flat = z_layers.reshape(1, N * 4, h, w)  # (1, 24, h, w)
        mask_spatial = layer_mask.unsqueeze(-1).unsqueeze(-1).expand(1, N, h, w)  # (1, 6, h, w)
        input_with_mask = torch.cat([z_flat, mask_spatial], dim=1)  # (1, 30, h, w)

        # Input projection
        merged_latent = model.input_proj(input_with_mask)  # (1, 4, h, w)

        print(f"\n    === Latent Statistics ===")
        print(f"    Input (z_layers): mean={z_layers.mean():.4f}, std={z_layers.std():.4f}, min={z_layers.min():.4f}, max={z_layers.max():.4f}")
        print(f"    Merged latent: mean={merged_latent.mean():.4f}, std={merged_latent.std():.4f}, min={merged_latent.min():.4f}, max={merged_latent.max():.4f}")

        # Decode merged latent
        merged_decoded = vae.decode(merged_latent / scale_factor).sample

        # ========================================
        # 3. Test Output Projection (merged → layers)
        # ========================================
        layers_recon_flat = model.output_proj(merged_latent)  # (1, 24, h, w)
        layers_recon = layers_recon_flat.reshape(1, N, 4, h, w)

        print(f"    Recon layers: mean={layers_recon.mean():.4f}, std={layers_recon.std():.4f}, min={layers_recon.min():.4f}, max={layers_recon.max():.4f}")

        # Per-layer reconstruction error
        print(f"\n    === Reconstruction Error (MSE) ===")
        total_mse = 0
        for i in range(n_valid):
            mse = F.mse_loss(layers_recon[0, i], z_layers[0, i]).item()
            total_mse += mse
            print(f"    Layer {i}: MSE={mse:.6f}")
        print(f"    Average MSE: {total_mse / n_valid:.6f}")

        # Decode reconstructed layers
        layers_recon_decoded = []
        for i in range(N):
            z_i = layers_recon[0, i:i+1] / scale_factor
            img_i = vae.decode(z_i).sample[0]
            layers_recon_decoded.append(img_i)

        # ========================================
        # 4. Compute optimal assignment
        # ========================================
        cost = torch.zeros(N, N)
        for i in range(N):
            for j in range(n_valid):
                cost[i, j] = F.mse_loss(layers_recon[0, i], z_layers[0, j]).item()
            for j in range(n_valid, N):
                cost[i, j] = 1e8 if i < n_valid else 0

        pred_idx, gt_idx = linear_sum_assignment(cost.numpy())

        print(f"\n    Optimal Assignment:")
        for p, g in zip(pred_idx, gt_idx):
            if g < n_valid:
                mse = cost[p, g].item()
                marker = " ✓" if p == g else " ✗ (ORDER MISMATCH!)"
                print(f"      Output[{p}] → GT[{g}]: MSE={mse:.4f}{marker}")

        # Count mismatches
        mismatches = sum(1 for p, g in zip(pred_idx, gt_idx) if g < n_valid and p != g)
        if mismatches > 0:
            print(f"    ⚠️  {mismatches}/{n_valid} layers have ORDER MISMATCH!")
        else:
            print(f"    ✓ All layers in correct order")

        # ========================================
        # 5. Visualize
        # ========================================
        # 5a. Ground truth layers
        fig, axes = plt.subplots(1, N, figsize=(3*N, 3))
        for i in range(N):
            img = layers_rgb[0, i].cpu().permute(1, 2, 0).numpy()
            img = (img + 1) / 2  # [-1, 1] → [0, 1]
            axes[i].imshow(img.clip(0, 1))
            axes[i].set_title(f'GT Layer {i}' if i < n_valid else 'Padding')
            axes[i].axis('off')
        plt.suptitle('Ground Truth Layers')
        plt.tight_layout()
        plt.savefig(os.path.join(sample_dir, '1_gt_layers.png'), dpi=150)
        plt.close()

        # 5b. Merged image from input projection
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))

        # Alpha blend GT for comparison
        layers_01 = (layers + 1) / 2  # [-1, 1] → [0, 1]
        merged_gt = torch.ones(1, 3, H, W, device=device)
        for i in range(n_valid):
            rgb = layers_01[0, i, :3]
            alpha = layers_01[0, i, 3:4]
            merged_gt[0] = alpha * rgb + (1 - alpha) * merged_gt[0]

        img_gt = merged_gt[0].cpu().permute(1, 2, 0).numpy()
        axes[0].imshow(img_gt.clip(0, 1))
        axes[0].set_title('GT Merged (Alpha Blend)')
        axes[0].axis('off')

        img_pred = merged_decoded[0].cpu().permute(1, 2, 0).numpy()
        img_pred = (img_pred + 1) / 2
        axes[1].imshow(img_pred.clip(0, 1))
        axes[1].set_title('Input Projection Output\n(VAE Decoded)')
        axes[1].axis('off')

        plt.suptitle('Merged Image Comparison')
        plt.tight_layout()
        plt.savefig(os.path.join(sample_dir, '2_merged_comparison.png'), dpi=150)
        plt.close()

        # 5c. Reconstructed layers (with optimal assignment info)
        fig, axes = plt.subplots(2, N, figsize=(3*N, 6))

        for i in range(N):
            # GT
            img_gt = layers_rgb[0, i].cpu().permute(1, 2, 0).numpy()
            img_gt = (img_gt + 1) / 2
            axes[0, i].imshow(img_gt.clip(0, 1))
            axes[0, i].set_title(f'GT[{i}]' if i < n_valid else 'Pad')
            axes[0, i].axis('off')

            # Reconstructed (using optimal assignment)
            if i < n_valid:
                # Find which output matches this GT
                matched_pred = None
                for p, g in zip(pred_idx, gt_idx):
                    if g == i:
                        matched_pred = p
                        break

                if matched_pred is not None:
                    img_recon = layers_recon_decoded[matched_pred].cpu().permute(1, 2, 0).numpy()
                    img_recon = (img_recon + 1) / 2
                    axes[1, i].imshow(img_recon.clip(0, 1))
                    order_ok = "✓" if matched_pred == i else f"✗ (from Out[{matched_pred}])"
                    axes[1, i].set_title(f'Recon {order_ok}')
            else:
                axes[1, i].imshow(torch.zeros(H, W, 3).numpy())
                axes[1, i].set_title('Pad')
            axes[1, i].axis('off')

        plt.suptitle('Layer Reconstruction (Optimal Assignment)')
        plt.tight_layout()
        plt.savefig(os.path.join(sample_dir, '3_layers_reconstruction.png'), dpi=150)
        plt.close()

        # 5d. Direct comparison (same index, no assignment)
        fig, axes = plt.subplots(2, N, figsize=(3*N, 6))

        for i in range(N):
            # GT
            img_gt = layers_rgb[0, i].cpu().permute(1, 2, 0).numpy()
            img_gt = (img_gt + 1) / 2
            axes[0, i].imshow(img_gt.clip(0, 1))
            axes[0, i].set_title(f'GT[{i}]' if i < n_valid else 'Pad')
            axes[0, i].axis('off')

            # Reconstructed (same index)
            img_recon = layers_recon_decoded[i].cpu().permute(1, 2, 0).numpy()
            img_recon = (img_recon + 1) / 2
            axes[1, i].imshow(img_recon.clip(0, 1))
            mse = F.mse_loss(layers_recon[0, i], z_layers[0, i]).item() if i < n_valid else 0
            axes[1, i].set_title(f'Out[{i}] MSE={mse:.3f}')
            axes[1, i].axis('off')

        plt.suptitle('Direct Comparison (Same Index - No Assignment)')
        plt.tight_layout()
        plt.savefig(os.path.join(sample_dir, '4_direct_comparison.png'), dpi=150)
        plt.close()

        print(f"    ✓ Saved to {sample_dir}")

    print("\n" + "="*60)
    print("Test complete!")
    print(f"Results: {args.output_dir}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_roots', type=str, nargs='+', default=['../data/mulan_coco'])
    parser.add_argument('--output_dir', type=str, default='output/projection_test')
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--max_layers', type=int, default=6)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--vae_path', type=str, default='PixArt-alpha/sd-vae-ft-ema')
    parser.add_argument('--max_samples', type=int, default=50, help='Max samples to load (for fast testing)')
    args = parser.parse_args()

    test_projection(args)


if __name__ == '__main__':
    main()
