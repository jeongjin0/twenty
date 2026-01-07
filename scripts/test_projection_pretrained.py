"""
Projection Test - Pretrained Projections Only

Tests the projection weights BEFORE main training to verify:
1. Does output projection maintain layer order?
2. How well does input projection merge layers?
3. How well does output projection decompose back to layers?
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusion.model.nets.projection_unet import ProjectionAutoencoder
from diffusion.data.multilayer_builder import build_mulan_dataloader
from scipy.optimize import linear_sum_assignment


class PretrainProjectionModel(nn.Module):
    """Same model structure as used in pretrain_projections.py"""
    def __init__(self, max_layers=6):
        super().__init__()
        self.max_layers = max_layers
        self.autoencoder = ProjectionAutoencoder(in_channels=24, out_channels=24)

    def forward(self, layers):
        """
        Args:
            layers: (B, N, 4, h, w) - layer latents

        Returns:
            merged: (B, 4, h, w) - merged latent
            layers_recon: (B, N, 4, h, w) - reconstructed layers
        """
        B, N, C, h, w = layers.shape
        layers_flat = layers.reshape(B, N * C, h, w)  # (B, 24, h, w)

        # Encode (merge)
        merged, skips = self.autoencoder.encode(layers_flat)  # (B, 4, h, w)

        # Decode (decompose)
        layers_recon_flat = self.autoencoder.decode(merged, skips)  # (B, 24, h, w)
        layers_recon = layers_recon_flat.reshape(B, N, C, h, w)

        return merged, layers_recon


def alpha_blend_layers(layers, num_layers):
    """Alpha blend RGBA layers to create merged image"""
    B, N, _, H, W = layers.shape
    device = layers.device

    merged = torch.ones(B, 3, H, W, device=device)

    for b in range(B):
        n_valid = num_layers[b].item()
        for i in range(n_valid):
            layer_rgba = layers[b, i]
            rgb = layer_rgba[:3]
            alpha = layer_rgba[3:4]
            merged[b] = alpha * rgb + (1 - alpha) * merged[b]

    return merged


@torch.no_grad()
def test_pretrained_projection(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("="*60)
    print("Projection Test - PRETRAINED (before main training)")
    print("="*60)

    # ========================================
    # Load Pretrained Projection Model
    # ========================================
    print("\n[1/4] Loading pretrained projection model...")

    model = PretrainProjectionModel(max_layers=args.max_layers).to(device).eval()

    ckpt = torch.load(args.checkpoint, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)

    # Load weights
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"  ✓ Loaded from {args.checkpoint}")
    if missing:
        print(f"  ⚠ Missing keys: {len(missing)}")
    if unexpected:
        print(f"  ⚠ Unexpected keys: {len(unexpected)}")

    # Check weights
    enc_weight = model.autoencoder.enc_conv1.weight
    dec_weight = model.autoencoder.dec_conv1.weight
    print(f"\n  Encoder conv1: mean={enc_weight.mean():.6f}, std={enc_weight.std():.6f}")
    print(f"  Decoder conv1: mean={dec_weight.mean():.6f}, std={dec_weight.std():.6f}")

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
        caption_type='blip2'
    )
    print(f"  ✓ Dataset loaded")

    # ========================================
    # Test
    # ========================================
    print("\n[4/4] Testing projections...")

    os.makedirs(args.output_dir, exist_ok=True)
    scale_factor = 0.18215

    total_order_mismatches = 0
    total_valid_layers = 0

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= args.num_samples:
            break

        layers, captions, num_layers, image_ids = batch
        layers = layers.to(device)

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
        layers_rgb = layers[:, :, :3, :, :]
        layers_flat = layers_rgb.reshape(N, 3, H, W)
        z_layers = vae.encode(layers_flat).latent_dist.mode() * scale_factor
        z_layers = z_layers.unsqueeze(0)  # (1, N, 4, h, w)

        # ========================================
        # 2. Forward through projection model
        # ========================================
        merged_latent, layers_recon = model(z_layers)

        print(f"    Merged latent: mean={merged_latent.mean():.4f}, std={merged_latent.std():.4f}")

        # ========================================
        # 3. Compute optimal assignment
        # ========================================
        cost = torch.zeros(N, N)
        for i in range(N):
            for j in range(n_valid):
                cost[i, j] = F.mse_loss(layers_recon[0, i], z_layers[0, j]).item()
            for j in range(n_valid, N):
                cost[i, j] = 1e8 if i < n_valid else 0

        pred_idx, gt_idx = linear_sum_assignment(cost.numpy())

        print(f"\n    Optimal Assignment:")
        mismatches = 0
        for p, g in zip(pred_idx, gt_idx):
            if g < n_valid:
                mse = cost[p, g].item()
                if p == g:
                    print(f"      Output[{p}] → GT[{g}]: MSE={mse:.4f} ✓")
                else:
                    print(f"      Output[{p}] → GT[{g}]: MSE={mse:.4f} ✗ ORDER MISMATCH!")
                    mismatches += 1

        total_order_mismatches += mismatches
        total_valid_layers += n_valid

        if mismatches > 0:
            print(f"    ⚠️  {mismatches}/{n_valid} layers have ORDER MISMATCH!")
        else:
            print(f"    ✓ All layers in correct order")

        # ========================================
        # 4. Decode and visualize
        # ========================================
        # Decode merged
        merged_decoded = vae.decode(merged_latent / scale_factor).sample

        # Decode reconstructed layers
        layers_recon_decoded = []
        for i in range(N):
            z_i = layers_recon[0, i:i+1] / scale_factor
            img_i = vae.decode(z_i).sample[0]
            layers_recon_decoded.append(img_i)

        # ========================================
        # 5. Create visualizations
        # ========================================
        # 5a. GT layers
        fig, axes = plt.subplots(1, N, figsize=(3*N, 3))
        for i in range(N):
            img = layers_rgb[0, i].cpu().permute(1, 2, 0).numpy()
            img = (img + 1) / 2
            axes[i].imshow(img.clip(0, 1))
            axes[i].set_title(f'GT[{i}]' if i < n_valid else 'Pad')
            axes[i].axis('off')
        plt.suptitle('Ground Truth Layers')
        plt.tight_layout()
        plt.savefig(os.path.join(sample_dir, '1_gt_layers.png'), dpi=150)
        plt.close()

        # 5b. Merged comparison
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))

        # Alpha blend GT
        layers_01 = (layers + 1) / 2
        merged_gt = torch.ones(1, 3, H, W, device=device)
        for i in range(n_valid):
            rgb = layers_01[0, i, :3]
            alpha = layers_01[0, i, 3:4]
            merged_gt[0] = alpha * rgb + (1 - alpha) * merged_gt[0]

        img_gt = merged_gt[0].cpu().permute(1, 2, 0).numpy()
        axes[0].imshow(img_gt.clip(0, 1))
        axes[0].set_title('GT Merged\n(Alpha Blend)')
        axes[0].axis('off')

        img_pred = merged_decoded[0].cpu().permute(1, 2, 0).numpy()
        img_pred = (img_pred + 1) / 2
        axes[1].imshow(img_pred.clip(0, 1))
        axes[1].set_title('Projection Merged\n(VAE Decoded)')
        axes[1].axis('off')

        plt.suptitle('Merge Comparison')
        plt.tight_layout()
        plt.savefig(os.path.join(sample_dir, '2_merged.png'), dpi=150)
        plt.close()

        # 5c. Direct comparison (same index - no assignment)
        fig, axes = plt.subplots(2, N, figsize=(3*N, 6))

        for i in range(N):
            # GT
            img_gt = layers_rgb[0, i].cpu().permute(1, 2, 0).numpy()
            img_gt = (img_gt + 1) / 2
            axes[0, i].imshow(img_gt.clip(0, 1))
            axes[0, i].set_title(f'GT[{i}]')
            axes[0, i].axis('off')

            # Recon (same index)
            img_recon = layers_recon_decoded[i].cpu().permute(1, 2, 0).numpy()
            img_recon = (img_recon + 1) / 2
            axes[1, i].imshow(img_recon.clip(0, 1))
            mse = cost[i, i].item() if i < n_valid else 0
            axes[1, i].set_title(f'Out[{i}] MSE={mse:.3f}')
            axes[1, i].axis('off')

        plt.suptitle('Direct Comparison (Same Index - How Training Sees It)')
        plt.tight_layout()
        plt.savefig(os.path.join(sample_dir, '3_direct_comparison.png'), dpi=150)
        plt.close()

        # 5d. Optimal assignment comparison
        fig, axes = plt.subplots(2, N, figsize=(3*N, 6))

        for i in range(N):
            # GT
            img_gt = layers_rgb[0, i].cpu().permute(1, 2, 0).numpy()
            img_gt = (img_gt + 1) / 2
            axes[0, i].imshow(img_gt.clip(0, 1))
            axes[0, i].set_title(f'GT[{i}]')
            axes[0, i].axis('off')

            # Matched recon
            if i < n_valid:
                matched_pred = None
                for p, g in zip(pred_idx, gt_idx):
                    if g == i:
                        matched_pred = p
                        break
                if matched_pred is not None:
                    img_recon = layers_recon_decoded[matched_pred].cpu().permute(1, 2, 0).numpy()
                    img_recon = (img_recon + 1) / 2
                    axes[1, i].imshow(img_recon.clip(0, 1))
                    order_str = "✓" if matched_pred == i else f"from Out[{matched_pred}]"
                    axes[1, i].set_title(f'Matched {order_str}')
            else:
                axes[1, i].imshow(torch.zeros(H, W, 3).numpy())
                axes[1, i].set_title('Pad')
            axes[1, i].axis('off')

        plt.suptitle('Optimal Assignment (Best Matching)')
        plt.tight_layout()
        plt.savefig(os.path.join(sample_dir, '4_optimal_assignment.png'), dpi=150)
        plt.close()

        print(f"    ✓ Saved to {sample_dir}")

    # ========================================
    # Summary
    # ========================================
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    mismatch_rate = total_order_mismatches / total_valid_layers * 100 if total_valid_layers > 0 else 0
    print(f"Total order mismatches: {total_order_mismatches}/{total_valid_layers} ({mismatch_rate:.1f}%)")

    if mismatch_rate > 20:
        print("\n⚠️  HIGH MISMATCH RATE!")
        print("   Pretrained projection does NOT preserve layer order.")
        print("   This causes problems in main training which assumes order is preserved.")
        print("\n   SOLUTION: Add optimal_assignment_loss to main training,")
        print("   or retrain projections with order preservation.")
    else:
        print("\n✓ Layer order is mostly preserved.")

    print(f"\nResults: {args.output_dir}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_roots', type=str, nargs='+', default=['../data/mulan_coco'])
    parser.add_argument('--output_dir', type=str, default='output/projection_test_pretrained')
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--max_layers', type=int, default=6)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--vae_path', type=str, default='PixArt-alpha/sd-vae-ft-ema')
    args = parser.parse_args()

    test_pretrained_projection(args)


if __name__ == '__main__':
    main()
