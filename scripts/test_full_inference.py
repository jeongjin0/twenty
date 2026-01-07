"""
Full Inference Test - Tests actual denoising pipeline

This tests the REAL model behavior:
1. Take clean layers from dataset
2. Add noise to one layer (masked layer)
3. Run through full model: input_proj → PixArt → output_proj
4. Use DDIM to denoise
5. Compare result with ground truth

This is different from test_projection.py which only tests input_proj → output_proj
"""

import argparse
import os
import sys
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusers.models import AutoencoderKL
from diffusion.model.nets.PixArt_layer_inpainting import PixArtLayerInpainting
from diffusion.model.nets.PixArt import PixArt_XL_2
from diffusion.model.t5 import T5Embedder
from diffusion.data.multilayer_builder import build_mulan_dataloader
from diffusion import IDDPM


@torch.no_grad()
def ddim_sample_one_step(model, x_t, t, t_prev, y, mask, layer_mask, alphas_cumprod):
    """
    One step of DDIM sampling
    """
    alpha_t = alphas_cumprod[t]
    alpha_prev = alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0)

    # Get noise prediction from model
    t_tensor = torch.tensor([t], device=x_t.device, dtype=torch.long)
    noise_pred = model(x_t, layer_mask, t_tensor, y, mask=mask)

    # DDIM update
    # x0_pred = (x_t - sqrt(1-alpha_t) * noise_pred) / sqrt(alpha_t)
    sqrt_alpha_t = torch.sqrt(alpha_t)
    sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)

    x0_pred = (x_t - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t

    # Clip x0_pred
    x0_pred = torch.clamp(x0_pred, -3.0, 3.0)

    # Direction pointing to x_t
    sqrt_alpha_prev = torch.sqrt(alpha_prev)
    sqrt_one_minus_alpha_prev = torch.sqrt(1 - alpha_prev)

    # x_{t-1} = sqrt(alpha_{t-1}) * x0_pred + sqrt(1 - alpha_{t-1}) * noise_pred
    x_prev = sqrt_alpha_prev * x0_pred + sqrt_one_minus_alpha_prev * noise_pred

    return x_prev, x0_pred, noise_pred


@torch.no_grad()
def test_full_inference(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 60)
    print("Full Inference Test")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Steps: {args.steps}")
    print("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)

    # ========================================
    # Load Model
    # ========================================
    print("\n[1/5] Loading model...")

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
    state_dict = ckpt.get('state_dict_ema', ckpt.get('state_dict', ckpt))
    model.load_state_dict(state_dict, strict=False)
    print(f"  ✓ Model loaded")

    # ========================================
    # Load VAE & T5
    # ========================================
    print("\n[2/5] Loading VAE and T5...")
    vae = AutoencoderKL.from_pretrained(args.vae_path).to(device).eval()

    t5 = T5Embedder(device=device, local_cache=True, cache_dir=args.t5_path)
    print(f"  ✓ VAE and T5 loaded")

    # ========================================
    # Setup Diffusion
    # ========================================
    print("\n[3/5] Setting up diffusion...")
    diffusion = IDDPM(str(1000))
    alphas_cumprod = torch.tensor(diffusion.alphas_cumprod, device=device)
    print(f"  ✓ Diffusion ready")

    # ========================================
    # Load Dataset
    # ========================================
    print("\n[4/5] Loading dataset...")
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
    # Run Inference Test
    # ========================================
    print("\n[5/5] Running inference test...")

    scale_factor = 0.18215
    timesteps = list(range(999, -1, -1000 // args.steps))
    if timesteps[-1] != 0:
        timesteps.append(0)

    print(f"  Timesteps: {timesteps[:5]}...{timesteps[-3:]}")

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= args.num_samples:
            break

        layers, captions, num_layers, image_ids = batch
        layers = layers.to(device)

        B, N, C, H, W = layers.shape
        n_valid = num_layers[0].item()
        image_id = image_ids[0]
        h, w = H // 8, W // 8

        print(f"\n  --- Sample {batch_idx}: {image_id} ({n_valid} layers) ---")

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
        # 2. Select masked layer and add noise
        # ========================================
        # Use layer 1 (first foreground) as target
        masked_idx = min(1, n_valid - 1)
        layer_mask = torch.zeros(1, N, device=device)
        layer_mask[0, masked_idx] = 1.0

        # Get text embedding
        caption = captions[0][masked_idx] if captions[0][masked_idx] else "a layer"
        caption_embs, emb_masks = t5.get_text_embeddings([caption])
        caption_embs = caption_embs.to(device)
        emb_masks = emb_masks.to(device)

        # Add noise to masked layer at t=999
        t_start = 999
        noise = torch.randn_like(z_layers[:, masked_idx])
        alpha_t = alphas_cumprod[t_start]
        sqrt_alpha = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha = torch.sqrt(1 - alpha_t)

        z_noisy = z_layers.clone()
        z_noisy[:, masked_idx] = sqrt_alpha * z_layers[:, masked_idx] + sqrt_one_minus_alpha * noise

        print(f"    Masked layer: {masked_idx}")
        print(f"    Caption: '{caption}'")
        print(f"    Noise added at t={t_start}")
        print(f"    GT layer stats: mean={z_layers[0, masked_idx].mean():.4f}, std={z_layers[0, masked_idx].std():.4f}")
        print(f"    Noisy layer stats: mean={z_noisy[0, masked_idx].mean():.4f}, std={z_noisy[0, masked_idx].std():.4f}")

        # ========================================
        # 3. DDIM Sampling
        # ========================================
        print(f"\n    Running DDIM sampling ({args.steps} steps)...")

        x_t = z_noisy.clone()

        # Track statistics during sampling
        step_stats = []

        for i in range(len(timesteps) - 1):
            t = timesteps[i]
            t_prev = timesteps[i + 1]

            x_t, x0_pred, noise_pred = ddim_sample_one_step(
                model, x_t, t, t_prev, caption_embs, emb_masks, layer_mask, alphas_cumprod
            )

            # Update only masked layer, keep visible layers fixed
            for layer_idx in range(N):
                if layer_mask[0, layer_idx] == 0:  # Visible layer
                    x_t[:, layer_idx] = z_layers[:, layer_idx]

            if i == 0 or i == len(timesteps) - 2:
                masked_x0 = x0_pred[0, masked_idx]
                masked_noise = noise_pred[0, masked_idx]
                stats = {
                    'step': i,
                    't': t,
                    'x0_mean': masked_x0.mean().item(),
                    'x0_std': masked_x0.std().item(),
                    'noise_mean': masked_noise.mean().item(),
                    'noise_std': masked_noise.std().item(),
                }
                step_stats.append(stats)
                print(f"      Step {i} (t={t}): x0_pred mean={stats['x0_mean']:.4f}, std={stats['x0_std']:.4f}, noise_pred std={stats['noise_std']:.4f}")

        # Final result
        z_denoised = x_t

        # ========================================
        # 4. Compute metrics
        # ========================================
        gt_layer = z_layers[0, masked_idx]
        pred_layer = z_denoised[0, masked_idx]

        mse = F.mse_loss(pred_layer, gt_layer).item()
        mae = (pred_layer - gt_layer).abs().mean().item()

        print(f"\n    === Results ===")
        print(f"    MSE: {mse:.6f}")
        print(f"    MAE: {mae:.6f}")
        print(f"    GT: mean={gt_layer.mean():.4f}, std={gt_layer.std():.4f}")
        print(f"    Pred: mean={pred_layer.mean():.4f}, std={pred_layer.std():.4f}")

        # ========================================
        # 5. Decode and visualize
        # ========================================
        # Decode GT
        gt_decoded = vae.decode(gt_layer.unsqueeze(0) / scale_factor).sample[0]

        # Decode prediction
        pred_decoded = vae.decode(pred_layer.unsqueeze(0) / scale_factor).sample[0]

        # Decode noisy input
        noisy_decoded = vae.decode(z_noisy[0, masked_idx].unsqueeze(0) / scale_factor).sample[0]

        # Create visualization
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        # GT layer
        img = gt_decoded.cpu().permute(1, 2, 0).numpy()
        img = (img + 1) / 2
        axes[0].imshow(img.clip(0, 1))
        axes[0].set_title(f'Ground Truth\n(Layer {masked_idx})')
        axes[0].axis('off')

        # Noisy input
        img = noisy_decoded.cpu().permute(1, 2, 0).numpy()
        img = (img + 1) / 2
        axes[1].imshow(img.clip(0, 1))
        axes[1].set_title(f'Noisy Input\n(t={t_start})')
        axes[1].axis('off')

        # Denoised prediction
        img = pred_decoded.cpu().permute(1, 2, 0).numpy()
        img = (img + 1) / 2
        axes[2].imshow(img.clip(0, 1))
        axes[2].set_title(f'Denoised\nMSE={mse:.4f}')
        axes[2].axis('off')

        # Difference
        diff = (pred_decoded - gt_decoded).abs().mean(dim=0).cpu().numpy()
        axes[3].imshow(diff, cmap='hot')
        axes[3].set_title(f'|Pred - GT|\nMAE={mae:.4f}')
        axes[3].axis('off')

        plt.suptitle(f'Full Inference Test - {image_id}', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(sample_dir, 'inference_result.png'), dpi=150)
        plt.close()

        # Save all layers comparison
        fig, axes = plt.subplots(2, N, figsize=(3 * N, 6))

        for i in range(N):
            # GT
            if i < n_valid:
                img = vae.decode(z_layers[0, i].unsqueeze(0) / scale_factor).sample[0]
                img = img.cpu().permute(1, 2, 0).numpy()
                img = (img + 1) / 2
                axes[0, i].imshow(img.clip(0, 1))
                marker = " [MASKED]" if i == masked_idx else ""
                axes[0, i].set_title(f'GT Layer {i}{marker}')
            else:
                axes[0, i].imshow(torch.zeros(H, W, 3).numpy())
                axes[0, i].set_title('Padding')
            axes[0, i].axis('off')

            # Denoised
            if i < n_valid:
                img = vae.decode(z_denoised[0, i].unsqueeze(0) / scale_factor).sample[0]
                img = img.cpu().permute(1, 2, 0).numpy()
                img = (img + 1) / 2
                axes[1, i].imshow(img.clip(0, 1))
                if i == masked_idx:
                    axes[1, i].set_title(f'Denoised MSE={mse:.4f}')
                else:
                    layer_mse = F.mse_loss(z_denoised[0, i], z_layers[0, i]).item()
                    axes[1, i].set_title(f'Kept (MSE={layer_mse:.6f})')
            else:
                axes[1, i].imshow(torch.zeros(H, W, 3).numpy())
                axes[1, i].set_title('Padding')
            axes[1, i].axis('off')

        plt.suptitle('All Layers Comparison', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(sample_dir, 'all_layers.png'), dpi=150)
        plt.close()

        print(f"    ✓ Saved to {sample_dir}")

    print("\n" + "=" * 60)
    print("Test complete!")
    print(f"Results: {args.output_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_roots', type=str, nargs='+', default=['../data/mulan_coco'])
    parser.add_argument('--output_dir', type=str, default='output/full_inference_test')
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--max_layers', type=int, default=6)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--vae_path', type=str, default='PixArt-alpha/sd-vae-ft-ema')
    parser.add_argument('--t5_path', type=str, default='PixArt-alpha')
    parser.add_argument('--max_samples', type=int, default=50)
    args = parser.parse_args()

    test_full_inference(args)


if __name__ == '__main__':
    main()
