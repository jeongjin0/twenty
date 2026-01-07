"""
Noise Prediction Diagnostic Test

Simple test to check if the model can predict noise correctly:
1. Take clean layer
2. Add KNOWN noise
3. Ask model to predict that noise
4. Compare predicted vs actual noise

This will tell us if the fundamental noise prediction is working.
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
def diagnose_noise_prediction(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 60)
    print("Noise Prediction Diagnostic")
    print("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
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

    ckpt = torch.load(args.checkpoint, map_location='cpu')
    state_dict = ckpt.get('state_dict_ema', ckpt.get('state_dict', ckpt))
    model.load_state_dict(state_dict, strict=False)
    print(f"  ✓ Model loaded")

    # Load VAE & T5
    print("\n[2/4] Loading VAE and T5...")
    vae = AutoencoderKL.from_pretrained(args.vae_path).to(device).eval()
    t5 = T5Embedder(device=device, local_cache=True, cache_dir=args.t5_path)
    print(f"  ✓ Loaded")

    # Setup diffusion
    print("\n[3/4] Setting up diffusion...")
    diffusion = IDDPM(str(1000))
    alphas_cumprod = torch.tensor(diffusion.alphas_cumprod, device=device)
    print(f"  ✓ Ready")

    # Load dataset
    print("\n[4/4] Loading dataset...")
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
    print(f"  ✓ Loaded")

    scale_factor = 0.18215

    # Test at different timesteps
    test_timesteps = [999, 800, 500, 200, 50]

    print("\n" + "=" * 60)
    print("DIAGNOSTIC RESULTS")
    print("=" * 60)

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 1:  # Just one sample
            break

        layers, captions, num_layers, image_ids = batch
        layers = layers.to(device)

        B, N, C, H, W = layers.shape
        n_valid = num_layers[0].item()
        h, w = H // 8, W // 8

        # Encode layers
        layers_rgb = layers[:, :, :3, :, :]
        layers_flat = layers_rgb.reshape(N, 3, H, W)
        z_layers = vae.encode(layers_flat).latent_dist.mode() * scale_factor
        z_layers = z_layers.unsqueeze(0)

        # Setup mask (layer 1 is masked)
        masked_idx = min(1, n_valid - 1)
        layer_mask = torch.zeros(1, N, device=device)
        layer_mask[0, masked_idx] = 1.0

        # Get text embedding
        caption = captions[0][masked_idx] if captions[0][masked_idx] else "a layer"
        caption_embs, emb_masks = t5.get_text_embeddings([caption])
        caption_embs = caption_embs.to(device)
        emb_masks = emb_masks.to(device)

        print(f"\nSample: {image_ids[0]}")
        print(f"Masked layer: {masked_idx}")
        print(f"GT layer stats: mean={z_layers[0, masked_idx].mean():.4f}, std={z_layers[0, masked_idx].std():.4f}")

        results = []

        for t in test_timesteps:
            # Generate known noise
            torch.manual_seed(42)  # Fixed seed for reproducibility
            noise = torch.randn_like(z_layers[:, masked_idx])

            # Add noise to get x_t
            alpha_t = alphas_cumprod[t]
            sqrt_alpha = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha = torch.sqrt(1 - alpha_t)

            z_noisy = z_layers.clone()
            z_noisy[:, masked_idx] = sqrt_alpha * z_layers[:, masked_idx] + sqrt_one_minus_alpha * noise

            # Model prediction
            t_tensor = torch.tensor([t], device=device, dtype=torch.long)
            noise_pred = model(z_noisy, layer_mask, t_tensor, caption_embs, mask=emb_masks)

            # Get prediction for masked layer only
            pred_masked = noise_pred[0, masked_idx]
            actual_noise = noise[0]

            # Compute metrics
            mse = F.mse_loss(pred_masked, actual_noise).item()
            mae = (pred_masked - actual_noise).abs().mean().item()

            pred_std = pred_masked.std().item()
            actual_std = actual_noise.std().item()

            pred_mean = pred_masked.mean().item()
            actual_mean = actual_noise.mean().item()

            # Correlation
            pred_flat = pred_masked.flatten()
            actual_flat = actual_noise.flatten()
            correlation = torch.corrcoef(torch.stack([pred_flat, actual_flat]))[0, 1].item()

            results.append({
                't': t,
                'mse': mse,
                'mae': mae,
                'pred_std': pred_std,
                'actual_std': actual_std,
                'pred_mean': pred_mean,
                'actual_mean': actual_mean,
                'correlation': correlation,
                'sqrt_alpha': sqrt_alpha.item(),
            })

            print(f"\n  t={t} (sqrt_alpha={sqrt_alpha.item():.4f}):")
            print(f"    Actual noise: mean={actual_mean:.4f}, std={actual_std:.4f}")
            print(f"    Pred noise:   mean={pred_mean:.4f}, std={pred_std:.4f}")
            print(f"    MSE: {mse:.4f}, MAE: {mae:.4f}")
            print(f"    Correlation: {correlation:.4f}")

            if pred_std < actual_std * 0.8:
                print(f"    ⚠️ Predicted std too LOW! ({pred_std:.3f} vs {actual_std:.3f})")
            elif pred_std > actual_std * 1.2:
                print(f"    ⚠️ Predicted std too HIGH! ({pred_std:.3f} vs {actual_std:.3f})")

        # Also check visible layer prediction (should be ~zero)
        print(f"\n  Visible layer noise prediction (should be ~0):")
        for vis_idx in range(n_valid):
            if vis_idx != masked_idx:
                vis_pred = noise_pred[0, vis_idx]
                print(f"    Layer {vis_idx}: mean={vis_pred.mean():.4f}, std={vis_pred.std():.4f}")

        # Visualization
        fig, axes = plt.subplots(2, len(test_timesteps), figsize=(4*len(test_timesteps), 8))

        for i, r in enumerate(results):
            t = r['t']

            # Regenerate for visualization
            torch.manual_seed(42)
            noise = torch.randn_like(z_layers[:, masked_idx])
            alpha_t = alphas_cumprod[t]
            sqrt_alpha = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha = torch.sqrt(1 - alpha_t)
            z_noisy = z_layers.clone()
            z_noisy[:, masked_idx] = sqrt_alpha * z_layers[:, masked_idx] + sqrt_one_minus_alpha * noise

            t_tensor = torch.tensor([t], device=device, dtype=torch.long)
            noise_pred = model(z_noisy, layer_mask, t_tensor, caption_embs, mask=emb_masks)
            pred_masked = noise_pred[0, masked_idx]

            # Show actual vs predicted noise (first channel)
            axes[0, i].imshow(noise[0, 0].cpu().numpy(), cmap='RdBu', vmin=-3, vmax=3)
            axes[0, i].set_title(f't={t}\nActual noise\nstd={r["actual_std"]:.3f}')
            axes[0, i].axis('off')

            axes[1, i].imshow(pred_masked[0].cpu().numpy(), cmap='RdBu', vmin=-3, vmax=3)
            axes[1, i].set_title(f'Predicted noise\nstd={r["pred_std"]:.3f}\ncorr={r["correlation"]:.3f}')
            axes[1, i].axis('off')

        plt.suptitle('Noise Prediction Diagnostic\nTop: Actual noise, Bottom: Predicted noise', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'noise_diagnostic.png'), dpi=150)
        plt.close()

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        avg_correlation = sum(r['correlation'] for r in results) / len(results)
        avg_std_ratio = sum(r['pred_std'] / r['actual_std'] for r in results) / len(results)

        print(f"Average correlation: {avg_correlation:.4f}")
        print(f"Average std ratio (pred/actual): {avg_std_ratio:.4f}")

        if avg_correlation < 0.5:
            print("\n❌ CRITICAL: Low correlation - model is NOT predicting noise correctly!")
        elif avg_correlation < 0.8:
            print("\n⚠️ WARNING: Moderate correlation - noise prediction is partially working")
        else:
            print("\n✓ Good correlation - noise prediction direction is correct")

        if avg_std_ratio < 0.8:
            print("❌ CRITICAL: Predicted noise magnitude too LOW!")
            print("   This causes x0_pred to be wrong, leading to bad sampling.")
        elif avg_std_ratio > 1.2:
            print("⚠️ WARNING: Predicted noise magnitude too HIGH!")
        else:
            print("✓ Noise magnitude is reasonable")

        print(f"\nResults saved to: {args.output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_roots', type=str, nargs='+', default=['../data/mulan_coco'])
    parser.add_argument('--output_dir', type=str, default='output/noise_diagnostic')
    parser.add_argument('--max_layers', type=int, default=6)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--vae_path', type=str, default='PixArt-alpha/sd-vae-ft-ema')
    parser.add_argument('--t5_path', type=str, default='PixArt-alpha')
    parser.add_argument('--max_samples', type=int, default=50)
    args = parser.parse_args()

    diagnose_noise_prediction(args)


if __name__ == '__main__':
    main()
