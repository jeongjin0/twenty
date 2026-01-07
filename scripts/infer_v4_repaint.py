"""
Inference V4 - Repaint-style with visible layer renoising
Fix mixed timestep issue by adding noise to visible layers at each step
"""

import argparse
import os
import sys
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusion.model.nets.PixArt_layer_inpainting import PixArtLayerInpainting
from diffusion.model.nets.PixArt import PixArt_XL_2
from diffusion.model.t5 import T5Embedder
from diffusion import IDDPM
from diffusion.data.multilayer_builder import build_mulan_dataloader


@torch.no_grad()
def ddim_sample_repaint(
    model,
    clean_layers,
    layer_mask,
    prompt,
    text_encoder,
    diffusion,
    alphas_cumprod,
    cfg_scale=1.0,
    steps=50,
    device='cuda'
):
    """
    DDIM sampling with Repaint-style renoising for visible layers

    Key fix: Add noise to visible layers at each timestep to match
    the noise level of masked layer, preventing mixed-timestep confusion
    """
    B, N, C, h, w = clean_layers.shape

    # Expand layer mask for broadcasting
    layer_mask_expanded = layer_mask[:, :, None, None, None].float()  # (B, N, 1, 1, 1)

    # Encode text
    caption_embs, emb_masks = text_encoder.get_text_embeddings([prompt])
    y = caption_embs.float()[:, None].to(device)
    y_mask = emb_masks.to(device)
    del caption_embs, emb_masks

    # DDIM timesteps
    timesteps = torch.linspace(999, 0, steps + 1, dtype=torch.long, device=device)

    # Initialize: ALL layers start with noise at t=999
    # (This matches training where all layers receive q_sample at same timestep)
    x_t = torch.randn(B, N, C, h, w, device=device)
    masked_idx = torch.where(layer_mask[0] == 1)[0][0].item()

    print(f"\n  Masked layer: {masked_idx}")
    print(f"  Initial x_t (all noisy at t=999): mean={x_t[0, masked_idx].mean().item():.4f}")

    # Sampling loop
    for i in tqdm(range(steps), desc="Sampling (Repaint)", leave=False):
        t = timesteps[i].item()
        t_next = timesteps[i + 1].item()

        t_batch = torch.full((B,), t, device=device, dtype=torch.long)

        # Model prediction (all layers at same timestep t)
        noise_pred_raw = model(x_t, layer_mask, t_batch, y, mask=y_mask)

        # Debug first step
        if i == 0:
            print(f"  Step 0 at t={t}:")
            print(f"    Masked layer {masked_idx} noise pred: std={noise_pred_raw[0, masked_idx].std().item():.4f}")
            vis_idx = 0 if masked_idx != 0 else 2
            print(f"    Visible layer {vis_idx} noise pred: std={noise_pred_raw[0, vis_idx].std().item():.4f}")

        # Zero out visible layer predictions (as trained)
        noise_pred = torch.zeros_like(noise_pred_raw)
        for b in range(B):
            for i_layer in range(N):
                if layer_mask[b, i_layer] == 1:
                    noise_pred[b, i_layer] = noise_pred_raw[b, i_layer]

        # DDIM update
        alpha_t = alphas_cumprod[t]
        alpha_next = alphas_cumprod[t_next] if t_next >= 0 else torch.tensor(1.0, device=device)

        # Predict x0
        x0_pred = (x_t - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
        x0_pred = torch.clamp(x0_pred, -3.0, 3.0)

        # Replace visible layers' x0 prediction with ground truth
        x0_pred = x0_pred * layer_mask_expanded + clean_layers * (1 - layer_mask_expanded)

        # Next step (deterministic DDIM)
        dir_xt = torch.sqrt(1 - alpha_next) * noise_pred
        x_next = torch.sqrt(alpha_next) * x0_pred + dir_xt

        # CRITICAL: Renoise visible layers to match t_next
        # This keeps all layers at the same noise level
        if t_next > 0:
            # Add noise to visible layers: x_t = sqrt(alpha) * x0 + sqrt(1-alpha) * noise
            visible_noise = torch.randn_like(clean_layers, device=device)
            x_visible_t_next = (torch.sqrt(alpha_next) * clean_layers +
                               torch.sqrt(1 - alpha_next) * visible_noise)
            # Combine: masked from DDIM, visible from renoising
            x_t = x_next * layer_mask_expanded + x_visible_t_next * (1 - layer_mask_expanded)
        else:
            # Final step: no renoising, use clean for visible
            x_t = x_next * layer_mask_expanded + clean_layers * (1 - layer_mask_expanded)

        # Debug middle step
        if i == steps // 2:
            print(f"  Step {i} at t={t_next}:")
            print(f"    Masked layer: std={x_t[0, masked_idx].std().item():.4f}")
            print(f"    Visible layer: std={x_t[0, vis_idx].std().item():.4f}")

    print(f"  Final x_t[masked]: mean={x_t[0, masked_idx].mean().item():.4f}, std={x_t[0, masked_idx].std().item():.4f}")

    return x_t


def main():
    parser = argparse.ArgumentParser(description='Inpainting V4 - Repaint-style Inference')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    parser.add_argument('--data_roots', type=str, nargs='+', required=True, help='Dataset directories')
    parser.add_argument('--output_dir', type=str, default='output/inference_v4_repaint', help='Output directory')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples')
    parser.add_argument('--cfg_scale', type=float, default=1.0, help='CFG scale')
    parser.add_argument('--steps', type=int, default=50, help='Sampling steps')
    parser.add_argument('--max_layers', type=int, default=6, help='Max layers')
    parser.add_argument('--image_size', type=int, default=256, help='Image size')
    parser.add_argument('--vae_path', type=str, default='PixArt-alpha/sd-vae-ft-ema')
    parser.add_argument('--t5_path', type=str, default='PixArt-alpha')
    parser.add_argument('--max_samples', type=int, default=50, help='Max samples to load (for fast testing)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("="*60)
    print("Inpainting V4 - Repaint-style Inference")
    print("="*60)
    print("Fix: Renoise visible layers to match masked layer timestep")
    print("="*60)

    # Load models
    print("\n[1/3] Loading models...")

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

    vae = AutoencoderKL.from_pretrained(args.vae_path).to(device).eval()
    print(f"  ✓ VAE loaded")

    t5 = T5Embedder(device=device, local_cache=True, cache_dir=args.t5_path, torch_dtype=torch.float16)
    print(f"  ✓ T5 loaded")

    diffusion = IDDPM(str(1000))
    alphas_cumprod = torch.from_numpy(diffusion.alphas_cumprod).float().to(device)
    print(f"  ✓ Diffusion ready")

    # Load dataset
    print("\n[2/3] Loading dataset...")
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

    # Run inference
    print("\n[3/3] Running inference...")
    os.makedirs(args.output_dir, exist_ok=True)

    sample_count = 0
    scale_factor = 0.18215

    for batch_idx, batch in enumerate(dataloader):
        if sample_count >= args.num_samples:
            break

        layers, captions, num_layers, image_ids = batch
        layers = layers.to(device)
        B, N, H, W = layers.shape[0], layers.shape[1], layers.shape[3], layers.shape[4]
        h, w = H // 8, W // 8

        # Encode to VAE latent
        layers_rgb = layers[:, :, :3, :, :]
        layers_flat = layers_rgb.reshape(B * N, 3, H, W)
        z_flat = vae.encode(layers_flat).latent_dist.mode() * scale_factor
        z_clean = z_flat.reshape(B, N, 4, h, w)
        del layers_rgb, layers_flat, z_flat

        for b in range(B):
            if sample_count >= args.num_samples:
                break

            n_valid = num_layers[b].item()
            image_id = image_ids[b]

            # Test first 2 layers
            for masked_idx in range(1, min(n_valid, 3)):
                if sample_count >= args.num_samples:
                    break

                print(f"\n[Sample {sample_count + 1}/{args.num_samples}] {image_id}, layer {masked_idx}")

                layer_mask = torch.zeros(1, args.max_layers, device=device)
                layer_mask[0, masked_idx] = 1

                clean_layers = z_clean[b:b+1].clone()
                prompt = captions[b][masked_idx]

                # Generate with Repaint-style renoising
                x_t = ddim_sample_repaint(
                    model=model,
                    clean_layers=clean_layers,
                    layer_mask=layer_mask,
                    prompt=prompt,
                    text_encoder=t5,
                    diffusion=diffusion,
                    alphas_cumprod=alphas_cumprod,
                    cfg_scale=args.cfg_scale,
                    steps=args.steps,
                    device=device
                )

                # Decode
                z_generated = x_t[0, masked_idx:masked_idx+1] / scale_factor
                img_generated = vae.decode(z_generated).sample[0]

                z_gt = z_clean[b, masked_idx:masked_idx+1] / scale_factor
                img_gt = vae.decode(z_gt).sample[0]

                # Save
                sample_dir = os.path.join(args.output_dir, f"sample_{sample_count:03d}")
                os.makedirs(sample_dir, exist_ok=True)

                save_image(img_generated, os.path.join(sample_dir, 'generated.png'),
                          normalize=True, value_range=(-1, 1))
                save_image(img_gt, os.path.join(sample_dir, 'ground_truth.png'),
                          normalize=True, value_range=(-1, 1))

                comparison = torch.stack([img_generated, img_gt], dim=0)
                save_image(comparison, os.path.join(sample_dir, 'comparison.png'),
                          nrow=2, normalize=True, value_range=(-1, 1))

                with open(os.path.join(sample_dir, 'info.txt'), 'w') as f:
                    f.write(f"Image ID: {image_id}\n")
                    f.write(f"Masked layer: {masked_idx}\n")
                    f.write(f"Prompt: {prompt}\n")
                    f.write(f"Method: Repaint-style (visible layer renoising)\n")

                print(f"  ✓ Saved to: {sample_dir}")
                sample_count += 1

    print(f"\n✓ Complete! {sample_count} samples saved to {args.output_dir}")


if __name__ == '__main__':
    main()
