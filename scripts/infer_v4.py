"""
Inference V4 - Dataset-based Testing
Load samples from MuLan dataset and test layer inpainting with reference layers
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
def ddim_sample(
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
    DDIM sampling for layer inpainting

    CRITICAL: Training used zero-noise prediction for visible layers!
    Must replicate this behavior during inference.
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

    # Initialize: noise at masked position, clean at visible positions
    x_t = clean_layers.clone()
    masked_idx = torch.where(layer_mask[0] == 1)[0][0].item()
    x_t[0, masked_idx] = torch.randn(C, h, w, device=device)

    print(f"\n  Masked layer: {masked_idx}")
    print(f"  Initial noise mean: {x_t[0, masked_idx].mean().item():.4f}, std: {x_t[0, masked_idx].std().item():.4f}")

    # Sampling loop
    for i in tqdm(range(steps), desc="Sampling", leave=False):
        t = timesteps[i].item()
        t_next = timesteps[i + 1].item()

        t_batch = torch.full((B,), t, device=device, dtype=torch.long)

        # No CFG (model not trained with unconditional)
        noise_pred_raw = model(x_t, layer_mask, t_batch, y, mask=y_mask)

        # Debug first step - CHECK WHAT MODEL ACTUALLY PREDICTS
        if i == 0:
            print(f"  Step 0 noise_pred_raw (BEFORE zeroing):")
            print(f"    Masked layer {masked_idx}: mean={noise_pred_raw[0, masked_idx].mean().item():.4f}, std={noise_pred_raw[0, masked_idx].std().item():.4f}")
            visible_noise_norms = []
            for j in range(N):
                if layer_mask[0, j] == 0 and j < 4:  # Only check first 4 layers
                    norm = noise_pred_raw[0, j].abs().mean().item()
                    visible_noise_norms.append(norm)
                    print(f"    Visible layer {j}: mean={noise_pred_raw[0, j].mean().item():.4f}, abs_mean={norm:.4f}")

            avg_visible_noise = sum(visible_noise_norms) / len(visible_noise_norms) if visible_noise_norms else 0
            print(f"    Average visible layer |noise|: {avg_visible_noise:.4f}")
            if avg_visible_noise > 0.1:
                print(f"    ⚠️  Model predicts high noise for visible layers at t={t}")
                print(f"    This is likely a timestep distribution mismatch issue")

        # CRITICAL: Use ONLY masked layer prediction, completely ignore visible layers
        # Create noise_pred with zeros for all layers
        noise_pred = torch.zeros_like(noise_pred_raw)
        # Only copy masked layer prediction
        for b in range(B):
            for i_layer in range(N):
                if layer_mask[b, i_layer] == 1:
                    noise_pred[b, i_layer] = noise_pred_raw[b, i_layer]

        # Debug first step - show AFTER zeroing
        if i == 0:
            print(f"  Step 0 noise_pred (AFTER zeroing - should all be 0 for visible):")
            print(f"    Masked layer {masked_idx}: mean={noise_pred[0, masked_idx].mean().item():.4f}, std={noise_pred[0, masked_idx].std().item():.4f}")
            print(f"    Visible layers: all forced to 0.0 (as trained)")

        # DDIM update (only affects masked layer now)
        alpha_t = alphas_cumprod[t]
        alpha_next = alphas_cumprod[t_next] if t_next >= 0 else torch.tensor(1.0, device=device)

        # Predict x0 (only for masked layer due to noise_pred zeroing)
        x0_pred_raw = (x_t - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)

        # Debug: Check if clamp is causing information loss
        if i == 0:
            x0_masked = x0_pred_raw[0, masked_idx]
            clipped_ratio = ((x0_masked.abs() > 3.0).float().mean() * 100).item()
            print(f"  Step 0 x0_pred (BEFORE clamp):")
            print(f"    Masked layer: mean={x0_masked.mean().item():.4f}, std={x0_masked.std().item():.4f}")
            print(f"    min={x0_masked.min().item():.4f}, max={x0_masked.max().item():.4f}")
            print(f"    Clipped ratio: {clipped_ratio:.1f}%")
            if clipped_ratio > 10:
                print(f"    ⚠️  HIGH CLIPPING! This may cause texture artifacts")

        x0_pred = torch.clamp(x0_pred_raw, -3.0, 3.0)

        # Next step
        dir_xt = torch.sqrt(1 - alpha_next) * noise_pred
        x_next = torch.sqrt(alpha_next) * x0_pred + dir_xt

        # Keep visible layers clean (only update masked layer)
        x_t = x_next * layer_mask_expanded + clean_layers * (1 - layer_mask_expanded)

        # Debug middle step
        if i == steps // 2:
            print(f"  Step {i} x_t[masked] stats: mean={x_t[0, masked_idx].mean().item():.4f}, std={x_t[0, masked_idx].std().item():.4f}")

    print(f"  Final x_t[masked] stats: mean={x_t[0, masked_idx].mean().item():.4f}, std={x_t[0, masked_idx].std().item():.4f}")

    return x_t


@torch.no_grad()
def run_inference(
    model,
    vae,
    diffusion,
    dataloader,
    text_encoder,
    output_dir,
    num_samples=5,
    max_layers=6,
    cfg_scale=1.0,
    steps=50,
    device='cuda'
):
    """Run inference on dataset samples"""
    os.makedirs(output_dir, exist_ok=True)

    model.eval()

    # Alpha schedule
    alphas_cumprod = torch.from_numpy(diffusion.alphas_cumprod).float().to(device)

    print("="*60)
    print("Inpainting V4 - Dataset Inference")
    print("="*60)
    print(f"Samples: {num_samples}")
    print(f"CFG Scale: {cfg_scale}")
    print(f"Steps: {steps}")
    print("="*60)

    sample_count = 0
    scale_factor = 0.18215

    for batch_idx, batch in enumerate(dataloader):
        if sample_count >= num_samples:
            break

        layers, captions, num_layers, image_ids = batch
        layers = layers.to(device)
        num_layers = num_layers.to(device)

        B = layers.shape[0]
        N = layers.shape[1]
        H, W = layers.shape[3], layers.shape[4]
        h, w = H // 8, W // 8

        # Encode to VAE latent
        layers_rgb = layers[:, :, :3, :, :]
        layers_flat = layers_rgb.reshape(B * N, 3, H, W)
        z_flat = vae.encode(layers_flat).latent_dist.mode() * scale_factor
        z_clean = z_flat.reshape(B, N, 4, h, w)
        del layers_rgb, layers_flat, z_flat

        # Process each sample
        for b in range(B):
            if sample_count >= num_samples:
                break

            n_valid = num_layers[b].item()
            image_id = image_ids[b]

            # Test each layer as masked (except background for now)
            for masked_idx in range(1, min(n_valid, 3)):  # Test first 2 foreground layers
                if sample_count >= num_samples:
                    break

                print(f"\n[Sample {sample_count + 1}/{num_samples}]")
                print(f"  Image ID: {image_id}")
                print(f"  Valid layers: {n_valid}")
                print(f"  Masked layer: {masked_idx}")

                # Prepare inputs
                layer_mask = torch.zeros(1, max_layers, device=device)
                layer_mask[0, masked_idx] = 1

                clean_layers = z_clean[b:b+1].clone()
                prompt = captions[b][masked_idx]
                print(f"  Prompt: '{prompt}'")

                # Generate
                x_t = ddim_sample(
                    model=model,
                    clean_layers=clean_layers,
                    layer_mask=layer_mask,
                    prompt=prompt,
                    text_encoder=text_encoder,
                    diffusion=diffusion,
                    alphas_cumprod=alphas_cumprod,
                    cfg_scale=cfg_scale,
                    steps=steps,
                    device=device
                )

                # Decode generated layer
                z_generated = x_t[0, masked_idx:masked_idx+1] / scale_factor
                img_generated = vae.decode(z_generated).sample[0]

                # Decode ground truth
                z_gt = z_clean[b, masked_idx:masked_idx+1] / scale_factor
                img_gt = vae.decode(z_gt).sample[0]

                # Decode all layers (for comparison)
                all_imgs = []
                all_imgs_gt = []
                for i in range(n_valid):
                    # Generated version
                    z = x_t[0, i:i+1] / scale_factor
                    img = vae.decode(z).sample[0]
                    all_imgs.append(img)

                    # Ground truth version
                    z_gt_layer = z_clean[b, i:i+1] / scale_factor
                    img_gt_layer = vae.decode(z_gt_layer).sample[0]
                    all_imgs_gt.append(img_gt_layer)

                # Save results
                sample_dir = os.path.join(output_dir, f"sample_{sample_count:03d}_{image_id}_layer{masked_idx}")
                os.makedirs(sample_dir, exist_ok=True)

                # Generated masked layer
                save_image(img_generated,
                          os.path.join(sample_dir, 'generated.png'),
                          normalize=True, value_range=(-1, 1))

                # Ground truth masked layer
                save_image(img_gt,
                          os.path.join(sample_dir, 'ground_truth.png'),
                          normalize=True, value_range=(-1, 1))

                # Comparison: generated vs GT
                comparison = torch.stack([img_generated, img_gt], dim=0)
                save_image(comparison,
                          os.path.join(sample_dir, 'comparison.png'),
                          nrow=2, normalize=True, value_range=(-1, 1))

                # All layers (with generated)
                if all_imgs:
                    all_imgs_tensor = torch.stack(all_imgs, dim=0)
                    save_image(all_imgs_tensor,
                              os.path.join(sample_dir, 'all_layers.png'),
                              nrow=n_valid, normalize=True, value_range=(-1, 1))

                # All layers (ground truth)
                if all_imgs_gt:
                    all_imgs_gt_tensor = torch.stack(all_imgs_gt, dim=0)
                    save_image(all_imgs_gt_tensor,
                              os.path.join(sample_dir, 'all_layers_gt.png'),
                              nrow=n_valid, normalize=True, value_range=(-1, 1))

                # Save info
                with open(os.path.join(sample_dir, 'info.txt'), 'w') as f:
                    f.write(f"Image ID: {image_id}\n")
                    f.write(f"Masked layer: {masked_idx}/{n_valid-1}\n")
                    f.write(f"Prompt: {prompt}\n")
                    f.write(f"CFG scale: {cfg_scale}\n")
                    f.write(f"Steps: {steps}\n")

                print(f"  ✓ Saved to: {sample_dir}")
                sample_count += 1

    print("\n" + "="*60)
    print(f"Inference complete! Processed {sample_count} samples")
    print(f"Results: {output_dir}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Inpainting V4 - Dataset Inference')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    parser.add_argument('--data_roots', type=str, nargs='+', required=True, help='Dataset directories')
    parser.add_argument('--output_dir', type=str, default='output/inference_v4', help='Output directory')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples')
    parser.add_argument('--cfg_scale', type=float, default=1.0, help='CFG scale (1.0 recommended)')
    parser.add_argument('--steps', type=int, default=50, help='Sampling steps')
    parser.add_argument('--max_layers', type=int, default=6, help='Max layers')
    parser.add_argument('--image_size', type=int, default=256, help='Image size')
    parser.add_argument('--vae_path', type=str, default='PixArt-alpha/sd-vae-ft-ema')
    parser.add_argument('--t5_path', type=str, default='PixArt-alpha')
    parser.add_argument('--max_samples', type=int, default=50, help='Max samples to load (for fast testing)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("="*60)
    print("Inpainting V4 - Dataset-based Inference")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data: {args.data_roots}")
    print("="*60)

    # Load models
    print("\n[1/4] Loading models...")

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
    print(f"  Loading from: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu')

    if 'state_dict_ema' in ckpt:
        state_dict = ckpt['state_dict_ema']
        print(f"  ✓ Using EMA model")
    elif 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
        print(f"  ✓ Using regular model")
    else:
        state_dict = ckpt

    # Check what's in the checkpoint
    proj_keys = [k for k in state_dict.keys() if 'proj' in k.lower()]
    print(f"  Projection keys in checkpoint: {len(proj_keys)}")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"  ✓ Model loaded: {len(state_dict)} keys")

    if missing:
        proj_missing = [k for k in missing if 'proj' in k.lower()]
        if proj_missing:
            print(f"  ❌ ERROR: {len(proj_missing)} projection keys missing!")
            print(f"     First few: {proj_missing[:5]}")
            print(f"  This means projections are RANDOM - results will be garbage!")
        else:
            print(f"  ⚠️  {len(missing)} non-critical keys missing")

    if unexpected:
        print(f"  ℹ️  {len(unexpected)} unexpected keys")

    # Sanity check: verify projection weights are not random
    input_proj_weight = model.input_proj.enc_conv1.weight
    print(f"\n  Projection sanity check:")
    print(f"    input_proj.enc_conv1.weight: mean={input_proj_weight.mean().item():.6f}, std={input_proj_weight.std().item():.6f}")
    if abs(input_proj_weight.mean().item()) < 0.001 and abs(input_proj_weight.std().item() - 0.02) < 0.01:
        print(f"    ⚠️  Looks like random init! (mean~0, std~0.02)")
    else:
        print(f"    ✓ Looks pretrained")

    # Load VAE
    vae = AutoencoderKL.from_pretrained(args.vae_path).to(device).eval()
    print(f"  ✓ VAE loaded")

    # Load T5
    t5 = T5Embedder(
        device=device,
        local_cache=True,
        cache_dir=args.t5_path,
        torch_dtype=torch.float16
    )
    print(f"  ✓ T5 loaded")

    # Diffusion
    diffusion = IDDPM(str(1000))
    print(f"  ✓ Diffusion ready")

    # Load dataset
    print("\n[2/4] Loading dataset...")
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
    print(f"  ✓ Dataset: {len(dataloader)} images")

    # Run inference
    print("\n[3/4] Running inference...")
    run_inference(
        model=model,
        vae=vae,
        diffusion=diffusion,
        dataloader=dataloader,
        text_encoder=t5,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        max_layers=args.max_layers,
        cfg_scale=args.cfg_scale,
        steps=args.steps,
        device=device
    )

    print("\n[4/4] Done!")


if __name__ == '__main__':
    main()
