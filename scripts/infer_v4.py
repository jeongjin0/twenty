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

    Args:
        model: PixArtLayerInpainting
        clean_layers: (1, N, 4, h, w) - clean latents (visible layers)
        layer_mask: (1, N) - binary mask (1 for masked layer)
        prompt: Text prompt for masked layer
        text_encoder: T5Embedder
        diffusion: IDDPM
        alphas_cumprod: Alpha schedule
        cfg_scale: CFG scale
        steps: DDIM steps
        device: Device

    Returns:
        x_t: Final latent with generated masked layer
    """
    B, N, C, h, w = clean_layers.shape

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

    # Sampling loop
    for i in tqdm(range(steps), desc="Sampling", leave=False):
        t = timesteps[i].item()
        t_next = timesteps[i + 1].item()

        t_batch = torch.full((B,), t, device=device, dtype=torch.long)

        # CFG
        if cfg_scale != 1.0:
            x_in = torch.cat([x_t, x_t], dim=0)
            t_in = torch.cat([t_batch, t_batch], dim=0)
            mask_in = torch.cat([layer_mask, layer_mask], dim=0)

            # Null embedding
            null_y = model.y_embedder.y_embedding.unsqueeze(0).unsqueeze(0)
            null_y = null_y.to(y.device).to(y.dtype)
            y_in = torch.cat([y, null_y], dim=0)

            null_mask = torch.ones(B, y_mask.shape[1], device=device, dtype=y_mask.dtype)
            y_mask_in = torch.cat([y_mask, null_mask], dim=0)

            # Predict
            noise_pred = model(x_in, mask_in, t_in, y_in, mask=y_mask_in)
            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2, dim=0)
            noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
        else:
            # No CFG
            noise_pred = model(x_t, layer_mask, t_batch, y, mask=y_mask)

        # DDIM update
        alpha_t = alphas_cumprod[t]
        alpha_next = alphas_cumprod[t_next] if t_next >= 0 else torch.tensor(1.0, device=device)

        # Predict x0
        x0_pred = (x_t - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
        x0_pred = torch.clamp(x0_pred, -3.0, 3.0)

        # Next step
        dir_xt = torch.sqrt(1 - alpha_next) * noise_pred
        x_next = torch.sqrt(alpha_next) * x0_pred + dir_xt

        # Keep visible layers clean (only update masked layer)
        layer_mask_expanded = layer_mask.view(B, N, 1, 1, 1)
        x_t = x_next * layer_mask_expanded + clean_layers * (1 - layer_mask_expanded)

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
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    if 'state_dict_ema' in ckpt:
        state_dict = ckpt['state_dict_ema']
        print(f"  ✓ Using EMA model")
    elif 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
        print(f"  ✓ Using regular model")
    else:
        state_dict = ckpt

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"  ✓ Model loaded")

    if missing:
        proj_missing = [k for k in missing if 'proj' in k.lower()]
        if proj_missing:
            print(f"  ⚠️  WARNING: {len(proj_missing)} projection keys missing")

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
        caption_type='blip2'
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
