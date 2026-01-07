"""
Inference script for Layer-wise Inpainting (Dataset-based)
Loads samples directly from MuLan dataset
"""

import argparse
import os
import sys
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL
from tqdm import tqdm
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusion.model.nets.PixArt_layer_inpainting import PixArtLayerInpainting
from diffusion.model.nets.PixArt import PixArt_XL_2
from diffusion.model.t5 import T5Embedder
from diffusion import IDDPM
from diffusion.data.multilayer_builder import build_mulan_dataloader


@torch.no_grad()
def ddim_sample_step(
    model,
    x_t,
    t,
    t_next,
    y,
    y_mask,
    layer_mask,
    alphas_cumprod,
    cfg_scale,
    clean_layers
):
    """Single DDIM sampling step with CFG"""
    B = x_t.shape[0]

    # Prepare timesteps
    t_batch = torch.full((B,), t, device=x_t.device, dtype=torch.long)

    # CFG: conditional + unconditional
    x_in = torch.cat([x_t, x_t], dim=0)
    t_in = torch.cat([t_batch, t_batch], dim=0)
    mask_in = torch.cat([layer_mask, layer_mask], dim=0)

    # Get null text embedding
    null_y = model.y_embedder.y_embedding.unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1)
    null_y = null_y.to(y.device).to(y.dtype)
    y_in = torch.cat([y, null_y], dim=0)

    # Text mask
    if y_mask is not None:
        null_mask = torch.ones(B, y_mask.shape[1], device=y_mask.device, dtype=y_mask.dtype)
        text_mask_in = torch.cat([y_mask, null_mask], dim=0)
    else:
        text_mask_in = None

    # Predict noise
    noise_pred = model(x_in, mask_in, t_in, y_in, mask=text_mask_in)

    # CFG
    noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2, dim=0)

    # CRITICAL FIX: Model was never trained with unconditional (null text)!
    if cfg_scale == 1.0:
        noise_pred = noise_pred_cond
    else:
        noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)

    # Get alpha values
    alpha_t = alphas_cumprod[t]
    alpha_next = alphas_cumprod[t_next] if t_next >= 0 else torch.tensor(1.0, device=x_t.device)

    # Initialize two versions:
    # 1. x_next: Ground truth visible + Generated masked (final output)
    # 2. x_next_model_all: Model predictions for ALL layers (for analysis)
    x_next = x_t.clone()
    x_next_model_all = x_t.clone()

    # Apply DDIM to ALL layers (for visualization)
    for b in range(B):
        for layer_idx in range(x_t.shape[1]):
            x_t_layer = x_t[b, layer_idx]
            noise_pred_layer = noise_pred[b, layer_idx]

            # DDIM formula
            x0_pred_layer = (x_t_layer - torch.sqrt(1 - alpha_t) * noise_pred_layer) / torch.sqrt(alpha_t)

            # Clip for stability
            x0_pred_layer = torch.clamp(x0_pred_layer, min=-3.0, max=3.0)

            dir_xt_layer = torch.sqrt(1 - alpha_next) * noise_pred_layer
            x_next_layer = torch.sqrt(alpha_next) * x0_pred_layer + dir_xt_layer

            # Update model_all version for ALL layers
            x_next_model_all[b, layer_idx] = x_next_layer

            # Update final version ONLY for masked layer
            if layer_mask[b, layer_idx] == 1:
                x_next[b, layer_idx] = x_next_layer

    # Keep visible layers clean (only update masked layer in final output)
    layer_mask_expanded = layer_mask.view(B, -1, 1, 1, 1)
    x_next = x_next * layer_mask_expanded + clean_layers * (1 - layer_mask_expanded)

    return x_next, x_next_model_all


@torch.no_grad()
def inpaint_from_dataset(
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
    """
    Run inference on samples from dataset

    Args:
        model: PixArtLayerInpainting model
        vae: VAE model
        diffusion: IDDPM diffusion
        dataloader: MuLan dataloader
        text_encoder: T5Embedder
        output_dir: Output directory
        num_samples: Number of samples to process
        max_layers: Max number of layers
        cfg_scale: CFG scale
        steps: DDIM steps
        device: Device
    """
    os.makedirs(output_dir, exist_ok=True)

    model.eval()

    # Get alpha schedule
    alphas_cumprod = torch.from_numpy(diffusion.alphas_cumprod).float().to(device)

    # DDIM timestep schedule
    timesteps = torch.linspace(999, 0, steps + 1, dtype=torch.long, device=device)

    print("="*60)
    print("Layer Inpainting from Dataset")
    print("="*60)
    print(f"Samples: {num_samples}")
    print(f"CFG Scale: {cfg_scale}")
    print(f"Steps: {steps}")
    print("="*60)

    sample_count = 0

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

        # ========================================
        # Encode to VAE latent
        # ========================================
        with torch.no_grad():
            layers_rgb = layers[:, :, :3, :, :]  # (B, N, 3, H, W)
            layers_flat = layers_rgb.reshape(B * N, 3, H, W)
            scale_factor = 0.18215
            z_flat = vae.encode(layers_flat).latent_dist.mode() * scale_factor
            z_clean = z_flat.reshape(B, N, 4, h, w)

        # Process each sample in batch
        for b in range(B):
            if sample_count >= num_samples:
                break

            n_valid = num_layers[b].item()

            # Iterate through each layer as masked
            for masked_idx in range(n_valid):
                if sample_count >= num_samples:
                    break

                print(f"\n[Sample {sample_count + 1}/{num_samples}]")
                print(f"  Image ID: {image_ids[b]}")
                print(f"  Masked layer: {masked_idx}/{n_valid-1}")

                # Build layer mask
                layer_mask = torch.zeros(1, max_layers, device=device)
                layer_mask[0, masked_idx] = 1

                # Get clean visible layers
                clean_layers = z_clean[b:b+1].clone()

                # Text prompt for masked layer
                prompt = captions[b][masked_idx]
                print(f"  Prompt: {prompt}")

                # Encode text
                caption_embs, emb_masks = text_encoder.get_text_embeddings([prompt])
                y = caption_embs.float()[:, None].to(device)
                y_mask = emb_masks.to(device)

                # Initialize: random noise for masked layer, clean for visible
                x_t = clean_layers.clone()
                x_t[0, masked_idx] = torch.randn(4, h, w, device=device)

                # Track model predictions
                x_t_model_all = x_t.clone()

                # DDIM sampling loop
                for i in tqdm(range(steps), desc="  Sampling", leave=False):
                    t = timesteps[i].item()
                    t_next = timesteps[i + 1].item()

                    x_t, x_t_model_all = ddim_sample_step(
                        model=model,
                        x_t=x_t,
                        t=t,
                        t_next=t_next,
                        y=y,
                        y_mask=y_mask,
                        layer_mask=layer_mask,
                        alphas_cumprod=alphas_cumprod,
                        cfg_scale=cfg_scale,
                        clean_layers=clean_layers
                    )

                # Decode layers
                print(f"  Decoding...")

                # Generated masked layer
                z_generated = x_t[0, masked_idx:masked_idx+1] / scale_factor
                img_generated = vae.decode(z_generated).sample[0]

                # Ground truth masked layer
                z_gt = z_clean[b, masked_idx:masked_idx+1] / scale_factor
                img_gt = vae.decode(z_gt).sample[0]

                # All layers (GT visible)
                all_imgs_gt = []
                for i in range(max_layers):
                    if i < n_valid:
                        z = x_t[0, i:i+1] / scale_factor
                        img = vae.decode(z).sample[0]
                        all_imgs_gt.append(img)

                # All layers (model predictions)
                all_imgs_model = []
                for i in range(max_layers):
                    if i < n_valid:
                        z = x_t_model_all[0, i:i+1] / scale_factor
                        img = vae.decode(z).sample[0]
                        all_imgs_model.append(img)

                # Save results
                sample_dir = os.path.join(output_dir, f"{image_ids[b]}_layer{masked_idx}")
                os.makedirs(sample_dir, exist_ok=True)

                # Generated
                save_image(img_generated,
                          os.path.join(sample_dir, 'generated.png'),
                          normalize=True, value_range=(-1, 1))

                # Ground truth
                save_image(img_gt,
                          os.path.join(sample_dir, 'ground_truth.png'),
                          normalize=True, value_range=(-1, 1))

                # All layers (GT visible)
                if all_imgs_gt:
                    all_imgs_gt_tensor = torch.stack(all_imgs_gt, dim=0)
                    save_image(all_imgs_gt_tensor,
                              os.path.join(sample_dir, 'all_layers_gt.png'),
                              nrow=len(all_imgs_gt), normalize=True, value_range=(-1, 1))

                # All layers (model predictions)
                if all_imgs_model:
                    all_imgs_model_tensor = torch.stack(all_imgs_model, dim=0)
                    save_image(all_imgs_model_tensor,
                              os.path.join(sample_dir, 'all_layers_model_pred.png'),
                              nrow=len(all_imgs_model), normalize=True, value_range=(-1, 1))

                # Save info
                with open(os.path.join(sample_dir, 'info.txt'), 'w') as f:
                    f.write(f"Image ID: {image_ids[b]}\n")
                    f.write(f"Masked layer: {masked_idx}/{n_valid-1}\n")
                    f.write(f"Prompt: {prompt}\n")
                    f.write(f"CFG scale: {cfg_scale}\n")
                    f.write(f"Steps: {steps}\n")

                print(f"  ✓ Saved to: {sample_dir}")

                sample_count += 1

    print("\n" + "="*60)
    print(f"Inference complete! Processed {sample_count} samples")
    print(f"Results saved to: {output_dir}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    parser.add_argument('--data_roots', type=str, nargs='+', required=True, help='Data directories')
    parser.add_argument('--output_dir', type=str, default='output/inference_from_dataset', help='Output directory')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to process')
    parser.add_argument('--cfg_scale', type=float, default=1.0, help='CFG scale (1.0 = no CFG)')
    parser.add_argument('--steps', type=int, default=50, help='Sampling steps')
    parser.add_argument('--max_layers', type=int, default=6, help='Max layers')
    parser.add_argument('--image_size', type=int, default=256, help='Image size')
    parser.add_argument('--vae_path', type=str, default='/workspace/twenty/PixArt-alpha/sd-vae-ft-ema')
    parser.add_argument('--t5_path', type=str, default='/workspace/twenty/PixArt-alpha')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("="*60)
    print("Layer-wise Inpainting (Dataset-based)")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data roots: {args.data_roots}")
    print("="*60)

    # Load models
    print("\n[1/4] Loading models...")

    # Create pretrained PixArt
    pretrained_pixart = PixArt_XL_2(
        input_size=args.image_size // 8,
        in_channels=4,
        caption_channels=4096,
        model_max_length=120,
        pred_sigma=True,
    )

    # Create layer inpainting model
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

    model.load_state_dict(state_dict, strict=False)
    print(f"  ✓ Model loaded")

    # Load VAE
    vae = AutoencoderKL.from_pretrained(args.vae_path).to(device).eval()
    print(f"  ✓ VAE loaded")

    # Load T5
    t5 = T5Embedder(device=device, local_cache=True, cache_dir=args.t5_path, torch_dtype=torch.float16)
    print(f"  ✓ T5 loaded")

    # Create diffusion
    diffusion = IDDPM(str(1000))
    print(f"  ✓ Diffusion created")

    # Load dataset
    print("\n[2/4] Loading dataset...")
    dataloader = build_mulan_dataloader(
        data_roots=args.data_roots,
        batch_size=1,  # Process one image at a time
        resolution=args.image_size,
        max_layers=args.max_layers,
        num_workers=0,
        shuffle=False,
        caption_type='blip2'
    )
    print(f"  ✓ Dataset loaded: {len(dataloader)} images")

    # Run inference
    print("\n[3/4] Running inference...")
    inpaint_from_dataset(
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
