"""
Inference script for Layer-wise Inpainting
Given visible layers, generate a missing layer
"""

import argparse
import os
import sys
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image
from diffusers.models import AutoencoderKL
from tqdm import tqdm
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusion.model.nets.PixArt_layer_inpainting import PixArtLayerInpainting
from diffusion.model.nets.PixArt import PixArt_XL_2
from diffusion.model.t5 import T5Embedder
from diffusion import IDDPM


def load_image(path, size=256):
    """Load and preprocess image"""
    img = Image.open(path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    return transform(img)


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
        null_mask = torch.zeros(B, y_mask.shape[1], device=y_mask.device, dtype=y_mask.dtype)
        text_mask_in = torch.cat([y_mask, null_mask], dim=0)
    else:
        text_mask_in = None

    # Predict noise
    noise_pred = model(x_in, mask_in, t_in, y_in, mask=text_mask_in)

    # CFG
    noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2, dim=0)
    noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)

    # Get alpha values
    alpha_t = alphas_cumprod[t]
    alpha_next = alphas_cumprod[t_next] if t_next >= 0 else torch.tensor(1.0, device=x_t.device)

    # DDIM update
    # Predict x0
    x0_pred = (x_t - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)

    # Direction pointing to x_t
    dir_xt = torch.sqrt(1 - alpha_next) * noise_pred

    # Next sample
    x_next = torch.sqrt(alpha_next) * x0_pred + dir_xt

    # Keep visible layers clean (only update masked layer)
    layer_mask_expanded = layer_mask.view(B, -1, 1, 1, 1)
    x_next = x_next * layer_mask_expanded + clean_layers * (1 - layer_mask_expanded)

    return x_next


@torch.no_grad()
def inpaint_layer(
    model,
    vae,
    diffusion,
    visible_layers,
    masked_idx,
    prompt,
    text_encoder,
    max_layers=6,
    cfg_scale=4.5,
    steps=50,
    device='cuda'
):
    """
    Inpaint a single layer

    Args:
        model: PixArtLayerInpainting model
        vae: VAE model
        diffusion: IDDPM diffusion for noise schedule
        visible_layers: List of (3, H, W) tensors - visible layer images
        masked_idx: int - index where to generate the layer
        prompt: str - text prompt for the masked layer
        text_encoder: T5Embedder
        max_layers: int - max number of layers
        cfg_scale: float - classifier-free guidance scale
        steps: int - number of DDIM steps
        device: str

    Returns:
        generated_layer: (3, H, W) - generated layer in pixel space
        all_layers: (max_layers, 3, H, W) - all layers including generated
    """
    H, W = visible_layers[0].shape[1:]
    h, w = H // 8, W // 8

    # Build layer mask
    num_visible = len(visible_layers)
    assert num_visible < max_layers, "Need at least one slot for generation"
    assert masked_idx < max_layers, f"masked_idx {masked_idx} >= max_layers {max_layers}"

    # Encode visible layers to latent
    visible_latents = []
    for img in visible_layers:
        img_device = img.unsqueeze(0).to(device)
        z = vae.encode(img_device).latent_dist.mode() * 0.18215
        visible_latents.append(z.squeeze(0))

    # Build full layer set with padding
    clean_layers = torch.zeros(1, max_layers, 4, h, w, device=device)
    layer_mask = torch.zeros(1, max_layers, device=device)

    # Place visible layers
    visible_idx = 0
    for i in range(max_layers):
        if i == masked_idx:
            # This is the layer to generate
            layer_mask[0, i] = 1
        elif visible_idx < num_visible:
            # Place visible layer
            clean_layers[0, i] = visible_latents[visible_idx]
            visible_idx += 1
        # else: keep as zero (black padding)

    # Encode text
    caption_embs, emb_masks = text_encoder.get_text_embeddings([prompt])
    y = caption_embs.float()[:, None].to(device)
    y_mask = emb_masks.to(device)

    # Get alpha schedule from IDDPM
    alphas_cumprod = torch.from_numpy(diffusion.alphas_cumprod).float().to(device)

    # DDIM timestep schedule
    timesteps = torch.linspace(999, 0, steps + 1, dtype=torch.long, device=device)

    # Initialize masked layer with random noise
    x_t = clean_layers.clone()
    x_t[0, masked_idx] = torch.randn(4, h, w, device=device)

    print(f"\nInitialization:")
    print(f"  clean_layers stats: mean={clean_layers.mean().item():.4f}, std={clean_layers.std().item():.4f}")
    print(f"  x_t stats: mean={x_t.mean().item():.4f}, std={x_t.std().item():.4f}")
    print(f"  masked_idx: {masked_idx}, layer_mask: {layer_mask}")
    print(f"  num_visible: {num_visible}, max_layers: {max_layers}")

    # DDIM sampling loop
    for i in tqdm(range(steps), desc="Inpainting"):
        t = timesteps[i].item()
        t_next = timesteps[i + 1].item()

        x_t = ddim_sample_step(
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

        # Debug: print stats every 10 steps
        if i % 10 == 0 or i == steps - 1:
            print(f"  Step {i}/{steps}: x_t mean={x_t.mean().item():.4f}, std={x_t.std().item():.4f}, "
                  f"min={x_t.min().item():.4f}, max={x_t.max().item():.4f}")

    # Decode generated layer
    print(f"\nDecoding:")
    print(f"  Final x_t[masked={masked_idx}] stats: mean={x_t[0, masked_idx].mean().item():.4f}, std={x_t[0, masked_idx].std().item():.4f}")

    z_generated = x_t[0, masked_idx:masked_idx+1] / 0.18215
    print(f"  z_generated stats: mean={z_generated.mean().item():.4f}, std={z_generated.std().item():.4f}")

    img_generated = vae.decode(z_generated).sample[0]
    print(f"  img_generated stats: mean={img_generated.mean().item():.4f}, std={img_generated.std().item():.4f}, "
          f"min={img_generated.min().item():.4f}, max={img_generated.max().item():.4f}")

    # Decode all layers for visualization
    all_imgs = []
    for i in range(max_layers):
        z = x_t[0, i:i+1] / 0.18215
        img = vae.decode(z).sample[0]
        all_imgs.append(img)
        print(f"  Layer {i}: img mean={img.mean().item():.4f}, std={img.std().item():.4f}")

    all_imgs = torch.stack(all_imgs, dim=0)

    return img_generated, all_imgs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    parser.add_argument('--visible_layers', type=str, nargs='+', required=True, help='Paths to visible layer images')
    parser.add_argument('--masked_idx', type=int, default=0, help='Index where to generate the layer')
    parser.add_argument('--prompt', type=str, required=True, help='Text prompt for the masked layer')
    parser.add_argument('--output', type=str, default='./inpainted_layer.png', help='Output path')
    parser.add_argument('--cfg_scale', type=float, default=4.5, help='CFG scale')
    parser.add_argument('--steps', type=int, default=50, help='Sampling steps')
    parser.add_argument('--max_layers', type=int, default=6, help='Max layers')
    parser.add_argument('--image_size', type=int, default=256, help='Image size')
    parser.add_argument('--vae_path', type=str, default='/workspace/twenty/PixArt-alpha/sd-vae-ft-ema')
    parser.add_argument('--t5_path', type=str, default='/workspace/twenty/PixArt-alpha')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("="*60)
    print("Layer-wise Inpainting")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Visible layers: {args.visible_layers}")
    print(f"Generate at index: {args.masked_idx}")
    print(f"Prompt: {args.prompt}")
    print("="*60)

    # Load models
    print("\n[1/5] Loading models...")

    # Create pretrained PixArt (needed for proper initialization)
    print("  - Creating PixArt backbone...")
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
    # Try to load EMA model first (better quality), fallback to regular model
    if 'state_dict_ema' in ckpt:
        state_dict = ckpt['state_dict_ema']
        print(f"  ✓ Using EMA model")
    elif 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
        print(f"  ✓ Using regular model")
    else:
        state_dict = ckpt
        print(f"  ✓ Using checkpoint as-is")

    model.load_state_dict(state_dict, strict=False)
    print(f"  ✓ Model loaded from {args.checkpoint}")

    # Load VAE
    vae = AutoencoderKL.from_pretrained(args.vae_path).to(device).eval()
    print(f"  ✓ VAE loaded")

    # Load T5
    t5 = T5Embedder(device=device, local_cache=True, cache_dir=args.t5_path, torch_dtype=torch.float16)
    print(f"  ✓ T5 loaded")

    # Create diffusion for noise schedule
    diffusion = IDDPM(str(1000))
    print(f"  ✓ Diffusion created")

    # Load visible layers
    print("\n[2/5] Loading visible layers...")
    visible_layers = []
    for path in args.visible_layers:
        img = load_image(path, args.image_size)
        visible_layers.append(img)
        print(f"  ✓ {path}")

    # Generate
    print("\n[3/5] Generating masked layer...")
    generated, all_layers = inpaint_layer(
        model=model,
        vae=vae,
        diffusion=diffusion,
        visible_layers=visible_layers,
        masked_idx=args.masked_idx,
        prompt=args.prompt,
        text_encoder=t5,
        max_layers=args.max_layers,
        cfg_scale=args.cfg_scale,
        steps=args.steps,
        device=device
    )

    # Save
    print("\n[4/5] Saving results...")
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    save_image(generated, args.output, normalize=True, value_range=(-1, 1))
    print(f"  ✓ Generated layer: {args.output}")

    if all_layers is not None:
        base_name = os.path.splitext(args.output)[0]
        comparison_path = f"{base_name}_all_layers.png"
        save_image(all_layers, comparison_path, nrow=len(all_layers), normalize=True, value_range=(-1, 1))
        print(f"  ✓ All layers: {comparison_path}")

    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == '__main__':
    main()
