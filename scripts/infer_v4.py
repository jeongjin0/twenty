"""
Simple Inference Script for Inpainting V4
Generate a single layer from text prompt
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusion.model.nets.PixArt_layer_inpainting import PixArtLayerInpainting
from diffusion.model.nets.PixArt import PixArt_XL_2
from diffusion.model.t5 import T5Embedder
from diffusion import IDDPM


def load_image(path, size=256):
    """Load and preprocess image to [-1, 1]"""
    img = Image.open(path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    return transform(img)


@torch.no_grad()
def ddim_sample(
    model,
    vae,
    diffusion,
    prompt,
    text_encoder,
    visible_layers=None,
    masked_idx=0,
    max_layers=6,
    cfg_scale=1.0,
    steps=50,
    image_size=256,
    device='cuda'
):
    """
    Generate a layer using DDIM sampling

    Args:
        model: PixArtLayerInpainting
        vae: VAE model
        diffusion: IDDPM
        prompt: Text prompt for generation
        text_encoder: T5Embedder
        visible_layers: Optional list of visible layer images (PIL or tensor)
        masked_idx: Index to generate (default 0)
        max_layers: Max number of layers (default 6)
        cfg_scale: Classifier-free guidance scale (use 1.0 for no CFG)
        steps: Number of DDIM steps
        image_size: Image size
        device: Device

    Returns:
        generated_img: Generated layer image (3, H, W) in [-1, 1]
    """
    h, w = image_size // 8, image_size // 8

    # Encode visible layers if provided
    clean_layers = torch.zeros(1, max_layers, 4, h, w, device=device)
    layer_mask = torch.zeros(1, max_layers, device=device)
    layer_mask[0, masked_idx] = 1  # Mark which layer to generate

    if visible_layers:
        for i, layer_img in enumerate(visible_layers):
            if i == masked_idx:
                continue  # Skip masked position
            if i >= max_layers:
                break

            # Convert to tensor if needed
            if isinstance(layer_img, (str, Image.Image)):
                if isinstance(layer_img, str):
                    layer_img = Image.open(layer_img)
                layer_img = load_image(layer_img, image_size)

            # Encode to latent
            layer_img_device = layer_img.unsqueeze(0).to(device)
            z = vae.encode(layer_img_device).latent_dist.mode() * 0.18215
            clean_layers[0, i] = z.squeeze(0)

    # Encode text
    caption_embs, emb_masks = text_encoder.get_text_embeddings([prompt])
    y = caption_embs.float()[:, None].to(device)
    y_mask = emb_masks.to(device)

    # Get alpha schedule
    alphas_cumprod = torch.from_numpy(diffusion.alphas_cumprod).float().to(device)

    # DDIM timesteps
    timesteps = torch.linspace(999, 0, steps + 1, dtype=torch.long, device=device)

    # Initialize with noise at masked position
    x_t = clean_layers.clone()
    x_t[0, masked_idx] = torch.randn(4, h, w, device=device)

    print(f"Starting generation...")
    print(f"  Prompt: '{prompt}'")
    print(f"  Masked index: {masked_idx}")
    print(f"  CFG scale: {cfg_scale}")
    print(f"  Steps: {steps}")

    # Sampling loop
    for i in tqdm(range(steps), desc="Sampling"):
        t = timesteps[i].item()
        t_next = timesteps[i + 1].item()

        t_batch = torch.full((1,), t, device=device, dtype=torch.long)

        # CFG: conditional + unconditional
        if cfg_scale != 1.0:
            x_in = torch.cat([x_t, x_t], dim=0)
            t_in = torch.cat([t_batch, t_batch], dim=0)
            mask_in = torch.cat([layer_mask, layer_mask], dim=0)

            # Get null embedding
            null_y = model.y_embedder.y_embedding.unsqueeze(0).unsqueeze(0)
            null_y = null_y.to(y.device).to(y.dtype)
            y_in = torch.cat([y, null_y], dim=0)

            null_mask = torch.ones(1, y_mask.shape[1], device=device, dtype=y_mask.dtype)
            y_mask_in = torch.cat([y_mask, null_mask], dim=0)

            # Predict
            noise_pred = model(x_in, mask_in, t_in, y_in, mask=y_mask_in)
            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2, dim=0)
            noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
        else:
            # No CFG
            noise_pred = model(x_t, layer_mask, t_batch, y, mask=y_mask)

        # DDIM update (only for masked layer)
        alpha_t = alphas_cumprod[t]
        alpha_next = alphas_cumprod[t_next] if t_next >= 0 else torch.tensor(1.0, device=device)

        # Predict x0
        x0_pred = (x_t - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
        x0_pred = torch.clamp(x0_pred, -3.0, 3.0)

        # Compute x_next
        dir_xt = torch.sqrt(1 - alpha_next) * noise_pred
        x_next = torch.sqrt(alpha_next) * x0_pred + dir_xt

        # Keep visible layers clean
        layer_mask_expanded = layer_mask.view(1, max_layers, 1, 1, 1)
        x_t = x_next * layer_mask_expanded + clean_layers * (1 - layer_mask_expanded)

    # Decode generated layer
    z_generated = x_t[0, masked_idx:masked_idx+1] / 0.18215
    img_generated = vae.decode(z_generated).sample[0]

    return img_generated


def main():
    parser = argparse.ArgumentParser(description='Inpainting V4 Inference')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--prompt', type=str, required=True, help='Text prompt for generation')
    parser.add_argument('--output', type=str, default='output.png', help='Output image path')

    # Optional reference layers
    parser.add_argument('--visible_layers', type=str, nargs='*', help='Optional visible layer images')
    parser.add_argument('--masked_idx', type=int, default=0, help='Index to generate (default 0)')

    # Generation parameters
    parser.add_argument('--cfg_scale', type=float, default=1.0, help='CFG scale (default 1.0 = no CFG)')
    parser.add_argument('--steps', type=int, default=50, help='Sampling steps')
    parser.add_argument('--image_size', type=int, default=256, help='Image size')
    parser.add_argument('--max_layers', type=int, default=6, help='Max layers')

    # Model paths
    parser.add_argument('--vae_path', type=str, default='PixArt-alpha/sd-vae-ft-ema')
    parser.add_argument('--t5_path', type=str, default='PixArt-alpha')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("="*60)
    print("Inpainting V4 - Simple Inference")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Prompt: {args.prompt}")
    print(f"Output: {args.output}")
    print("="*60)

    # Load models
    print("\n[1/4] Loading models...")

    # Create model
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
    print(f"  Loading checkpoint...")
    ckpt = torch.load(args.checkpoint, map_location='cpu')

    # Try EMA first
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
            print(f"  ⚠️  WARNING: {len(proj_missing)} projection keys missing!")
            print(f"      This may cause poor results.")

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

    # Load visible layers if provided
    print("\n[2/4] Preparing inputs...")
    visible_layers = None
    if args.visible_layers:
        visible_layers = []
        for path in args.visible_layers:
            img = load_image(path, args.image_size)
            visible_layers.append(img)
            print(f"  ✓ Loaded: {path}")

    # Generate
    print("\n[3/4] Generating...")
    generated = ddim_sample(
        model=model,
        vae=vae,
        diffusion=diffusion,
        prompt=args.prompt,
        text_encoder=t5,
        visible_layers=visible_layers,
        masked_idx=args.masked_idx,
        max_layers=args.max_layers,
        cfg_scale=args.cfg_scale,
        steps=args.steps,
        image_size=args.image_size,
        device=device
    )

    # Save
    print("\n[4/4] Saving...")
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    save_image(generated, args.output, normalize=True, value_range=(-1, 1))
    print(f"  ✓ Saved: {args.output}")

    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == '__main__':
    main()
