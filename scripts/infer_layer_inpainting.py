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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusion.model.nets.PixArt_layer_inpainting import PixArtLayerInpainting
from diffusion.model.t5 import T5Embedder


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
def inpaint_layer(
    model,
    vae,
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
    layers = torch.zeros(1, max_layers, 4, h, w, device=device)
    layer_mask = torch.zeros(1, max_layers, device=device)

    # Place visible layers
    visible_idx = 0
    for i in range(max_layers):
        if i == masked_idx:
            # This is the layer to generate
            layer_mask[0, i] = 1
        elif visible_idx < num_visible:
            # Place visible layer
            layers[0, i] = visible_latents[visible_idx]
            visible_idx += 1
        # else: keep as zero (black padding)

    # Initialize masked layer with noise
    layers[0, masked_idx] = torch.randn(4, h, w, device=device)

    # Encode text
    caption_embs, emb_masks = text_encoder.get_text_embeddings([prompt])
    y = caption_embs.float()[:, None].to(device)
    y_mask = emb_masks.to(device)

    # Build DDIM schedule
    betas = torch.linspace(0.0001, 0.02, 1000)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0).to(device)

    timesteps = torch.linspace(999, 0, steps + 1, dtype=torch.long, device=device)

    # DDIM sampling
    x_t = layers.clone()

    for i in tqdm(range(steps), desc="Inpainting"):
        t = timesteps[i]
        t_next = timesteps[i + 1]
        t_batch = t.expand(1)

        # CFG
        x_in = torch.cat([x_t, x_t], dim=0)
        t_in = torch.cat([t_batch, t_batch], dim=0)

        # Null text embedding
        null_y = model.y_embedder.y_embedding.unsqueeze(0).unsqueeze(0)
        null_y = null_y.to(y.device).to(y.dtype)
        y_in = torch.cat([y, null_y], dim=0)

        # Mask
        mask_in = torch.cat([layer_mask, layer_mask], dim=0)

        # Text mask
        if y_mask is not None:
            null_mask = torch.ones(1, y_mask.shape[1], device=y_mask.device, dtype=y_mask.dtype)
            text_mask_in = torch.cat([y_mask, null_mask], dim=0)
        else:
            text_mask_in = None

        # Predict noise
        noise_pred = model(x_in, mask_in, t_in, y_in, mask=text_mask_in)

        # CFG
        noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2, dim=0)
        noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)

        # DDIM step
        alpha_t = alphas_cumprod[t]
        alpha_next = alphas_cumprod[t_next] if t_next >= 0 else torch.tensor(1.0, device=device)

        # Predict x0
        x0_pred = (x_t - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)

        # Next step
        x_next = torch.sqrt(alpha_next) * x0_pred + torch.sqrt(1 - alpha_next) * noise_pred

        # Update only masked layer (keep visible layers clean)
        layer_mask_expanded = layer_mask.view(1, max_layers, 1, 1, 1)
        x_t = x_next * layer_mask_expanded + layers * (1 - layer_mask_expanded)

    # Decode generated layer
    z_generated = x_t[0, masked_idx:masked_idx+1] / 0.18215
    img_generated = vae.decode(z_generated).sample[0]

    # Decode all layers
    all_imgs = []
    for i in range(max_layers):
        if layer_mask[0, i] == 0 and i < num_visible + 1:  # Visible or generated
            z = x_t[0, i:i+1] / 0.18215
            img = vae.decode(z).sample[0]
            all_imgs.append(img)

    all_imgs = torch.stack(all_imgs, dim=0) if all_imgs else None

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
    print("\n[1/4] Loading models...")

    model = PixArtLayerInpainting(
        max_layers=args.max_layers,
        input_size=args.image_size // 8,
        pred_sigma=True,
    ).to(device).eval()

    ckpt = torch.load(args.checkpoint, map_location='cpu')
    state_dict = ckpt.get('model', ckpt)
    model.load_state_dict(state_dict, strict=True)
    print(f"  ✓ Model loaded")

    vae = AutoencoderKL.from_pretrained(args.vae_path).to(device).eval()
    print(f"  ✓ VAE loaded")

    t5 = T5Embedder(device=device, local_cache=True, cache_dir=args.t5_path, torch_dtype=torch.float16)
    print(f"  ✓ T5 loaded")

    # Load visible layers
    print("\n[2/4] Loading visible layers...")
    visible_layers = []
    for path in args.visible_layers:
        img = load_image(path, args.image_size)
        visible_layers.append(img)
        print(f"  ✓ {path}")

    # Generate
    print("\n[3/4] Generating masked layer...")
    generated, all_layers = inpaint_layer(
        model=model,
        vae=vae,
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
    print("\n[4/4] Saving results...")
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
