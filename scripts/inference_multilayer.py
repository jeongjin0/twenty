"""
ReferencePixArt - Minimal Inference Script
가장 간단한 테스트용 코드

사용법:
    python inference_simple.py --checkpoint path/to/checkpoint.pth
"""

import argparse
import os
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusion.model.t5 import T5Embedder
from diffusion.model.nets.PixArt_multilayer import ReferencePixArt_XL_2


class SimpleDDIMSampler:
    """간단한 DDIM Sampler"""
    def __init__(self, num_timesteps=1000):
        self.num_timesteps = num_timesteps
        betas = torch.linspace(0.0001, 0.02, num_timesteps)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    @torch.no_grad()
    def sample(self, model, shape, y, mask, x_ref=None, cfg_scale=4.5, steps=20, device='cuda'):
        """
        DDIM Sampling
        """
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        
        # Initial noise
        x = torch.randn(shape, device=device)
        
        # Time steps (reversed)
        timesteps = torch.linspace(self.num_timesteps - 1, 0, steps + 1, dtype=torch.long, device=device)
        
        for i in tqdm(range(steps), desc="Sampling"):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            
            t_batch = t.expand(shape[0])
            
            # CFG: concat conditional and unconditional
            x_in = torch.cat([x, x], dim=0)
            t_in = torch.cat([t_batch, t_batch], dim=0)
            
            # Null embedding for unconditional
            null_y = torch.zeros_like(y)
            y_in = torch.cat([y, null_y], dim=0)
            
            if x_ref is not None:
                x_ref_in = torch.cat([x_ref, x_ref], dim=0)
                noise_pred = model(x_in, t_in, y_in, x_ref_in, mask=None)
            else:
                noise_pred = model.forward_without_ref(x_in, t_in, y_in, mask=None)
            
            # Split and apply CFG
            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2, dim=0)
            
            # Handle pred_sigma (8 channels → 4 channels)
            if noise_pred_cond.shape[1] == 8:
                noise_pred_cond = noise_pred_cond[:, :4]
                noise_pred_uncond = noise_pred_uncond[:, :4]
            
            noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
            
            # DDIM step
            alpha_t = self.alphas_cumprod[t]
            alpha_t_next = self.alphas_cumprod[t_next] if t_next >= 0 else torch.tensor(1.0, device=device)
            
            x0_pred = (x - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
            x = torch.sqrt(alpha_t_next) * x0_pred + torch.sqrt(1 - alpha_t_next) * noise_pred
        
        return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--prompt', type=str, default="a cat sitting on green grass, digital art")
    parser.add_argument('--output', type=str, default='./output.png')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--cfg_scale', type=float, default=4.5)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--t5_path', type=str, default='./pretrained_models/t5')
    parser.add_argument('--vae_path', type=str, default='stabilityai/sd-vae-ft-ema')
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    
    print(f"Prompt: {args.prompt}")
    print(f"Layers: {args.num_layers}")
    
    # ============================================
    # Load Models
    # ============================================
    print("\nLoading models...")
    
    latent_size = args.image_size // 8
    
    # Model
    model = ReferencePixArt_XL_2(
        input_size=latent_size,
        in_channels=4,
        max_ref_layers=args.num_layers - 1,
        ref_encoder_depth=4,
        caption_channels=4096,
        model_max_length=120,
        pred_sigma=True,
    ).to(device).eval()
    
    # Load weights
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt.get('model', ckpt))
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    print(f"✓ Model loaded")
    
    # VAE
    vae = AutoencoderKL.from_pretrained(args.vae_path).to(device).eval()
    print(f"✓ VAE loaded")
    
    # T5
    t5 = T5Embedder(device=device, local_cache=True, cache_dir=args.t5_path, torch_dtype=torch.float16)
    print(f"✓ T5 loaded")
    
    # ============================================
    # Encode Text
    # ============================================
    print("\nEncoding text...")
    caption_embs, _ = t5.get_text_embeddings([args.prompt])
    y = caption_embs.float()[:, None]  # (1, 1, L, 4096)
    print(f"✓ Text encoded: {y.shape}")
    
    # ============================================
    # Generate Layers
    # ============================================
    print(f"\nGenerating {args.num_layers} layers...")
    
    sampler = SimpleDDIMSampler()
    generated_latents = []
    
    for i in range(args.num_layers):
        print(f"\n--- Layer {i+1}/{args.num_layers} ---")
        
        if i == 0:
            # First layer: no reference
            x_ref = None
            print("  (no reference)")
        else:
            # Use previous layers as reference
            x_ref = torch.stack(generated_latents, dim=1)  # (1, N_ref, 4, h, w)
            print(f"  (reference: {x_ref.shape[1]} layers)")
        
        z = sampler.sample(
            model=model,
            shape=(1, 4, latent_size, latent_size),
            y=y,
            mask=None,
            x_ref=x_ref,
            cfg_scale=args.cfg_scale,
            steps=args.steps,
            device=device,
        )
        
        generated_latents.append(z.squeeze(0))
    
    # ============================================
    # Decode & Save
    # ============================================
    print("\nDecoding...")
    
    decoded_images = []
    for i, z in enumerate(generated_latents):
        img = vae.decode(z.unsqueeze(0) / 0.18215).sample
        decoded_images.append(img.squeeze(0))
    
    # Save individual layers
    output_dir = os.path.dirname(args.output) or '.'
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.splitext(args.output)[0]
    
    for i, img in enumerate(decoded_images):
        path = f"{base_name}_layer{i}.png"
        save_image(img, path, normalize=True, value_range=(-1, 1))
        print(f"✓ Saved: {path}")
    
    # Save grid
    grid = torch.stack(decoded_images, dim=0)
    grid_path = f"{base_name}_grid.png"
    save_image(grid, grid_path, nrow=len(decoded_images), normalize=True, value_range=(-1, 1))
    print(f"✓ Saved grid: {grid_path}")
    
    print("\nDone!")


if __name__ == '__main__':
    main()