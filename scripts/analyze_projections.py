"""
Analyze and visualize Input/Output Projections
"""

import argparse
import os
import sys
import torch
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid
from diffusers.models import AutoencoderKL
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusion.model.nets.PixArt_layer_inpainting import PixArtLayerInpainting
from diffusion.model.nets.PixArt import PixArt_XL_2
from diffusion.model.t5 import T5Embedder
from diffusion.data.multilayer_builder import build_mulan_dataloader
from diffusion import IDDPM


class ProjectionHook:
    """Hook to capture input/output of projection layers"""
    def __init__(self):
        self.input = None
        self.output = None

    def __call__(self, module, input, output):
        self.input = input[0].detach().cpu()
        self.output = output.detach().cpu()


@torch.no_grad()
def analyze_projections(
    model,
    dataloader,
    vae,
    text_encoder,
    diffusion,
    output_dir,
    num_samples=4,
    device='cuda'
):
    """
    Analyze input/output projections

    Args:
        model: PixArtLayerInpainting model
        dataloader: Data loader
        vae: VAE for encoding
        text_encoder: T5 encoder
        diffusion: IDDPM
        output_dir: Output directory
        num_samples: Number of samples to analyze
        device: Device
    """
    os.makedirs(output_dir, exist_ok=True)

    # Register hooks
    input_proj_hook = ProjectionHook()
    output_proj_hook = ProjectionHook()

    input_handle = model.input_proj.register_forward_hook(input_proj_hook)
    output_handle = model.output_proj.register_forward_hook(output_proj_hook)

    model.eval()

    print("="*60)
    print("Analyzing Input/Output Projections")
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

        # Encode to VAE latent
        with torch.no_grad():
            layers_rgb = layers[:, :, :3, :, :]  # (B, N, 3, H, W)
            layers_flat = layers_rgb.reshape(B * N, 3, H, W)
            z_flat = vae.encode(layers_flat).latent_dist.mode() * 0.18215
            z_clean = z_flat.reshape(B, N, 4, h, w)

        # Random layer masking
        layer_mask = torch.zeros(B, N, device=device)
        masked_captions = []
        for b in range(B):
            n_valid = num_layers[b].item()
            masked_idx = torch.randint(0, n_valid, (1,)).item()
            layer_mask[b, masked_idx] = 1
            masked_captions.append(captions[b][masked_idx])

        # Encode text
        caption_embs, emb_masks = text_encoder.get_text_embeddings(masked_captions)
        y = caption_embs.float()[:, None].to(device)
        y_mask = emb_masks.to(device)

        # Use CLEAN latents directly (no noise)
        # This shows what projections process with clean visible layers
        z_input = z_clean.clone()

        # Use a fixed timestep (e.g., t=500) for consistency
        timesteps = torch.full((B,), 500, device=device, dtype=torch.long)

        # Forward pass (hooks will capture input/output)
        noise_pred = model(
            layers=z_input,
            layer_mask=layer_mask,
            timestep=timesteps,
            y=y,
            mask=y_mask
        )

        # Process each sample in batch
        for b in range(B):
            if sample_count >= num_samples:
                break

            print(f"\n[Sample {sample_count + 1}/{num_samples}] Image ID: {image_ids[b]}")
            print(f"  Timestep: {timesteps[b].item()}")
            print(f"  Masked layer: {layer_mask[b].nonzero().item()}")

            # Get captured tensors for this batch element
            input_proj_in = input_proj_hook.input[b]    # (N*4+N, h, w)
            input_proj_out = input_proj_hook.output[b]  # (4, h, w)
            output_proj_in = output_proj_hook.input[b]  # (4, h, w)
            output_proj_out = output_proj_hook.output[b] # (N*4, h, w)

            sample_dir = os.path.join(output_dir, f"sample_{sample_count:03d}")
            os.makedirs(sample_dir, exist_ok=True)

            # ========================================
            # 1. Input Projection Visualization
            # ========================================
            print(f"\n  Input Projection:")
            print(f"    Input shape: {input_proj_in.shape}")  # (30, h, w)
            print(f"    Output shape: {input_proj_out.shape}") # (4, h, w)

            # Input: (30, h, w) = 24 layer channels + 6 mask channels
            layer_channels = input_proj_in[:24]  # (24, h, w)
            mask_channels = input_proj_in[24:]   # (6, h, w)

            # Reshape layer channels to (6, 4, h, w)
            layer_channels_reshaped = layer_channels.reshape(6, 4, h, w)

            # ========================================
            # 1a. Input Projection Input - Latent Visualization
            # ========================================
            fig, axes = plt.subplots(2, 6, figsize=(18, 6))
            fig.suptitle(f'Input Projection Input (Latent) - Sample {sample_count}', fontsize=16)

            for i in range(6):
                # Row 1: Average of 4 channels per layer
                layer_avg = layer_channels_reshaped[i].mean(dim=0)
                axes[0, i].imshow(layer_avg, cmap='viridis')
                axes[0, i].set_title(f'Layer {i}\n(avg 4ch latent)')
                axes[0, i].axis('off')

                # Row 2: Mask for each layer
                axes[1, i].imshow(mask_channels[i], cmap='gray', vmin=0, vmax=1)
                axes[1, i].set_title(f'Mask {i}')
                axes[1, i].axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(sample_dir, 'input_proj_input_latent.png'), dpi=150, bbox_inches='tight')
            plt.close()

            # ========================================
            # 1b. Input Projection Input - Decoded Images
            # ========================================
            print(f"    Decoding input layers...")
            scale_factor = 0.18215

            # Decode 6 layers
            input_layers_decoded = []
            for i in range(6):
                z = layer_channels_reshaped[i:i+1].to(device) / scale_factor  # (1, 4, h, w)
                img = vae.decode(z).sample[0]  # (3, H, W)
                input_layers_decoded.append(img.cpu())

            # Save decoded layers
            input_layers_tensor = torch.stack(input_layers_decoded, dim=0)
            save_image(input_layers_tensor,
                      os.path.join(sample_dir, 'input_proj_input_decoded.png'),
                      nrow=6, normalize=True, value_range=(-1, 1))
            print(f"    ✓ Input decoded saved")

            # ========================================
            # 1c. Input Projection Output - Latent Visualization
            # ========================================
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            fig.suptitle(f'Input Projection Output (Latent) - Sample {sample_count}', fontsize=16)

            for i in range(4):
                axes[i].imshow(input_proj_out[i], cmap='viridis')
                axes[i].set_title(f'Channel {i}')
                axes[i].axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(sample_dir, 'input_proj_output_latent.png'), dpi=150, bbox_inches='tight')
            plt.close()

            # ========================================
            # 1d. Input Projection Output - Decoded Image
            # ========================================
            print(f"    Decoding input projection output...")
            z = input_proj_out.unsqueeze(0).to(device) / scale_factor  # (1, 4, h, w)
            img = vae.decode(z).sample[0]  # (3, H, W)
            save_image(img.cpu(),
                      os.path.join(sample_dir, 'input_proj_output_decoded.png'),
                      normalize=True, value_range=(-1, 1))
            print(f"    ✓ Input projection output decoded saved")

            # ========================================
            # 2. Output Projection Visualization
            # ========================================
            print(f"\n  Output Projection:")
            print(f"    Input shape: {output_proj_in.shape}")   # (4, h, w)
            print(f"    Output shape: {output_proj_out.shape}") # (24, h, w)

            # ========================================
            # 2a. Output Projection Input - Latent Visualization
            # ========================================
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            fig.suptitle(f'Output Projection Input (Latent) - Sample {sample_count}', fontsize=16)

            for i in range(4):
                axes[i].imshow(output_proj_in[i], cmap='viridis')
                axes[i].set_title(f'Channel {i}')
                axes[i].axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(sample_dir, 'output_proj_input_latent.png'), dpi=150, bbox_inches='tight')
            plt.close()

            # ========================================
            # 2b. Output Projection Input - Decoded Image
            # ========================================
            print(f"    Decoding output projection input...")
            z = output_proj_in.unsqueeze(0).to(device) / scale_factor  # (1, 4, h, w)
            img = vae.decode(z).sample[0]  # (3, H, W)
            save_image(img.cpu(),
                      os.path.join(sample_dir, 'output_proj_input_decoded.png'),
                      normalize=True, value_range=(-1, 1))
            print(f"    ✓ Output projection input decoded saved")

            # ========================================
            # 2c. Output Projection Output - Latent Visualization
            # ========================================
            output_channels_reshaped = output_proj_out.reshape(6, 4, h, w)

            fig, axes = plt.subplots(1, 6, figsize=(18, 3))
            fig.suptitle(f'Output Projection Output (Latent) - Sample {sample_count}', fontsize=16)

            for i in range(6):
                # Average of 4 channels per layer
                layer_avg = output_channels_reshaped[i].mean(dim=0)
                axes[i].imshow(layer_avg, cmap='viridis')
                is_masked = layer_mask[b, i].item() == 1
                axes[i].set_title(f'Layer {i}\n{"[MASKED]" if is_masked else "[visible]"}')
                axes[i].axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(sample_dir, 'output_proj_output_latent.png'), dpi=150, bbox_inches='tight')
            plt.close()

            # ========================================
            # 2d. Output Projection Output - Decoded Images
            # ========================================
            print(f"    Decoding output layers...")

            # Decode 6 layers
            output_layers_decoded = []
            for i in range(6):
                z = output_channels_reshaped[i:i+1].to(device) / scale_factor  # (1, 4, h, w)
                img = vae.decode(z).sample[0]  # (3, H, W)
                output_layers_decoded.append(img.cpu())

            # Save decoded layers
            output_layers_tensor = torch.stack(output_layers_decoded, dim=0)
            save_image(output_layers_tensor,
                      os.path.join(sample_dir, 'output_proj_output_decoded.png'),
                      nrow=6, normalize=True, value_range=(-1, 1))
            print(f"    ✓ Output decoded saved")

            # ========================================
            # 3. Statistics
            # ========================================
            stats_file = os.path.join(sample_dir, 'statistics.txt')
            with open(stats_file, 'w') as f:
                f.write(f"Sample {sample_count} - {image_ids[b]}\n")
                f.write("="*60 + "\n\n")

                f.write("Input Projection:\n")
                f.write(f"  Input shape: {input_proj_in.shape}\n")
                f.write(f"  Input mean: {input_proj_in.mean().item():.6f}\n")
                f.write(f"  Input std: {input_proj_in.std().item():.6f}\n")
                f.write(f"  Input min: {input_proj_in.min().item():.6f}\n")
                f.write(f"  Input max: {input_proj_in.max().item():.6f}\n\n")

                f.write(f"  Output shape: {input_proj_out.shape}\n")
                f.write(f"  Output mean: {input_proj_out.mean().item():.6f}\n")
                f.write(f"  Output std: {input_proj_out.std().item():.6f}\n")
                f.write(f"  Output min: {input_proj_out.min().item():.6f}\n")
                f.write(f"  Output max: {input_proj_out.max().item():.6f}\n\n")

                f.write("Output Projection:\n")
                f.write(f"  Input shape: {output_proj_in.shape}\n")
                f.write(f"  Input mean: {output_proj_in.mean().item():.6f}\n")
                f.write(f"  Input std: {output_proj_in.std().item():.6f}\n")
                f.write(f"  Input min: {output_proj_in.min().item():.6f}\n")
                f.write(f"  Input max: {output_proj_in.max().item():.6f}\n\n")

                f.write(f"  Output shape: {output_proj_out.shape}\n")
                f.write(f"  Output mean: {output_proj_out.mean().item():.6f}\n")
                f.write(f"  Output std: {output_proj_out.std().item():.6f}\n")
                f.write(f"  Output min: {output_proj_out.min().item():.6f}\n")
                f.write(f"  Output max: {output_proj_out.max().item():.6f}\n\n")

                f.write(f"Timestep: {timesteps[b].item()}\n")
                f.write(f"Masked layer index: {layer_mask[b].nonzero().item()}\n")

            print(f"  ✓ Saved to: {sample_dir}")

            sample_count += 1

        if sample_count >= num_samples:
            break

    # Remove hooks
    input_handle.remove()
    output_handle.remove()

    print("\n" + "="*60)
    print(f"Analysis complete! Results saved to: {output_dir}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    parser.add_argument('--data_roots', type=str, nargs='+', required=True, help='Data directories')
    parser.add_argument('--output_dir', type=str, default='output/projection_analysis', help='Output directory')
    parser.add_argument('--num_samples', type=int, default=4, help='Number of samples to analyze')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--max_layers', type=int, default=6, help='Max layers')
    parser.add_argument('--image_size', type=int, default=256, help='Image size')
    parser.add_argument('--vae_path', type=str, default='/workspace/twenty/PixArt-alpha/sd-vae-ft-ema')
    parser.add_argument('--t5_path', type=str, default='/workspace/twenty/PixArt-alpha')
    parser.add_argument('--max_samples', type=int, default=50, help='Max samples to load (for fast testing)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("="*60)
    print("Projection Analysis")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output directory: {args.output_dir}")
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
        batch_size=args.batch_size,
        resolution=args.image_size,
        max_layers=args.max_layers,
        num_workers=0,
        shuffle=False,
        caption_type='blip2',
        max_samples=args.max_samples,
    )
    print(f"  ✓ Dataset loaded: {len(dataloader)} batches")

    # Analyze projections
    print("\n[3/4] Analyzing projections...")
    analyze_projections(
        model=model,
        dataloader=dataloader,
        vae=vae,
        text_encoder=t5,
        diffusion=diffusion,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        device=device
    )

    print("\n[4/4] Done!")


if __name__ == '__main__':
    main()
