"""
Test script for multilayer reference model.

Purpose:
- Test with same prompt but different reference sets
- Visualize reference influence on generation
- Compare multiple generations with different references
"""

import argparse
import os
import sys
import torch
import torch.nn.functional as F
from pathlib import Path
from torchvision.utils import save_image, make_grid
import numpy as np

from diffusion import IDDPM, ReferenceIDDPM
from diffusion.model.nets.PixArt_reference_crossattn import ReferencePixArtCrossAttn_XL_2
from diffusion.model.nets.PixArt_multilayer import ReferencePixArt_XL_2
from diffusers.models import AutoencoderKL
from diffusion.model.t5 import T5Embedder
from diffusion.utils.misc import read_config
from diffusion.data.multilayer_builder import build_mulan_dataloader


def load_model_and_checkpoint(args, device):
    """Load model from checkpoint."""
    config = read_config(args.config)

    # Determine model type
    image_size = config.image_size
    latent_size = int(image_size) // 8
    pred_sigma = getattr(config, 'pred_sigma', True)
    max_layers = getattr(config, 'max_layers', 8)

    print(f"Building model: {args.model_type}")

    if args.model_type == 'crossattn':
        model = ReferencePixArtCrossAttn_XL_2(
            input_size=latent_size,
            in_channels=4,
            max_ref_layers=max_layers - 1,
            ref_encoder_depth=4,
            ref_compression_ratio=4,
            caption_channels=4096,
            model_max_length=config.model_max_length,
            pred_sigma=pred_sigma,
        ).to(device).eval()
    elif args.model_type == 'adaln':
        model = ReferencePixArt_XL_2(
            input_size=latent_size,
            in_channels=4,
            max_ref_layers=max_layers - 1,
            ref_encoder_depth=4,
            caption_channels=4096,
            model_max_length=config.model_max_length,
            pred_sigma=pred_sigma,
            use_ref=True,
        ).to(device).eval()
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    # Load checkpoint
    print(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')

    if 'state_dict_ema' in checkpoint:
        state_dict = checkpoint['state_dict_ema']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix if exists
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Missing keys: {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")

    return model, config


def load_vae_and_t5(config, device):
    """Load VAE and T5 encoder."""
    print(f"Loading VAE from: {config.vae_pretrained}")
    vae = AutoencoderKL.from_pretrained(config.vae_pretrained).to(device).eval()
    for param in vae.parameters():
        param.requires_grad = False

    t5_pretrained = getattr(config, 't5_pretrained', 'google/flan-t5-xxl')
    print(f"Loading T5 from: {t5_pretrained}")
    text_encoder = T5Embedder(
        device=device,
        local_cache=True,
        cache_dir=t5_pretrained,
        torch_dtype=torch.float16
    )

    return vae, text_encoder


def load_reference_images(dataloader, image_idx, layer_indices, device):
    """
    Load reference layers from a single image.

    Args:
        dataloader: MultiLayer dataloader
        image_idx: Dataset index of the image to load
        layer_indices: List of layer indices to use as references (e.g., [0, 1, 2, 3])
        device: torch device

    Returns:
        ref_images: (num_layers, C, H, W) - reference images
        ref_captions: List[str] - captions for each reference layer
    """
    dataset = dataloader.dataset

    if image_idx >= len(dataset):
        raise ValueError(f"Image index {image_idx} is out of bounds (dataset size: {len(dataset)})")

    # Load one image with all its layers
    layers, captions, num_layers, image_id = dataset[image_idx]
    # layers: (max_layers, C, H, W)

    # Get specified layers as references
    all_ref_images = []
    all_ref_captions = []

    for layer_idx in layer_indices:
        if layer_idx >= num_layers:
            print(f"Warning: Layer {layer_idx} exceeds available layers ({num_layers}). Skipping.")
            continue

        ref_image = layers[layer_idx]  # (C, H, W)
        ref_caption = captions[layer_idx]

        all_ref_images.append(ref_image)
        all_ref_captions.append(ref_caption)

    if len(all_ref_images) == 0:
        raise ValueError("No valid reference layers loaded!")

    # Stack all references: (num_layers, C, H, W)
    ref_images = torch.stack(all_ref_images, dim=0).to(device)

    return ref_images, all_ref_captions


def generate_with_references(
    model,
    vae,
    text_encoder,
    diffusion,
    prompt,
    ref_images,
    device,
    cfg_scale=4.5,
    steps=20,
    scale_factor=0.18215
):
    """
    Generate image with given prompt and references.

    Args:
        model: The diffusion model
        vae: VAE model
        text_encoder: T5 encoder
        diffusion: ReferenceIDDPM
        prompt: Text prompt
        ref_images: (N, C, H, W) reference images in pixel space
        device: torch device
        cfg_scale: Classifier-free guidance scale
        steps: Number of sampling steps
        scale_factor: VAE scale factor

    Returns:
        generated_image: (3, H, W) in pixel space
        ref_images_decoded: (N, 3, H, W) in pixel space
    """
    with torch.no_grad():
        # Encode text
        caption_embs, emb_masks = text_encoder.get_text_embeddings([prompt])
        y = caption_embs.float()[:, None].to(device)

        # Encode references to latent space
        # ref_images: (N, 3, H, W)
        N, C, H, W = ref_images.shape
        if C == 4:
            # Already in RGBA, extract RGB
            ref_rgb = ref_images[:, :3]
        else:
            ref_rgb = ref_images

        z_ref = vae.encode(ref_rgb).latent_dist.mode() * scale_factor
        # z_ref: (N, 4, h, w)
        z_ref = z_ref.unsqueeze(0)  # (1, N, 4, h, w)

        # Sample
        h, w = z_ref.shape[-2:]
        z_gen = diffusion.ddim_sample(
            model=model,
            shape=(1, 4, h, w),
            y=y,
            x_ref=z_ref,
            steps=steps,
            cfg_scale=cfg_scale,
            device=device,
        )

        # Decode
        img_gen = vae.decode(z_gen / scale_factor).sample[0]  # (3, H, W)

        # Decode references
        z_ref_flat = z_ref.squeeze(0)  # (N, 4, h, w)
        img_refs = vae.decode(z_ref_flat / scale_factor).sample  # (N, 3, H, W)

    return img_gen, img_refs


def main():
    parser = argparse.ArgumentParser(description="Test MultiLayer Reference Model")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--output_dir', type=str, default='./test_results', help='Output directory')
    parser.add_argument('--model_type', type=str, default='crossattn', choices=['crossattn', 'adaln'])
    parser.add_argument('--prompt', type=str, default='a photo of a zebra', help='Text prompt')
    parser.add_argument('--cfg_scale', type=float, default=4.5, help='CFG scale')
    parser.add_argument('--steps', type=int, default=20, help='Sampling steps')
    parser.add_argument('--num_refs', type=int, default=3, help='Number of reference images per set')
    parser.add_argument('--num_samples', type=int, default=4, help='Number of samples per reference set')
    parser.add_argument('--device', type=str, default='cuda', help='Device')

    args = parser.parse_args()

    # Create output directory based on prompt
    import re
    prompt_folder = re.sub(r'[^\w\s-]', '', args.prompt).strip().replace(' ', '_')[:50]
    output_dir = os.path.join(args.output_dir, prompt_folder)
    os.makedirs(output_dir, exist_ok=True)

    # ============================================
    # TODO: Fill in your reference image sets!
    # Each set specifies:
    #   - image_idx: which image from dataset to use
    #   - layer_indices: which layers from that image to use as references
    # ============================================
    reference_sets = [
        # Example sets - REPLACE WITH YOUR ACTUAL VALUES
        {'name': 'set1_image0', 'image_idx': 0, 'layer_indices': [0, 1, 2, 3]},
        {'name': 'set2_image1', 'image_idx': 1, 'layer_indices': [0, 1, 2, 3]},
        {'name': 'set3_image2', 'image_idx': 2, 'layer_indices': [0, 1, 2, 3]},
        # Add more sets here...
    ]

    print("="*60)
    print("MultiLayer Reference Model Test")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Model type: {args.model_type}")
    print(f"Prompt: '{args.prompt}'")
    print(f"CFG scale: {args.cfg_scale}")
    print(f"Steps: {args.steps}")
    print(f"Num samples per set: {args.num_samples}")
    print(f"Output dir: {output_dir}")
    print("="*60)

    device = torch.device(args.device)

    # Load model
    model, config = load_model_and_checkpoint(args, device)

    # Load VAE and T5
    vae, text_encoder = load_vae_and_t5(config, device)

    # Build diffusion
    base_diffusion = IDDPM(
        str(config.train_sampling_steps),
        learn_sigma=getattr(config, 'learn_sigma', True),
        pred_sigma=getattr(config, 'pred_sigma', True),
        snr=config.snr_loss
    )
    diffusion = ReferenceIDDPM(base_diffusion)

    # Load dataloader (for getting reference images)
    print("Loading dataset...")
    from diffusion.data.multilayer_builder import build_mulan_dataloader
    dataloader = build_mulan_dataloader(
        data_roots=config.data_roots,
        batch_size=1,
        resolution=config.image_size,
        max_layers=getattr(config, 'max_layers', 8),
        num_workers=2,
        shuffle=False,
        caption_type=getattr(config, 'caption_type', 'blip2')
    )

    # Generate with each reference set
    all_generated = []

    for set_idx, ref_set in enumerate(reference_sets):
        set_name = ref_set['name']
        image_idx = ref_set['image_idx']
        layer_indices = ref_set['layer_indices']

        print(f"\n[{set_idx+1}/{len(reference_sets)}] Generating with {set_name}...")
        print(f"  Image index: {image_idx}, Layer indices: {layer_indices}")

        # Load reference layers from the specified image
        ref_images, ref_captions = load_reference_images(
            dataloader,
            image_idx,
            layer_indices,
            device
        )

        print(f"  Reference shape: {ref_images.shape}")
        print(f"  Reference captions:")
        for i, cap in enumerate(ref_captions):
            print(f"    [Layer {layer_indices[i]}] {cap[:80]}...")  # Print first 80 chars of each caption

        # Generate multiple samples with different seeds
        all_samples = []
        img_refs = None

        for sample_idx in range(args.num_samples):
            print(f"  Generating sample {sample_idx + 1}/{args.num_samples}...")

            # Set different seed for each sample
            torch.manual_seed(42 + sample_idx)

            img_gen, img_refs_tmp = generate_with_references(
                model=model,
                vae=vae,
                text_encoder=text_encoder,
                diffusion=diffusion,
                prompt=args.prompt,
                ref_images=ref_images,
                device=device,
                cfg_scale=args.cfg_scale,
                steps=args.steps,
                scale_factor=config.scale_factor
            )

            all_samples.append(img_gen)
            if img_refs is None:
                img_refs = img_refs_tmp

        # Save result: [ref1, ref2, ref3, ..., sample1, sample2, sample3, sample4]
        num_refs_loaded = img_refs.shape[0]
        images = [img_refs[i] for i in range(num_refs_loaded)] + all_samples
        grid = make_grid(images, nrow=num_refs_loaded + args.num_samples, normalize=True, value_range=(-1, 1))

        save_path = os.path.join(output_dir, f'{set_name}.png')
        save_image(grid, save_path)
        print(f"  Saved to: {save_path}")

        all_generated.extend(all_samples)

    # Save comparison: all generated images together
    if len(all_generated) > 0:
        # Limit number of images per row for better visualization
        nrow = min(args.num_samples, 8)
        comparison_grid = make_grid(all_generated, nrow=nrow, normalize=True, value_range=(-1, 1))
        comparison_path = os.path.join(output_dir, 'all_generated_comparison.png')
        save_image(comparison_grid, comparison_path)
        print(f"\n[Comparison] All generated images saved to: {comparison_path}")

    print("\n" + "="*60)
    print("Test completed!")
    print(f"Results saved in: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
