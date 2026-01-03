"""
Pretrain Input/Output Projections with Merge/Decompose Strategy

New approach:
- Input Projection: 6 layers → Merged image latent (alpha blending)
- Output Projection: Merged image latent → 6 layers (decomposition)

This ensures:
1. Input projection output = real image latent (PixArt's learned space!)
2. Semantic meaning: merge (compositing) ↔ decompose (layer separation)
3. Better alignment with diffusion training
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models import AutoencoderKL
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusion.data.multilayer_builder import build_mulan_dataloader
from diffusion.utils.logger import get_root_logger
from diffusion.utils.misc import set_random_seed, read_config
from accelerate import Accelerator


def alpha_blend_layers(layers, num_layers):
    """
    Alpha blend RGBA layers to create merged image

    Args:
        layers: (B, N, 4, H, W) - RGBA layers in [0, 1] range
        num_layers: (B,) - actual number of valid layers

    Returns:
        merged: (B, 3, H, W) - RGB merged image
    """
    B, N, _, H, W = layers.shape
    device = layers.device

    # Initialize with white background
    merged = torch.ones(B, 3, H, W, device=device)

    for b in range(B):
        n_valid = num_layers[b].item()

        # Blend from bottom to top
        for i in range(n_valid):
            layer_rgba = layers[b, i]  # (4, H, W)
            rgb = layer_rgba[:3]       # (3, H, W)
            alpha = layer_rgba[3:4]    # (1, H, W)

            # Alpha compositing: result = alpha * fg + (1 - alpha) * bg
            merged[b] = alpha * rgb + (1 - alpha) * merged[b]

    return merged


class ProjectionAutoencoder(nn.Module):
    """
    Autoencoder using input_proj (merge) and output_proj (decompose)
    """

    def __init__(self, max_layers=6):
        super().__init__()
        self.max_layers = max_layers

        input_channels = max_layers * 4 + max_layers  # 24 + 6 = 30
        output_channels = max_layers * 4  # 24

        # Input projection: 6 layers → merged image
        # (30, h, w) → (4, h, w)
        self.input_proj = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 4, kernel_size=1),
        )

        # Output projection: merged image → 6 layers
        # (4, h, w) → (24, h, w)
        self.output_proj = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, output_channels, kernel_size=1),
        )

    def forward(self, layers, layer_mask):
        """
        Args:
            layers: (B, N, 4, h, w) - layer latents
            layer_mask: (B, N) - binary mask

        Returns:
            merged_pred: (B, 4, h, w) - predicted merged image latent
            layers_recon: (B, N, 4, h, w) - reconstructed layer latents
        """
        B, N, C, h, w = layers.shape

        # Flatten layers
        layers_flat = layers.reshape(B, N * C, h, w)  # (B, 24, h, w)

        # Expand mask
        mask_spatial = layer_mask.unsqueeze(-1).unsqueeze(-1).expand(B, N, h, w)

        # Concatenate
        x = torch.cat([layers_flat, mask_spatial], dim=1)  # (B, 30, h, w)

        # Input projection: layers → merged image latent
        merged_pred = self.input_proj(x)  # (B, 4, h, w)

        # Output projection: merged image latent → layers
        layers_recon_flat = self.output_proj(merged_pred)  # (B, 24, h, w)
        layers_recon = layers_recon_flat.reshape(B, N, C, h, w)

        return merged_pred, layers_recon


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='output/projection_pretrain')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--merge_weight', type=float, default=1.0, help='Weight for merge loss')
    parser.add_argument('--decompose_weight', type=float, default=1.0, help='Weight for decompose loss')
    args = parser.parse_args()

    config = read_config(args.config)

    # Accelerator
    accelerator = Accelerator(mixed_precision='fp16')

    set_random_seed(42)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)

    logger = get_root_logger(
        log_file=os.path.join(args.output_dir, 'train.log') if accelerator.is_main_process else None
    )

    logger.info("="*60)
    logger.info("Projection Pretraining: Merge/Decompose Strategy")
    logger.info("="*60)
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Epochs: {args.epochs}, LR: {args.lr}")
    logger.info(f"Merge weight: {args.merge_weight}, Decompose weight: {args.decompose_weight}")
    logger.info("="*60)

    # Model
    model = ProjectionAutoencoder(max_layers=config.max_layers)
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Dataset
    train_dataloader = build_mulan_dataloader(
        data_roots=config.data_roots,
        batch_size=args.batch_size,
        resolution=config.image_size,
        max_layers=config.max_layers,
        num_workers=4,
        shuffle=True,
        caption_type='blip2'
    )

    # VAE
    vae = AutoencoderKL.from_pretrained(config.vae_pretrained).to(accelerator.device)
    vae.eval()
    vae.requires_grad_(False)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Scheduler
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_dataloader))

    # Prepare
    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, scheduler
    )

    # Training
    logger.info("\nStarting training...")

    scale_factor = 0.18215

    for epoch in range(args.epochs):
        model.train()

        total_loss_sum = 0
        merge_loss_sum = 0
        decompose_loss_sum = 0
        count = 0

        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs}",
                   disable=not accelerator.is_local_main_process)

        for step, batch in enumerate(pbar):
            layers, captions, num_layers, image_ids = batch
            layers = layers.to(accelerator.device)
            num_layers = num_layers.to(accelerator.device)

            B = layers.shape[0]
            N = layers.shape[1]
            H, W = config.image_size, config.image_size
            h, w = H // 8, W // 8

            # ========================================
            # 1. Ground Truth: Merged image latent
            # ========================================
            with torch.no_grad():
                # Alpha blend layers to create merged image
                # layers: (B, N, 4, H, W) - RGBA in [-1, 1] from normalization
                # Convert to [0, 1] for alpha blending
                layers_01 = (layers + 1.0) / 2.0  # [-1, 1] → [0, 1]

                merged_rgb = alpha_blend_layers(layers_01, num_layers)  # (B, 3, H, W)

                # Convert back to [-1, 1]
                merged_rgb = merged_rgb * 2.0 - 1.0

                # Encode merged image to latent
                merged_latent_gt = vae.encode(merged_rgb).latent_dist.mode() * scale_factor
                # (B, 4, h, w)

            # ========================================
            # 2. Ground Truth: Individual layer latents
            # ========================================
            with torch.no_grad():
                # Extract RGB from layers
                layers_rgb = layers[:, :, :3, :, :]  # (B, N, 3, H, W)
                layers_flat = layers_rgb.reshape(B * N, 3, H, W)

                # Encode to latent
                z_flat = vae.encode(layers_flat).latent_dist.mode() * scale_factor
                layers_latent_gt = z_flat.reshape(B, N, 4, h, w)

            # ========================================
            # 3. Random layer masking
            # ========================================
            layer_mask = torch.zeros(B, N, device=accelerator.device)
            for b in range(B):
                n_valid = num_layers[b].item()
                masked_idx = torch.randint(0, n_valid, (1,)).item()
                layer_mask[b, masked_idx] = 1

            # ========================================
            # 4. Forward pass
            # ========================================
            optimizer.zero_grad()

            merged_pred, layers_recon = model(layers_latent_gt, layer_mask)

            # ========================================
            # 5. Loss computation
            # ========================================
            # Loss 1: Merge loss (input projection)
            # Predicted merged latent should match ground truth
            merge_loss = F.mse_loss(merged_pred, merged_latent_gt)

            # Loss 2: Decompose loss (output projection)
            # Reconstructed layers should match ground truth
            # Only compute loss on valid layers
            valid_mask = torch.zeros(B, N, device=accelerator.device)
            for b in range(B):
                n_valid = num_layers[b].item()
                valid_mask[b, :n_valid] = 1

            decompose_loss_per_layer = F.mse_loss(layers_recon, layers_latent_gt, reduction='none')
            decompose_loss_per_layer = decompose_loss_per_layer.mean(dim=[2, 3, 4])  # (B, N)
            decompose_loss = (decompose_loss_per_layer * valid_mask).sum() / valid_mask.sum()

            # Total loss
            total_loss = args.merge_weight * merge_loss + args.decompose_weight * decompose_loss

            # Backward
            accelerator.backward(total_loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Logging
            total_loss_sum += total_loss.item()
            merge_loss_sum += merge_loss.item()
            decompose_loss_sum += decompose_loss.item()
            count += 1

            if (step + 1) % 50 == 0:
                avg_total = total_loss_sum / count
                avg_merge = merge_loss_sum / count
                avg_decompose = decompose_loss_sum / count

                pbar.set_postfix({
                    'total': f'{avg_total:.4f}',
                    'merge': f'{avg_merge:.4f}',
                    'decomp': f'{avg_decompose:.4f}'
                })

                total_loss_sum = 0
                merge_loss_sum = 0
                decompose_loss_sum = 0
                count = 0

        # Save checkpoint
        if accelerator.is_main_process:
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': accelerator.unwrap_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }

            save_path = os.path.join(args.output_dir, 'checkpoints', f'epoch_{epoch+1:02d}.pth')
            torch.save(checkpoint, save_path)
            logger.info(f"Saved: {save_path}")

    logger.info("\nTraining finished!")
    logger.info(f"Pretrained projections saved to: {args.output_dir}/checkpoints")
    logger.info("\nTo use in main training:")
    logger.info("  python train_scripts/train_layer_inpainting.py \\")
    logger.info(f"    --pretrained_projections {args.output_dir}/checkpoints/epoch_{args.epochs:02d}.pth")


if __name__ == '__main__':
    train()
