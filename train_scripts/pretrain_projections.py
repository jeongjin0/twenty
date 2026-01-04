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
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


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


@torch.no_grad()
def visualize_predictions(model, dataloader, vae, epoch, output_dir, num_samples=4):
    """
    시각화: Ground truth vs 예측 비교

    저장되는 이미지:
    1. merged_gt.png: Ground truth merged image (alpha blending)
    2. merged_pred.png: 예측된 merged image
    3. layers_gt.png: Ground truth 6 layers
    4. layers_recon.png: 재구성된 6 layers
    """
    model.eval()
    vae.eval()

    vis_dir = os.path.join(output_dir, 'visualizations', f'epoch_{epoch:02d}')
    os.makedirs(vis_dir, exist_ok=True)

    scale_factor = 0.18215
    device = next(model.parameters()).device

    sample_count = 0

    for batch in dataloader:
        if sample_count >= num_samples:
            break

        layers, captions, num_layers, image_ids = batch
        layers = layers.to(device)
        num_layers = num_layers.to(device)

        B, N, C, H, W = layers.shape
        h, w = H // 8, W // 8

        # ========================================
        # 1. Ground Truth: Merged image
        # ========================================
        # Alpha blend layers
        layers_01 = (layers + 1.0) / 2.0  # [-1, 1] → [0, 1]
        merged_rgb_gt = alpha_blend_layers(layers_01, num_layers)  # (B, 3, H, W)
        merged_rgb_gt = merged_rgb_gt * 2.0 - 1.0  # [0, 1] → [-1, 1]

        # Encode to latent
        merged_latent_gt = vae.encode(merged_rgb_gt).latent_dist.mode() * scale_factor

        # ========================================
        # 2. Ground Truth: Layer latents
        # ========================================
        layers_rgb = layers[:, :, :3, :, :]  # (B, N, 3, H, W)
        layers_flat = layers_rgb.reshape(B * N, 3, H, W)
        z_flat = vae.encode(layers_flat).latent_dist.mode() * scale_factor
        layers_latent_gt = z_flat.reshape(B, N, 4, h, w)

        # ========================================
        # 3. Model prediction
        # ========================================
        # Create random layer mask
        layer_mask = torch.zeros(B, N, device=device)
        for b in range(B):
            n_valid = num_layers[b].item()
            masked_idx = torch.randint(0, n_valid, (1,)).item()
            layer_mask[b, masked_idx] = 1

        # Forward pass
        merged_pred, layers_recon = model(layers_latent_gt, layer_mask)

        # ========================================
        # 4. Decode predictions
        # ========================================
        # Decode merged image
        merged_rgb_pred = vae.decode(merged_pred / scale_factor).sample

        # Decode reconstructed layers
        layers_recon_flat = layers_recon.reshape(B * N, 4, h, w)
        layers_rgb_recon_flat = vae.decode(layers_recon_flat / scale_factor).sample
        layers_rgb_recon = layers_rgb_recon_flat.reshape(B, N, 3, H, W)

        # ========================================
        # 5. Save visualizations
        # ========================================
        for b in range(B):
            if sample_count >= num_samples:
                break

            sample_dir = os.path.join(vis_dir, f'sample_{sample_count:02d}')
            os.makedirs(sample_dir, exist_ok=True)

            n_valid = num_layers[b].item()

            # 5a. Merged images comparison
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))

            # GT merged
            img_gt = merged_rgb_gt[b].cpu().permute(1, 2, 0).numpy()
            img_gt = (img_gt + 1.0) / 2.0  # [-1, 1] → [0, 1]
            axes[0].imshow(img_gt.clip(0, 1))
            axes[0].set_title('GT Merged Image\n(Alpha Blending)')
            axes[0].axis('off')

            # Predicted merged
            img_pred = merged_rgb_pred[b].cpu().permute(1, 2, 0).numpy()
            img_pred = (img_pred + 1.0) / 2.0
            axes[1].imshow(img_pred.clip(0, 1))
            axes[1].set_title('Predicted Merged Image\n(Input Projection Output)')
            axes[1].axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(sample_dir, 'merged_comparison.png'), dpi=150, bbox_inches='tight')
            plt.close()

            # 5b. Layers comparison
            fig, axes = plt.subplots(2, 6, figsize=(18, 6))
            fig.suptitle(f'Layers Comparison - Sample {sample_count}', fontsize=16)

            for i in range(6):
                # GT layer
                if i < n_valid:
                    img_gt = layers_rgb[b, i].cpu().permute(1, 2, 0).numpy()
                    img_gt = (img_gt + 1.0) / 2.0
                    axes[0, i].imshow(img_gt.clip(0, 1))
                    axes[0, i].set_title(f'GT Layer {i}')
                else:
                    axes[0, i].imshow(torch.zeros(H, W, 3).numpy())
                    axes[0, i].set_title(f'Padding {i}')
                axes[0, i].axis('off')

                # Reconstructed layer
                if i < n_valid:
                    img_recon = layers_rgb_recon[b, i].cpu().permute(1, 2, 0).numpy()
                    img_recon = (img_recon + 1.0) / 2.0
                    axes[1, i].imshow(img_recon.clip(0, 1))
                    is_masked = layer_mask[b, i].item() == 1
                    axes[1, i].set_title(f'Recon {i}\n{"[MASKED]" if is_masked else "[visible]"}')
                else:
                    axes[1, i].imshow(torch.zeros(H, W, 3).numpy())
                    axes[1, i].set_title(f'Padding {i}')
                axes[1, i].axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(sample_dir, 'layers_comparison.png'), dpi=150, bbox_inches='tight')
            plt.close()

            # 5c. Loss statistics
            with open(os.path.join(sample_dir, 'stats.txt'), 'w') as f:
                merge_mse = F.mse_loss(merged_pred[b], merged_latent_gt[b]).item()

                # Per-layer decompose loss
                f.write(f"Sample {sample_count} - {image_ids[b]}\n")
                f.write("="*60 + "\n\n")
                f.write(f"Merge Loss (MSE): {merge_mse:.6f}\n\n")
                f.write("Decompose Loss per Layer:\n")
                for i in range(n_valid):
                    layer_mse = F.mse_loss(layers_recon[b, i], layers_latent_gt[b, i]).item()
                    is_masked = layer_mask[b, i].item() == 1
                    f.write(f"  Layer {i} {'[MASKED]' if is_masked else '[visible]'}: {layer_mse:.6f}\n")

            print(f"  ✓ Saved visualization: {sample_dir}")
            sample_count += 1

    model.train()
    return vis_dir


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='output/projection_pretrain')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--merge_weight', type=float, default=1.0, help='Weight for merge loss')
    parser.add_argument('--decompose_weight', type=float, default=1.0, help='Weight for decompose loss')
    parser.add_argument('--masked_layer_weight', type=float, default=2.0, help='Extra weight for masked layer in decompose loss')
    parser.add_argument('--background_layer_weight', type=float, default=0.5, help='Weight for layer 0 (background) in decompose loss')
    parser.add_argument('--shuffle_layers', action='store_true', help='Shuffle layer order for order-invariant learning')
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
            # 3. Layer order shuffling (optional)
            # ========================================
            shuffle_indices = []
            if args.shuffle_layers:
                # Shuffle layer order for each sample
                layers_latent_input = torch.zeros_like(layers_latent_gt)
                for b in range(B):
                    n_valid = num_layers[b].item()
                    # Random permutation of valid layers
                    perm = torch.randperm(n_valid)
                    shuffle_indices.append(perm)
                    # Apply permutation
                    layers_latent_input[b, :n_valid] = layers_latent_gt[b, perm]
            else:
                layers_latent_input = layers_latent_gt
                shuffle_indices = [torch.arange(num_layers[b].item()) for b in range(B)]

            # ========================================
            # 4. Random layer masking
            # ========================================
            layer_mask = torch.zeros(B, N, device=accelerator.device)
            original_masked_indices = []  # Store original (pre-shuffle) masked index
            for b in range(B):
                n_valid = num_layers[b].item()
                # Never mask layer 0 (background) - always keep it as reference
                # Sample from layers 1 to n_valid-1
                if n_valid > 1:
                    masked_idx = torch.randint(1, n_valid, (1,)).item()
                else:
                    # Edge case: only 1 layer (shouldn't happen with min_layers=2)
                    masked_idx = 0
                original_masked_indices.append(masked_idx)

                # If shuffled, find where this layer ended up
                if args.shuffle_layers:
                    shuffled_idx = (shuffle_indices[b] == masked_idx).nonzero(as_tuple=True)[0].item()
                    layer_mask[b, shuffled_idx] = 1
                else:
                    layer_mask[b, masked_idx] = 1

            # ========================================
            # 5. Forward pass
            # ========================================
            optimizer.zero_grad()

            merged_pred, layers_recon = model(layers_latent_input, layer_mask)

            # ========================================
            # 6. Loss computation
            # ========================================
            # Loss 1: Merge loss (input projection)
            # Predicted merged latent should match ground truth
            merge_loss = F.mse_loss(merged_pred, merged_latent_gt)

            # Loss 2: Decompose loss (output projection)
            # Reconstructed layers should match ORIGINAL (non-shuffled) ground truth
            # Need to unshuffle the reconstruction for comparison
            if args.shuffle_layers:
                layers_recon_unshuffled = torch.zeros_like(layers_recon)
                for b in range(B):
                    n_valid = num_layers[b].item()
                    # Inverse permutation
                    inv_perm = torch.argsort(shuffle_indices[b])
                    layers_recon_unshuffled[b, :n_valid] = layers_recon[b, inv_perm]
            else:
                layers_recon_unshuffled = layers_recon

            # Compute per-layer loss
            decompose_loss_per_layer = F.mse_loss(layers_recon_unshuffled, layers_latent_gt, reduction='none')
            decompose_loss_per_layer = decompose_loss_per_layer.mean(dim=[2, 3, 4])  # (B, N)

            # Create weighted mask
            loss_weight_mask = torch.zeros(B, N, device=accelerator.device)
            for b in range(B):
                n_valid = num_layers[b].item()
                for i in range(n_valid):
                    # Base weight: 1.0
                    weight = 1.0

                    # Layer 0 (background): lower weight
                    if i == 0:
                        weight *= args.background_layer_weight

                    # Masked layer: higher weight
                    if i == original_masked_indices[b]:
                        weight *= args.masked_layer_weight

                    loss_weight_mask[b, i] = weight

            # Weighted decompose loss
            weighted_decompose_loss = (decompose_loss_per_layer * loss_weight_mask).sum() / loss_weight_mask.sum()

            # Total loss
            total_loss = args.merge_weight * merge_loss + args.decompose_weight * weighted_decompose_loss

            # Backward
            accelerator.backward(total_loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Logging
            total_loss_sum += total_loss.item()
            merge_loss_sum += merge_loss.item()
            decompose_loss_sum += weighted_decompose_loss.item()
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

            # Visualize predictions
            logger.info(f"\nGenerating visualizations for epoch {epoch+1}...")
            model_unwrapped = accelerator.unwrap_model(model)
            vis_dir = visualize_predictions(
                model_unwrapped,
                train_dataloader,
                vae,
                epoch=epoch+1,
                output_dir=args.output_dir,
                num_samples=4
            )
            logger.info(f"Visualizations saved to: {vis_dir}")

    logger.info("\nTraining finished!")
    logger.info(f"Pretrained projections saved to: {args.output_dir}/checkpoints")
    logger.info("\nTo use in main training:")
    logger.info("  python train_scripts/train_layer_inpainting.py \\")
    logger.info(f"    --pretrained_projections {args.output_dir}/checkpoints/epoch_{args.epochs:02d}.pth")


if __name__ == '__main__':
    train()
