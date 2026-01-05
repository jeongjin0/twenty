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
from scipy.optimize import linear_sum_assignment

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


def optimal_assignment_loss(pred_layers, gt_layers, num_valid, background_weight=0.1, target_layer_weight=2.0, target_indices=None):
    """
    Compute loss using optimal assignment (Hungarian algorithm)

    Args:
        pred_layers: (B, N, C, H, W) - predicted layers (any order)
        gt_layers: (B, N, C, H, W) - ground truth layers (canonical order)
        num_valid: (B,) - number of valid layers per sample
        background_weight: weight for layer 0 (background, default=0.1)
        target_layer_weight: extra weight for target layer (default=2.0)
        target_indices: (B,) - which layer to prioritize (None = no priority)

    Returns:
        total_loss: scalar loss
        assignments: list of (pred_idx, gt_idx) tuples for each batch
        metrics: dict with assignment statistics
    """
    B, N, C, H, W = pred_layers.shape
    device = pred_layers.device

    total_loss = 0.0
    all_assignments = []

    # Metrics
    total_permutations = 0
    total_valid_layers = 0

    for b in range(B):
        n_valid = num_valid[b].item()

        # Compute cost matrix (without gradients for assignment)
        with torch.no_grad():
            cost = torch.zeros(N, N, device=device)

            for i in range(N):  # pred index
                for j in range(n_valid):  # valid GT index
                    # MSE between pred[i] and gt[j]
                    cost[i, j] = F.mse_loss(
                        pred_layers[b, i],
                        gt_layers[b, j],
                        reduction='mean'
                    ).item()

                # Padding GT layers
                for j in range(n_valid, N):
                    if i < n_valid:
                        # Valid pred should not match padding
                        cost[i, j] = 1e8
                    else:
                        # Padding pred -> padding GT (zero cost)
                        cost[i, j] = 0.0

            # Hungarian algorithm for optimal assignment
            pred_idx, gt_idx = linear_sum_assignment(cost.cpu().numpy())

        all_assignments.append((pred_idx, gt_idx))

        # Count permutations (how many layers NOT in canonical position)
        permutation_count = 0
        for p, g in zip(pred_idx, gt_idx):
            if g < n_valid and p != g:  # Not in canonical position
                permutation_count += 1
        total_permutations += permutation_count
        total_valid_layers += n_valid

        # Compute loss with gradients using the assignment
        for i, j in zip(pred_idx, gt_idx):
            if j < n_valid:  # Valid layer
                layer_loss = F.mse_loss(pred_layers[b, i], gt_layers[b, j])

                # Determine weight
                weight = 1.0

                # Background (layer 0): lower weight
                if j == 0:
                    weight = background_weight
                # Target layer: higher weight
                elif target_indices is not None and j == target_indices[b].item():
                    weight = target_layer_weight

                total_loss += weight * layer_loss

    # Metrics
    avg_permutation_rate = total_permutations / max(total_valid_layers, 1)
    metrics = {
        'permutation_rate': avg_permutation_rate,  # 0.0 = all canonical, 1.0 = all permuted
        'total_permutations': total_permutations,
        'total_valid_layers': total_valid_layers
    }

    return total_loss / B, all_assignments, metrics


class ProjectionAutoencoder(nn.Module):
    """
    Order-invariant Autoencoder for merge/decompose
    No mask input - pure compositional learning
    """

    def __init__(self, max_layers=6):
        super().__init__()
        self.max_layers = max_layers

        input_channels = max_layers * 4  # 24 (no mask!)
        output_channels = max_layers * 4  # 24

        # Input projection: 6 layers → merged image
        # (24, h, w) → (4, h, w)
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

    def forward(self, layers):
        """
        Args:
            layers: (B, N, 4, h, w) - layer latents (any order)

        Returns:
            merged_pred: (B, 4, h, w) - predicted merged image latent
            layers_recon: (B, N, 4, h, w) - reconstructed layer latents (any order)
        """
        B, N, C, h, w = layers.shape

        # Flatten layers
        layers_flat = layers.reshape(B, N * C, h, w)  # (B, 24, h, w)

        # Input projection: layers → merged image latent
        merged_pred = self.input_proj(layers_flat)  # (B, 4, h, w)

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
        # Forward pass (no mask needed)
        merged_pred, layers_recon = model(layers_latent_gt)

        # ========================================
        # 4. Compute optimal assignment for visualization
        # ========================================
        _, assignments, _ = optimal_assignment_loss(
            layers_recon,
            layers_latent_gt,
            num_layers,
            background_weight=0.1,
            target_layer_weight=1.0,
            target_indices=None
        )

        # ========================================
        # 5. Decode predictions
        # ========================================
        # Decode merged image
        merged_rgb_pred = vae.decode(merged_pred / scale_factor).sample

        # Decode reconstructed layers
        layers_recon_flat = layers_recon.reshape(B * N, 4, h, w)
        layers_rgb_recon_flat = vae.decode(layers_recon_flat / scale_factor).sample
        layers_rgb_recon = layers_rgb_recon_flat.reshape(B, N, 3, H, W)

        # ========================================
        # 6. Save visualizations
        # ========================================
        for b in range(B):
            if sample_count >= num_samples:
                break

            sample_dir = os.path.join(vis_dir, f'sample_{sample_count:02d}')
            os.makedirs(sample_dir, exist_ok=True)

            n_valid = num_layers[b].item()
            pred_idx, gt_idx = assignments[b]

            # 6a. Merged images comparison
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

            # 6b. Layers comparison (using optimal assignment)
            fig, axes = plt.subplots(2, 6, figsize=(18, 6))
            fig.suptitle(f'Layers Comparison - Sample {sample_count} (Optimal Assignment)', fontsize=16)

            for gt_i in range(6):
                # GT layer
                if gt_i < n_valid:
                    img_gt = layers_rgb[b, gt_i].cpu().permute(1, 2, 0).numpy()
                    img_gt = (img_gt + 1.0) / 2.0
                    axes[0, gt_i].imshow(img_gt.clip(0, 1))
                    axes[0, gt_i].set_title(f'GT Layer {gt_i}')
                else:
                    axes[0, gt_i].imshow(torch.zeros(H, W, 3).numpy())
                    axes[0, gt_i].set_title(f'Padding')
                axes[0, gt_i].axis('off')

                # Find which pred matches this GT
                matching_pred = None
                for p, g in zip(pred_idx, gt_idx):
                    if g == gt_i:
                        matching_pred = p
                        break

                # Reconstructed layer (matched)
                if matching_pred is not None and gt_i < n_valid:
                    img_recon = layers_rgb_recon[b, matching_pred].cpu().permute(1, 2, 0).numpy()
                    img_recon = (img_recon + 1.0) / 2.0
                    axes[1, gt_i].imshow(img_recon.clip(0, 1))
                    axes[1, gt_i].set_title(f'Pred[{matching_pred}]→GT[{gt_i}]')
                else:
                    axes[1, gt_i].imshow(torch.zeros(H, W, 3).numpy())
                    axes[1, gt_i].set_title(f'Padding')
                axes[1, gt_i].axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(sample_dir, 'layers_comparison.png'), dpi=150, bbox_inches='tight')
            plt.close()

            # 6c. Loss statistics
            with open(os.path.join(sample_dir, 'stats.txt'), 'w') as f:
                merge_mse = F.mse_loss(merged_pred[b], merged_latent_gt[b]).item()

                f.write(f"Sample {sample_count} - {image_ids[b]}\n")
                f.write("="*60 + "\n\n")
                f.write(f"Merge Loss (MSE): {merge_mse:.6f}\n\n")
                f.write("Optimal Assignment:\n")
                for p, g in zip(pred_idx, gt_idx):
                    if g < n_valid:
                        layer_mse = F.mse_loss(layers_recon[b, p], layers_latent_gt[b, g]).item()
                        bg_marker = " [BG]" if g == 0 else ""
                        f.write(f"  Pred[{p}] -> GT[{g}]{bg_marker}: MSE={layer_mse:.6f}\n")

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
    parser.add_argument('--background_weight', type=float, default=0.1, help='Weight for layer 0 (background)')
    parser.add_argument('--target_layer_weight', type=float, default=2.0, help='Weight for target foreground layer')
    parser.add_argument('--enable_shuffle', action='store_true', help='Shuffle non-target foreground layers (pos2+)')
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
            # 3. Structured input with fixed positions
            # ========================================
            # Position 0: Background (layer 0) - always fixed
            # Position 1: Target layer - always fixed (for high weight learning)
            # Position 2+: Other foreground layers - shuffled (optional)

            target_indices = torch.zeros(B, dtype=torch.long, device=accelerator.device)
            layers_latent_input = torch.zeros_like(layers_latent_gt)

            for b in range(B):
                n_valid = num_layers[b].item()

                # Position 0: Always background (layer 0)
                layers_latent_input[b, 0] = layers_latent_gt[b, 0]

                if n_valid > 1:
                    # Select target from foreground layers (1 to n_valid-1)
                    target_idx = torch.randint(1, n_valid, (1,), device=accelerator.device).item()
                    target_indices[b] = target_idx

                    # Position 1: Always target layer
                    layers_latent_input[b, 1] = layers_latent_gt[b, target_idx]

                    # Remaining foreground layers (exclude background and target)
                    remaining_indices = [i for i in range(1, n_valid) if i != target_idx]

                    if len(remaining_indices) > 0:
                        if args.enable_shuffle:
                            # Shuffle remaining layers
                            perm = torch.randperm(len(remaining_indices), device=accelerator.device)
                            shuffled_indices = [remaining_indices[p] for p in perm]
                        else:
                            # Keep order
                            shuffled_indices = remaining_indices

                        # Fill positions 2, 3, 4, ... with remaining layers
                        for new_pos, old_idx in enumerate(shuffled_indices):
                            layers_latent_input[b, 2 + new_pos] = layers_latent_gt[b, old_idx]

                # Padding stays as is
                if n_valid < N:
                    layers_latent_input[b, n_valid:] = layers_latent_gt[b, n_valid:]

            # ========================================
            # 4. Forward pass
            # ========================================
            optimizer.zero_grad()

            merged_pred, layers_recon = model(layers_latent_input)

            # ========================================
            # 5. Loss computation
            # ========================================
            # Loss 1: Merge loss (input projection)
            merge_loss = F.mse_loss(merged_pred, merged_latent_gt)

            # Loss 2: Decompose loss with optimal assignment
            decompose_loss, assignments, metrics = optimal_assignment_loss(
                layers_recon,
                layers_latent_gt,
                num_layers,
                background_weight=args.background_weight,
                target_layer_weight=args.target_layer_weight,
                target_indices=target_indices
            )

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

            # Calculate running average
            avg_total = total_loss_sum / count
            avg_merge = merge_loss_sum / count
            avg_decompose = decompose_loss_sum / count

            # Update progress bar with both current and average
            pbar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'loss_avg': f'{avg_total:.4f}',
                'perm': f'{metrics["permutation_rate"]:.2f}'
            })

            if (step + 1) % 50 == 0:
                logger.info(
                    f"Epoch [{epoch+1}/{args.epochs}] Step [{step+1}/{len(train_dataloader)}] "
                    f"Loss: {total_loss.item():.4f} (avg: {avg_total:.4f}) "
                    f"Merge: {merge_loss.item():.4f} (avg: {avg_merge:.4f}) "
                    f"Decomp: {decompose_loss.item():.4f} (avg: {avg_decompose:.4f}) "
                    f"Perm: {metrics['permutation_rate']:.2%}"
                )

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
