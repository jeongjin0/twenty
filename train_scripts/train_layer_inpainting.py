"""
Training script for Layer-wise Inpainting
"""

import argparse
import datetime
import gc
import os
import sys
import time
import random
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.models import AutoencoderKL
from mmcv.runner import LogBuffer

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['NCCL_P2P_DISABLE'] = '1'

from diffusion.model.nets.PixArt_layer_inpainting import PixArtLayerInpainting, load_pretrained_pixart
from diffusion.model.t5 import T5Embedder
from diffusion import IDDPM
from diffusion.data.multilayer_builder import build_mulan_dataloader
from diffusion.utils.logger import get_root_logger
from diffusion.utils.misc import set_random_seed, read_config, DebugUnderflowOverflow
from diffusion.utils.checkpoint import save_checkpoint, load_checkpoint
from diffusion.utils.lr_scheduler import build_lr_scheduler
from diffusion.utils.optimizer import build_optimizer, auto_scale_lr
from diffusion.utils.dist_utils import get_world_size
from accelerate import Accelerator, InitProcessGroupKwargs
from copy import deepcopy

warnings.filterwarnings("ignore")


def ema_update(model_dest, model_src, rate):
    """Update EMA model parameters"""
    param_dict_src = dict(model_src.named_parameters())
    for p_name, p_dest in model_dest.named_parameters():
        p_src = param_dict_src[p_name]
        assert p_src is not p_dest
        p_dest.data.mul_(rate).add_((1 - rate) * p_src.data)


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--pretrained_pixart', type=str, default=None, help='Pretrained PixArt checkpoint')
    parser.add_argument('--pretrained_projections', type=str, default=None, help='Pretrained projection weights')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    args = parser.parse_args()

    # Read config
    config = read_config(args.config)

    # Initialize accelerator
    kwargs = InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=7200))
    accelerator = Accelerator(
        mixed_precision=getattr(config, 'mixed_precision', 'fp16'),
        gradient_accumulation_steps=getattr(config, 'gradient_accumulation_steps', 1),
        kwargs_handlers=[kwargs],
    )

    # Set seed
    set_random_seed(getattr(config, 'seed', 42))

    # Create work directory first
    if accelerator.is_main_process:
        os.makedirs(config.work_dir, exist_ok=True)
        os.makedirs(os.path.join(config.work_dir, 'checkpoints'), exist_ok=True)

    # Logger
    logger = get_root_logger(
        log_file=os.path.join(config.work_dir, 'train.log') if accelerator.is_main_process else None,
        log_level='INFO'
    )

    if accelerator.is_main_process:
        logger.info(f"Config: {args.config}")
        logger.info(f"Working directory: {config.work_dir}")

    # ============================================
    # Build Model
    # ============================================
    logger.info("Building model...")

    # Load pretrained PixArt
    pretrained_path = args.pretrained_pixart or getattr(config, 'pretrained_pixart_path', None)
    if pretrained_path:
        logger.info(f"Loading pretrained PixArt from: {pretrained_path}")
        pretrained_pixart = load_pretrained_pixart(
            pretrained_path,
            input_size=config.image_size // 8
        )
    else:
        logger.info("No pretrained PixArt specified, training from scratch")
        pretrained_pixart = None

    # Build layer inpainting model
    model = PixArtLayerInpainting(
        pretrained_pixart=pretrained_pixart,
        max_layers=config.max_layers,
        input_size=config.image_size // 8,
        pred_sigma=config.pred_sigma,
        caption_channels=4096,
        model_max_length=config.model_max_length,
    )

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Enable gradient checkpointing
    if getattr(config, 'gradient_checkpointing', False):
        model.enable_gradient_checkpointing()
        logger.info("Gradient checkpointing enabled")

    # ============================================
    # Load Pretrained Projection Weights
    # ============================================
    if args.pretrained_projections:
        logger.info(f"Loading pretrained projections from: {args.pretrained_projections}")
        pretrained_proj = torch.load(args.pretrained_projections, map_location='cpu')

        # Extract state dict
        if 'state_dict' in pretrained_proj:
            proj_state_dict = pretrained_proj['state_dict']
        else:
            proj_state_dict = pretrained_proj

        # Load only projection weights
        model_dict = model.state_dict()
        proj_dict = {}

        for k, v in proj_state_dict.items():
            if 'input_proj' in k or 'output_proj' in k:
                proj_dict[k] = v

        logger.info(f"Loading {len(proj_dict)} projection parameters")

        model_dict.update(proj_dict)
        model.load_state_dict(model_dict)

        logger.info("✓ Pretrained projections loaded successfully")
        logger.info("  - Input projection: 6 layers → merged image latent")
        logger.info("  - Output projection: merged image latent → 6 layers")

    # ============================================
    # Build Dataset & Dataloader
    # ============================================
    logger.info("Loading dataset...")

    train_dataloader = build_mulan_dataloader(
        data_roots=config.data_roots,
        batch_size=config.batch_size,
        resolution=config.image_size,
        max_layers=config.max_layers,
        num_workers=config.num_workers,
        shuffle=True,
        caption_type=getattr(config, 'caption_type', 'blip2')
    )

    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Total steps per epoch: {len(train_dataloader)}")

    # ============================================
    # VAE & Text Encoder
    # ============================================
    logger.info("Loading VAE and T5...")

    vae = AutoencoderKL.from_pretrained(config.vae_pretrained).to(accelerator.device)
    vae.eval()
    vae.requires_grad_(False)

    text_encoder = T5Embedder(
        device=accelerator.device,
        local_cache=True,
        cache_dir=config.text_encoder_name,
        torch_dtype=torch.float16
    )

    # ============================================
    # Diffusion
    # ============================================
    pred_sigma = getattr(config, 'pred_sigma', True)
    learn_sigma = pred_sigma and getattr(config, 'learn_sigma', True)
    diffusion = IDDPM(
        str(config.train_sampling_steps),
        learn_sigma=learn_sigma,
        pred_sigma=pred_sigma,
        snr=getattr(config, 'snr_loss', False)
    )

    # ============================================
    # Optimizer & LR Scheduler
    # ============================================
    # Auto-scale learning rate for distributed training
    lr_scale_ratio = 1
    if config.get('auto_lr', None):
        lr_scale_ratio = auto_scale_lr(
            config.batch_size * get_world_size() * config.gradient_accumulation_steps,
            config.optimizer,
            **config.auto_lr
        )
        logger.info(f"Auto scaling lr by ratio: {lr_scale_ratio:.2f}")

    optimizer = build_optimizer(model, config.optimizer)
    lr_scheduler = build_lr_scheduler(config, optimizer, train_dataloader, lr_scale_ratio)

    # Create EMA model
    model_ema = deepcopy(model).eval()
    ema_update(model_ema, model, 0.)
    logger.info("EMA model created")

    # ============================================
    # 2-Stage Training: Initial freeze (before DDP)
    # ============================================
    pixart_freeze_epochs = getattr(config, 'pixart_freeze_epochs', 0)
    if pixart_freeze_epochs > 0:
        # Freeze PixArt for Stage 1 (must be done BEFORE accelerator.prepare)
        model.pixart.requires_grad_(False)
        logger.info(f"2-Stage Training: PixArt frozen for first {pixart_freeze_epochs} epochs")

    # ============================================
    # Prepare with Accelerator
    # ============================================
    model, model_ema = accelerator.prepare(model, model_ema)
    optimizer, train_dataloader, lr_scheduler = accelerator.prepare(optimizer, train_dataloader, lr_scheduler)

    # ============================================
    # Resume from checkpoint
    # ============================================
    start_epoch = 0

    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        start_epoch, missing, unexpected = load_checkpoint(
            checkpoint=args.resume,
            model=model,
            model_ema=model_ema,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            load_ema=False,
            resume_optimizer=True,
            resume_lr_scheduler=True
        )
        logger.warning(f'Missing keys: {missing}')
        logger.warning(f'Unexpected keys: {unexpected}')
        logger.info(f"Resumed from epoch {start_epoch}")

    # ============================================
    # Training Loop
    # ============================================
    logger.info("Starting training...")

    if getattr(config, 'debug_nan', False):
        DebugUnderflowOverflow(model)
        logger.info('NaN debugger registered. Start to detect overflow during training.')

    time_start, last_tic = time.time(), time.time()
    log_buffer = LogBuffer()

    grad_norm = None
    global_step = start_epoch * len(train_dataloader)
    total_steps = len(train_dataloader) * config.num_epochs

    # 2-Stage Training Configuration
    pixart_freeze_epochs = getattr(config, 'pixart_freeze_epochs', 0)
    if pixart_freeze_epochs > 0:
        logger.info(f"2-Stage Training enabled:")
        logger.info(f"  Stage 1 (Epoch 0-{pixart_freeze_epochs-1}): Train projections only (PixArt frozen)")
        logger.info(f"  Stage 2 (Epoch {pixart_freeze_epochs}+): Train full model (PixArt unfrozen)")

    for epoch in range(start_epoch, config.num_epochs):
        # ============================================
        # 2-Stage Training: Unfreeze at Stage 2
        # ============================================
        if pixart_freeze_epochs > 0 and epoch == pixart_freeze_epochs:
            # Stage 2: Unfreeze PixArt
            accelerator.unwrap_model(model).pixart.requires_grad_(True)
            logger.info(f"[Epoch {epoch}] Stage 2: PixArt unfrozen, training full model")
            # Optionally reduce learning rate when unfreezing
            if hasattr(config, 'pixart_unfreeze_lr_scale'):
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= config.pixart_unfreeze_lr_scale
                logger.info(f"  Scaled learning rate by {config.pixart_unfreeze_lr_scale}")

        model.train()
        data_time_all = 0

        for step, batch in enumerate(train_dataloader):
            data_time_start = time.time()

            # Unpack batch
            layers, captions, num_layers, image_ids = batch
            layers = layers.to(accelerator.device)
            num_layers = num_layers.to(accelerator.device)

            data_time_all += time.time() - data_time_start

            B = layers.shape[0]  # Batch size
            N = layers.shape[1]  # max_layers
            max_layers = config.max_layers
            H, W = config.image_size, config.image_size
            h, w = H // 8, W // 8

            # ========================================
            # Encode to VAE latent with padding
            # ========================================
            with torch.no_grad():
                # layers: (B, N, 4, H, W) - RGBA
                # Extract RGB only (first 3 channels)
                layers_rgb = layers[:, :, :3, :, :]  # (B, N, 3, H, W)

                # Flatten: (B, N, 3, H, W) → (B*N, 3, H, W)
                layers_flat = layers_rgb.reshape(B * N, 3, H, W)

                # VAE encode
                scale_factor = getattr(config, 'scale_factor', 0.18215)
                z_flat = vae.encode(layers_flat).latent_dist.mode() * scale_factor
                # (B*N, 4, h, w)

                # Reshape back: (B*N, 4, h, w) → (B, N, 4, h, w)
                z_clean = z_flat.reshape(B, N, 4, h, w)

                # Memory optimization
                torch.cuda.empty_cache()

            # ========================================
            # Random layer masking (exactly 1 layer)
            # ========================================
            layer_mask = torch.zeros(B, max_layers, device=accelerator.device)
            masked_captions = []

            for b in range(B):
                # num_layers[b] is the actual number of valid layers for this sample
                n_valid = num_layers[b].item()
                # Randomly select ONE layer to mask (from valid layers only)
                masked_idx = random.randint(0, n_valid - 1)
                layer_mask[b, masked_idx] = 1

                # Use the masked layer's caption
                masked_captions.append(captions[b][masked_idx])

            # ========================================
            # Encode text (masked layer captions)
            # ========================================
            with torch.no_grad():
                caption_embs, emb_masks = text_encoder.get_text_embeddings(masked_captions)
                y = caption_embs.float()[:, None].to(accelerator.device)
                y_mask = emb_masks.to(accelerator.device)

            # ========================================
            # Sample timesteps
            # ========================================
            timesteps = torch.randint(
                0, config.train_sampling_steps, (B,),
                device=accelerator.device
            ).long()

            # ========================================
            # Training step
            # ========================================
            with accelerator.accumulate(model):
                optimizer.zero_grad()

                # Forward diffusion: add noise using q_sample
                noise = torch.randn_like(z_clean)

                # Apply q_sample to all layers
                # Need to flatten to (B*N, 4, h, w) for q_sample
                B, N, C, h, w = z_clean.shape
                z_clean_flat = z_clean.reshape(B * N, C, h, w)
                noise_flat = noise.reshape(B * N, C, h, w)
                timesteps_expanded = timesteps.unsqueeze(1).expand(B, N).reshape(B * N)

                z_noisy_flat = diffusion.q_sample(z_clean_flat, timesteps_expanded, noise=noise_flat)
                z_noisy = z_noisy_flat.reshape(B, N, C, h, w)

                # Replace visible layers with clean latents
                layer_mask_expanded = layer_mask.view(B, max_layers, 1, 1, 1)
                z_input = z_noisy * layer_mask_expanded + z_clean * (1 - layer_mask_expanded)

                # Predict noise
                noise_pred = model(
                    layers=z_input,
                    layer_mask=layer_mask,
                    timestep=timesteps,
                    y=y,
                    mask=y_mask
                )

                # ========================================
                # Compute loss (masked + visible layers)
                # ========================================
                # If pred_sigma=True, output is (B, max_layers, 8, h, w)
                if config.pred_sigma and noise_pred.shape[2] == 8:
                    noise_pred = noise_pred[:, :, :4]  # Take only noise prediction

                # Create valid layer mask (excluding padding)
                valid_mask = torch.zeros(B, max_layers, device=accelerator.device)
                for b in range(B):
                    n_valid = num_layers[b].item()
                    valid_mask[b, :n_valid] = 1

                # Compute noise prediction loss (per layer)
                noise_loss = F.mse_loss(
                    noise_pred,
                    noise,
                    reduction='none'
                ).mean(dim=[2, 3, 4])  # (B, max_layers)

                # Compute zero-noise loss for visible layers
                zero_noise_loss = F.mse_loss(
                    noise_pred,
                    torch.zeros_like(noise),
                    reduction='none'
                ).mean(dim=[2, 3, 4])  # (B, max_layers)

                # Create weighted loss mask
                loss_weight_mask = torch.zeros(B, max_layers, device=accelerator.device)

                masked_layer_weight = getattr(config, 'masked_layer_weight', 1.0)
                visible_layer_weight = getattr(config, 'visible_layer_weight', 0.5)
                background_layer_weight = getattr(config, 'background_layer_weight', 0.3)

                for b in range(B):
                    n_valid = num_layers[b].item()
                    for i in range(n_valid):
                        # Determine layer type and weight
                        is_masked = layer_mask[b, i].item() == 1
                        is_background = (i == 0)

                        if is_masked:
                            # Masked (target) layer: highest weight
                            weight = masked_layer_weight
                        elif is_background:
                            # Background layer: lowest weight
                            weight = visible_layer_weight * background_layer_weight
                        else:
                            # Visible (reference) layers: medium weight
                            weight = visible_layer_weight

                        loss_weight_mask[b, i] = weight

                # Compute weighted loss
                # Masked layer: predict actual noise
                # Visible layers: predict zero noise (since they're clean)
                combined_loss = torch.zeros(B, max_layers, device=accelerator.device)
                for b in range(B):
                    for i in range(max_layers):
                        if layer_mask[b, i] == 1:  # Masked
                            combined_loss[b, i] = noise_loss[b, i]
                        elif valid_mask[b, i] == 1:  # Visible
                            combined_loss[b, i] = zero_noise_loss[b, i]

                total_loss = (combined_loss * loss_weight_mask).sum() / (loss_weight_mask.sum() + 1e-8)

                # For logging
                masked_loss = (noise_loss * layer_mask).sum() / (layer_mask.sum() + 1e-8)
                visible_mask = (1 - layer_mask) * valid_mask
                visible_loss = (zero_noise_loss * visible_mask).sum() / (visible_mask.sum() + 1e-8)

                # Backward
                accelerator.backward(total_loss)

                # Gradient clipping
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), config.gradient_clip)

                optimizer.step()
                lr_scheduler.step()

                # Update EMA model
                if accelerator.sync_gradients:
                    ema_rate = getattr(config, 'ema_rate', 0.9999)
                    ema_update(model_ema, model, ema_rate)

            # ========================================
            # Logging
            # ========================================
            lr = lr_scheduler.get_last_lr()[0]
            logs = {
                'loss': accelerator.gather(total_loss).mean().item(),
                'masked': accelerator.gather(masked_loss).mean().item(),
                'visible': accelerator.gather(visible_loss).mean().item()
            }
            if grad_norm is not None:
                logs.update(grad_norm=accelerator.gather(grad_norm).mean().item())
            log_buffer.update(logs)

            # Log every N steps
            if (step + 1) % config.log_interval == 0 or (step + 1) == 1:
                t = (time.time() - last_tic) / config.log_interval
                t_d = data_time_all / config.log_interval
                avg_time = (time.time() - time_start) / (global_step + 1)
                start_step = start_epoch * len(train_dataloader)
                eta = str(datetime.timedelta(seconds=int(avg_time * (total_steps - start_step - global_step - 1))))
                eta_epoch = str(datetime.timedelta(seconds=int(avg_time * (len(train_dataloader) - step - 1))))

                log_buffer.average()

                info = f"Step/Epoch [{global_step + 1}/{epoch}][{step + 1}/{len(train_dataloader)}]: " \
                       f"total_eta: {eta}, epoch_eta: {eta_epoch}, time_all: {t:.3f}, time_data: {t_d:.3f}, " \
                       f"lr: {lr:.3e}, "
                info += ', '.join([f"{k}: {v:.4f}" for k, v in log_buffer.output.items()])
                logger.info(info)

                last_tic = time.time()
                log_buffer.clear()
                data_time_all = 0

            global_step += 1

            # Memory optimization: periodic cache cleanup every 10 steps
            if (step + 1) % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()

            # ========================================
            # Save checkpoint (by steps)
            # ========================================
            save_model_steps = getattr(config, 'save_model_steps', None)
            if save_model_steps is not None and global_step % save_model_steps == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    os.umask(0o000)
                    save_checkpoint(
                        os.path.join(config.work_dir, 'checkpoints'),
                        epoch=epoch,
                        step=global_step,
                        model=accelerator.unwrap_model(model),
                        model_ema=accelerator.unwrap_model(model_ema),
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler
                    )
                    logger.info(f"Saved checkpoint at step {global_step}")

        # ========================================
        # Save checkpoint (by epochs)
        # ========================================
        if (epoch + 1) % config.save_model_epochs == 0 or (epoch + 1) == config.num_epochs:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                os.umask(0o000)
                save_checkpoint(
                    os.path.join(config.work_dir, 'checkpoints'),
                    epoch=epoch + 1,
                    step=global_step,
                    model=accelerator.unwrap_model(model),
                    model_ema=accelerator.unwrap_model(model_ema),
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler
                )
                logger.info(f"Saved checkpoint at epoch {epoch + 1}")

    logger.info("Training finished!")


if __name__ == '__main__':
    train()
