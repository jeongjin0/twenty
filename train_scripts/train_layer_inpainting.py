"""
Training script for Layer-wise Inpainting
"""

import os
import sys
import time
import datetime
import random
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from diffusers.models import AutoencoderKL

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusion.model.nets.PixArt_layer_inpainting import PixArtLayerInpainting, load_pretrained_pixart
from diffusion.model.t5 import T5Embedder
from diffusion.iddpm import IDDPM
from dataset.mulan_dataset import MuLanDataset
from tools.logger import get_root_logger
from tools.train_utils import (
    AverageMeter,
    DebugUnderflowOverflow,
)

try:
    from accelerate import Accelerator
    from accelerate.utils import set_seed
except ImportError:
    print("Please install accelerate: pip install accelerate")
    sys.exit(1)


def read_config(config_path):
    """Read config file"""
    import importlib.util
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


def collate_fn(batch):
    """Custom collate function for variable number of layers"""
    # Each item: {'layers': (N, 3, H, W), 'captions': List[str], 'num_layers': int}
    return batch


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--pretrained_pixart', type=str, default=None, help='Pretrained PixArt checkpoint')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    args = parser.parse_args()

    # Read config
    config = read_config(args.config)

    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=getattr(config, 'gradient_accumulation_steps', 1),
        mixed_precision=getattr(config, 'mixed_precision', 'fp16'),
    )

    # Set seed
    if accelerator.is_main_process:
        set_seed(getattr(config, 'seed', 42))

    # Logger
    logger = get_root_logger(
        log_file=os.path.join(config.work_dir, 'train.log') if accelerator.is_main_process else None,
        log_level='INFO'
    )

    if accelerator.is_main_process:
        logger.info(f"Config: {args.config}")
        logger.info(f"Working directory: {config.work_dir}")
        os.makedirs(config.work_dir, exist_ok=True)
        os.makedirs(os.path.join(config.work_dir, 'checkpoints'), exist_ok=True)

    # ============================================
    # Build Model
    # ============================================
    logger.info("Building model...")

    # Load pretrained PixArt
    if args.pretrained_pixart:
        logger.info(f"Loading pretrained PixArt from: {args.pretrained_pixart}")
        pretrained_pixart = load_pretrained_pixart(
            args.pretrained_pixart,
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
        model.pixart.enable_gradient_checkpointing()
        logger.info("Gradient checkpointing enabled")

    # ============================================
    # Build Dataset & Dataloader
    # ============================================
    logger.info("Loading dataset...")

    dataset = MuLanDataset(
        csv_files=config.data_root,
        resolution=config.image_size,
        max_layers=config.max_layers,
        caption_type=config.caption_type,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    logger.info(f"Dataset size: {len(dataset)}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Total steps per epoch: {len(dataloader)}")

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
    diffusion = IDDPM(str(config.train_sampling_steps))

    # ============================================
    # Optimizer
    # ============================================
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
    )

    # ============================================
    # Prepare with Accelerator
    # ============================================
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    # ============================================
    # Resume from checkpoint
    # ============================================
    start_epoch = 0
    global_step = 0

    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt.get('epoch', 0)
        global_step = ckpt.get('global_step', 0)
        logger.info(f"Resumed from epoch {start_epoch}, step {global_step}")

    # ============================================
    # Training Loop
    # ============================================
    logger.info("Starting training...")

    log_buffer = AverageMeter()
    data_time_all = 0
    last_tic = time.time()

    for epoch in range(start_epoch, config.num_epochs):
        model.train()

        for step, batch_list in enumerate(dataloader):
            # batch_list is a list of dict, each with variable number of layers
            data_time_start = last_tic
            data_time_all += time.time() - data_time_start

            # ========================================
            # Process batch with variable layers
            # ========================================
            B = len(batch_list)
            max_layers = config.max_layers
            H, W = config.image_size, config.image_size
            h, w = H // 8, W // 8

            # Collect layers and captions
            all_layers = []  # List of (N_i, 3, H, W)
            all_captions = []  # List of List[str]
            all_num_layers = []  # List of int

            for item in batch_list:
                layers = item['layers']  # (N_i, 3, H, W)
                captions = item['captions']  # List[str], length N_i
                num_layers = item['num_layers']  # int

                # Random shuffle layers (order doesn't matter)
                perm = torch.randperm(num_layers)
                layers = layers[perm]
                captions = [captions[i] for i in perm]

                all_layers.append(layers)
                all_captions.append(captions)
                all_num_layers.append(num_layers)

            # ========================================
            # Encode to VAE latent with padding
            # ========================================
            with torch.no_grad():
                z_all = []

                for b in range(B):
                    layers = all_layers[b]  # (N_i, 3, H, W)
                    num_layers = all_num_layers[b]

                    # Encode existing layers
                    layers_device = layers.to(accelerator.device)
                    z = vae.encode(layers_device).latent_dist.mode() * 0.18215
                    # (N_i, 4, h, w)

                    # Pad to max_layers with zeros (black)
                    if num_layers < max_layers:
                        padding = torch.zeros(
                            max_layers - num_layers, 4, h, w,
                            device=z.device, dtype=z.dtype
                        )
                        z = torch.cat([z, padding], dim=0)

                    z_all.append(z)

                # Stack: (B, max_layers, 4, h, w)
                z_clean = torch.stack(z_all, dim=0)

            # ========================================
            # Random layer masking (exactly 1 layer)
            # ========================================
            layer_mask = torch.zeros(B, max_layers, device=accelerator.device)
            masked_captions = []

            for b in range(B):
                num_layers = all_num_layers[b]
                # Randomly select ONE layer to mask (from valid layers only)
                masked_idx = random.randint(0, num_layers - 1)
                layer_mask[b, masked_idx] = 1

                # Use the masked layer's caption
                masked_captions.append(all_captions[b][masked_idx])

            # ========================================
            # Sample timesteps
            # ========================================
            timesteps = torch.randint(
                0, config.train_sampling_steps, (B,),
                device=accelerator.device
            ).long()

            # ========================================
            # Add noise
            # ========================================
            noise = torch.randn_like(z_clean)

            # Get alpha values
            alpha_t = diffusion.iddpm.alphas_cumprod[timesteps]  # (B,)
            alpha_t = alpha_t.view(B, 1, 1, 1, 1)

            # Add noise to ALL layers
            z_noisy = torch.sqrt(alpha_t) * z_clean + torch.sqrt(1 - alpha_t) * noise

            # Replace visible layers with clean latents
            # Only masked layer has noise
            layer_mask_expanded = layer_mask.view(B, max_layers, 1, 1, 1)
            z_input = z_noisy * layer_mask_expanded + z_clean * (1 - layer_mask_expanded)

            # ========================================
            # Encode text (masked layer captions)
            # ========================================
            with torch.no_grad():
                caption_embs, emb_masks = text_encoder.get_text_embeddings(masked_captions)
                y = caption_embs.float()[:, None].to(accelerator.device)
                y_mask = emb_masks.to(accelerator.device)

            # ========================================
            # Forward pass
            # ========================================
            with accelerator.accumulate(model):
                optimizer.zero_grad()

                noise_pred = model(
                    layers=z_input,
                    layer_mask=layer_mask,
                    timestep=timesteps,
                    y=y,
                    mask=y_mask
                )

                # ========================================
                # Compute loss (only on masked layer)
                # ========================================
                # If pred_sigma=True, output is (B, max_layers, 8, h, w)
                if config.pred_sigma and noise_pred.shape[2] == 8:
                    noise_pred = noise_pred[:, :, :4]  # Take only noise prediction

                # MSE loss
                layer_loss = F.mse_loss(
                    noise_pred,
                    noise,
                    reduction='none'
                )  # (B, max_layers, 4, h, w)

                # Average over spatial and channel dims
                layer_loss = layer_loss.mean(dim=[2, 3, 4])  # (B, max_layers)

                # Apply layer mask (only masked layer)
                masked_loss = (layer_loss * layer_mask).sum() / layer_mask.sum()

                # Backward
                accelerator.backward(masked_loss)

                # Gradient clipping
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.gradient_clip)

                optimizer.step()

            # ========================================
            # Logging
            # ========================================
            log_buffer.update({'loss': masked_loss.item()})

            global_step += 1

            # Log every N steps
            if step % config.log_interval == 0 and step > 0:
                lr = optimizer.param_groups[0]['lr']
                t = time.time() - last_tic
                t_d = data_time_all
                avg_time = log_buffer.meters['loss'].avg_time if hasattr(log_buffer.meters['loss'], 'avg_time') else t

                eta_epoch = str(datetime.timedelta(seconds=int(avg_time * (len(dataloader) - step - 1))))

                log_buffer.average()

                info = f"Step/Epoch [{global_step}/{epoch}][{step + 1}/{len(dataloader)}]: " \
                       f"epoch_eta: {eta_epoch}, time: {t:.3f}, time_data: {t_d:.3f}, " \
                       f"lr: {lr:.3e}, "
                info += ', '.join([f"{k}: {v:.4f}" for k, v in log_buffer.output.items()])
                logger.info(info)

                last_tic = time.time()
                log_buffer.clear()
                data_time_all = 0

        # ========================================
        # Save checkpoint
        # ========================================
        if accelerator.is_main_process and (epoch + 1) % config.save_model_epochs == 0:
            save_path = os.path.join(
                config.work_dir,
                'checkpoints',
                f'epoch_{epoch + 1}_step_{global_step}.pth'
            )
            accelerator.save({
                'model': accelerator.unwrap_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
                'global_step': global_step,
                'config': args.config,
            }, save_path)
            logger.info(f"Saved checkpoint: {save_path}")

    logger.info("Training finished!")


if __name__ == '__main__':
    train()
