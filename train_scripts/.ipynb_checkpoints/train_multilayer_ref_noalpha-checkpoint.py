import argparse
import datetime
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['NCCL_P2P_DISABLE'] = '1'

import time
import types
import warnings
from copy import deepcopy
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator, InitProcessGroupKwargs, DistributedDataParallelKwargs
from accelerate.utils import DistributedType
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt


from diffusers.models import AutoencoderKL
from mmcv.runner import LogBuffer
from torch.utils.data import RandomSampler

from diffusion import IDDPM, ReferenceIDDPM
from diffusion.data.builder import build_dataset, build_dataloader, set_data_root
from diffusion.model.builder import build_model
from diffusion.model.t5 import T5Embedder
from diffusion.utils.checkpoint import save_checkpoint, load_checkpoint
from diffusion.utils.data_sampler import AspectRatioBatchSampler, BalancedAspectRatioBatchSampler
from diffusion.utils.dist_utils import get_world_size, clip_grad_norm_
from diffusion.utils.logger import get_root_logger
from diffusion.utils.lr_scheduler import build_lr_scheduler
from diffusion.utils.misc import set_random_seed, read_config, init_random_seed, DebugUnderflowOverflow
from diffusion.utils.optimizer import build_optimizer, auto_scale_lr
from accelerate import Accelerator, InitProcessGroupKwargs, DistributedDataParallelKwargs
warnings.filterwarnings("ignore")  # ignore warning

current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))


def setup_freeze(model, freeze_pretrained=True):
    must_train = [
        'ref_encoder',    # 새로 추가됨
        'pos_embed',      # Resize됨 → 학습 필요!
    ]
    
    frozen_count = 0
    trainable_count = 0
    
    for name, param in model.named_parameters():
        if any(key in name for key in must_train):
            param.requires_grad = True
            trainable_count += 1
        elif freeze_pretrained:
            param.requires_grad = False
            frozen_count += 1
        else:
            param.requires_grad = True
            trainable_count += 1
    
    print(f"[Freeze] Frozen: {frozen_count}, Trainable: {trainable_count}")
    
    # 학습되는 파라미터 확인
    trainable_params = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f"[Trainable params]: {trainable_params[:10]}...")  # 처음 10개만
    
    return model


def print_gpu_memory(tag=""):
    """GPU 메모리 사용량 출력"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        print(f"[{tag}] Allocated: {allocated:.2f}GB | Reserved: {reserved:.2f}GB | Max: {max_allocated:.2f}GB")

def reset_memory_stats():
    """메모리 통계 초기화"""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()


def encode_reference_vae_rgb_batch(vae, layers, num_layers, target_indices, scale_factor=0.18215):
    """
    Alpha 없이 RGB만 처리하는 VAE 인코딩
    
    Returns:
        z_target: (B, 4, h, w)
        z_ref: (B, max_ref, 4, h, w)
    """
    B, N, C, H, W = layers.shape
    device = layers.device
    
    # RGB만 VAE encoding
    rgb = layers[:, :, :3, :, :]  # (B, N, 3, H, W)
    
    rgb_flat = rgb.reshape(B * N, 3, H, W)
    z_rgb_flat = vae.encode(rgb_flat).latent_dist.mode() * scale_factor
    
    _, C_latent, h, w = z_rgb_flat.shape  # C_latent = 4
    z_all = z_rgb_flat.reshape(B, N, C_latent, h, w)  # (B, N, 4, h, w)
    
    # Batch별로 target/reference 분리
    z_target_list = []
    z_ref_list = []
    
    for b in range(B):
        t_idx = target_indices[b]
        z_target_list.append(z_all[b, t_idx])  # (4, h, w)
        
        ref_indices = [i for i in range(N) if i != t_idx]
        z_ref_b = z_all[b, ref_indices]  # (N-1, 4, h, w)
        z_ref_list.append(z_ref_b)
    
    z_target = torch.stack(z_target_list, dim=0)  # (B, 4, h, w)
    z_ref = torch.stack(z_ref_list, dim=0)  # (B, N-1, 4, h, w)
    
    return z_target, z_ref


def set_fsdp_env():
    os.environ["ACCELERATE_USE_FSDP"] = 'true'
    os.environ["FSDP_AUTO_WRAP_POLICY"] = 'TRANSFORMER_BASED_WRAP'
    os.environ["FSDP_BACKWARD_PREFETCH"] = 'BACKWARD_PRE'
    os.environ["FSDP_TRANSFORMER_CLS_TO_WRAP"] = 'PixArtBlock'


def ema_update(model_dest: nn.Module, model_src: nn.Module, rate):
    param_dict_src = dict(model_src.named_parameters())
    for p_name, p_dest in model_dest.named_parameters():
        p_src = param_dict_src[p_name]
        assert p_src is not p_dest
        p_dest.data.mul_(rate).add_((1 - rate) * p_src.data)


def run_evaluation(model, vae, multilayer_diffusion, dataloader, text_encoder, 
                   accelerator, config, epoch, step, save_dir):
    """Training 중 evaluation 실행"""
    model.eval()
    
    # 첫 번째 batch만 사용
    batch = next(iter(dataloader))
    layers, captions, num_layers, image_ids = batch
    layers = layers.to(accelerator.device)
    
    B, N = layers.shape[0], layers.shape[1]
    
    # VAE encoding
    with torch.no_grad():
        actual_layers = num_layers[0].item()
        target_idx = torch.randint(0, actual_layers, (1,)).item() #target layer is 0 or 1
        target_indices = [min(target_idx, num_layers[b].item() - 1) for b in range(B)]

        z_target, z_ref = encode_reference_vae_rgb_batch(
            vae, layers, num_layers,
            target_indices=target_indices,
            scale_factor=config.scale_factor
        )

        target_captions = [captions[b][target_indices[b]] for b in range(B)]
        # T5 encoding
        caption_embs, emb_masks = text_encoder.get_text_embeddings(target_captions)
        y = caption_embs.float()[:, None]
    

    with torch.no_grad():
          
        # Evaluate
        results = multilayer_diffusion.evaluate(
            model=accelerator.unwrap_model(model),
            vae=vae,
            z_target=z_target,
            z_ref=z_ref,
            y=y,
            steps=20,
            cfg_scale=4.5,
            scale_factor=config.scale_factor,
        )
    
    # Plot and save
    os.makedirs(save_dir, exist_ok=True)
    
    for b in range(min(B, 4)):  # 최대 4개 샘플
        n_ref = z_ref.shape[1]
        
        # Collect images: refs + target(GT) + generated
        images = []
        for r in range(n_ref):
            images.append(results['img_ref'][b, r])  # (3, H, W)
        images.append(results['img_target'][b])  # GT
        images.append(results['img_gen'][b])  # Generated
        
        # Make grid
        grid = make_grid(torch.stack(images), nrow=n_ref + 2, normalize=True, value_range=(-1, 1))
        
        save_path = os.path.join(save_dir, f'eval_epoch{epoch}_step{step}_sample{b}.png')
        save_image(grid, save_path)
    
    # Log
    logger.info(f"[Eval] Epoch {epoch}, Step {step}: MSE={results['mse']:.6f}")
    logger.info(f"  z_gen stats - mean: {results['z_gen'].mean():.4f}, std: {results['z_gen'].std():.4f}")
    logger.info(f"  z_target stats - mean: {results['z_target'].mean():.4f}, std: {results['z_target'].std():.4f}")
    
    model.train()
    return results['mse']

def init_ref_proj_zero(model):
    """ref_proj 출력을 0으로 초기화"""
    if hasattr(model, 'ref_proj'):
        ref_proj = model.ref_proj
        if isinstance(ref_proj, nn.Linear):
            nn.init.zeros_(ref_proj.weight)
            if ref_proj.bias is not None:
                nn.init.zeros_(ref_proj.bias)
            print("✓ ref_proj (Linear) initialized to zero")
        elif isinstance(ref_proj, nn.Sequential):
            # 마지막 layer만 0으로
            for layer in reversed(list(ref_proj.modules())):
                if isinstance(layer, nn.Linear):
                    nn.init.zeros_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                    print("✓ ref_proj (Sequential) last Linear initialized to zero")
                    break
    else:
        print("⚠ model has no ref_proj attribute")

def train():
    global optimizer, model
    #init_ref_proj_zero(model)
    #model = setup_freeze(model, freeze_pretrained=True)

    if config.get('debug_nan', False):
        DebugUnderflowOverflow(model)
        logger.info('NaN debugger registered. Start to detect overflow during training.')
    
    time_start, last_tic = time.time(), time.time()
    log_buffer = LogBuffer()

    grad_norm = None
    start_step = start_epoch * len(train_dataloader)
    global_step = 0
    total_steps = len(train_dataloader) * config.num_epochs

    for epoch in range(start_epoch + 1, config.num_epochs + 1):
        data_time_start = time.time()
        data_time_all = 0
        
        for step, batch in enumerate(train_dataloader):
            data_time_all += time.time() - data_time_start
            
            # 1. Unpack batch
            layers, captions, num_layers, image_ids = batch
            layers = layers.to(accelerator.device)
            num_layers = num_layers.to(accelerator.device)
            B = layers.shape[0]
            N = layers.shape[1]


            with torch.no_grad():
                # 각 샘플마다 유효 레이어 범위 내에서 랜덤 target 선택
                target_indices = [torch.randint(0, num_layers[b].item(), (1,)).item() for b in range(B)]
                z_target, z_ref = encode_reference_vae_rgb_batch(
                    vae, layers, num_layers,
                    target_indices=target_indices,
                    scale_factor=config.scale_factor
                )                

            with torch.no_grad():
                target_captions = [captions[b][target_indices[b]] for b in range(B)]
                caption_embs, emb_masks = text_encoder.get_text_embeddings(target_captions)
                y = caption_embs.float()[:, None]
                y_mask = emb_masks

            # 4. Sample timesteps
            timesteps = torch.randint(
                0, config.train_sampling_steps, (B,),
                device=accelerator.device
            ).long()
            
            # 5. Training step
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                # 기본 IDDPM loss 사용 (단일 레이어)
                loss_dict = multilayer_diffusion.training_losses(
                    model=model,
                    x=z_target,  # (B, 4, h, w)
                    t=timesteps,
                    model_kwargs=dict(
                        y=y,
                        mask=y_mask,
                        x_ref=z_ref,  # (B, N-1, 4, h, w)
                    ),
                )
                
                loss = loss_dict['loss'].mean()
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), config.gradient_clip)
                


                def debug_check_parameters(model, optimizer, accelerator, step):
                    accelerator.print(f"\n[DEBUG] Step {step}")

                    total = 0
                    frozen = 0
                    has_grad = 0

                    for name, p in model.named_parameters():
                        total += 1

                        req = p.requires_grad
                        grad = p.grad

                        if not req:
                            frozen += 1

                        if grad is not None:
                            has_grad += 1

                        if not req or grad is None:
                            accelerator.print(
                                f"{name:60s} | requires_grad={req} | grad={'YES' if grad is not None else 'NO'}"
                            )

                    accelerator.print(
                        f"\n[SUMMARY] total={total}, frozen={frozen}, grad_present={has_grad}\n"
                    )
                

                optimizer.step()
                lr_scheduler.step()
                
                if accelerator.sync_gradients:
                    ema_update(model_ema, model, config.ema_rate)

            if global_step % 100 == 0:
                torch.cuda.empty_cache()

            if global_step >= 10000:
                logger.info(f"[Step {global_step}] Unfreezing all layers")
                for param in model.parameters():
                    param.requires_grad = True
                
                optimizer = build_optimizer(accelerator.unwrap_model(model), config.optimizer)
                optimizer = accelerator.prepare(optimizer)

            # ============================================
            # 6. Logging
            # ============================================
            lr = lr_scheduler.get_last_lr()[0]
            logs = {args.loss_report_name: accelerator.gather(loss).mean().item()}
            if grad_norm is not None:
                logs.update(grad_norm=accelerator.gather(grad_norm).mean().item())
            log_buffer.update(logs)
            
            if (step + 1) % config.log_interval == 0 or (step + 1) == 1:
                t = (time.time() - last_tic) / config.log_interval
                t_d = data_time_all / config.log_interval
                avg_time = (time.time() - time_start) / (global_step + 1)
                eta = str(datetime.timedelta(seconds=int(avg_time * (total_steps - start_step - global_step - 1))))
                eta_epoch = str(datetime.timedelta(seconds=int(avg_time * (len(train_dataloader) - step - 1))))
                
                log_buffer.average()
                
                # Get latent size info
                h, w = z_target.shape[-2], z_target.shape[-1]
                avg_layers = num_layers.float().mean().item()

                info = f"Step/Epoch [{(epoch-1)*len(train_dataloader)+step+1}/{epoch}][{step + 1}/{len(train_dataloader)}]: " \
                       f"total_eta: {eta}, epoch_eta: {eta_epoch}, time_all: {t:.3f}, time_data: {t_d:.3f}, " \
                       f"lr: {lr:.3e}, latent: ({h}, {w}), layers: {N} (avg: {avg_layers:.1f}), "
                info += ', '.join([f"{k}: {v:.4f}" for k, v in log_buffer.output.items()])
                logger.info(info)
                
                last_tic = time.time()
                log_buffer.clear()
                data_time_all = 0
            
            logs.update(lr=lr)
            accelerator
            
            # ============================================
            # Periodic Evaluation
            # ============================================
            eval_interval = getattr(config, 'eval_interval', 1000)
            if (global_step) % eval_interval == 0:
                if accelerator.is_main_process:
                    eval_save_dir = os.path.join(config.work_dir, 'eval_samples')
                    run_evaluation(
                        model=model,
                        vae=vae,
                        multilayer_diffusion=multilayer_diffusion,
                        dataloader=train_dataloader,
                        text_encoder=text_encoder,
                        accelerator=accelerator,
                        config=config,
                        epoch=epoch,
                        step=global_step + 1,
                        save_dir=eval_save_dir,
                    )

            global_step += 1
            data_time_start = time.time()

            # ============================================
            # 7. Save checkpoint (by steps)
            # ============================================
            if ((epoch - 1) * len(train_dataloader) + step + 1) % config.save_model_steps == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    os.umask(0o000)
                    save_checkpoint(
                        os.path.join(config.work_dir, 'checkpoints'),
                        epoch=epoch,
                        step=(epoch - 1) * len(train_dataloader) + step + 1,
                        model=accelerator.unwrap_model(model),
                        model_ema=accelerator.unwrap_model(model_ema),
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler
                    )

        # ============================================
        # 8. Save checkpoint (by epochs)
        # ============================================
        if epoch % config.save_model_epochs == 0 or epoch == config.num_epochs:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                os.umask(0o000)
                save_checkpoint(
                    os.path.join(config.work_dir, 'checkpoints'),
                    epoch=epoch,
                    step=(epoch - 1) * len(train_dataloader) + step + 1,
                    model=accelerator.unwrap_model(model),
                    model_ema=accelerator.unwrap_model(model_ema),
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler
                )


def parse_args():
    parser = argparse.ArgumentParser(description="Train MultiLayerPixArt")
    parser.add_argument("config", type=str, help="config file path")
    parser.add_argument("--cloud", action='store_true', default=False, help="cloud or local machine")
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--resume-from', help='the dir to resume the training')
    parser.add_argument('--load-from', default=None, help='the dir to load a ckpt for training')
    parser.add_argument('--local-rank', type=int, default=-1)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--use_ref', default=True)

    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help='The integration to report the results and logs to.'
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="multilayer-pixart",
        help="The project name for tracking"
    )
    parser.add_argument("--loss_report_name", type=str, default="loss")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    config = read_config(args.config)
    
    if args.work_dir is not None:
        config.work_dir = args.work_dir
    if args.cloud:
        config.data_root = '/data/data'
    if args.resume_from is not None:
        config.load_from = None
        config.resume_from = dict(
            checkpoint=args.resume_from,
            load_ema=False,
            resume_optimizer=True,
            resume_lr_scheduler=True
        )
    if args.debug:
        config.log_interval = 1
        config.train_batch_size = 2
        config.valid_num = 100

    os.umask(0o000)
    os.makedirs(config.work_dir, exist_ok=True)

    # ============================================
    # Initialize Accelerator
    # ============================================
    init_handler = InitProcessGroupKwargs()
    init_handler.timeout = datetime.timedelta(seconds=5400)
    
    ddp_kwargs = DistributedDataParallelKwargs(
        broadcast_buffers=False,
        gradient_as_bucket_view=False  # 명시적으로 False
    )

    if config.use_fsdp:
        init_train = 'FSDP'
        from accelerate import FullyShardedDataParallelPlugin
        from torch.distributed.fsdp.fully_sharded_data_parallel import FullStateDictConfig
        set_fsdp_env()
        fsdp_plugin = FullyShardedDataParallelPlugin(
            state_dict_config=FullStateDictConfig(offload_to_cpu=False, rank0_only=False)
        )
    else:
        init_train = 'DDP'
        fsdp_plugin = None

    even_batches = True
    if config.multi_scale:
        even_batches = False

    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with=args.report_to,
        project_dir=os.path.join(config.work_dir, "logs"),
        fsdp_plugin=fsdp_plugin,
        even_batches=even_batches,
        kwargs_handlers=[init_handler, ddp_kwargs]
    )

    logger = get_root_logger(os.path.join(config.work_dir, 'train_log.log'))

    config.seed = init_random_seed(config.get('seed', None))
    set_random_seed(config.seed)

    if accelerator.is_main_process:
        config.dump(os.path.join(config.work_dir, 'config.py'))

    logger.info(f"Config: \n{config.pretty_text}")
    logger.info(f"World_size: {get_world_size()}, seed: {config.seed}")
    logger.info(f"Initializing: {init_train} for training")
    
    # ============================================
    # Model Configuration
    # ============================================
    image_size = config.image_size  # 256, 512, or 1024
    latent_size = int(image_size) // 8
    pred_sigma = getattr(config, 'pred_sigma', True)
    learn_sigma = getattr(config, 'learn_sigma', True) and pred_sigma
    max_layers = getattr(config, 'max_layers', 8)
    
    logger.info(f"Image size: {image_size}, Latent size: {latent_size}, Max layers: {max_layers}")

    # ============================================
    # Build Base Diffusion (for noise schedule)
    # ============================================
    base_diffusion = IDDPM(
        str(config.train_sampling_steps),
        learn_sigma=learn_sigma,
        pred_sigma=pred_sigma,
        snr=config.snr_loss
    )
    
    # Wrap with MultiLayer handler
    multilayer_diffusion = ReferenceIDDPM(base_diffusion)    
    # ============================================
    # Build MultiLayerPixArt Model
    # ============================================


    model_type = 'adaln'
    logger.info(f"Building MultiLayerPixArt model of type: {model_type}")

    if model_type == 'adaln':
        from diffusion.model.nets.PixArt_multilayer import ReferencePixArt_XL_2
        model = ReferencePixArt_XL_2(
            input_size=latent_size,
            in_channels=4,
            max_ref_layers=max_layers - 1,
            ref_encoder_depth=4,
            caption_channels=4096,
            model_max_length=config.model_max_length,
            pred_sigma=pred_sigma,
            use_ref=args.use_ref,
        ).train()
    elif model_type == 'crossattn':
        from diffusion.model.nets.PixArt_reference_crossattn import ReferencePixArtCrossAttn_XL_2
        model = ReferencePixArtCrossAttn_XL_2(
            input_size=latent_size,\
            in_channels=4,
            max_ref_layers=max_layers - 1,
            ref_encoder_depth=4,
            ref_compression_ratio=4, 
            caption_channels=4096,
            model_max_length=config.model_max_length,
            pred_sigma=pred_sigma,
        ).train()


    model.enable_gradient_checkpointing()

    logger.info(f"Model: {model.__class__.__name__}")
    logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")





    # ============================================
    # Load Pretrained PixArt Weights
    # ============================================
    # pretrained_path = getattr(config, 'pretrained_pixart', None)

    # if pretrained_path is not None:
    #     logger.info(f"Loading pretrained weights from: {pretrained_path}")

    #     if pretrained_path.endswith('.pth'):
    #         checkpoint = torch.load(pretrained_path, map_location='cpu')
    #         pretrained_state_dict = checkpoint['state_dict']
            
    #         # Position embedding resize
    #         if 'pos_embed' in pretrained_state_dict:
    #             old_pos_embed = pretrained_state_dict['pos_embed']
    #             new_size = model.pos_embed.shape[1]
    #             old_size = old_pos_embed.shape[1]
                
    #             if old_size != new_size:
    #                 print(f"Resizing pos_embed: {old_size} -> {new_size}")
    #                 pretrained_state_dict['pos_embed'] = F.interpolate(
    #                     old_pos_embed.reshape(1, int(old_size**0.5), int(old_size**0.5), -1).permute(0,3,1,2),
    #                     size=(int(new_size**0.5), int(new_size**0.5)),
    #                     mode='bilinear'
    #                 ).permute(0,2,3,1).reshape(1, new_size, -1)
    #     else:
    #         raise ValueError("Unsupported pretrained format. Use .pth files.")

    #     missing_keys = model.load_pretrained_pixart(pretrained_state_dict)
    #     logger.info(f"Loaded pretrained. Missing keys (new layers): {len(missing_keys)}")
    #     if len(missing_keys) > 0 and len(missing_keys) <= 20:
    #         logger.info(f"Missing keys: {missing_keys}")
    # else:
    #     logger.info("Training from scratch.")



    pretrained_path = getattr(config, 'pretrained_pixart', None)
    resume_training = getattr(config, 'resume_training', None)

    if pretrained_path is not None and resume_training == None:
        logger.info(f"Loading pretrained weights from: {pretrained_path}")

        if pretrained_path.endswith('.pth'):
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            pretrained_state_dict = checkpoint['state_dict']
            
            # ============================================================
            # 1. x_embedder adaptation (기존)
            # ============================================================
            if 'x_embedder.proj.weight' in pretrained_state_dict:
                pretrained_weight = pretrained_state_dict['x_embedder.proj.weight']
                model_weight = model.x_embedder.proj.weight
                
                if pretrained_weight.shape[1] != model_weight.shape[1]:
                    logger.info(f"Adapting x_embedder: {pretrained_weight.shape[1]}ch -> {model_weight.shape[1]}ch")
                    
                    new_weight = torch.zeros_like(model_weight)
                    # RGB 복사
                    new_weight[:, :4, :, :] = pretrained_weight[:, :4, :, :]
                    # Alpha를 RGB 평균으로 (논문 방식)
                    rgb_mean = pretrained_weight[:, :3, :, :].mean(dim=1, keepdim=True)
                    new_weight[:, 4:5, :, :] = rgb_mean * 0.1  # 작게!
                    
                    pretrained_state_dict['x_embedder.proj.weight'] = new_weight
                    logger.info(f"  ✓ x_embedder adapted (alpha = RGB mean × 0.1)")
            
            # ============================================================
            # 2. final_layer adaptation (새로 추가!) ⭐
            # ============================================================
            if 'final_layer.linear.weight' in pretrained_state_dict:
                pretrained_weight = pretrained_state_dict['final_layer.linear.weight']
                model_weight = model.final_layer.linear.weight
                
                if pretrained_weight.shape[0] != model_weight.shape[0]:
                    logger.info(f"Adapting final_layer: {pretrained_weight.shape[0]} -> {model_weight.shape[0]} out_features")
                    
                    patch_size = model.patch_size
                    old_channels = pretrained_weight.shape[0] // (patch_size * patch_size)
                    new_channels = model_weight.shape[0] // (patch_size * patch_size)
                    
                    logger.info(f"  Output channels: {old_channels} -> {new_channels}")
                    
                    # Reshape: (out_features, hidden) -> (patch^2, channels, hidden)
                    pre_w = pretrained_weight.reshape(patch_size * patch_size, old_channels, -1)
                    new_w = torch.zeros(patch_size * patch_size, new_channels, pre_w.shape[-1])
                    
                    # RGB/RGBA 복사
                    copy_ch = min(old_channels, new_channels, 4)
                    new_w[:, :copy_ch, :] = pre_w[:, :copy_ch, :]
                    
                    # Alpha 채널 작게 초기화
                    if new_channels > old_channels:
                        torch.nn.init.normal_(new_w[:, old_channels:, :], std=0.0001)
                    
                    # Reshape back
                    new_w = new_w.reshape(-1, pre_w.shape[-1])
                    pretrained_state_dict['final_layer.linear.weight'] = new_w
                    logger.info(f"  ✓ final_layer weight adapted")
                    
                    # Bias도 마찬가지
                    if 'final_layer.linear.bias' in pretrained_state_dict:
                        pre_b = pretrained_state_dict['final_layer.linear.bias']
                        pre_b = pre_b.reshape(patch_size * patch_size, old_channels)
                        new_b = torch.zeros(patch_size * patch_size, new_channels)
                        new_b[:, :copy_ch] = pre_b[:, :copy_ch]
                        new_b = new_b.reshape(-1)
                        pretrained_state_dict['final_layer.linear.bias'] = new_b
                        logger.info(f"  ✓ final_layer bias adapted")
            
            # Position embedding resize
            if 'pos_embed' in pretrained_state_dict:
                old_pos_embed = pretrained_state_dict['pos_embed']
                new_size = model.pos_embed.shape[1]
                old_size = old_pos_embed.shape[1]
                
                if old_size != new_size:
                    logger.info(f"Resizing pos_embed: {old_size} -> {new_size}")
                    pretrained_state_dict['pos_embed'] = F.interpolate(
                        old_pos_embed.reshape(1, int(old_size**0.5), int(old_size**0.5), -1).permute(0,3,1,2),
                        size=(int(new_size**0.5), int(new_size**0.5)),
                        mode='bilinear'
                    ).permute(0,2,3,1).reshape(1, new_size, -1)
                    logger.info(f"  ✓ pos_embed resized")
            
            # ============================================================
            # 0. 디버깅: y_embedder 키 확인 ⭐
            # ============================================================
            y_keys = [k for k in pretrained_state_dict.keys() if 'y_embedder' in k]
            logger.info(f"🔍 Pretrained y_embedder keys ({len(y_keys)}):")
            for k in y_keys:
                logger.info(f"    {k}")
            
            # Shape 비교
            model_dict = model.state_dict()
            logger.info(f"🔍 Shape comparison:")
            for k in y_keys:
                if k in model_dict:
                    pre_shape = pretrained_state_dict[k].shape
                    model_shape = model_dict[k].shape
                    match = "✓" if pre_shape == model_shape else "✗"
                    logger.info(f"    {match} {k}")
                    logger.info(f"        Pretrained: {pre_shape}")
                    logger.info(f"        Model:      {model_shape}")
            
        else:
            raise ValueError("Unsupported pretrained format.")

        missing_keys = model.load_pretrained_pixart(pretrained_state_dict)
        logger.info(f"Loaded pretrained. Missing keys: {len(missing_keys)}")
        if len(missing_keys) > 0 and len(missing_keys) <= 20:
            logger.info(f"Missing keys: {missing_keys}")
    
    if resume_training is not None:
        #load weight from resume_training path
        logger.info("Resuming training, skipping pretrained weight loading.")
        state_dict = torch.load(resume_training, map_location='cpu')['state_dict_ema']
        
        missing_keys = model.load_pretrained_pixart(state_dict)
        logger.info(f"Resumed training. Missing keys: {len(missing_keys)}")
        if len(missing_keys) > 0 and len(missing_keys) <= 20:
            logger.info(f"Missing keys: {missing_keys}")


    # Create EMA model
    model_ema = deepcopy(model).eval()
    ema_update(model_ema, model, 0.)
    print_gpu_memory("After model load")

    # ============================================
    # Load Additional Checkpoint (optional)
    # ============================================
    if config.load_from is not None:
        if args.load_from is not None:
            config.load_from = args.load_from
        missing, unexpected = load_checkpoint(
            config.load_from, model, 
            load_ema=config.get('load_ema', False)
        )
        logger.warning(f'Missing keys: {missing}')
        logger.warning(f'Unexpected keys: {unexpected}')

    # ============================================
    # Load VAE
    # ============================================
    logger.info(f"Loading VAE from: {config.vae_pretrained}")
    vae = AutoencoderKL.from_pretrained(config.vae_pretrained).to(accelerator.device).eval()
    for param in vae.parameters():
        param.requires_grad = False
    print_gpu_memory("After vae load")

    # ============================================
    # Load T5 Text Encoder
    # ============================================
    t5_pretrained = getattr(config, 't5_pretrained', 'google/flan-t5-xxl')
    logger.info(f"Loading T5 from: {t5_pretrained}")

    text_encoder = T5Embedder(device=accelerator.device, local_cache=True, cache_dir=t5_pretrained, torch_dtype=torch.float16)
    print_gpu_memory("After t5 load")

    # ============================================
    # Prepare for FSDP
    # ============================================
    if accelerator.distributed_type == DistributedType.FSDP:
        for m in accelerator._models:
            m.clip_grad_norm_ = types.MethodType(clip_grad_norm_, m)

    # ============================================
    # Build Dataloader
    # ============================================
    set_data_root(config.data_root)
    
    from diffusion.data.multilayer_builder import build_mulan_dataloader
    train_dataloader = build_mulan_dataloader(
        data_roots=config.data_roots,  # e.g., ["../data/mulan_coco", "../data/mulan_laion"]
        batch_size=config.train_batch_size,
        resolution=image_size,
        max_layers=max_layers,
        num_workers=config.num_workers,
        shuffle=True,
        caption_type=getattr(config, 'caption_type', 'blip2')
    )
    
    logger.info(f"Dataset size: {len(train_dataloader.dataset)}")
    logger.info(f"Dataloader batches: {len(train_dataloader)}")

    # ============================================
    # Build Optimizer and LR Scheduler
    # ============================================
    lr_scale_ratio = 1
    if config.get('auto_lr', None):
        lr_scale_ratio = auto_scale_lr(
            config.train_batch_size * get_world_size() * config.gradient_accumulation_steps,
            config.optimizer,
            **config.auto_lr
        )
        logger.info(f"Auto scaling lr by ratio: {lr_scale_ratio:.2f}")
    
    optimizer = build_optimizer(model, config.optimizer)
    lr_scheduler = build_lr_scheduler(config, optimizer, train_dataloader, lr_scale_ratio)

    # ============================================
    # Resume Training
    # ============================================
    start_epoch = 0
    if config.resume_from is not None and config.resume_from['checkpoint'] is not None:
        start_epoch, missing, unexpected = load_checkpoint(
            **config.resume_from,
            model=model,
            model_ema=model_ema,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )
        logger.warning(f'Missing keys: {missing}')
        logger.warning(f'Unexpected keys: {unexpected}')

    # ============================================
    # Accelerator Prepare
    # ============================================
    model, model_ema = accelerator.prepare(model, model_ema)
    optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        optimizer, train_dataloader, lr_scheduler
    )


    # ============================================
    # Initialize Tracker
    # ============================================
    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    if accelerator.is_main_process:
        tracker_config = dict(vars(config))
        try:
            accelerator.init_trackers(args.tracker_project_name, tracker_config)
        except:
            accelerator.init_trackers(f"tb_{timestamp}")

    # ============================================
    # Start Training
    # ============================================
    logger.info("=" * 50)
    logger.info("Starting MultiLayerPixArt Training!")
    logger.info("=" * 50)
    train()
    logger.info("Training completed!")