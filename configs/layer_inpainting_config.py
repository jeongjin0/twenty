"""
Configuration for Layer-wise Inpainting Training
"""

# Model
max_layers = 6  # Maximum number of layers (pad with black if less)
image_size = 256
pred_sigma = True
model_max_length = 120

# Data
data_roots = [
    '../data/mulan_coco',
    '../data/mulan_laion',
]
caption_type = 'blip2'  # or 't5', 'blip2'

# Training
batch_size = 4
num_workers = 4
num_epochs = 60
gradient_clip = 1.0
gradient_accumulation_steps = 4
gradient_checkpointing = True

# 2-Stage Training: Freeze PixArt for first N epochs
# Stage 1: Train projections only (pretrained PixArt frozen)
# Stage 2: Train full model (PixArt unfrozen)
pixart_freeze_epochs = 5  # Freeze PixArt for first 5 epochs
pixart_unfreeze_lr_scale = 0.5  # Reduce LR by 50% when unfreezing (optional)

# Optimizer
optimizer = dict(
    type='AdamW',
    lr=5e-5,
    weight_decay=0.0,
    betas=(0.9, 0.999),
)

# LR Scheduler
lr_schedule = 'constant'
lr_schedule_args = dict(
    num_warmup_steps=0,  # No warmup for constant schedule
)

# EMA
ema_rate = 0.9999

# Diffusion
train_sampling_steps = 1000
scale_factor = 0.18215
snr_loss = False

# Pretrained models
pretrained_pixart_path = '/workspace/twenty/PixArt-alpha/PixArt-XL-2-256x256.pth'
vae_pretrained = '/workspace/twenty/PixArt-alpha/sd-vae-ft-ema'
text_encoder_name = '/workspace/twenty/PixArt-alpha'

# Logging & Saving
log_interval = 50
save_model_epochs = 5  # Save every 5 epochs
save_model_steps = 5000  # Save every 5000 steps

# Output
work_dir = 'output/layer_inpainting_v1'

# Misc
seed = 42
mixed_precision = 'fp16'  # 'no', 'fp16', 'bf16'
