# config/multilayer_pixart_512.py
# MultiLayerPixArt Training Configuration

_base_ = ['../PixArt_xl2_internal.py']

# ============================================
# Data Settings
# ============================================
data_root = './data'
data_roots = ['../data/mulan_coco', '../data/mulan_laion']  # MuLan dataset paths
caption_type = 'blip2'  # or 'original'

# ============================================
# Model Settings
# ============================================
image_size = 256
max_layers = 8
model_max_length = 120

# Pretrained paths
pretrained_pixart = 'PixArt-alpha/PixArt-XL-2-256x256.pth'  # or HuggingFace id
vae_pretrained = '/workspace/twenty/PixArt-alpha/sd-vae-ft-ema'
t5_pretrained = '/workspace/twenty/PixArt-alpha'

# Model architecture
pred_sigma = True
learn_sigma = True

# ============================================
# Training Settings
# ============================================
num_epochs = 100
train_batch_size = 4  # per GPU
gradient_accumulation_steps = 4  # effective batch = 4 * 4 * num_gpus

# Diffusion
train_sampling_steps = 1000
snr_loss = False
scale_factor = 0.18215  # VAE scale factor

# Optimizer
optimizer = dict(
    type='AdamW',
    lr=2e-5,
    weight_decay=0.03,
    eps=1e-8,
    betas=(0.9, 0.999)
)

# LR Scheduler
lr_schedule = 'constant'
lr_schedule_args = dict(
    num_warmup_steps=1000
)

# EMA
ema_rate = 0.9999

# Gradient clipping
gradient_clip = 1.0

# ============================================
# Logging & Saving
# ============================================
work_dir = './output/multilayer_pixart_512'
log_interval = 50
save_model_epochs = 5
save_model_steps = 5000

# ============================================
# Distributed Training
# ============================================
use_fsdp = False
mixed_precision = 'fp16'  # or 'bf16' for A100
multi_scale = False
num_workers = 4

# ============================================
# Optional: Window Attention (from PixArt)
# ============================================
window_block_indexes = []
window_size = 0
use_rel_pos = False
lewei_scale = 1.0