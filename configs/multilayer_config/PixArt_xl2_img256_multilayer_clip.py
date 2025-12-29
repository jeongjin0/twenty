# config/multilayer_pixart_256_clip.py
# MultiLayerPixArt Training with CLIP Reference Encoder and Text Dropout

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

resume_training = None

# Model architecture
pred_sigma = True
learn_sigma = True

# ============================================
# 🔥 NEW: CLIP Reference Encoder Settings
# ============================================
use_clip_ref_encoder = True  # Use pretrained CLIP vision encoder
clip_model_name = "openai/clip-vit-large-patch14"  # CLIP model name
freeze_clip = True  # Freeze CLIP weights (only train projection layer)

# ============================================
# 🔥 NEW: Text Dropout Settings
# ============================================
text_dropout_prob = 0.0  # 50% 확률로 text를 빈 문자열로 대체
                         # Reference를 필수로 만들어서 학습 강제

# ============================================
# Training Settings
# ============================================
num_epochs = 100
train_batch_size = 4  # per GPU
gradient_accumulation_steps = 4  # effective batch = 8 * 2 * num_gpus
eval_interval = 1000

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
gradient_clip = 0.5

# ============================================
# Logging & Saving
# ============================================
work_dir = './output/multilayer_pixart_256_clip'
log_interval = 50
save_model_epochs = 5
save_model_steps = 5000

# ============================================
# Distributed Training
# ============================================
use_fsdp = False
mixed_precision = "fp16"
multi_scale = False
num_workers = 4

# ============================================
# Optional: Window Attention
# ============================================
window_block_indexes = []
window_size = 0
use_rel_pos = False
lewei_scale = 1.0

# ============================================
# Layer-wise Augmentation
# ============================================
shuffle_ref = True
merge_augmentation_prob = 0.0
