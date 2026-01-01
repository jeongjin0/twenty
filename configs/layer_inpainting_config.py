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
batch_size = 16
num_workers = 4
num_epochs = 60
lr = 5e-5
weight_decay = 0.0
gradient_clip = 1.0
gradient_accumulation_steps = 1
gradient_checkpointing = True

# Diffusion
train_sampling_steps = 1000

# Pretrained models
pretrained_pixart_path = '/workspace/twenty/PixArt-alpha/PixArt-XL-2-256x256.pth'
vae_pretrained = '/workspace/twenty/PixArt-alpha/sd-vae-ft-ema'
text_encoder_name = '/workspace/twenty/PixArt-alpha'

# Logging & Saving
log_interval = 50
save_model_epochs = 5  # Save every 5 epochs

# Output
work_dir = 'output/layer_inpainting_v1'

# Misc
seed = 42
mixed_precision = 'fp16'  # 'no', 'fp16', 'bf16'
