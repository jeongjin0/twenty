"""
Debug script to check what's wrong with inference
"""

import torch
from diffusers.models import AutoencoderKL
from PIL import Image
from torchvision import transforms
import sys
import os

sys.path.insert(0, '/workspace/twenty')

from diffusion.model.nets.PixArt_layer_inpainting import PixArtLayerInpainting
from diffusion.model.nets.PixArt import PixArt_XL_2
from diffusion.model.t5 import T5Embedder
from diffusion import IDDPM

# Load a test image
img_path = "/workspace/data/mulan_coco/000000581346-layer_0.png"
img = Image.open(img_path).convert('RGB')
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])
img_tensor = transform(img).unsqueeze(0).cuda()

print("="*60)
print("DEBUG: Layer-wise Inpainting")
print("="*60)

# Load VAE
print("\n1. Testing VAE encode/decode...")
vae = AutoencoderKL.from_pretrained("/workspace/twenty/PixArt-alpha/sd-vae-ft-ema").cuda().eval()

with torch.no_grad():
    z = vae.encode(img_tensor).latent_dist.mode() * 0.18215
    print(f"   Latent shape: {z.shape}")
    print(f"   Latent stats: mean={z.mean():.4f}, std={z.std():.4f}, min={z.min():.4f}, max={z.max():.4f}")

    img_recon = vae.decode(z / 0.18215).sample
    print(f"   Reconstructed shape: {img_recon.shape}")
    print(f"   Reconstructed stats: mean={img_recon.mean():.4f}, std={img_recon.std():.4f}, min={img_recon.min():.4f}, max={img_recon.max():.4f}")

    # Save reconstruction
    from torchvision.utils import save_image
    save_image(img_recon, "debug_vae_recon.png", normalize=True, value_range=(-1, 1))
    print(f"   ✓ VAE reconstruction saved to debug_vae_recon.png")

# Load model
print("\n2. Loading model...")
checkpoint_path = "output/layer_inpainting_v1/checkpoints/epoch_7_step_35000.pth"

pretrained_pixart = PixArt_XL_2(
    input_size=32,
    in_channels=4,
    caption_channels=4096,
    model_max_length=120,
    pred_sigma=True,
)

model = PixArtLayerInpainting(
    pretrained_pixart=pretrained_pixart,
    max_layers=6,
    input_size=32,
    pred_sigma=True,
).cuda().eval()

ckpt = torch.load(checkpoint_path, map_location='cpu')
if 'state_dict_ema' in ckpt:
    state_dict = ckpt['state_dict_ema']
    print(f"   Using EMA model")
elif 'state_dict' in ckpt:
    state_dict = ckpt['state_dict']
    print(f"   Using regular model")
else:
    state_dict = ckpt
    print(f"   Using checkpoint as-is")

# Check keys
model_keys = set(model.state_dict().keys())
ckpt_keys = set(state_dict.keys())
missing = model_keys - ckpt_keys
unexpected = ckpt_keys - model_keys

print(f"   Total model keys: {len(model_keys)}")
print(f"   Total checkpoint keys: {len(ckpt_keys)}")
print(f"   Missing keys: {len(missing)}")
print(f"   Unexpected keys: {len(unexpected)}")

if len(missing) > 0 and len(missing) <= 10:
    print(f"   Missing: {list(missing)[:10]}")
if len(unexpected) > 0 and len(unexpected) <= 10:
    print(f"   Unexpected: {list(unexpected)[:10]}")

result = model.load_state_dict(state_dict, strict=False)
print(f"   ✓ Model loaded")
print(f"   Missing keys from load: {len(result.missing_keys)}")
print(f"   Unexpected keys from load: {len(result.unexpected_keys)}")

# Test forward pass
print("\n3. Testing model forward pass...")
t5 = T5Embedder(device="cuda", local_cache=True, cache_dir="/workspace/twenty/PixArt-alpha", torch_dtype=torch.float16)

with torch.no_grad():
    # Prepare inputs
    layers = torch.randn(1, 6, 4, 32, 32).cuda()  # Random latents
    layer_mask = torch.zeros(1, 6).cuda()
    layer_mask[0, 1] = 1  # Mask layer 1
    timestep = torch.tensor([500], device='cuda')

    caption_embs, emb_masks = t5.get_text_embeddings(["a test prompt"])
    y = caption_embs.float()[:, None].cuda()
    y_mask = emb_masks.cuda()

    print(f"   Input layers shape: {layers.shape}")
    print(f"   Layer mask: {layer_mask}")
    print(f"   Timestep: {timestep}")
    print(f"   Y shape: {y.shape}")

    noise_pred = model(layers, layer_mask, timestep, y, mask=y_mask)
    print(f"   Output shape: {noise_pred.shape}")
    print(f"   Output stats: mean={noise_pred.mean():.4f}, std={noise_pred.std():.4f}, min={noise_pred.min():.4f}, max={noise_pred.max():.4f}")

    # Check if output is all zeros or constant
    if noise_pred.std() < 0.01:
        print(f"   ⚠️  WARNING: Output has very low variance! Model might not be working.")
    else:
        print(f"   ✓ Model forward pass seems OK")

# Test DDIM step
print("\n4. Testing DDIM sampling step...")
diffusion = IDDPM(str(1000))
alphas_cumprod = torch.from_numpy(diffusion.alphas_cumprod).float().cuda()

with torch.no_grad():
    # Start with random noise
    x_t = torch.randn(1, 6, 4, 32, 32).cuda()
    t = 500
    t_batch = torch.tensor([t], device='cuda')

    noise_pred = model(x_t, layer_mask, t_batch, y, mask=y_mask)

    alpha_t = alphas_cumprod[t]
    alpha_next = alphas_cumprod[t-1]

    x0_pred = (x_t - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
    dir_xt = torch.sqrt(1 - alpha_next) * noise_pred
    x_next = torch.sqrt(alpha_next) * x0_pred + dir_xt

    print(f"   x_t stats: mean={x_t.mean():.4f}, std={x_t.std():.4f}")
    print(f"   noise_pred stats: mean={noise_pred.mean():.4f}, std={noise_pred.std():.4f}")
    print(f"   x0_pred stats: mean={x0_pred.mean():.4f}, std={x0_pred.std():.4f}")
    print(f"   x_next stats: mean={x_next.mean():.4f}, std={x_next.std():.4f}")

    # Decode to check
    z_test = x_next[0, 0:1] / 0.18215
    img_test = vae.decode(z_test).sample[0]
    print(f"   Decoded image stats: mean={img_test.mean():.4f}, std={img_test.std():.4f}")

    save_image(img_test.unsqueeze(0), "debug_ddim_step.png", normalize=True, value_range=(-1, 1))
    print(f"   ✓ DDIM step output saved to debug_ddim_step.png")

print("\n" + "="*60)
print("Debug complete! Check the generated images:")
print("  - debug_vae_recon.png (should look like original)")
print("  - debug_ddim_step.png (random but not black)")
print("="*60)
