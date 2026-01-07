"""
Quick test to verify layer inpainting model builds correctly
"""

import torch
from diffusion.model.nets.PixArt_layer_inpainting import PixArtLayerInpainting

print("Testing PixArtLayerInpainting model...")

# Build model
model = PixArtLayerInpainting(
    pretrained_pixart=None,  # No pretrained for this test
    max_layers=6,
    input_size=32,  # 256 // 8
    pred_sigma=True,
)

print(f"✓ Model created")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

# Test forward pass
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

B = 2
max_layers = 6
h, w = 32, 32

# Create dummy inputs
layers = torch.randn(B, max_layers, 4, h, w, device=device)
layer_mask = torch.zeros(B, max_layers, device=device)
layer_mask[:, 0] = 1  # Mask first layer
timestep = torch.randint(0, 1000, (B,), device=device)
y = torch.randn(B, 1, 120, 4096, device=device)
mask = torch.ones(B, 120, device=device)

print(f"✓ Dummy inputs created")
print(f"  layers: {layers.shape}")
print(f"  layer_mask: {layer_mask.shape}")
print(f"  timestep: {timestep.shape}")
print(f"  y: {y.shape}")

# Forward pass
with torch.no_grad():
    output = model(layers, layer_mask, timestep, y, mask)

print(f"✓ Forward pass successful")
print(f"  output: {output.shape}")
print(f"  Expected: ({B}, {max_layers}, 4, {h}, {w})")

assert output.shape == (B, max_layers, 4, h, w), f"Shape mismatch: {output.shape}"

print("\n" + "="*60)
print("All tests passed! ✓")
print("="*60)
print("\nReady to train with:")
print("  bash bash_scripts/train_inpainting.sh")
