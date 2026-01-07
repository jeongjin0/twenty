"""
Test if model actually uses reference layers
Compare inference with real references vs random references
"""

import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusion.model.nets.PixArt_layer_inpainting import PixArtLayerInpainting
from diffusion.model.nets.PixArt import PixArt_XL_2

def test_reference_usage(checkpoint_path):
    """
    Test if model uses reference layers by comparing predictions
    with real vs random reference layers
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model
    print("Loading model...")
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
    ).to(device).eval()

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    if 'state_dict_ema' in ckpt:
        state_dict = ckpt['state_dict_ema']
    elif 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=False)
    print(f"✓ Model loaded from {checkpoint_path}")

    # Create dummy inputs
    B, N, C, h, w = 1, 6, 4, 32, 32

    # Real reference layers (some pattern)
    real_refs = torch.randn(B, N, C, h, w, device=device) * 0.5  # Smaller variance

    # Random reference layers
    random_refs = torch.randn(B, N, C, h, w, device=device)

    # Layer mask: mask layer 2
    layer_mask = torch.zeros(B, N, device=device)
    layer_mask[0, 2] = 1

    # Same noisy masked layer for both
    noisy_masked = torch.randn(1, C, h, w, device=device)

    # Build inputs
    input_real = real_refs.clone()
    input_real[0, 2] = noisy_masked

    input_random = random_refs.clone()
    input_random[0, 2] = noisy_masked

    # Dummy text embedding
    y = torch.randn(B, 1, 120, 4096, device=device)
    y_mask = torch.ones(B, 120, device=device)

    # Forward pass
    timestep = torch.tensor([500], device=device)

    with torch.no_grad():
        pred_real = model(input_real, layer_mask, timestep, y, mask=y_mask)
        pred_random = model(input_random, layer_mask, timestep, y, mask=y_mask)

    # Compare predictions for masked layer
    diff = (pred_real[0, 2] - pred_random[0, 2]).abs().mean().item()

    print(f"\n" + "="*60)
    print(f"Reference Usage Test")
    print(f"="*60)
    print(f"Prediction difference (real refs vs random refs): {diff:.6f}")
    print(f"")

    if diff < 0.001:
        print(f"⚠️  WARNING: Difference is very small!")
        print(f"   Model may NOT be using reference layers properly")
        print(f"   Expected: diff > 0.01")
    elif diff < 0.01:
        print(f"⚠️  Difference is small but non-zero")
        print(f"   Model might be using references weakly")
    else:
        print(f"✓ Model is using reference layers!")
        print(f"  Predictions differ significantly based on references")

    print(f"="*60)

    return diff


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    args = parser.parse_args()

    test_reference_usage(args.checkpoint)
