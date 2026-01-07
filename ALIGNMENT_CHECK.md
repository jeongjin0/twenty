# Alignment Verification: Training vs Inference

## ✅ Dimension Alignment Check

### 1. Dataset → VAE Encoding
- **Dataset output**: `(B, max_layers, 4, H, W)` - always padded to `max_layers=6`
- **VAE input**: `(B*N, 3, H, W)` - RGB only, reshaped from `(B, N, 3, H, W)`
- **VAE output**: `(B*N, 4, h, w)` where `h=H/8, w=W/8`
- **After reshape**: `z_clean (B, N, 4, h, w)` where `N=max_layers=6`
- ✅ **Aligned**

### 2. Layer Mask Creation
- **Inference**: `layer_mask = torch.zeros(1, max_layers)` → `(1, 6)`
- **Training**: `layer_mask` from dataset collation → `(B, max_layers)` → `(B, 6)`
- **Expanded**: `layer_mask_expanded = layer_mask[:, :, None, None, None]` → `(B, 6, 1, 1, 1)`
- ✅ **Aligned** - broadcasts correctly to `(B, 6, 4, h, w)`

### 3. Model Input/Output
- **Input**: `layers (B, 6, 4, h, w)`, `layer_mask (B, 6)`
- **Model forward** (PixArt_layer_inpainting.py:91):
  ```python
  assert N == self.max_layers  # Enforces N=6
  ```
- **Output**: `noise_pred (B, 6, 4, h, w)`
- ✅ **Aligned**

### 4. DDIM Sampling
- **x_t**: `(B, 6, 4, h, w)` - same shape throughout sampling
- **layer_mask_expanded**: `(B, 6, 1, 1, 1)` - broadcasts to match x_t
- **clean_layers**: `(B, 6, 4, h, w)` - keeps visible layers clean
- **Blending** (line 117): `x_t = x_next * layer_mask_expanded + clean_layers * (1 - layer_mask_expanded)`
- ✅ **Aligned**

## ✅ Training vs Inference Behavior

### Training (train_layer_inpainting.py:422-424)
```python
# Input preparation
z_input = z_noisy * layer_mask_expanded + z_clean * (1 - layer_mask_expanded)
# Masked layers: z_noisy (noisy at timestep t)
# Visible layers: z_clean (always clean, no noise)
```

### Training Loss (train_layer_inpainting.py:498-501)
```python
if layer_mask[b, i] == 1:  # Masked layer
    combined_loss[b, i] = noise_loss[b, i]  # Predict actual noise
elif valid_mask[b, i] == 1:  # Visible layer
    combined_loss[b, i] = zero_noise_loss[b, i]  # Predict ZERO noise
```

**Key insight**: Model is trained to predict **zero noise** for visible (clean) layers!

### Inference (infer_v4.py:54-57, 117)
```python
# Initialize: noise at masked position, clean at visible positions
x_t = clean_layers.clone()
x_t[0, masked_idx] = torch.randn(C, h, w, device=device)

# During sampling: keep visible layers clean
x_t = x_next * layer_mask_expanded + clean_layers * (1 - layer_mask_expanded)
```

✅ **Aligned**: Visible layers are clean in both training and inference

## ✅ Zero-Noise Enforcement Fix

### Issue Identified
- Training log shows `visible_loss = 0.005` ✓ (model learned well on average)
- But at `t=999`, model predicts `avg |noise| = 0.2` for visible layers ✗
- **Root cause**: Timestep conditioning affects predictions across all layers
  - High timestep (t=999) → model predicts higher noise everywhere
  - Even though visible layers are always clean during training

### Solution Implemented (infer_v4.py:91-96)
```python
# CRITICAL: Use ONLY masked layer prediction, completely ignore visible layers
noise_pred = torch.zeros_like(noise_pred_raw)
for b in range(B):
    for i_layer in range(N):
        if layer_mask[b, i_layer] == 1:  # Only copy masked layer
            noise_pred[b, i_layer] = noise_pred_raw[b, i_layer]
# Visible layers: forced to zero (as trained)
```

✅ **Correctly enforces** zero-noise constraint for visible layers

## Summary

All dimensions align correctly:
- ✅ Dataset → VAE → Model → Sampling: all use `max_layers=6`
- ✅ layer_mask broadcasting: `(B, 6)` → `(B, 6, 1, 1, 1)` → broadcasts to `(B, 6, 4, h, w)`
- ✅ Training input preparation matches inference initialization
- ✅ Zero-noise enforcement fixes timestep conditioning issue

**No alignment mismatches found!**
