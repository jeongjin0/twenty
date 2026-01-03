# Visible Layer Loss Implementation

## Problem

Previously, only the masked (target) layer had loss during diffusion training:
- **Masked layer**: Noisy → predict noise → compute loss ✓
- **Visible layers**: Clean → predict noise → **no loss** ✗

This created the "reference ignore" problem where the model could ignore visible reference layers.

## Solution

Added loss on visible layers to force the model to preserve them correctly.

### Loss Formulation

**Masked Layer Loss** (existing):
```python
# Input: z_clean + noise (at timestep t)
# Target: predict the actual noise
masked_loss = mse_loss(noise_pred[masked], noise[masked])
```

**Visible Layer Loss** (new):
```python
# Input: z_clean (no noise added)
# Target: predict zero noise (since no noise was added)
visible_loss = mse_loss(noise_pred[visible], zeros)
```

**Total Loss**:
```python
total_loss = masked_loss + visible_weight * visible_loss
```

### Why Zero Noise for Visible Layers?

During training:
1. Visible layers receive **clean latents** (no noise added)
2. This is equivalent to timestep t=0 in the diffusion process
3. At t=0, there is zero noise
4. Therefore, the model should predict zero noise for visible layers

This forces the model to:
- Recognize that visible layers are clean
- Preserve them during forward pass
- Actually use reference information instead of ignoring it

## Configuration

**New Parameter**: `visible_loss_weight`
- Default: 0.5
- Controls the strength of visible layer preservation
- Higher values → stronger preservation of references

Example:
```python
# In config file
visible_loss_weight = 0.5  # Balance between masked and visible losses
```

## Benefits

1. **Prevents Reference Ignore**: Model must preserve visible layers
2. **Better Reference Usage**: Forces model to incorporate context from other layers
3. **Improved Consistency**: Output layers remain consistent with input references
4. **Complementary to Pretraining**: Works together with merge/decompose pretraining

## Training Logs

Now shows three metrics:
- `loss`: Total loss (masked + visible)
- `masked`: Loss on target layer (noise prediction)
- `visible`: Loss on reference layers (preservation)

Example output:
```
Step [1000/50000]: lr: 1e-4, loss: 0.0234, masked: 0.0156, visible: 0.0078
```
