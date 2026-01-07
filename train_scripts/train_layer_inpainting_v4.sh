#!/bin/bash

# Layer-wise Inpainting v4 Training
# Uses pretrained UNet-style projections with skip connections

CONFIG="configs/layer_inpainting_config.py"
OUTPUT_DIR="output/inpainting_v4"
PRETRAINED_PIXART="PixArt-alpha/PixArt-XL-2-256x256.pth"  # Update this path
PRETRAINED_PROJECTIONS="output/projection_pretrain_v3/checkpoints/epoch_05.pth"  # From projection pretraining

EPOCHS=50
LR=1e-4
BATCH_SIZE=4
GRADIENT_ACCUMULATION=4

# Weight configuration (loss weights for different layer types)
MASKED_LAYER_WEIGHT=1.0      # Target (masked) layer: highest priority
VISIBLE_LAYER_WEIGHT=0.5     # Reference (visible) layers: medium priority
BACKGROUND_LAYER_WEIGHT=0.3  # Background (layer 0): lower priority within visible

# 2-Stage Training (optional): Train projections first, then unfreeze PixArt
PIXART_FREEZE_EPOCHS=0  # Freeze PixArt for first N epochs (0 = disabled)

echo "============================================"
echo "Layer-wise Inpainting v4 Training"
echo "============================================"
echo "Strategy: Reference-based inpainting with layer-wise control"
echo ""
echo "Architecture:"
echo "  - Input Projection: UNet-style (6 layers + mask → 4ch merged latent)"
echo "  - Backbone: PixArt-XL (pretrained diffusion transformer)"
echo "  - Output Projection: UNet-style (4ch merged latent → 6 layers)"
echo ""
echo "Key Features:"
echo "  - Pretrained projections with skip connections"
echo "  - Layer weight hierarchy: masked (1.0) > visible (0.5) > background (0.15)"
echo "  - Visible layer loss: prevent model ignoring references"
echo "  - 2-Stage training: projections → full model"
echo ""
echo "Config: ${CONFIG}"
echo "Output: ${OUTPUT_DIR}"
echo "Pretrained PixArt: ${PRETRAINED_PIXART}"
echo "Pretrained Projections: ${PRETRAINED_PROJECTIONS}"
echo "Epochs: ${EPOCHS}"
echo "LR: ${LR}"
echo "Batch size: ${BATCH_SIZE} × ${GRADIENT_ACCUMULATION} accumulation"
echo ""
if [ ${PIXART_FREEZE_EPOCHS} -gt 0 ]; then
    echo "2-Stage Training:"
    echo "  Stage 1 (Epoch 0-$((PIXART_FREEZE_EPOCHS-1))): Train projections only"
    echo "  Stage 2 (Epoch ${PIXART_FREEZE_EPOCHS}+): Train full model"
    echo ""
fi
echo "Loss Weights:"
echo "  Masked layer: ${MASKED_LAYER_WEIGHT}"
echo "  Visible layers: ${VISIBLE_LAYER_WEIGHT}"
echo "  Background (within visible): ${BACKGROUND_LAYER_WEIGHT}"
echo ""
echo "Multi-GPU: 2 GPUs, mixed precision FP16"
echo "============================================"
echo ""

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export NCCL_P2P_DISABLE=1

# Multi-GPU training (2 GPUs)
accelerate launch \
    --num_processes=2 \
    --num_machines=1 \
    --mixed_precision=fp16 \
    --multi_gpu \
    train_scripts/train_layer_inpainting.py \
    --config ${CONFIG} \
    --pretrained_pixart ${PRETRAINED_PIXART} \
    --pretrained_projections ${PRETRAINED_PROJECTIONS}

echo ""
echo "============================================"
echo "Training complete!"
echo "============================================"
echo "Checkpoints saved to: ${OUTPUT_DIR}/checkpoints/"
echo ""
echo "Next: Inference"
echo "  python inference_scripts/infer_layer_inpainting.py \\\\"
echo "    --config ${CONFIG} \\\\"
echo "    --checkpoint ${OUTPUT_DIR}/checkpoints/epoch_${EPOCHS}.pth \\\\"
echo "    --prompt 'your prompt here' \\\\"
echo "    --reference_layers path/to/layers/ \\\\"
echo "    --output output_image.png"
echo "============================================"
