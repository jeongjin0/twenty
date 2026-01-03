#!/bin/bash

# Layer-wise Inpainting Inference (Dataset-based)
# Loads samples directly from MuLan dataset

# ============================================
# Configuration
# ============================================
CHECKPOINT="output/layer_inpainting_v3/checkpoints/epoch_9_step_45000.pth"
DATA_ROOTS="/workspace/data/mulan_coco"
OUTPUT_DIR="output/inference_from_dataset"
VAE_PATH="/workspace/twenty/PixArt-alpha/sd-vae-ft-ema"
T5_PATH="/workspace/twenty/PixArt-alpha"

# Inference parameters
NUM_SAMPLES=5  # Number of samples to process
CFG_SCALE=1.0  # No CFG (model not trained with unconditional)
STEPS=50
MAX_LAYERS=6
IMAGE_SIZE=256

# ============================================
# Run Inference
# ============================================
echo "============================================"
echo "Layer-wise Inpainting from Dataset"
echo "============================================"
echo "Checkpoint: ${CHECKPOINT}"
echo "Data: ${DATA_ROOTS}"
echo "Output: ${OUTPUT_DIR}"
echo "Samples: ${NUM_SAMPLES}"
echo "CFG Scale: ${CFG_SCALE}"
echo "Steps: ${STEPS}"
echo "============================================"
echo ""

export PYTHONPATH="${PYTHONPATH}:/workspace/twenty"

python scripts/infer_layer_inpainting_dataset.py \
    --checkpoint ${CHECKPOINT} \
    --data_roots ${DATA_ROOTS} \
    --output_dir ${OUTPUT_DIR} \
    --num_samples ${NUM_SAMPLES} \
    --cfg_scale ${CFG_SCALE} \
    --steps ${STEPS} \
    --max_layers ${MAX_LAYERS} \
    --image_size ${IMAGE_SIZE} \
    --vae_path ${VAE_PATH} \
    --t5_path ${T5_PATH}

echo ""
echo "============================================"
echo "Inference complete!"
echo "============================================"
echo "Results saved to: ${OUTPUT_DIR}"
echo ""
echo "For each sample (image_id_layerN):"
echo "  - generated.png: Generated masked layer"
echo "  - ground_truth.png: Original masked layer"
echo "  - all_layers_gt.png: GT visible + generated masked"
echo "  - all_layers_model_pred.png: Model predictions for all layers"
echo "  - info.txt: Sample information"
echo "============================================"
