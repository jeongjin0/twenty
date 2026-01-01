#!/bin/bash

# Layer-wise Inpainting Inference Script (Using Dataset)
# This script loads a sample from the dataset and generates a masked layer

# ============================================
# Configuration
# ============================================
CHECKPOINT="output/layer_inpainting_v1/checkpoints/epoch_7_step_35000.pth"
VAE_PATH="/workspace/twenty/PixArt-alpha/sd-vae-ft-ema"
T5_PATH="/workspace/twenty/PixArt-alpha"

# Dataset configuration
DATA_ROOT="../data/mulan_coco"  # or mulan_laion
SAMPLE_ID="000000000139"  # Example image ID from COCO

# Which layer to mask and regenerate (0-5 for max_layers=6)
MASKED_IDX=1

# Inference parameters
CFG_SCALE=4.5
STEPS=50
MAX_LAYERS=6
IMAGE_SIZE=256

# Output
OUTPUT_DIR="output/layer_inpainting_v1/inference/${SAMPLE_ID}"
mkdir -p ${OUTPUT_DIR}

# ============================================
# Extract visible layers from dataset
# ============================================
echo "Extracting layers from dataset sample: ${SAMPLE_ID}"

# Find the sample directory
SAMPLE_DIR="${DATA_ROOT}/images/${SAMPLE_ID}"

if [ ! -d "${SAMPLE_DIR}" ]; then
    echo "Error: Sample directory not found: ${SAMPLE_DIR}"
    exit 1
fi

# Get all layer images (exclude the masked one)
LAYER_IMAGES=($(find ${SAMPLE_DIR} -name "layer_*.png" | sort))
NUM_LAYERS=${#LAYER_IMAGES[@]}

echo "Found ${NUM_LAYERS} layers in sample ${SAMPLE_ID}"

# Build visible layers list (exclude masked layer)
VISIBLE_LAYERS=""
MASKED_LAYER=""
for i in "${!LAYER_IMAGES[@]}"; do
    if [ $i -eq ${MASKED_IDX} ]; then
        MASKED_LAYER="${LAYER_IMAGES[$i]}"
        echo "  Layer $i: [MASKED] ${LAYER_IMAGES[$i]}"
    else
        VISIBLE_LAYERS="${VISIBLE_LAYERS} ${LAYER_IMAGES[$i]}"
        echo "  Layer $i: [VISIBLE] ${LAYER_IMAGES[$i]}"
    fi
done

# Get the caption for the masked layer
CAPTION_FILE="${SAMPLE_DIR}/captions.json"
if [ -f "${CAPTION_FILE}" ]; then
    # Extract caption for masked layer (assuming layer captions are in JSON)
    PROMPT=$(python -c "import json; data=json.load(open('${CAPTION_FILE}')); print(data[${MASKED_IDX}])" 2>/dev/null || echo "a layer")
    echo "Prompt for layer ${MASKED_IDX}: ${PROMPT}"
else
    PROMPT="a layer"
    echo "Warning: No caption file found, using default prompt"
fi

# ============================================
# Run Inference
# ============================================
export PYTHONPATH="${PYTHONPATH}:/workspace/twenty"

OUTPUT_FILE="${OUTPUT_DIR}/generated_layer_${MASKED_IDX}.png"

python scripts/infer_layer_inpainting.py \
    --checkpoint ${CHECKPOINT} \
    --visible_layers ${VISIBLE_LAYERS} \
    --masked_idx ${MASKED_IDX} \
    --prompt "${PROMPT}" \
    --output ${OUTPUT_FILE} \
    --cfg_scale ${CFG_SCALE} \
    --steps ${STEPS} \
    --max_layers ${MAX_LAYERS} \
    --image_size ${IMAGE_SIZE} \
    --vae_path ${VAE_PATH} \
    --t5_path ${T5_PATH}

# ============================================
# Create comparison (optional)
# ============================================
echo ""
echo "Creating comparison with ground truth..."

# Copy ground truth for comparison
if [ -f "${MASKED_LAYER}" ]; then
    cp "${MASKED_LAYER}" "${OUTPUT_DIR}/ground_truth_layer_${MASKED_IDX}.png"
    echo "Ground truth saved to: ${OUTPUT_DIR}/ground_truth_layer_${MASKED_IDX}.png"
fi

echo ""
echo "Inference complete!"
echo "Generated layer: ${OUTPUT_FILE}"
echo "All results in: ${OUTPUT_DIR}"
