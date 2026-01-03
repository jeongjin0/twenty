#!/bin/bash

# Analyze Input/Output Projections
# Visualizes what input_proj and output_proj are processing

# ============================================
# Configuration
# ============================================
CHECKPOINT="output/layer_inpainting_v3/checkpoints/epoch_9_step_45000.pth"
DATA_ROOTS="/workspace/data/mulan_coco"
OUTPUT_DIR="output/projection_analysis"
VAE_PATH="/workspace/twenty/PixArt-alpha/sd-vae-ft-ema"
T5_PATH="/workspace/twenty/PixArt-alpha"

# Analysis parameters
NUM_SAMPLES=4  # Number of samples to analyze
BATCH_SIZE=2
MAX_LAYERS=6
IMAGE_SIZE=256

# ============================================
# Run Analysis
# ============================================
echo "============================================"
echo "Projection Analysis"
echo "============================================"
echo "Checkpoint: ${CHECKPOINT}"
echo "Data: ${DATA_ROOTS}"
echo "Output: ${OUTPUT_DIR}"
echo "Samples: ${NUM_SAMPLES}"
echo "============================================"
echo ""

export PYTHONPATH="${PYTHONPATH}:/workspace/twenty"

python scripts/analyze_projections.py \
    --checkpoint ${CHECKPOINT} \
    --data_roots ${DATA_ROOTS} \
    --output_dir ${OUTPUT_DIR} \
    --num_samples ${NUM_SAMPLES} \
    --batch_size ${BATCH_SIZE} \
    --max_layers ${MAX_LAYERS} \
    --image_size ${IMAGE_SIZE} \
    --vae_path ${VAE_PATH} \
    --t5_path ${T5_PATH}

echo ""
echo "============================================"
echo "Analysis complete!"
echo "============================================"
echo "Results saved to: ${OUTPUT_DIR}"
echo ""
echo "For each sample, you'll find:"
echo ""
echo "  Latent visualizations (raw latent space):"
echo "    - input_proj_input_latent.png"
echo "    - input_proj_output_latent.png"
echo "    - output_proj_input_latent.png"
echo "    - output_proj_output_latent.png"
echo ""
echo "  Decoded visualizations (actual images via VAE):"
echo "    - input_proj_input_decoded.png: 6 input layers"
echo "    - input_proj_output_decoded.png: PixArt input"
echo "    - output_proj_input_decoded.png: PixArt output"
echo "    - output_proj_output_decoded.png: 6 output layers"
echo ""
echo "  Statistics:"
echo "    - statistics.txt"
echo "============================================"
