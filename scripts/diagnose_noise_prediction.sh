#!/bin/bash

# Noise Prediction Diagnostic
# Tests if the model can correctly predict noise at different timesteps

CHECKPOINT="${1:-output/inpainting_v4/checkpoints/epoch_10_step_50000.pth}"

echo "============================================"
echo "Noise Prediction Diagnostic"
echo "============================================"
echo "Checkpoint: ${CHECKPOINT}"
echo ""
echo "This tests the FUNDAMENTAL noise prediction:"
echo "  1. Add KNOWN noise to a layer"
echo "  2. Ask model to predict that exact noise"
echo "  3. Compare predicted vs actual"
echo "============================================"

python scripts/diagnose_noise_prediction.py \
    --checkpoint ${CHECKPOINT} \
    --data_roots ../data/mulan_coco \
    --output_dir output/noise_diagnostic
