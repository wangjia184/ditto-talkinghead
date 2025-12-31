#!/bin/bash

# Script to export LMDM checkpoint to ONNX format
# Usage: ./export_checkpoint_to_onnx.sh

set -e  # Exit on error

# Configuration
CHECKPOINT_PATH="/data/experiments/s2/ditto_s2_baseline/ckpts/train_626.pt"
OUTPUT_DIR="/data/experiments/s2/ditto_s2_baseline/onnx"
OUTPUT_NAME="lmdm_train_626.onnx"
OUTPUT_PATH="${OUTPUT_DIR}/${OUTPUT_NAME}"

# Model parameters (adjust if needed)
MOTION_FEAT_DIM=265
AUDIO_FEAT_DIM=1103
SEQ_FRAMES=80  # 3.2 * 25

# Device (use 'cpu' if CUDA is not available)
DEVICE="cuda"

echo "=========================================="
echo "Exporting LMDM Checkpoint to ONNX"
echo "=========================================="
echo ""
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Output: ${OUTPUT_PATH}"
echo ""
echo "Model Parameters:"
echo "  Motion Feature Dim: ${MOTION_FEAT_DIM}"
echo "  Audio Feature Dim: ${AUDIO_FEAT_DIM}"
echo "  Sequence Frames: ${SEQ_FRAMES}"
echo "  Device: ${DEVICE}"
echo ""

# Check if checkpoint exists
if [ ! -f "${CHECKPOINT_PATH}" ]; then
    echo "Error: Checkpoint not found at ${CHECKPOINT_PATH}"
    exit 1
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Run export script
# Note: By default, fixed size is used (better for Netron visualization)
# Add --use_dynamic_axes if you need variable batch_size/seq_len
python export_to_onnx.py \
    --checkpoint "${CHECKPOINT_PATH}" \
    --output "${OUTPUT_PATH}" \
    --motion_feat_dim ${MOTION_FEAT_DIM} \
    --audio_feat_dim ${AUDIO_FEAT_DIM} \
    --seq_frames ${SEQ_FRAMES} \
    --device ${DEVICE}
    # --use_dynamic_axes  # Uncomment this if you need dynamic batch_size/seq_len

echo ""
echo "=========================================="
echo "Export completed successfully!"
echo "=========================================="
echo "ONNX model saved to: ${OUTPUT_PATH}"
echo ""
