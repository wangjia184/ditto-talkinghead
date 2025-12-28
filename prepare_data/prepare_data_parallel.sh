#!/bin/bash
# Process videos in parallel using multiprocessing
# This is the fastest way to process large datasets

set -euo pipefail

SECONDS=0

DITTO_ROOT_DIR="$(dirname "$(dirname "$(readlink -f "$0")")")"

DITTO_PYTORCH_PATH="${DITTO_ROOT_DIR}/checkpoints/ditto_pytorch"
HUBERT_ONNX="${DITTO_PYTORCH_PATH}/aux_models/hubert_streaming_fix_kv.onnx"
MP_FACE_LMK_TASK="${DITTO_PYTORCH_PATH}/aux_models/face_landmarker.task"

cd "${DITTO_ROOT_DIR}/prepare_data"

data_info_json="$1"
data_list_json="$2"
data_preload_pkl="$3"
num_workers="${4:-0}"  # Optional: number of workers (0 = auto-detect)

echo "=========================================="
echo "Starting parallel data preparation..."
echo "Input:  ${data_info_json}"
echo "Output: ${data_list_json}"
echo "Output: ${data_preload_pkl}"
echo "Workers: ${num_workers} (0 = auto-detect)"
echo "=========================================="
echo ""

# check ckpt
echo "[Step 0] Checking checkpoints..."
python scripts/check_ckpt_path.py --ditto_pytorch_path ${DITTO_PYTORCH_PATH}
echo "[Step 0] ✓ Checkpoints check completed"
echo ""

# Process videos in parallel
echo "[Step 1-6] Processing all videos in parallel (all steps per video)..."
python scripts/process_videos_parallel.py \
    -i "${data_info_json}" \
    --ditto_pytorch_path "${DITTO_PYTORCH_PATH}" \
    --Hubert_onnx "${HUBERT_ONNX}" \
    --MP_face_landmarker_task_path "${MP_FACE_LMK_TASK}" \
    --num_workers "${num_workers}" \
    --skip_existing

if [ $? -ne 0 ]; then
    echo "Warning: Some videos failed to process. Continuing anyway..."
fi

echo ""
echo "[Step 7] Gathering data list JSON for training..."
python scripts/gather_data_list_json_for_train.py \
    -i "${data_info_json}" \
    -o "${data_list_json}" \
    --use_emo \
    --use_eye_open \
    --use_eye_ball \
    --with_flip
echo "[Step 7] ✓ Data list JSON generation completed"
echo ""

echo "[Step 8] Preloading training data to PKL..."
python scripts/preload_train_data_to_pkl.py \
    --data_list_json "${data_list_json}" \
    --data_preload_pkl "${data_preload_pkl}" \
    --use_sc \
    --use_emo \
    --use_eye_open \
    --use_eye_ball \
    --motion_feat_dim 265
echo "[Step 8] ✓ Data preloading completed"
echo ""

cd "${DITTO_ROOT_DIR}"

echo "=========================================="
echo "[prepare_data] All steps completed!"
echo ""
echo "Output files:"
echo "  data_list_json: ${data_list_json}"
echo "  data_preload_pkl: ${data_preload_pkl}"
echo "=========================================="
echo ""
echo "Elapsed time: $SECONDS seconds"

