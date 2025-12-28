#!/bin/bash
# Process videos one by one, completing all steps for each video before moving to the next
# This is more efficient and allows you to see progress immediately

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

echo "=========================================="
echo "Starting data preparation (per-video mode)..."
echo "Input:  ${data_info_json}"
echo "Output: ${data_list_json}"
echo "Output: ${data_preload_pkl}"
echo "=========================================="
echo ""

# check ckpt
echo "[Step 0] Checking checkpoints..."
python scripts/check_ckpt_path.py --ditto_pytorch_path ${DITTO_PYTORCH_PATH}
echo "[Step 0] ✓ Checkpoints check completed"
echo ""

# Load data_info.json and process each video
python3 << EOF
import json
import os
import subprocess
import sys

# Load data_info.json
with open("${data_info_json}", 'r') as f:
    data_info = json.load(f)

fps25_video_list = data_info.get('fps25_video_list', [])
video_list = data_info.get('video_list', [])
wav_list = data_info.get('wav_list', [])
hubert_aud_npy_list = data_info.get('hubert_aud_npy_list', [])
LP_pkl_list = data_info.get('LP_pkl_list', [])
LP_npy_list = data_info.get('LP_npy_list', [])
MP_lmk_npy_list = data_info.get('MP_lmk_npy_list', [])
eye_open_npy_list = data_info.get('eye_open_npy_list', [])
eye_ball_npy_list = data_info.get('eye_ball_npy_list', [])
emo_npy_list = data_info.get('emo_npy_list', [])

# Generate flip paths
def flip_path(p):
    items = p.split('/')
    if len(items) >= 2:
        items[-2] = items[-2] + '_flip'
    return '/'.join(items)

LP_pkl_flip_list = [flip_path(p) for p in LP_pkl_list]
LP_npy_flip_list = [flip_path(p) for p in LP_npy_list]
MP_lmk_npy_flip_list = [flip_path(p) for p in MP_lmk_npy_list]
eye_open_npy_flip_list = [flip_path(p) for p in eye_open_npy_list]
eye_ball_npy_flip_list = [flip_path(p) for p in eye_ball_npy_list]

total = len(fps25_video_list)
success_count = 0
fail_count = 0

print(f"Total videos to process: {total}")
print("")

for idx in range(total):
    fps25_video = fps25_video_list[idx]
    video = video_list[idx]
    wav = wav_list[idx]
    hubert_aud_npy = hubert_aud_npy_list[idx]
    LP_pkl = LP_pkl_list[idx]
    LP_npy = LP_npy_list[idx]
    LP_pkl_flip = LP_pkl_flip_list[idx]
    LP_npy_flip = LP_npy_flip_list[idx]
    MP_lmk_npy = MP_lmk_npy_list[idx]
    eye_open_npy = eye_open_npy_list[idx]
    eye_ball_npy = eye_ball_npy_list[idx]
    MP_lmk_npy_flip = MP_lmk_npy_flip_list[idx]
    eye_open_npy_flip = eye_open_npy_flip_list[idx]
    eye_ball_npy_flip = eye_ball_npy_flip_list[idx]
    emo_npy = emo_npy_list[idx]
    
    cmd = [
        sys.executable, "scripts/process_one_video_complete.py",
        "--fps25_video", fps25_video,
        "--video", video,
        "--wav", wav,
        "--hubert_aud_npy", hubert_aud_npy,
        "--LP_pkl", LP_pkl,
        "--LP_npy", LP_npy,
        "--LP_pkl_flip", LP_pkl_flip,
        "--LP_npy_flip", LP_npy_flip,
        "--MP_lmk_npy", MP_lmk_npy,
        "--eye_open_npy", eye_open_npy,
        "--eye_ball_npy", eye_ball_npy,
        "--MP_lmk_npy_flip", MP_lmk_npy_flip,
        "--eye_open_npy_flip", eye_open_npy_flip,
        "--eye_ball_npy_flip", eye_ball_npy_flip,
        "--emo_npy", emo_npy,
        "--ditto_pytorch_path", "${DITTO_PYTORCH_PATH}",
        "--Hubert_onnx", "${HUBERT_ONNX}",
        "--MP_face_landmarker_task_path", "${MP_FACE_LMK_TASK}",
        "--skip_existing"
    ]
    
    print(f"[{idx+1}/{total}] Processing video: {os.path.basename(fps25_video)}")
    sys.stdout.flush()
    
    try:
        result = subprocess.run(cmd, check=False, capture_output=False)
        if result.returncode == 0:
            success_count += 1
            print(f"[{idx+1}/{total}] ✓ Completed successfully\n")
        else:
            fail_count += 1
            print(f"[{idx+1}/{total}] ✗ Failed (return code: {result.returncode})\n")
    except Exception as e:
        fail_count += 1
        print(f"[{idx+1}/{total}] ✗ Exception: {e}\n")

print("="*60)
print(f"Processing summary: {success_count} succeeded, {fail_count} failed out of {total} total")
print("="*60)
EOF

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

