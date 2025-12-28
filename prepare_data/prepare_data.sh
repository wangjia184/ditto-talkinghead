set -euo pipefail
# set -x
 

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
echo "Starting data preparation..."
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

###############################################
# process data

### VIDEO ###
# crop video: 'fps25_video_list' -> 'video_list'
echo "[Step 1] Cropping videos: 'fps25_video_list' -> 'video_list'"
python scripts/crop_video_by_LP.py -i "${data_info_json}" --ditto_pytorch_path "${DITTO_PYTORCH_PATH}"
echo "[Step 1] ✓ Video cropping completed"
echo ""

### AUDIO ###
# extract audio: 'video_list' -> 'wav_list'
echo "[Step 2] Extracting audio: 'video_list' -> 'wav_list'"
python scripts/extract_audio_from_video.py -i "${data_info_json}"
echo "[Step 2] ✓ Audio extraction completed"
echo ""

### Feature ###
# audio feat: 'wav_list' -> 'hubert_aud_npy_list'
echo "[Step 3] Extracting audio features: 'wav_list' -> 'hubert_aud_npy_list'"
python scripts/extract_audio_feat_by_Hubert.py -i "${data_info_json}" --Hubert_onnx "${HUBERT_ONNX}"
echo "[Step 3] ✓ Audio feature extraction completed"
echo ""

# motion feat: 'video_list' -> {'LP_pkl_list', 'LP_npy_list'} (_flip)
echo "[Step 4a] Extracting motion features (normal): 'video_list' -> {'LP_pkl_list', 'LP_npy_list'}"
python scripts/extract_motion_feat_by_LP.py -i "${data_info_json}" --ditto_pytorch_path "${DITTO_PYTORCH_PATH}"
echo "[Step 4a] ✓ Motion feature extraction (normal) completed"
echo ""

echo "[Step 4b] Extracting motion features (flip): 'video_list' -> {'LP_pkl_list', 'LP_npy_list'} (_flip)"
python scripts/extract_motion_feat_by_LP.py -i "${data_info_json}" --ditto_pytorch_path "${DITTO_PYTORCH_PATH}" --flip_flag
echo "[Step 4b] ✓ Motion feature extraction (flip) completed"
echo ""

# eye feat: 'video_list' -> {'MP_lmk_npy_list', 'eye_open_npy_list', 'eye_ball_npy_list'} (_flip)
echo "[Step 5a] Extracting eye features (normal): 'video_list' -> {'MP_lmk_npy_list', 'eye_open_npy_list', 'eye_ball_npy_list'}"
python scripts/extract_eye_ratio_from_video.py -i "${data_info_json}" --MP_face_landmarker_task_path "${MP_FACE_LMK_TASK}"
echo "[Step 5a] ✓ Eye feature extraction (normal) completed"
echo ""

echo "[Step 5b] Extracting eye features (flip): 'video_list' -> {'MP_lmk_npy_list', 'eye_open_npy_list', 'eye_ball_npy_list'} (_flip)"
python scripts/extract_eye_ratio_from_video.py -i "${data_info_json}" --MP_face_landmarker_task_path "${MP_FACE_LMK_TASK}" --flip_lmk_flag
echo "[Step 5b] ✓ Eye feature extraction (flip) completed"
echo ""

# emo feat: 'video_list' -> 'emo_npy_list'
echo "[Step 6] Extracting emotion features: 'video_list' -> 'emo_npy_list'"
python scripts/extract_emo_feat_from_video.py -i "${data_info_json}"
echo "[Step 6] ✓ Emotion feature extraction completed"
echo ""


###############################################
# get data_list_json for train
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

# get preload data_pkl ([option] for faster training speed)
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

###############################################

echo "=========================================="
echo "[prepare_data] All steps completed successfully!"
echo ""
echo "Output files:"
echo "  data_list_json: ${data_list_json}"
echo "  data_preload_pkl: ${data_preload_pkl}"
echo ""
echo "Expected output directories (relative to save_dir in data_info.json):"
echo "  - video/          (mp4 files - cropped videos)"
echo "  - wav/            (wav files - extracted audio)"
echo "  - hubert_aud_npy/ (npy files - audio features)"
echo "  - LP_pkl/         (pkl files - motion features)"
echo "  - LP_npy/         (npy files - motion features)"
echo "  - MP_lmk_npy/     (npy files - face landmarks)"
echo "  - eye_open_npy/   (npy files - eye open state)"
echo "  - eye_ball_npy/   (npy files - eye ball state)"
echo "  - emo_npy/        (npy files - emotion features)"
echo "=========================================="
echo ""
echo "Elapsed time: $SECONDS seconds"
