#!/bin/bash

# Ditto Training Script
# This script launches the training inside a Docker container

# Training configuration
EXPERIMENT_NAME="${1:-ditto_s2_baseline}"
BATCH_SIZE="${2:-32}"
EPOCHS="${3:-1000}"

# Data and checkpoint paths
DATA_ROOT="/hdtf_data"
DATA_INFO_JSON="/data/data_info.json"
DATA_LIST_JSON="/data/data_list_train.json"
EXPERIMENT_DIR="/data/experiments/s2"
CHECKPOINT_DIR="/data/checkpoints/ditto_pytorch/models"

# Training hyperparameters
SAVE_CKPT_FREQ=50
LR=1e-4
NUM_WORKERS=4

echo "=========================================="
echo "Ditto Training Script"
echo "=========================================="
echo "Experiment: $EXPERIMENT_NAME"
echo "Batch Size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "=========================================="

# Step 1: Generate data info JSON from the prepared data structure
if [ ! -f "$DATA_INFO_JSON" ]; then
    echo "Step 1: Generating data info JSON from prepared data..."
    python << 'EOF'
import os
import json

data_root = "/hdtf_data"
save_dir = data_root

# Check what directories exist
available_dirs = {}
for dirname in ['LP_npy', 'LP_pkl', 'MP_lmk_npy', 'hubert_aud_npy', 
                'video', 'wav', 'emo_npy', 'eye_open_npy', 'eye_ball_npy']:
    dirpath = os.path.join(data_root, dirname)
    if os.path.exists(dirpath):
        available_dirs[dirname] = True

# Build data lists based on available directories
video_list = []
if os.path.exists(os.path.join(data_root, 'video')):
    video_files = [f for f in os.listdir(os.path.join(data_root, 'video')) if f.endswith('.mp4')]
    video_list = [os.path.join(data_root, 'video', f) for f in video_files]

# Get the base names to build all paths
names = set()
for dirname in ['LP_npy', 'hubert_aud_npy', 'video']:
    dirpath = os.path.join(data_root, dirname)
    if os.path.exists(dirpath):
        files = os.listdir(dirpath)
        for f in files:
            if '.' in f:
                name = f.rsplit('.', 1)[0]
                names.add(name)

names = sorted(list(names))

# Build the data info dictionary
data_info = {
    'video_list': [os.path.join(data_root, 'video', n + '.mp4') for n in names],
    'wav_list': [os.path.join(data_root, 'wav', n + '.wav') for n in names] if os.path.exists(os.path.join(data_root, 'wav')) else [],
    'LP_pkl_list': [os.path.join(data_root, 'LP_pkl', n + '.pkl') for n in names] if os.path.exists(os.path.join(data_root, 'LP_pkl')) else [],
    'LP_npy_list': [os.path.join(data_root, 'LP_npy', n + '.npy') for n in names],
    'hubert_aud_npy_list': [os.path.join(data_root, 'hubert_aud_npy', n + '.npy') for n in names],
    'MP_lmk_npy_list': [os.path.join(data_root, 'MP_lmk_npy', n + '.npy') for n in names] if os.path.exists(os.path.join(data_root, 'MP_lmk_npy')) else [],
    'eye_open_npy_list': [os.path.join(data_root, 'eye_open_npy', n + '.npy') for n in names] if os.path.exists(os.path.join(data_root, 'eye_open_npy')) else [],
    'eye_ball_npy_list': [os.path.join(data_root, 'eye_ball_npy', n + '.npy') for n in names] if os.path.exists(os.path.join(data_root, 'eye_ball_npy')) else [],
    'emo_npy_list': [os.path.join(data_root, 'emo_npy', n + '.npy') for n in names] if os.path.exists(os.path.join(data_root, 'emo_npy')) else [],
}

with open('/data/data_info.json', 'w') as f:
    json.dump(data_info, f, indent=2)

print(f"Generated data_info.json with {len(names)} samples")
print(f"Available modalities: {list(available_dirs.keys())}")
EOF
else
    echo "data_info.json already exists"
fi

# Step 2: Generate training data list from the data info
if [ ! -f "$DATA_LIST_JSON" ]; then
    echo "Step 2: Generating training data list..."
    cd /data
    python prepare_data/scripts/gather_data_list_json_for_train.py \
        --input-data-json $DATA_INFO_JSON \
        --output-data-json $DATA_LIST_JSON \
        --dataset-version v2 \
        --use-emo \
        --use-eye-open \
        --use-eye-ball \
        --with-flip
    
    if [ ! -f "$DATA_LIST_JSON" ]; then
        echo "ERROR: Failed to generate data list!"
        exit 1
    fi
    
    # Count the number of training samples
    SAMPLE_COUNT=$(python -c "import json; print(len(json.load(open('$DATA_LIST_JSON'))))")
    echo "Generated training data list with $SAMPLE_COUNT samples"
else
    echo "Training data list already exists"
fi

# Step 3: Create experiment directory
mkdir -p $EXPERIMENT_DIR
echo "Created experiment directory: $EXPERIMENT_DIR"

# Step 4: Launch training
echo ""
echo "=========================================="
echo "Step 3: Starting training..."
echo "=========================================="
cd /data

python MotionDiT/train.py \
    --experiment_dir $EXPERIMENT_DIR \
    --experiment_name $EXPERIMENT_NAME \
    --data_list_json $DATA_LIST_JSON \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --save_ckpt_freq $SAVE_CKPT_FREQ \
    --lr $LR \
    --motion_feat_dim 265 \
    --audio_feat_dim 1103 \
    --seq_frames 80 \
    --num_workers $NUM_WORKERS \
    --use_sc \
    --use_last_frame \
    --use_last_frame_loss \
    --use_emo \
    --use_eye_open \
    --use_eye_ball

echo ""
echo "=========================================="
echo "Training completed!"
echo "Checkpoints saved to: $EXPERIMENT_DIR/$EXPERIMENT_NAME"
echo "=========================================="
