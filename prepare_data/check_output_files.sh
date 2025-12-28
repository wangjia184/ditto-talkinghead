#!/bin/bash

# Script to check which output files have been generated
# Usage: bash prepare_data/check_output_files.sh <data_info_json>

if [ $# -lt 1 ]; then
    echo "Usage: $0 <data_info_json>"
    echo "Example: $0 /data/example/trainset_example/HDTF/data_info.json"
    exit 1
fi

data_info_json="$1"

if [ ! -f "$data_info_json" ]; then
    echo "Error: data_info_json file not found: $data_info_json"
    exit 1
fi

echo "=========================================="
echo "Checking output files from data_info.json"
echo "=========================================="
echo ""

# Load JSON and extract paths
python3 << EOF
import json
import os

with open("$data_info_json", 'r') as f:
    data_info = json.load(f)

# Get base directory from first video path
if data_info.get('video_list') and len(data_info['video_list']) > 0:
    first_video = data_info['video_list'][0]
    base_dir = os.path.dirname(os.path.dirname(first_video))
    print(f"Base directory: {base_dir}")
    print("")

# Check each file type
file_types = {
    'video_list': 'video/',
    'wav_list': 'wav/',
    'hubert_aud_npy_list': 'hubert_aud_npy/',
    'LP_pkl_list': 'LP_pkl/',
    'LP_npy_list': 'LP_npy/',
    'MP_lmk_npy_list': 'MP_lmk_npy/',
    'eye_open_npy_list': 'eye_open_npy/',
    'eye_ball_npy_list': 'eye_ball_npy/',
    'emo_npy_list': 'emo_npy/',
}

for file_type, dir_name in file_types.items():
    if file_type not in data_info:
        continue
    
    file_list = data_info[file_type]
    if not file_list:
        continue
    
    total = len(file_list)
    existing = sum(1 for f in file_list if os.path.isfile(f))
    missing = total - existing
    
    status = "✓" if existing == total else "✗"
    print(f"{status} {dir_name:20s} {existing:4d}/{total:4d} files exist ({missing} missing)")
    
    # Show first few missing files if any
    if missing > 0 and missing <= 5:
        missing_files = [f for f in file_list if not os.path.isfile(f)]
        for mf in missing_files[:3]:
            print(f"    Missing: {os.path.basename(mf)}")
    elif missing > 5:
        missing_files = [f for f in file_list if not os.path.isfile(f)]
        print(f"    Missing: {missing} files (showing first 3)")
        for mf in missing_files[:3]:
            print(f"    Missing: {os.path.basename(mf)}")
    print("")

EOF

echo "=========================================="
echo "Check completed!"
echo "=========================================="

