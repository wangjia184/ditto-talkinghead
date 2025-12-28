#!/bin/bash

python3 example/prepare_hdtf_data.py \
    --video_dir /data/example/trainset_example/HDTF \
    --save_dir /data/example/trainset_example/HDTF_processed \
    --output_json /data/example/trainset_example/HDTF/data_info.json

conda init
.  /root/.bashrc
conda activate ditto_train

bash prepare_data/prepare_data_parallel.sh \
    /data/example/trainset_example/HDTF/data_info.json \
    /data/example/trainset_example/HDTF/data_list.json \
    /data/example/trainset_example/HDTF/data_preload.pkl \
    4  # 使用12个进程