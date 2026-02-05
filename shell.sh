#!/bin/bash

docker run -it --rm --runtime=nvidia --gpus all \
    --ipc host \
    --privileged \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --ulimit nofile=65536:65536 \
    --shm-size=8g \
    --name audio2head-trainer \
    -v $(pwd):/data \
    -v /mnt/z/HDTF_processed:/data/HDTF_processed:ro \
    everymatrix.jfrog.io/emlab-docker/ayida/audio2head:trainer /bin/bash