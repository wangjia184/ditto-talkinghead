#!/bin/bash

docker run -it --rm --runtime=nvidia --gpus all \
    --name audio2head-trainer \
    -v $(pwd):/data \
    -v /mnt/z/HDTF_processed:/data/HDTF_processed:ro \
    -v /mnt/z/hallo3_processed:/data/hallo3_processed:ro \
    everymatrix.jfrog.io/emlab-docker/ayida/audio2head:trainer /bin/bash