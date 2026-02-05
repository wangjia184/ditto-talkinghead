#!/bin/bash

docker run -it --rm --runtime=nvidia --gpus all \
    --name audio2head-trainer \
    -v $(pwd):/data \
    -v /mnt/z/HDTF_processed:/data/HDTF_processed:ro \
    everymatrix.jfrog.io/emlab-docker/ayida/audio2head:trainer /bin/bash