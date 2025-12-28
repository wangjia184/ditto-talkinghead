#!/bin/bash

IMAGE_URL=everymatrix.jfrog.io/emlab-docker/ayida/audio2head:trainer
docker build  --progress=plain -t=$IMAGE_URL .
docker push $IMAGE_URL