FROM everymatrix.jfrog.io/emlab-docker-remote-hub/ubuntu:24.04
# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-11.html

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl git build-essential \
    ca-certificates \
    pkg-config \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libavfilter-dev \
    libswscale-dev \
    libswresample-dev \
    && rm -rf /var/lib/apt/lists/*

ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py310_25.11.1-1-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh && \
    $CONDA_DIR/bin/conda clean -ya

ENV PATH=$CONDA_DIR/bin:$PATH

ADD environment.yaml /tmp/
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
 && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r \
 && conda env create -f /tmp/environment.yaml

# Upgrade pip, setuptools, wheel to fix build isolation issues
RUN /opt/conda/envs/ditto_train/bin/pip install --upgrade pip setuptools wheel

# Install NVIDIA CUDA packages separately as a dedicated layer for better caching
RUN /opt/conda/envs/ditto_train/bin/pip install \
    nvidia-cublas-cu12==12.1.3.1 \
    nvidia-cuda-cupti-cu12==12.1.105 \
    nvidia-cuda-nvrtc-cu12==12.1.105 \
    nvidia-cuda-runtime-cu12==12.1.105 \
    nvidia-cudnn-cu12==8.9.2.26 \
    nvidia-cufft-cu12==11.0.2.54 \
    nvidia-curand-cu12==10.3.2.106 \
    nvidia-cusolver-cu12==11.4.5.107 \
    nvidia-cusparse-cu12==12.1.0.106 \
    nvidia-nccl-cu12==2.19.3 \
    nvidia-nvjitlink-cu12==12.9.86 \
    nvidia-nvtx-cu12==12.1.105

# Install chumpy separately with --no-build-isolation flag (required for chumpy==0.70)
RUN /opt/conda/envs/ditto_train/bin/pip install chumpy==0.70 --no-build-isolation

# Install pip dependencies (excluding nvidia packages which will be installed separately)
RUN /opt/conda/envs/ditto_train/bin/pip install \
    absl-py==2.0.0 \
    accelerate==0.33.0 \
    aliyun-python-sdk-core==2.14.0 \
    aliyun-python-sdk-kms==2.16.2 \
    attrs==23.1.0 \
    audioread==3.0.1 \
    av==12.3.0 \
    cachetools==5.3.2 \
    coloredlogs==15.0.1 \
    configargparse==1.7 \
    contourpy==1.2.0 \
    crcmod==1.7 \
    cycler==0.12.1 \
    dearpygui==1.10.1 \
    decorator==4.4.2 \
    dill==0.3.8 \
    docstring-parser==0.16 \
    einops==0.7.0 \
    face-alignment==1.4.1 \
    facenet-pytorch==2.6.0 \
    filelock==3.13.1 \
    flatbuffers==23.5.26 \
    fonttools==4.46.0 \
    fsspec==2023.12.1 \
    google-auth==2.25.2 \
    google-auth-oauthlib==1.1.0 \
    grpcio==1.60.0 \
    hsemotion==0.3.0 \
    huggingface-hub==0.24.5 \
    humanfriendly==10.0 \
    imageio==2.33.1 \
    imageio-ffmpeg==0.4.9 \
    jinja2==3.1.6 \
    jmespath==0.10.0 \
    joblib==1.3.2 \
    kiwisolver==1.4.5 \
    kornia==0.7.0 \
    lazy-loader==0.3 \
    librosa==0.10.1 \
    llvmlite==0.41.1 \
    lpips==0.1.4 \
    markdown==3.5.1 \
    markdown-it-py==3.0.0 \
    markupsafe==2.1.3 \
    matplotlib==3.8.2 \
    mdurl==0.1.2 \
    mediapipe==0.10.8 \
    moviepy==1.0.3 \
    mpmath==1.3.0 \
    msgpack==1.0.7 \
    multiprocess==0.70.16 \
    natsort==8.4.0 \
    networkx==3.2.1 \
    ninja==1.11.1.1 \
    numba==0.58.1 \
    oauthlib==3.2.2 \
    onnx==1.16.2 \
    onnxruntime-gpu==1.16.3 \
    opencv-contrib-python==4.10.0.84 \
    opencv-python==4.10.0.84 \
    opencv-python-headless==4.8.1.78 \
    oss2==2.18.3 \
    packaging==23.2 \
    pillow==10.2.0 \
    pip==24.0 \
    platformdirs==4.1.0 \
    pooch==1.8.0 \
    praat-parselmouth==0.4.3 \
    proglog==0.1.10 \
    protobuf==3.20.3 \
    psutil==6.0.0 \
    pyasn1==0.5.1 \
    pyasn1-modules==0.3.0 \
    pyaudio==0.2.11 \
    pycryptodome==3.19.0 \
    pygments==2.17.2 \
    pykalman==0.9.7 \
    pymcubes==0.1.4 \
    pyparsing==3.1.1 \
    python-speech-features==0.6 \
    regex==2023.10.3 \
    requests-oauthlib==1.3.1 \
    resampy==0.4.2 \
    rich==13.7.0 \
    rsa==4.9 \
    safetensors==0.4.1 \
    scikit-image==0.22.0 \
    scikit-learn==1.3.2 \
    scikit-video==1.1.11 \
    scipy==1.11.4 \
    shtab==1.7.1 \
    sounddevice==0.4.6 \
    soundfile==0.12.1 \
    soxr==0.3.7 \
    sympy==1.13.2 \
    tensorboard==2.15.1 \
    tensorboard-data-server==0.7.2 \
    tensorboardx==2.6.2.2 \
    threadpoolctl==3.2.0 \
    tifffile==2023.12.9 \
    timm==0.9.0 \
    tokenizers==0.15.0 \
    torch==2.2.2 \
    torch-ema==0.3 \
    torchvision==0.17.2 \
    transformers==4.36.0 \
    trimesh==4.0.5 \
    triton==2.2.0 \
    typing-extensions==4.15.0 \
    tyro==0.8.8 \
    werkzeug==3.0.1



ENV PATH=/opt/conda/envs/ditto_train/bin:$PATH