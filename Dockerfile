ARG PYTORCH="1.8.1"
ARG CUDA="11.1"
ARG CUDNN="8"
ARG MMCV="1.3.13"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN conda clean --all

# Install MMCV
ARG PYTORCH
ARG CUDA
ARG MMCV

RUN apt-get update
RUN apt-get install wget ffmpeg libsm6 libxext6  -y

RUN ["/bin/bash", "-c", "pip install mmcv-full==${MMCV} -f https://download.openmmlab.com/mmcv/dist/cu${CUDA//./}/torch${PYTORCH}/index.html"]

# Install MMSegmentation
RUN git clone https://github.com/open-mmlab/mmsegmentation.git /mmsegmentation
WORKDIR /mmsegmentation
ENV FORCE_CUDA="1"
RUN pip install -r requirements.txt
RUN pip install --no-cache-dir -e .

RUN wget https://github.com/git-lfs/git-lfs/releases/download/v3.0.1/git-lfs-linux-amd64-v3.0.1.tar.gz && \
    mkdir git-lfs-linux-amd64-v3.0.1 && \
    tar xf git-lfs-linux-amd64-v3.0.1.tar.gz --directory git-lfs-linux-amd64-v3.0.1 && \
    ./git-lfs-linux-amd64-v3.0.1/install.sh && \
    git lfs install && \
    rm -rf git-lfs-linux-amd64-v3.0.1*

WORKDIR /app