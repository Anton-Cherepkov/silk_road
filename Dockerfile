ARG PYTORCH="1.8.1"
ARG CUDA="11.1"
ARG CUDNN="8"
ARG MMCV="1.3.13"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
 && apt-get clean    \
 && rm -rf /var/lib/apt/lists/*
RUN conda clean --all
RUN apt-get update
RUN apt-get install wget ffmpeg libsm6 libxext6  -y
COPY requirements.txt /requirements.txt
RUN ["/bin/bash", "-c", "pip install -r /requirements.txt -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html"]

COPY . /silk_road_app
WORKDIR /silk_road_app

CMD ./run_web_ui.sh
EXPOSE 8011