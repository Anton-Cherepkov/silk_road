#!/usr/bin/env bash

set -e

# Names to identify images and containers of this app
TAG_NAME=silkroad:v01-${USER}
CONTAINER_NAME=silkroad_${USER}
PROJECT_NAME=/app

# Output colors
NORMAL="\\033[0;39m"
RED="\\033[1;31m"
BLUE="\\033[1;34m"


log() {
  echo -e "$BLUE > $1 $NORMAL"
}

error() {
  echo ""
  echo -e "$RED >>> ERROR - $1$NORMAL"
}

build() {
    log "Building Docker image"
    docker build \
      --network=host \
      -t ${TAG_NAME} .
}

# functions for run / stop web interface START
WEB_UI_PORT=8011
CONFIG="artifacts/ocrnet_hrnet.py"
CHECKPOINT="artifacts/iter_3800.pth"
FP16_MODE=False

run_web_ui() {
    log "Running WEB interface..."
    log "It will be available at http://localhost:${WEB_UI_PORT}"
	  docker run \
		  -e DISPLAY=unix${DISPLAY} \
      -v /tmp/.X11-unix:/tmp/.X11-unix \
      -v /etc/xdg/dvc:/etc/xdg/dvc:ro \
		  --ipc=host \
      --gpus all \
      -itd \
      --name=${CONTAINER_NAME}_web_ui \
      -p ${WEB_UI_PORT}:8011 \
      -v ${PWD}:${PROJECT_NAME} \
      --entrypoint '/bin/sh' \
      ${TAG_NAME} \
      -c "python web_ui.py ${CONFIG} ${CHECKPOINT} --fp16 ${FP16_MODE}"
}

stop_web_ui() {
    log "Stopping and removing the container ${CONTAINER_NAME}_web_ui"
    docker stop ${CONTAINER_NAME}_web_ui; docker rm ${CONTAINER_NAME}_web_ui
}
# functions for run / stop web interface END

# helper functions, used primarly for running traning START
run() {
    log "Running the container"
	  docker run \
		  -e DISPLAY=unix${DISPLAY} \
      -v /tmp/.X11-unix:/tmp/.X11-unix \
      -v /etc/xdg/dvc:/etc/xdg/dvc:ro \
		  --ipc=host \
      --gpus all \
      -itd \
      --name=${CONTAINER_NAME} \
      -p ${WEB_UI_PORT}:8011 \
      -v ${PWD}:${PROJECT_NAME} \
      ${TAG_NAME}
}

lfs_pull() {
    docker exec ${CONTAINER_NAME} git lfs pull
}

run_inference() {
    docker exec ${CONTAINER_NAME} bash -c "python3 tools/infer_big_images.py \"$1\" \"$2\" $CONFIG $CHECKPOINT && python3 tools/postprocess_masks.py --masks_folder \"$2\" --imgs_folder \"$1\" --output_folder \"$3\""
}

exec() {
    log "Entering the container"
    docker exec -it ${CONTAINER_NAME} bash
}

stop() {
    log "Stopping and removing the container ${CONTAINER_NAME}"
    docker stop ${CONTAINER_NAME}; docker rm ${CONTAINER_NAME}
}
# helper functions, used primarly for running traning END


help() {
  echo "-----------------------------------------------------------------------"
  echo "                      Available commands                              -"
  echo "-----------------------------------------------------------------------"
  echo -e -n "$BLUE"
  echo "   > build - To build the Docker image"
  echo "   > run - To start the Docker container"
  echo "   > exec - Log you into container"
  echo "   > stop - Remove the container"
  echo "   > help - Display this help"
  echo -e -n "$NORMAL"
  echo "-----------------------------------------------------------------------"

}

$*
