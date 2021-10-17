#!/usr/bin/env bash

set -e

# Names to identify images and containers of this app
TAG_NAME=silkroad:v01-${USER}
CONTAINER_NAME=silkroad_${USER}
PROJECT_NAME=/app
SSH_PORT=27035

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

run() {
    log "Running the container"
	  docker run \
		  -e DISPLAY=unix${DISPLAY} \
		  -e AWS_ACCESS_KEY_ID=${DVC_AWS_ACCESS_KEY_ID} \
      -e AWS_SECRET_ACCESS_KEY=${DVC_AWS_SECRET_ACCESS_KEY}\
      -e AWS_CA_BUNDLE=${DVC_AWS_CA_BUNDLE} \
      -e USERNAME=${USER} \
      -v /tmp/.X11-unix:/tmp/.X11-unix \
      -v /etc/xdg/dvc:/etc/xdg/dvc:ro \
      -v ${DVC_CACHE_DIR}:${DVC_CACHE_DIR} \
		  --ipc=host \
      --gpus all \
      -itd \
      --name=${CONTAINER_NAME} \
      -p ${SSH_PORT}:22 \
      -v ${PWD}:${PROJECT_NAME} \
      ${TAG_NAME}
}

exec() {
    log "Entering the container"
    docker exec -it ${CONTAINER_NAME} bash
}

stop() {
    log "Stopping and removing the container ${CONTAINER_NAME}"
    docker stop ${CONTAINER_NAME}; docker rm ${CONTAINER_NAME}
}


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
