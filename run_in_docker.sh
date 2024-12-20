#!/bin/bash
set -e

IMAGE_NAME="my_robot_image"
CONTAINER_NAME="my_robot_container"
LOCAL_DIR="$PWD"
MOUNT_POINT="/home/FlexivPy"
SCRIPT_PATH="/home/FlexivPy/execute_cmd.sh"

# Build the image
echo "Building Docker image '${IMAGE_NAME}'..."
docker build -t "${IMAGE_NAME}" .

# Remove any existing container with the same name
if [ "$(docker ps -aq -f name=${CONTAINER_NAME})" ]; then
    echo "Removing existing container '${CONTAINER_NAME}'..."
    docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
fi

echo "Running container '${CONTAINER_NAME}' and executing '${SCRIPT_PATH}'..."
docker run --rm -it \
    --net=host \
    --entrypoint "" \
    --privileged \
    --name "${CONTAINER_NAME}" \
    -v "${LOCAL_DIR}:${MOUNT_POINT}" \
    "${IMAGE_NAME}" \
    /bin/bash -i -c "${SCRIPT_PATH}"

# The container will run the script and then exit. If you need it to stay alive,
# you could append a command to keep it running (e.g., 'bash -c "${SCRIPT_PATH}; tail -f /dev/null"').
