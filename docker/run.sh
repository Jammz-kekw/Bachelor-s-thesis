#!/usr/bin/env bash

# Fail on error and unset variables.
set -eu -o pipefail

CWD=$(readlink -e "$(dirname "$0")")
cd "${CWD}/.." || exit $?
source ./docker/common.sh

DEVICE=0
echo "Using GPU devices: ${DEVICE}"


docker run \
    -d \
    --name "editable-stain-xaicyclegan2" \
    --gpus all \
    --privileged \
    --shm-size 8g \
    -v "/setup/.netrc":/root/.netrc \
    -v "/workspace/editable-stain-xaicyclegan2/..":/workspace/editable-stain-xaicyclegan2 \
    -v "/mnt/scratch/root/editable-stain-xaicyclegan2":/workspace/editable-stain-xaicyclegan2/.mnt/scratch \
    -v "/mnt/persist/root/editable-stain-xaicyclegan2":/workspace/editable-stain-xaicyclegan2/.mnt/persist \
    -e CUDA_VISIBLE_DEVICES=1 \
    editable-stain-xaicyclegan2:latest \
    "$@" || exit $?
