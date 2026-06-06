#!/usr/bin/env bash
set -euo pipefail

# Build and optionally push the worker-side images.
#
# Usage:
#   IMAGE_PREFIX=my-docker-user/elm PUSH=1 ./MA/onnode/dockerfiles/build_and_push.sh
#
# Environment:
#   IMAGE_PREFIX   image namespace/prefix, default: local/elm
#   PUSH           set to 1 to docker push after build, default: 0
#   DOCKER_USERNAME / DOCKER_PASSWORD
#                  optional credentials used only when PUSH=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ONNODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

IMAGE_PREFIX="${IMAGE_PREFIX:-local/elm}"
PUSH="${PUSH:-0}"

build_image() {
  local name="$1"
  local dockerfile="$2"
  local image="${IMAGE_PREFIX}-${name}:latest"

  echo "[build] $image from $dockerfile"
  docker build -f "$SCRIPT_DIR/$dockerfile" -t "$image" "$ONNODE_DIR"

  if [[ "$PUSH" == "1" ]]; then
    docker push "$image"
  fi
}

if [[ "$PUSH" == "1" && -n "${DOCKER_USERNAME:-}" && -n "${DOCKER_PASSWORD:-}" ]]; then
  echo "$DOCKER_PASSWORD" | docker login -u "$DOCKER_USERNAME" --password-stdin
fi

build_image "lm1" "Dockerfile.lm1"
build_image "lm2" "Dockerfile.lm2"
build_image "lm3" "Dockerfile.lm3"
build_image "lm4" "Dockerfile.lm4"
build_image "sadrl" "Dockerfile.sadrl"

echo "[done] images built with prefix: $IMAGE_PREFIX"
