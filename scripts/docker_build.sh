#!/usr/bin/env bash
# Build the lab-corpus GPU image on gomer.
#
# Context = parent directory containing both arxiv-radar-mcp/ and
# lab-corpus-mcp/, since the Dockerfile COPYs both. The Docker daemon
# streams the context over the gomer context link; .dockerignore keeps
# the payload tiny (only src/, pyproject, README ship per repo).
set -e
export MSYS_NO_PATHCONV=1
export MSYS2_ARG_CONV_EXCL='*'

PARENT_WIN="D:/home/ignat/project-third-matter/git"
DOCKERFILE="$PARENT_WIN/lab-corpus-mcp/Dockerfile"
TAG="${1:-exopoiesis/lab-corpus-gpu:latest}"

echo "[build] tag=$TAG  context=$PARENT_WIN  dockerfile=$DOCKERFILE"

docker --context gomer build \
    --tag "$TAG" \
    --file "$DOCKERFILE" \
    "$PARENT_WIN"

echo
docker --context gomer images "$TAG"
echo
echo "[build] next: bash scripts/docker_download_models.sh"
echo "        (one-time MinerU model fetch into a named volume)"
