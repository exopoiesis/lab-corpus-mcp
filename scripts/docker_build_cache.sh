#!/usr/bin/env bash
# Build an embedding cache inside the bundled image on gomer.
#
# Args:
#   $1  radar.toml path on gomer  (config with [sources.*], [embeddings])
#   $2  data dir on gomer         (mounted at /data/sources)
#   $3  cache dir on gomer        (output: embeddings.npy + index.json)
set -e
export MSYS_NO_PATHCONV=1
export MSYS2_ARG_CONV_EXCL='*'

CONFIG="${1:?radar.toml path required}"
DATA_DIR="${2:?data dir required}"
CACHE_DIR="${3:?cache dir required}"
TAG="${4:-exopoiesis/lab-corpus-gpu:latest}"

echo "[build-cache] config=$CONFIG  data=$DATA_DIR  cache=$CACHE_DIR"

docker --context gomer run --rm \
    --gpus all \
    -v "lab-corpus-hf:/root/.cache/huggingface" \
    -v "$CONFIG:/data/radar.toml:ro" \
    -v "$DATA_DIR:/data/sources:ro" \
    -v "$CACHE_DIR:/cache" \
    "$TAG" build-cache --config /data/radar.toml

echo
echo "[build-cache] done. cache at $CACHE_DIR"
