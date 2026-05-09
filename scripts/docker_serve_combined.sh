#!/usr/bin/env bash
# Spawn the COMBINED arxiv-radar + lab-corpus backend on gomer.
#
# One container, two HTTP MCP backends, one Qwen3-4B encoder shared in
# VRAM (saves ~8 GB vs running both servers in separate containers).
# Two MCP proxies on the host can connect to localhost:8765 (arxiv) and
# localhost:8766 (lab), reusing one Qwen instance.
#
# Args (paths are on gomer, not local):
#   $1  arxiv-radar radar.toml path  (default: /srv/arxiv-radar/radar.toml)
#   $2  lab-corpus  radar.toml path  (default: /srv/lab-corpus/radar.toml)
#   $3  arxiv data dir               (default: /srv/arxiv-radar/data)
#   $4  arxiv cache dir              (default: /srv/arxiv-radar/cache)
#   $5  lab cache dir                (default: /srv/lab-corpus/cache)
#   $6  image tag                    (default: exopoiesis/lab-corpus-gpu:latest)
#
# Container name: lab-corpus-combined (long-lived; subsequent runs
# replace it via `docker rm -f`). Use `docker --context gomer logs -f
# lab-corpus-combined` to watch.
set -e
export MSYS_NO_PATHCONV=1
export MSYS2_ARG_CONV_EXCL='*'

ARXIV_CONFIG="${1:-/srv/arxiv-radar/radar.toml}"
LAB_CONFIG="${2:-/srv/lab-corpus/radar.toml}"
ARXIV_DATA="${3:-/srv/arxiv-radar/data}"
ARXIV_CACHE="${4:-/srv/arxiv-radar/cache}"
LAB_CACHE="${5:-/srv/lab-corpus/cache}"
TAG="${6:-exopoiesis/lab-corpus-gpu:latest}"

NAME="lab-corpus-combined"

echo "[combined] killing any prior $NAME container"
docker --context gomer rm -f "$NAME" >/dev/null 2>&1 || true

echo "[combined] starting:"
echo "  arxiv-radar:  $ARXIV_CONFIG  →  :8765"
echo "  lab-corpus:   $LAB_CONFIG    →  :8766"
echo "  shared GPU encoder + lab-corpus-hf / lab-corpus-ms volumes"

exec docker --context gomer run -d \
    --name "$NAME" \
    --restart unless-stopped \
    --gpus all \
    -p 8765:8765 -p 8766:8766 \
    -v "lab-corpus-hf:/root/.cache/huggingface" \
    -v "lab-corpus-ms:/root/.cache/modelscope" \
    -v "$ARXIV_CONFIG:/srv/arxiv-radar/radar.toml:ro" \
    -v "$LAB_CONFIG:/srv/lab-corpus/radar.toml:ro" \
    -v "$ARXIV_DATA:/srv/arxiv-radar/data:ro" \
    -v "$ARXIV_CACHE:/srv/arxiv-radar/cache" \
    -v "$LAB_CACHE:/srv/lab-corpus/cache" \
    "$TAG" combined
