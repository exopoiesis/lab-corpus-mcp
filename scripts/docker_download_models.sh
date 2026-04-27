#!/usr/bin/env bash
# One-time: download MinerU's parsing models (~5–10 GB) into persistent
# named volumes so subsequent containers reuse them. Idempotent — re-running
# just re-checks revisions.
set -e
export MSYS_NO_PATHCONV=1
export MSYS2_ARG_CONV_EXCL='*'

TAG="${1:-exopoiesis/lab-corpus-gpu:latest}"

docker --context gomer run --rm \
    --gpus all \
    -v "lab-corpus-hf:/root/.cache/huggingface" \
    -v "lab-corpus-ms:/root/.cache/modelscope" \
    "$TAG" download-models

echo
echo "[models] cached in named volumes:"
docker --context gomer volume inspect lab-corpus-hf lab-corpus-ms \
    --format 'table {{.Name}}\t{{.Mountpoint}}'
