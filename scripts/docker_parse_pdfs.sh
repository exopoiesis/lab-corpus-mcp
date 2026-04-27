#!/usr/bin/env bash
# Run MinerU PDF parsing inside the bundled image on gomer.
#
# Args:
#   $1  input dir on gomer  (PDFs to parse, mounted read-only)
#   $2  output dir on gomer (one subdir per PDF will be created here)
#   $3  backend             default 'vlm-transformers' (95+ accuracy, GPU)
#                           use 'pipeline' for CPU-only fallback (85+)
#
# Example:
#   bash scripts/docker_parse_pdfs.sh /mnt/literature/pdfs /mnt/literature/parsed
set -e
export MSYS_NO_PATHCONV=1
export MSYS2_ARG_CONV_EXCL='*'

INPUT="${1:?input dir required (path on gomer)}"
OUTPUT="${2:?output dir required (path on gomer)}"
BACKEND="${3:-vlm-transformers}"
TAG="${4:-exopoiesis/lab-corpus-gpu:latest}"

echo "[parse] backend=$BACKEND  in=$INPUT  out=$OUTPUT  tag=$TAG"

docker --context gomer run --rm \
    --gpus all \
    -v "lab-corpus-hf:/root/.cache/huggingface" \
    -v "lab-corpus-ms:/root/.cache/modelscope" \
    -v "$INPUT:/in:ro" \
    -v "$OUTPUT:/out" \
    "$TAG" parse -p /in -o /out -b "$BACKEND"

echo
echo "[parse] done. Each PDF's output is at $OUTPUT/<pdf_basename>/"
