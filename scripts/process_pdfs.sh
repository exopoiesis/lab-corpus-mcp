#!/usr/bin/env bash
# Local-CPU MinerU runner — alternative to the gomer Docker path when
# you just want to spot-check parsing on a small set of PDFs without
# touching the GPU server.
#
# Usage:
#   bash scripts/process_pdfs.sh <input_dir> <output_dir> [backend]
#
# Args:
#   input_dir   directory containing .pdf files (recursive search by MinerU)
#   output_dir  destination; MinerU creates one subdir per PDF
#   backend     'pipeline' (default — CPU OK, 85+ accuracy) or
#               'vlm-transformers' (GPU, 8 GB+ VRAM, 95+ accuracy)
#
# Output layout (per PDF):
#   <output>/<pdf_basename>/
#     <pdf_basename>.md                    full markdown with inline figure refs
#     <pdf_basename>_content_list.json     flat reading-order JSON (used by loader)
#     <pdf_basename>_middle.json           full structure with bboxes
#     <pdf_basename>_layout.pdf            visual debug overlay
#     <pdf_basename>_origin.pdf            source PDF copy
#     images/                              extracted figure / table PNGs
#
# Prereq: bash tmp/install_mineru.sh (one-time).
set -e

INPUT="${1:?input_dir required}"
OUTPUT="${2:?output_dir required}"
BACKEND="${3:-pipeline}"

MINERU="D:/home/github/tertia/MinerU/.venv/Scripts/mineru.exe"

if [ ! -f "$MINERU" ]; then
    echo "[process] MinerU not installed yet — run:"
    echo "          bash tmp/install_mineru.sh"
    exit 1
fi

if [ ! -d "$INPUT" ]; then
    echo "[process] input dir does not exist: $INPUT"
    exit 1
fi

mkdir -p "$OUTPUT"

PDF_COUNT=$(find "$INPUT" -type f -iname '*.pdf' | wc -l)
echo "[process] backend=$BACKEND  input=$INPUT  output=$OUTPUT  pdfs=$PDF_COUNT"

"$MINERU" -p "$INPUT" -o "$OUTPUT" -b "$BACKEND"

echo
echo "[process] done. Output: $OUTPUT/<pdf_basename>/"
