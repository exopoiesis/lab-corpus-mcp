# lab-corpus-mcp + arxiv-radar-mcp + corpus-core + MinerU bundled image.
#
# Build context is the PARENT directory containing all three sibling
# repos so we can COPY them in one go:
#   <parent>/
#     ├── corpus-core/
#     ├── arxiv-radar-mcp/
#     └── lab-corpus-mcp/
#
# Build:
#   docker build -f lab-corpus-mcp/Dockerfile -t exopoiesis/lab-corpus-gpu:latest .
#
# (See scripts/docker_build.sh — it sets the right context.)
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

LABEL org.opencontainers.image.title="lab-corpus-gpu" \
      org.opencontainers.image.description="MinerU + arxiv-radar-mcp + lab admin tools" \
      org.opencontainers.image.source="https://github.com/exopoiesis/lab-corpus-mcp"

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface \
    MODELSCOPE_CACHE=/root/.cache/modelscope

# System deps:
#   libgl1 + libglib2.0-0 — opencv-python (MinerU image ops)
#   git/curl/ca-certs     — pip + HF model fetches
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# MinerU first — heaviest layer, changes least often.
# `[core]` covers vlm-transformers + pipeline + gradio; we skip the linux-only
# vllm extra (~3 GB of wheels) since vlm-transformers does the job for us.
RUN pip install --upgrade pip && \
    pip install "mineru[core]"

# Sibling 1: corpus-core (shared Encoder, EmbeddingIndex, search, MCP
# scaffold, JobRegistry). Phase 3 broke it out of arxiv-radar-mcp, so
# it now installs first as a peer.
COPY corpus-core/pyproject.toml /opt/corpus-core/
COPY corpus-core/README.md     /opt/corpus-core/
COPY corpus-core/src           /opt/corpus-core/src
RUN pip install -e /opt/corpus-core

# Sibling 2: arxiv-radar-mcp (RadarServer, arxiv shards, refresh).
# Required because lab_corpus_mcp.combined imports its server module
# for combined-mode (single shared Qwen across both backends).
COPY arxiv-radar-mcp/pyproject.toml /opt/arxiv-radar-mcp/
COPY arxiv-radar-mcp/README.md     /opt/arxiv-radar-mcp/
COPY arxiv-radar-mcp/src           /opt/arxiv-radar-mcp/src
RUN pip install -e /opt/arxiv-radar-mcp

# This repo: lab_corpus_mcp on top.
COPY lab-corpus-mcp/pyproject.toml /opt/lab-corpus-mcp/
COPY lab-corpus-mcp/README.md     /opt/lab-corpus-mcp/
COPY lab-corpus-mcp/src           /opt/lab-corpus-mcp/src
RUN pip install -e /opt/lab-corpus-mcp

# Tiny dispatcher that routes by first CMD arg (mcp / build-cache / parse).
COPY lab-corpus-mcp/scripts/docker_entrypoint.sh /usr/local/bin/lab-corpus-entrypoint
RUN chmod +x /usr/local/bin/lab-corpus-entrypoint

# Build-time audit — fail fast if pip somehow installed two torches,
# two sentence-transformers, or any of the three siblings is missing.
# See scripts/audit_image.py for the single source of truth on what
# the combined image is expected to contain.
COPY lab-corpus-mcp/scripts/audit_image.py /usr/local/bin/audit_image.py
RUN python /usr/local/bin/audit_image.py

RUN mkdir -p /data /cache /workspace
WORKDIR /workspace

# Persisted state lives in named volumes / bind-mounts:
#   /root/.cache/huggingface — sentence-transformers / transformers / Qwen
#   /root/.cache/modelscope  — MinerU layout / OCR / VLM models
#   /cache                   — embeddings + index (per-source shards once
#                              lab_corpus_mcp's sharded layout lands)
#   /data                    — corpus shards, parsed PDFs, radar.toml
VOLUME ["/root/.cache/huggingface", "/root/.cache/modelscope", "/cache", "/data"]

# Combined mode: arxiv-radar (8765) + lab-corpus (8766) on one shared
# Qwen3-4B encoder. Two MCP proxies on the host can connect to the
# different ports; one Qwen instance fits in 12 GB VRAM.
EXPOSE 8765 8766

ENTRYPOINT ["lab-corpus-entrypoint"]
# Default: combined mode (both backends, shared encoder). Override to
# `mcp` for lab-only single-server stdio, or `arxiv-radar` for
# arxiv-only HTTP backend without lab.
CMD ["combined"]
