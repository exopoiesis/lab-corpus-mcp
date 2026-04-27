#!/usr/bin/env bash
# Spawn the lab-corpus MCP server on gomer, bridging stdio to the local
# shell. Wire into a Claude Desktop / MCP-client config:
#
#   {
#     "mcpServers": {
#       "lab-corpus": {
#         "command": "bash",
#         "args": [
#           "<absolute path>/scripts/docker_serve_mcp.sh",
#           "/srv/lab-corpus/radar.toml",
#           "/srv/lab-corpus/data",
#           "/srv/lab-corpus/cache"
#         ]
#       }
#     }
#   }
#
# Args (paths are on gomer, not local):
#   $1  radar.toml path
#   $2  data dir
#   $3  cache dir
set -e
export MSYS_NO_PATHCONV=1
export MSYS2_ARG_CONV_EXCL='*'

CONFIG="${1:?radar.toml path on gomer required}"
DATA_DIR="${2:?data dir on gomer required}"
CACHE_DIR="${3:?cache dir on gomer required}"
TAG="${4:-exopoiesis/lab-corpus-gpu:latest}"

# `exec` so signals from the client (Ctrl+C, SIGTERM) propagate cleanly.
# `-i` keeps stdin open for MCP transport. NO `-t` — we want a clean
# binary stream to the MCP client.
exec docker --context gomer run --rm -i \
    --gpus all \
    -v "lab-corpus-hf:/root/.cache/huggingface" \
    -v "$CONFIG:/data/radar.toml:ro" \
    -v "$DATA_DIR:/data/sources:ro" \
    -v "$CACHE_DIR:/cache:ro" \
    "$TAG" mcp --config /data/radar.toml
