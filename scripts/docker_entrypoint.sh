#!/usr/bin/env bash
# Container entrypoint — routes the first arg to the right tool.
# Unknown first arg → exec the args verbatim (handy for ad-hoc shells,
# `python ...`, `--help`, debugging).
set -e

case "${1:-combined}" in
    combined)
        # Default: arxiv-radar + lab-corpus in one process, sharing one
        # Qwen3-4B Encoder (~10 GB peak VRAM with the threading.Lock on).
        # Two HTTP backends on different ports — wire two MCP proxies
        # to the same host on different ports.
        # Configs default to /srv/{arxiv-radar,lab-corpus}/radar.toml;
        # override with --arxiv-config / --lab-config / port flags.
        : "${ARXIV_CONFIG:=/srv/arxiv-radar/radar.toml}"
        : "${LAB_CONFIG:=/srv/lab-corpus/radar.toml}"
        : "${COMBINED_BIND:=0.0.0.0}"
        : "${ARXIV_PORT:=8765}"
        : "${LAB_PORT:=8766}"
        exec python -m lab_corpus_mcp \
            --mode combined \
            --arxiv-config "$ARXIV_CONFIG" \
            --lab-config   "$LAB_CONFIG" \
            --bind         "$COMBINED_BIND" \
            --arxiv-port   "$ARXIV_PORT" \
            --lab-port     "$LAB_PORT" \
            "${@:2}"
        ;;
    mcp)
        # Lab-only single-server. Stdio by default; pass `--transport http`
        # for an HTTP backend without arxiv-radar in the same process.
        exec python -m lab_corpus_mcp "${@:2}"
        ;;
    arxiv-radar)
        # Arxiv-radar-only single-server. Mirror of `mcp` but for the
        # arxiv-radar shell — useful when you DO want a single backend
        # but the arxiv side, not the lab side.
        exec python -m arxiv_radar_mcp "${@:2}"
        ;;
    build-cache)
        exec python -m arxiv_radar_mcp --build-cache "${@:2}"
        ;;
    download-models)
        # One-time fetch of MinerU's pipeline / VLM models into the
        # `lab-corpus-ms` persistent volume so subsequent ingests reuse
        # them. Invoked by scripts/docker_download_models.sh.
        exec mineru-models-download "${@:2}"
        ;;
    *)
        exec "$@"
        ;;
esac
