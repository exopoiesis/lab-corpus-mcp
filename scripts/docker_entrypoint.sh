#!/usr/bin/env bash
# Container entrypoint — routes the first arg to the right tool.
# Unknown first arg → exec the args verbatim (handy for ad-hoc shells,
# `python ...`, `--help`, debugging).
set -e

case "${1:-mcp}" in
    mcp)
        # Phase 2A: lab_corpus_mcp ships its own MCP server built on
        # corpus_core.mcp_scaffold (corpus_stats, list_corpus, job_status,
        # job_list). Ingest / search tools land in Phase 2B.
        exec python -m lab_corpus_mcp "${@:2}"
        ;;
    build-cache)
        exec python -m arxiv_radar_mcp --build-cache "${@:2}"
        ;;
    parse)
        exec mineru "${@:2}"
        ;;
    download-models)
        # Fetch MinerU's pipeline / VLM models into the persistent volume.
        exec mineru-models-download "${@:2}"
        ;;
    *)
        exec "$@"
        ;;
esac
