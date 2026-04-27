"""lab-corpus-mcp — personal research-OS layer on top of arxiv-radar-mcp.

The MCP server, embedding pipeline, search, and reranker all come from
`arxiv_radar_mcp`. This package adds:
  * MinerU-driven literature loaders (planned, src/loaders/)
  * Upload + job-queue admin tools (planned, src/upload.py + src/jobs.py)
  * Slides / video → MD pipeline hooks (future)

Until those are implemented, `python -m lab_corpus_mcp` is a thin
wrapper that delegates to `python -m arxiv_radar_mcp`.
"""
__version__ = "0.0.1"
