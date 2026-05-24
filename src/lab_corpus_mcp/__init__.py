"""lab-corpus-mcp — personal research-OS MCP server.

Built on `corpus_core` (the shared infrastructure that also powers
`arxiv-radar-mcp`). This package adds:

  * `LabCorpusServer` — handler that owns a parsed-corpus directory +
    a `corpus_core.JobRegistry` for long-running ingest / reindex jobs.
  * `LAB_TOOL_SPECS` — MCP tool catalogue. Phase 2A skeleton:
    `corpus_stats`, `list_corpus`, `job_status`, `job_list`.
  * `serve()` / `serve_http()` — stdio and streamable-HTTP transports,
    both wired through `corpus_core.mcp_scaffold`.

Planned (Phase 2B+):

  * MinerU-driven `ingest_pdf` / `ingest_local_dir` tools.
  * `search_paper_*` once the chunk-level index is populated.
  * Slide / video → markdown loaders.

`python -m lab_corpus_mcp` runs the stdio server by default; pass
`--transport http` for the long-running backend, or `--remote` for the
local stdio→remote-HTTP proxy.
"""
__version__ = "0.0.3"
