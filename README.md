# lab-corpus-mcp

Personal research-OS layer that shares the corpus-core stack with
[arxiv-radar-mcp](https://github.com/exopoiesis/arxiv-radar-mcp).

Where `arxiv-radar-mcp` is a narrow, public-data MCP feed for the
`daily-arxiv-*` fork family, **lab-corpus-mcp** is the private corpus
side: literature PDFs, presentation slides / videos, anything ingested
through MinerU, plus admin tooling (upload archives, kick off
re-indexing, query job status) — wired as MCP tools on top of the same
embedding stack.

## What it bundles

- **MinerU** for PDF / DOCX / PPTX / image parsing — same VLM-backed
  pipeline that produces `<file>_content_list.json` + extracted figures.
- **`corpus_core`** (shipped inside `arxiv-radar-mcp` until Phase 3 of
  the extraction plan) — provides the embedding `Encoder`, search
  primitives, cross-encoder `Reranker`, `JobRegistry`, and the generic
  MCP server scaffold (`make_method_dispatcher`, `build_mcp_app`,
  `serve_stdio`, `serve_streamable_http`). Same Qwen3-Embedding-4B
  native default empirically validated in
  `arxiv-radar-mcp/docs/MODEL_BENCHMARKS.md`.
- **`lab_corpus_mcp.server`** — `LabCorpusServer` handler + `LAB_TOOL_SPECS`
  catalogue, both wired through `corpus_core.mcp_scaffold`. The Phase 2A
  surface is `corpus_stats`, `list_corpus`, `job_status`, `job_list` —
  enough to verify dispatcher + transport + JobRegistry. Ingest / search
  tools land in Phase 2B.

## Architecture

```
              ┌──────────────────────────────────────────┐
              │   lab-corpus-mcp Docker image (gomer)    │
              │ ┌──────────────────────────────────────┐ │
              │ │ MinerU 2.x (PDF → md + figures)      │ │
              │ ├──────────────────────────────────────┤ │
              │ │ corpus_core  (shared with radar)     │ │
              │ │   Encoder · search · Reranker        │ │
              │ │   JobRegistry · mcp_scaffold         │ │
              │ ├──────────────────────────────────────┤ │
              │ │ lab_corpus_mcp                       │ │
              │ │   LabCorpusServer · LAB_TOOL_SPECS   │ │
              │ │   loaders · upload · jobs (planned)  │ │
              │ └──────────────────────────────────────┘ │
              └──────────────────────────────────────────┘
                       │                          ▲
                       │ stdio MCP                │ docker exec
                       ▼                          │
              ┌─────────────────────┐   ┌──────────────────┐
              │   Claude Desktop    │   │ scripts/docker_* │
              └─────────────────────┘   └──────────────────┘
```

The Docker image bundles MinerU + arxiv-radar-mcp + lab_corpus_mcp on a
single CUDA + torch base. One image, three workflows: PDF parsing,
embedding cache build, MCP server. See `docs/DEPLOY.md`.

## Layout

```
lab-corpus-mcp/
├── Dockerfile                  # GPU image — parent dir = build context
├── docs/DEPLOY.md              # operations + Claude Desktop wiring
├── radar.example.toml          # config template
├── scripts/                    # docker_*.sh wrappers, process_pdfs.sh
├── src/lab_corpus_mcp/
│   ├── __main__.py             # CLI: stdio (default) / --transport http / --remote
│   ├── config.py               # LabConfig: embeddings + parse + server
│   └── server.py               # LabCorpusServer + LAB_TOOL_SPECS + serve/serve_http
├── tests/
└── tmp/                        # local-iteration helpers (install_mineru.sh, …)
```

## Status

- **Phase 2A done (2026-05-09):** own MCP server on `corpus_core.mcp_scaffold`
  (no more pass-through to `arxiv_radar_mcp.__main__`). Skeleton tool
  surface: `corpus_stats`, `list_corpus`, `job_status`, `job_list`.
- **Phase 2B (next):** MinerU-driven `ingest_pdf` / `ingest_local_dir`,
  then `corpus_core.corpus_index.reindex` over the parsed tree.
- **Phase 2B+:** `search_paper_*` and slide / video loaders.

## License

MIT (same as arxiv-radar-mcp).
