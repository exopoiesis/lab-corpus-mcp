# lab-corpus-mcp

Personal research-OS layer on top of [arxiv-radar-mcp](https://github.com/exopoiesis/arxiv-radar-mcp).

Where `arxiv-radar-mcp` is a narrow, public-data MCP feed for the
`daily-arxiv-*` fork family, **lab-corpus-mcp** is the private corpus
side: literature PDFs, presentation slides / videos, anything ingested
through MinerU, plus admin tooling (upload archives, kick off
re-indexing, query job status) — wired as MCP tools on top of the same
embedding stack.

## What it bundles

- **MinerU** for PDF / DOCX / PPTX / image parsing — same VLM-backed
  pipeline that produces `<file>_content_list.json` + extracted figures.
- **arxiv-radar-mcp** as a pip dependency — provides the embedding
  Encoder, search primitives, cross-encoder reranker, and the base MCP
  server. (Same Qwen3-Embedding-4B native default we empirically
  validated; see `arxiv-radar-mcp/docs/MODEL_BENCHMARKS.md`.)
- Lab-specific tools (planned, see `docs/DEPLOY.md`):
  `upload_corpus`, `rebuild_index`, `corpus_stats`, `job_status`.

## Architecture

```
              ┌──────────────────────────────────────────┐
              │   lab-corpus-mcp Docker image (gomer)    │
              │ ┌──────────────────────────────────────┐ │
              │ │ MinerU 2.x (PDF → md + figures)      │ │
              │ ├──────────────────────────────────────┤ │
              │ │ arxiv-radar-mcp (pip dep)            │ │
              │ │   Encoder · search · Reranker · MCP  │ │
              │ ├──────────────────────────────────────┤ │
              │ │ lab_corpus_mcp                       │ │
              │ │   loaders · upload · jobs            │ │
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
├── src/lab_corpus_mcp/         # Python package (delegates to arxiv_radar_mcp for MCP for now)
├── tests/
└── tmp/                        # local-iteration helpers (install_mineru.sh, …)
```

## Status

Skeleton + Docker bundle migrated from `arxiv-radar-mcp`.
Lab-specific tools (upload, jobs, loaders) — pending. See task list.

## License

MIT (same as arxiv-radar-mcp).
