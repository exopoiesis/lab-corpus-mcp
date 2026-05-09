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
├── scripts/                    # docker_*.sh wrappers (build, serve, model fetch)
├── src/lab_corpus_mcp/
│   ├── __main__.py             # CLI: single (stdio/http) / combined / remote-proxy
│   ├── combined.py             # supervisor: arxiv-radar + lab-corpus, shared Encoder
│   ├── config.py               # LabConfig: embeddings + parse + server
│   ├── corpus.py               # LabPaper schema + paper_id derivation + on-disk loader
│   ├── ingest.py               # MinerU subprocess wrapper + ingest_one / ingest_dir
│   └── server.py               # LabCorpusServer + LAB_TOOL_SPECS + serve/serve_http
├── tests/                      # pytest suite, 99% coverage on lab_corpus_mcp
└── tmp/                        # local-iteration helpers (gitignored)
```

## Tool surface (Phase 2B, 11 tools)

| Tool | Phase | Notes |
|------|-------|-------|
| `corpus_stats` | 2A | parsed / indexed / chunks + last ingest |
| `list_corpus` | 2A | paper rows newest-ingest-first, `limit` arg |
| `paper_info` | 2B | full LabPaper + indexed status |
| `job_status` / `job_list` | 2A | delegate to `corpus_core.JobRegistry` |
| `ingest_pdf` | 2B-1 | async; MinerU on one file (PDF/DOCX/PPTX/image) |
| `ingest_local_dir` | 2B-1 | async; bulk ingest by glob, optional recursion |
| `rebuild_index` | 2B-2 | async; delegates to `corpus_core.corpus_index.reindex` |
| `search_paper_text` | 2B-2 | substring AND-scan over chunks |
| `search_paper_semantic` | 2B-2 | cosine over chunk embeddings (Qwen3-4B-native default) |
| `similar_to_paper` | 2B-2 | nearest-neighbour by mean-of-chunks |

`paper_id` ∈ {DOI, sha256-of-file, arxiv_id-from-filename, user-supplied},
distinguished by `paper_id_kind` in the `LabPaper` metadata sidecar.
arxiv-id pattern (`\d{4}\.\d{4,5}`) wins on filename; otherwise sha256
prefix of the file bytes. Explicit `paper_id` arg to `ingest_pdf`
overrides both.

## Combined mode — both backends on one Qwen

Running arxiv-radar-mcp + lab-corpus-mcp as separate containers on a
12 GB GPU doesn't fit (Qwen3-Embedding-4B ≈ 8 GB in bf16 each → 16 GB
total). The combined supervisor in `lab_corpus_mcp.combined` boots
both servers in one process, hands them the same `Encoder` instance,
and serializes encode calls with a `threading.Lock` so peak VRAM
stays at ~10 GB (weights + one batch's activations).

```bash
# On gomer, with both radar.toml configs in place:
bash scripts/docker_serve_combined.sh \
    /srv/arxiv-radar/radar.toml \
    /srv/lab-corpus/radar.toml \
    /srv/arxiv-radar/data \
    /srv/arxiv-radar/cache \
    /srv/lab-corpus/cache

# → arxiv-radar HTTP backend on  :8765
# → lab-corpus  HTTP backend on  :8766
# Two MCP proxies on the host can connect independently.
```

The supervisor refuses to start if the two configs disagree on
`[embeddings].model` or `target_dim` — they share one in-memory
copy. Disable the encode-call lock with `--no-encoder-lock` if you
have VRAM headroom and want concurrent encode throughput.

## Status

- **Phase 1.5 done (2026-05-09)** in arxiv-radar-mcp — `corpus_core.mcp_scaffold`
  extracted (commit `4eb5670`).
- **Phase 2A done (2026-05-09):** own MCP server on `corpus_core.mcp_scaffold`.
- **Phase 2B done (2026-05-09):** MinerU ingest + reindex + chunk search
  delegate to `corpus_core.corpus_index`. Test suite 99% coverage. Real
  MinerU runs only inside the Docker image on gomer; tests stub the
  subprocess via the `MineruRunner` injection seam.
- **Phase 2C done (2026-05-09):** combined-mode supervisor — one
  container, two HTTP backends, one Qwen instance. Default `CMD` of
  the lab-corpus-gpu image is now `combined`; lab-only single-server
  remains available as `mcp` arg.
- **Phase 2B+ (deferred):** PDF-content DOI extraction (currently filename
  arxiv-id or sha256 prefix), slide / video loaders.

## License

MIT (same as arxiv-radar-mcp).
