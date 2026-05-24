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
- **[`corpus-core`](https://github.com/exopoiesis/corpus-core)** —
  shared infrastructure extracted in Phase 3: `Encoder`,
  `EmbeddingIndex`, `JobRegistry`, search primitives, the chunker,
  `Reranker`, and the generic MCP server scaffold
  (`make_method_dispatcher`, `build_mcp_app`, `serve_stdio`,
  `serve_streamable_http`). Same Qwen3-Embedding-4B native default
  empirically validated in `arxiv-radar-mcp/docs/MODEL_BENCHMARKS.md`.
- **[`arxiv-radar-mcp`](https://github.com/exopoiesis/arxiv-radar-mcp)**
  — the public-data arxiv backend. Required at runtime because
  `lab_corpus_mcp.combined` builds the supervisor that runs both
  servers in one container with one shared Encoder.
- **`lab_corpus_mcp.server`** — `LabCorpusServer` handler + 13-tool
  `LAB_TOOL_SPECS` catalogue, wired through `corpus_core.mcp_scaffold`.

## Architecture

```
              ┌──────────────────────────────────────────┐
              │   lab-corpus-mcp Docker image (gomer)    │
              │ ┌──────────────────────────────────────┐ │
              │ │ MinerU 2.x — library mode (s153)     │ │
              │ │   do_parse() → md + figures          │ │
              │ │   AtomModelSingleton in-process      │ │
              │ ├──────────────────────────────────────┤ │
              │ │ corpus_core  (shared with radar)     │ │
              │ │   Encoder.unload · search · Reranker │ │
              │ │   JobRegistry · mcp_scaffold         │ │
              │ ├──────────────────────────────────────┤ │
              │ │ lab_corpus_mcp                       │ │
              │ │   LabCorpusServer · LAB_TOOL_SPECS   │ │
              │ │   _release_gpu_vram (encoder+MinerU) │ │
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
│   ├── ingest.py               # MinerU wrapper: ingest_one / ingest_dir / fetch_and_ingest (U14)
│   └── server.py               # LabCorpusServer + LAB_TOOL_SPECS + serve/serve_http
├── tests/                      # pytest suite, 99% coverage on lab_corpus_mcp
└── tmp/                        # local-iteration helpers (gitignored)
```

## Tool surface (Phase 2B + U14, 13 tools)

| Tool | Phase | Notes |
|------|-------|-------|
| `corpus_stats` | 2A | parsed / indexed / chunks + last ingest |
| `list_corpus` | 2A | paper rows newest-ingest-first, `limit` arg |
| `paper_info` | 2B | full LabPaper + indexed status |
| `job_status` / `job_list` | 2A | delegate to `corpus_core.JobRegistry` |
| `ingest_pdf` | 2B-1 | async; MinerU on one file (PDF/DOCX/PPTX/image) |
| `ingest_local_dir` | 2B-1 | async; bulk ingest by glob, optional recursion |
| `ingest_url` | 2B+ (U14) | async; download via `corpus_core.fetch_url` → MinerU. Any http(s) URL. |
| `ingest_arxiv_pdf` | 2B+ (U14) | async; convenience for arxiv preprints — forces `paper_id = arxiv_id` |
| `rebuild_index` | 2B-2 | async; delegates to `corpus_core.corpus_index.reindex` |
| `search_paper_text` | 2B-2 | substring AND-scan over chunks |
| `search_paper_semantic` | 2B-2 | cosine over chunk embeddings (Qwen3-4B-native default) |
| `similar_to_paper` | 2B-2 | nearest-neighbour by mean-of-chunks |

`paper_id` ∈ {DOI, sha256-of-file, arxiv_id-from-filename, user-supplied},
distinguished by `paper_id_kind` in the `LabPaper` metadata sidecar.
arxiv-id pattern (`\d{4}\.\d{4,5}`) wins on filename; otherwise sha256
prefix of the file bytes. Explicit `paper_id` arg to `ingest_pdf` /
`ingest_url` overrides both.

### Fetch-by-URL (U14, 2026-05-13)

`ingest_pdf` / `ingest_local_dir` need a path on the server's
filesystem — for fresh remote PDFs that meant `curl` + `docker cp` +
`ingest_local_dir` (the s142 dogfood pain). The U14 tools collapse
that to one MCP call:

```jsonc
// Arxiv preprint — paper_id forced to the arxiv id.
{"tool": "ingest_arxiv_pdf", "args": {"arxiv_id": "2512.14129"}}
// → {"job_id": "ef34ab…", "kind": "ingest_arxiv_pdf",
//    "arxiv_id": "2512.14129", "backend": "pipeline"}

// Generic URL — paper_id auto-derived from filename, or override.
{"tool": "ingest_url",
 "args": {"url": "https://example.org/preprints/ai4chem.pdf",
          "paper_id": "ai4chem-2026"}}
```

Downloads land under `<parse.dir>/inbox/<filename>` via
`corpus_core.http_fetch.fetch_url` (atomic write, 429/503 retry with
`Retry-After`). arxiv.org URLs go through the singleton arxiv throttle
so the combined image shares one 1 req / 3 sec budget between
arxiv-radar's HTML/LaTeX fetcher and lab-corpus's PDF downloader —
no double-spam. Closes arxiv-radar-mcp's U14.

## Combined mode — both backends on one Qwen

Running arxiv-radar-mcp + lab-corpus-mcp as separate containers on a
12 GB GPU doesn't fit (Qwen3-Embedding-4B ≈ 8 GB in bf16 each → 16 GB
total). The combined supervisor in `lab_corpus_mcp.combined` boots
both servers in one process, hands them the same `Encoder` instance,
and serializes encode calls with a `threading.Lock` so peak VRAM
stays at ~10 GB (weights + one batch's activations).

```bash
# On gomer, with both radar.toml configs in place at the canonical
# host paths:
bash scripts/docker_serve_combined.sh \
    /srv/arxiv-radar/radar.toml \
    /srv/lab-corpus/radar.toml \
    /srv/arxiv-radar/cache/sources \
    /srv/arxiv-radar/cache \
    /srv/lab-corpus/cache

# → arxiv-radar HTTP backend on  :8765
# → lab-corpus  HTTP backend on  :8766
# (or any host:port mapping you choose — the production deploy uses
# 127.0.0.1:18765 / :18766 to coexist with the legacy
# arxiv-radar-backend container on :8765 during migration.)
```

The supervisor refuses to start if the two configs disagree on
`[embeddings].model` or `target_dim` — they share one in-memory
copy. Disable the encode-call lock with `--no-encoder-lock` if you
have VRAM headroom and want concurrent encode throughput.

## GPU memory release (v0.0.3, 2026-05-24)

After every ingest job (`ingest_pdf` / `ingest_local_dir` /
`ingest_url` / `ingest_arxiv_pdf`) and every `rebuild_index`, the
server calls `LabCorpusServer._release_gpu_vram()` which:

1. `Encoder.unload()` from corpus-core ≥ 0.2.0 — drops the shared
   Qwen3-Embedding-4B (~7-8 GB bf16) + `torch.cuda.empty_cache()`.
2. `unload_mineru_models()` — clears MinerU's `AtomModelSingleton` and
   `HybridModelSingleton` model caches (~2-3 GB on pipeline backend)
   + same CUDA cache flush.

Net effect: between batches the lab-corpus container holds **near-zero
GPU VRAM**, so the same RTX 4070 can run unrelated DFT / MLIP /
training jobs on the host without OOM. First call after release pays
the cold-load cost again (~20 sec encoder, ~30 sec MinerU pipeline),
so this design optimises for **bursty ingest + occasional search**, not
sustained low-latency queries.

Concurrent jobs are safe: both unload functions are idempotent, and
`Encoder.unload()` takes the same internal lock as `_ensure_loaded`,
so a parallel encode either runs first or triggers a clean re-load on
its next call.

## MinerU library mode (s153, 2026-05-24)

Earlier releases shelled out to `mineru` CLI for every ingest, which
spawned a transient `LocalAPIServer` subprocess that loaded layout +
OCR + table models (~30 sec cold-load), parsed one PDF, then died.
Two PDFs in a row paid the cold-load twice; ten PDFs paid it ten
times. The granchild process could also zombie under SIGKILL and pin
VRAM.

`lab_corpus_mcp.ingest._default_mineru_runner` now calls
`mineru.cli.common.do_parse(...)` directly. MinerU's singleton model
cache lives in the lab-corpus-mcp Python process across calls; a bulk
`ingest_local_dir` of N PDFs loads models once for the whole batch.
After the batch completes, `_release_gpu_vram` clears the singletons
and frees VRAM (see above).

The `MineruRunner` injection seam is preserved for tests — fakes that
just write a stub markdown still plug in via the
`runner=fake_mineru_runner` kwarg, so the test suite has no real
MinerU dependency.

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
- **Phase 3 done (2026-05-09):** `corpus_core` extracted to its own
  repo at `git/corpus-core/`. arxiv-radar-mcp + lab-corpus-mcp
  declare it as a regular dependency. Combined image installs all
  three siblings editable in dep order; build-time
  `scripts/audit_image.py` enforces the no-duplicate-distribution
  invariant.
- **Production deploy done (2026-05-09 → 10):** combined image built
  on gomer (`exopoiesis/lab-corpus-gpu`, ~12 GB), supervisor running
  on `127.0.0.1:18765` (arxiv-radar) + `127.0.0.1:18766` (lab-corpus).
  PyTorch 2.7.1+cu126 base satisfies MinerU 3.x's `torch>=2.6,<3` —
  no parallel torch reinstall in the image. Single Qwen3-4B in VRAM
  (~10 GB peak) shared between both backends via `_LockedEncoder`.
  Migrated 34,627 abstract embeddings + 466 fulltext chunks (51
  papers) from the legacy `arxiv-radar-cache` volume into
  `/srv/arxiv-radar/cache/`; nightly refresh runs incremental
  (`full_rebuild=false`, `interval_hours=24`) so existing embeddings
  are preserved.
- **MinerU backend default = `pipeline`** (not `vlm-transformers`).
  The 1.2B Qwen2-VL backend wedges on a 12 GB GPU when sharing
  VRAM with our embedding Qwen; `pipeline` (layout-CNN + OCR)
  finishes a 2 MB PDF in ~90 sec. Override per-call via
  `backend="vlm-transformers"` if you have 24 GB+ headroom.
  End-to-end smoke verified on arxiv:2512.14129 (Yin et al.,
  (Cr,Fe)S pyrrhotite) — 16 chunks indexed, `search_paper_text` and
  `search_paper_semantic` return correct hits.
- **U14 fetch-by-URL done (2026-05-13):** new MCP tools `ingest_url`
  and `ingest_arxiv_pdf` download remote PDFs server-side via
  `corpus_core.http_fetch.fetch_url` and feed them straight into
  the MinerU pipeline. Combined image shares one process-wide
  `Throttle` instance for arxiv.org so both backends respect the
  ToS 1 req / 3 sec budget without coordinating.
- **s153 MinerU library mode + GPU unload (2026-05-24, v0.0.3):**
  Ingest no longer shells out to the `mineru` CLI; calls
  `mineru.cli.common.do_parse` directly so model loads are amortised
  across a batch. After every ingest / reindex job,
  `_release_gpu_vram()` drops both the Qwen3-Embedding-4B (via
  `Encoder.unload()` from corpus-core 0.2.0) and MinerU's singleton
  caches (via `unload_mineru_models()`), so the shared RTX 4070 on
  gomer is free for unrelated compute between batches. 156 lab-corpus
  tests + 119 corpus-core + 230 arxiv-radar = 505 green.
- **Phase 2B+ (deferred):** PDF-content DOI extraction (currently filename
  arxiv-id or sha256 prefix), slide / video loaders.

## License

MIT (same as arxiv-radar-mcp).
