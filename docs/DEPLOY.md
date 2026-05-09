# Deploying lab-corpus-mcp

Production deployment is a single GPU container on **gomer**, bundling
three workflows on the same CUDA + torch stack:

| Workflow | When to run | Time |
|---|---|---|
| MinerU PDF parsing | new literature batch arrives | minutes per PDF (vlm-transformers) |
| Embedding cache build | new corpus shards added or model swap | ~5 min per 14k papers (Qwen3-4B native) |
| MCP server (stdio) | always-on for Claude Desktop | runs as long as a query is open |

The image bundles **MinerU + arxiv-radar-mcp + lab_corpus_mcp** in one
go. Build context is the parent directory of both repos; see
`scripts/docker_build.sh`.

## One-time setup

```bash
# All three sibling repos must live under D:/home/ignat/project-third-matter/git/
#   ├── corpus-core/         (shared Encoder, JobRegistry, MCP scaffold)
#   ├── arxiv-radar-mcp/     (RadarServer + arxiv shards)
#   └── lab-corpus-mcp/      (this repo, MinerU + combined supervisor)

# Build the image on gomer (parent dir streams as Docker context)
bash scripts/docker_build.sh

# Pre-download MinerU's parsing models into a persistent named volume
bash scripts/docker_download_models.sh
```

After this, two named docker volumes exist on gomer:
* `lab-corpus-hf` — sentence-transformers / transformers / Qwen
* `lab-corpus-ms` — MinerU layout / OCR / VLM models

These persist across container restarts so we never re-download.

## Workflows

### Ingest PDFs (and DOCX / PPTX / images)

Through the MCP server's `ingest_local_dir` tool — runs MinerU as a
subprocess inside the already-warm lab-corpus container, persists a
LabPaper sidecar per file, makes the result discoverable via
`list_corpus` / `paper_info`, and is async (poll with `job_status`).

```jsonc
// from any MCP client (Claude Desktop, claude-code, …)
{
  "tool": "ingest_local_dir",
  "args": {
    "dir_path": "/data/pdfs/inbox",
    "glob": "*.pdf",
    "recursive": true
  }
}
// → {"job_id": "ab12cd…", "kind": "ingest_local_dir", "n_total": 42}
// later: {"tool": "job_status", "args": {"job_id": "ab12cd…"}}
```

Per ingested file this writes:
* `<parse.dir>/sources/<paper_id>.md` — flat markdown (MinerU output).
* `<parse.dir>/sources/<paper_id>.meta.json` — `LabPaper` sidecar
  ({paper_id_kind, title, source_path, n_chars, ingested_at, …}).
* `<parse.dir>/figures/<paper_id>/` — extracted PNG/JPG figures
  (when MinerU produces an `images/` subdir).

`paper_id` is auto-derived: arxiv-id pattern (`\d{4}\.\d{4,5}`) on
the filename, else sha256 prefix of the file bytes. Pass an explicit
`paper_id` to the `ingest_pdf` single-file tool to override.

### Build the embedding cache

```bash
bash scripts/docker_build_cache.sh \
    /path/to/radar.toml \
    /path/to/data_root \
    /path/to/cache_dir
```

The radar.toml's `[sources.*].path` should reference the in-container
mount paths (e.g. `/data/sources/ai4chem`). See `radar.example.toml`.

### Combined arxiv-radar + lab-corpus on one Qwen (recommended)

When you use both backends from the same client, run the **combined**
container instead of two separate ones — Qwen3-4B is ~8 GB in bf16,
so two copies don't fit in 12 GB VRAM. The combined supervisor in
`lab_corpus_mcp.combined` boots both servers in one process, hands
them the same `Encoder`, and serializes encode calls with a
`threading.Lock` (peak VRAM ≈ 10 GB).

```bash
# Start the long-lived container (auto-restart, two ports exposed)
bash scripts/docker_serve_combined.sh \
    /srv/arxiv-radar/radar.toml \
    /srv/lab-corpus/radar.toml \
    /srv/arxiv-radar/data \
    /srv/arxiv-radar/cache \
    /srv/lab-corpus/cache

# Watch logs
docker --context gomer logs -f lab-corpus-combined
```

Both backends now reachable on gomer at `:8765` (arxiv-radar) and
`:8766` (lab-corpus). For Claude Desktop, run two **stdio→HTTP
proxies** (one per backend) over an SSH tunnel:

```json
{
  "mcpServers": {
    "arxiv-radar": {
      "command": "lab-corpus-mcp",
      "args": ["--remote", "you@gomer", "--remote-port", "8765"]
    },
    "lab-corpus": {
      "command": "lab-corpus-mcp",
      "args": ["--remote", "you@gomer", "--remote-port", "8766"]
    }
  }
}
```

The supervisor refuses to start if the two `radar.toml` files
disagree on `[embeddings].model` or `[embeddings].target_dim` — they
share one in-memory copy of the model. Pass `--no-encoder-lock` to
the container CMD if you have VRAM headroom and want concurrent
encode throughput (default is on; cost is small because both servers
already hand the encoder off to `asyncio.to_thread`).

### Lab-only stdio (single-server)

`scripts/docker_serve_mcp.sh` spawns ONLY the lab-corpus server
bridged to stdio. Use this when you don't need arxiv-radar in the
same container, or when you want the stdio transport (Claude Desktop)
without an HTTP backend at all:

```json
{
  "mcpServers": {
    "lab-corpus": {
      "command": "bash",
      "args": [
        "D:/home/ignat/project-third-matter/git/lab-corpus-mcp/scripts/docker_serve_mcp.sh",
        "/srv/lab-corpus/radar.toml",
        "/srv/lab-corpus/data",
        "/srv/lab-corpus/cache"
      ]
    },
    "arxiv-radar": {
      "command": "bash",
      "args": [
        "D:/home/ignat/project-third-matter/git/arxiv-radar-mcp/tmp/setup.sh",
        "..."
      ]
    }
  }
}
```

Each Claude session spins up an ephemeral container; on exit it tears down
(`--rm`). Persistent state stays in the named volumes + bind mounts.

## Why a separate repo (split by content source)

The repos split along **what they consume**, not along complexity:

* `arxiv-radar-mcp` owns **all arXiv content** — abstracts via the
  `daily-arxiv-*` fork family, plus on-demand full-text fetched from
  `arxiv.org/html/<id>` and `arxiv.org/e-print/<id>` (HTML and LaTeX
  source). It chunks, indexes, and serves both abstract and fulltext
  search via MCP. Heavy MinerU is intentionally NOT included — HTML
  and LaTeX cover ~85-90% of arXiv submissions, the rest fails
  cleanly with a pointer here.

* `lab-corpus-mcp` owns **non-arXiv content** — uploaded PDFs (parsed
  via MinerU on gomer GPU), future YouTube/video transcripts,
  sideloaded books and conference preprints without arXiv IDs.

This boundary keeps the user flow consistent: a researcher who found
something in an abstract via `search_abstract_*` can deepen the
search with `search_paper_*` in the **same MCP catalog** — no need to
install a second server. The heavy stuff stays where it actually
needs to be heavy.

`lab-corpus-mcp` takes arxiv-radar-mcp as a pip dependency inside
this Docker image (`pip install -e /opt/arxiv-radar-mcp` in the
Dockerfile). Encoder, search, indexes are reused from upstream.

## Image size

Approximate breakdown (no preloaded models — those go in volumes):

| Layer | Size |
|---|---|
| pytorch/pytorch:2.5.1-cuda12.4 base | ~7 GB |
| MinerU `[core]` (vlm-transformers + pipeline + gradio) | ~2 GB |
| arxiv-radar-mcp + sentence-transformers + mcp + numpy | ~500 MB |
| lab_corpus_mcp glue (currently tiny) | ~5 MB |
| **Total** | **~9.5 GB** |

Pulled once to gomer; subsequent rebuilds reuse layers.

## Roadmap (lab-specific, not yet implemented)

After the 2026-05-01 boundary re-cut, the previously-planned literature
loader and on-demand full-text fetcher moved to `arxiv-radar-mcp`
(arxiv source has uniform IDs and HTML/LaTeX endpoints — fits the
public repo's "feed reader" pattern). What stays here:

1. **`upload_corpus` MCP tool** — accepts archive path / dir of PDFs,
   sniffs type, extracts to `/data/sources/<name>/`, runs MinerU
   subprocess. **Non-arXiv** PDFs (books, conference preprints
   without arXiv IDs, sideloaded reports). Job registry analogous to
   the one in arxiv-radar-mcp but for MinerU batch runs.
2. **YouTube / video → MD pipelines** — transcribe + chunk + add to
   a fulltext-style index. Probably a new MCP tool family
   `search_video_*` parallel to `search_paper_*`.
3. **Sideloaded books** — long-form PDFs with chapter headings; MinerU
   does the parse, then the chunker from arxiv-radar-mcp can be
   reused (heading-based split is the same problem).

The async job registry, ThreadPoolExecutor + persistence pattern, and
the chunker are all imported from `arxiv-radar-mcp` to avoid
duplicating that glue here.
