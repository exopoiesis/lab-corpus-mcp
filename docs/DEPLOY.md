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
# Both repos must be siblings under D:/home/ignat/project-third-matter/git/
#   ├── arxiv-radar-mcp/
#   └── lab-corpus-mcp/

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

### Parse PDFs into structured markdown + figures

```bash
bash scripts/docker_parse_pdfs.sh \
    /mnt/literature/pdfs \
    /mnt/literature/parsed \
    vlm-transformers
```

Per PDF this writes (under `/mnt/literature/parsed/<pdf_basename>/`):
* `<name>.md` — markdown with inline figure refs and captions
* `<name>_content_list.json` — flat reading-order JSON (consumed by the
  literature loader once it lands)
* `<name>_middle.json` — full structure with bboxes
* `images/` — extracted PNG/JPG figures and tables

### Build the embedding cache

```bash
bash scripts/docker_build_cache.sh \
    /path/to/radar.toml \
    /path/to/data_root \
    /path/to/cache_dir
```

The radar.toml's `[sources.*].path` should reference the in-container
mount paths (e.g. `/data/sources/ai4chem`). See `radar.example.toml`.

### MCP server for Claude Desktop

`scripts/docker_serve_mcp.sh` spawns the server bridged to stdio. Wire
into your Claude Desktop MCP config:

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

## Why a separate repo (and not a fork of arxiv-radar-mcp)

`arxiv-radar-mcp` has a clear identity: feed-consumer for the
`daily-arxiv-*` fork family. Public, narrow, stateless, MIT-licensed.
You can fork the data sources independently.

`lab-corpus-mcp` is the personal admin layer — write-mounts, MinerU
heavyweight, job queues, eventually slides + video pipelines. Bundling
this into arxiv-radar-mcp would inflate the simple feed-consumer with
dependencies the public users don't need, and force its evolution to
track lab-only requirements.

Two-repo split keeps:
* arxiv-radar-mcp narrow + publishable
* lab-corpus-mcp free to evolve aggressively (heavy deps, breaking
  tools) without touching the upstream

The cost is ~500 lines of duplicated glue eventually, or — current
choice — `lab-corpus-mcp` taking arxiv-radar-mcp as a pip dependency
inside its Docker image.

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

## Falling back without Docker

The pre-Docker scripts still exist for local Windows iteration:

* `tmp/install_mineru.sh` — MinerU venv + pip install + model fetch
* `scripts/process_pdfs.sh` — local CPU MinerU run (slower than GPU)

These predate the bundled image. If gomer is unavailable they're the
fallback path.

## Roadmap (lab-specific tools, not yet implemented)

The minimal scaffold today only re-exposes `arxiv_radar_mcp`'s search
tools through a new container. Lab-specific MCP tools that we agreed to
add (in this order):

1. **Per-source cache shards** — `/cache/<source>/embeddings.npy` instead
   of the current single monolithic cache. Allows incremental rebuilds.
2. **Dynamic source discovery** — scan `/data/sources/*` subdirs,
   detect type (abstracts JSON / MinerU output / raw PDFs) automatically.
3. **Job registry** — `src/lab_corpus_mcp/jobs.py` with persistence at
   `/data/jobs.json`, lockfile to prevent concurrent admin ops.
4. **`upload_corpus` MCP tool** — accepts archive path, sniffs type,
   extracts to `/data/sources/<name>/`, runs MinerU subprocess if PDFs.
5. **`rebuild_index`, `corpus_stats`, `job_status`** — admin tool surface.
6. **Literature loader** — once we see real MinerU output structure.

See arxiv-radar-mcp's task list (#23, #27) for related items.
