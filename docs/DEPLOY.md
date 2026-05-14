# Deploying lab-corpus-mcp

Production deployment is a single GPU container on **gomer**, bundling
three workflows on the same CUDA + torch stack:

| Workflow | When to run | Time on gomer (RTX 4070) |
|---|---|---|
| MinerU PDF parsing (`pipeline` backend) | new literature batch arrives | ~60-90 sec per 2 MB PDF |
| Reindex (chunk-level, incremental) | after ingest | ~70 sec for 16 chunks |
| MCP server (HTTP) | always-on for clients | runs while container is up |
| Daily refresh of arxiv shards | nightly | incremental — only new arxiv_ids re-encoded |

The image bundles **MinerU + corpus-core + arxiv-radar-mcp + lab_corpus_mcp**
on top of `pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime` (Python 3.11,
torch 2.7.1+cu126, satisfies MinerU 3.x's `torch>=2.6,<3` floor without
a parallel torch reinstall). Build context is the parent directory of
all three sibling repos; see `scripts/docker_build.sh`.

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

> **Sharing the HF volume with a legacy arxiv-radar-backend.** If you
> already have an `arxiv-radar-backend` container on the host using
> `arxiv-radar-hf` for its Qwen weights, point `docker_serve_combined.sh`
> at the SAME volume name (the script accepts `-v arxiv-radar-hf:...`)
> instead of `lab-corpus-hf`. That way Qwen3-4B weights stay at one
> copy on the host disk during the migration window.

## Migrating from a legacy arxiv-radar-backend

If you have a long-running `arxiv-radar-backend` container with
abstract/fulltext indexes already built, you can promote it to the
combined image without re-encoding. The combined supervisor reads the
exact same on-disk layout (`<cache_dir>/abstracts`,
`<cache_dir>/fulltext`, `<cache_dir>/sources`).

```bash
# 1. Spin up a transient container that mounts BOTH the old volume
#    (read-only) and the new host path:
docker --context gomer run --rm \
    -v arxiv-radar-cache:/src:ro \
    -v /srv/arxiv-radar:/dst \
    busybox sh -c '
        mkdir -p /dst/cache
        cp -a /src/. /dst/cache/
    '

# 2. Write the new radar.toml at /srv/arxiv-radar/radar.toml with paths
#    rewritten to /srv/arxiv-radar/cache/* (see radar.example.toml).
#    Set [refresh] enabled=true, full_rebuild=false so the nightly tick
#    only encodes new shards.

# 3. Boot the combined container — it loads the migrated indexes:
#    [INFO] loaded abstract index: (34627, 2560)
#    [INFO] loaded fulltext index: (466, 2560) (466 chunks across 51 papers)
#    [INFO] refresh loop: every 24h, full_rebuild=False
```

After verification, stop the legacy `arxiv-radar-backend` and repoint
your MCP clients at the combined image's two ports.

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
    "recursive": true,
    "backend": "pipeline"          // default; pass "vlm-transformers" if 24 GB+ GPU
  }
}
// → {"job_id": "ab12cd…", "kind": "ingest_local_dir", "n_total": 42, "backend": "pipeline"}
// later: {"tool": "job_status", "args": {"job_id": "ab12cd…"}}
```

### Ingest by URL (U14, 2026-05-13)

`ingest_local_dir` requires the PDF to already sit on the server's
filesystem. For one-shot ingest of a remote PDF (arxiv preprint,
journal supplement, OSF / Zenodo deposit), call **`ingest_url`** or its
arxiv-specific shorthand **`ingest_arxiv_pdf`** — both download server-
side via `corpus_core.http_fetch.fetch_url`, land the file under
`<parse.dir>/inbox/`, then run the existing MinerU + reindex path.

```jsonc
// Arxiv-specific shorthand — paper_id is forced to the arxiv id.
{
  "tool": "ingest_arxiv_pdf",
  "args": {"arxiv_id": "2512.14129"}
}
// → {"job_id": "ef34ab…", "kind": "ingest_arxiv_pdf",
//    "arxiv_id": "2512.14129", "backend": "pipeline"}

// Generic URL — paper_id auto-derived from filename (or sha256 fallback);
// pass `paper_id` to override.
{
  "tool": "ingest_url",
  "args": {
    "url": "https://example.org/preprints/ai4chem.pdf",
    "paper_id": "ai4chem-2026"
  }
}
// → {"job_id": "cd56ef…", "kind": "ingest_url", "backend": "pipeline"}
```

**Throttle sharing:** arxiv.org URLs go through the singleton
`corpus_core.http_fetch.get_arxiv_throttle()` so the combined image
(arxiv-radar's HTML/LaTeX fetcher + lab-corpus's PDF fetcher) shares
one 1 req / 3 sec budget instead of double-spamming. Non-arxiv hosts
are unthrottled by default.

**MinerU backend default = `pipeline`.** The `vlm-transformers` backend
loads MinerU's 1.2B Qwen2-VL into VRAM, which combined with our shared
Qwen3-Embedding-4B exhausts a 12 GB GPU and wedges the parse. The
`pipeline` backend (layout-CNN + OCR + table) parses a 2 MB arxiv
preprint in ~90 sec on RTX 4070 with no VRAM contention; quality is
slightly lower but still extracts title, abstract, sections, formulas,
and figures cleanly. Pass `backend="vlm-transformers"` per-call if you
have GPU headroom and want higher fidelity.

Per ingested file this writes:
* `<parse.dir>/sources/<paper_id>.md` — flat markdown (MinerU output).
* `<parse.dir>/sources/<paper_id>.meta.json` — `LabPaper` sidecar
  ({paper_id_kind, title, source_path, n_chars, ingested_at, …}).
  Tolerates extra fields written by `corpus_core.corpus_index.reindex`
  (`n_chunks_after_split`, `indexed_at`) — they fold into the
  `LabPaper.extra` dict on read.
* `<parse.dir>/figures/<paper_id>/` — extracted PNG/JPG figures
  (when MinerU produces an `images/` subdir).
* `<parse.dir>/embeddings.npy` + `<parse.dir>/index.json` — written by
  the next `rebuild_index` call. `corpus_core.corpus_index.reindex`
  writes the chunk index AT `parse.dir` (same level as `sources/`),
  not under `embeddings.cache_dir`. lab-corpus's `LabCorpusServer`
  loads from there.

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
    /srv/arxiv-radar/cache/sources \
    /srv/arxiv-radar/cache \
    /srv/lab-corpus/cache

# Watch logs (expect three index-load lines + supervisor banner)
docker --context gomer logs -f lab-corpus-combined
```

Both backends now reachable on gomer at `:8765` (arxiv-radar) and
`:8766` (lab-corpus). When migrating from a legacy
`arxiv-radar-backend` that already binds `:8765`, deploy the combined
container on alternate ports (e.g. `127.0.0.1:18765` + `:18766`) until
you cut over.

For Claude Desktop, run two **stdio→HTTP proxies** (one per backend)
over an SSH tunnel:

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
| pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime base | ~9 GB |
| MinerU `[core]` (vlm-transformers + pipeline + gradio) | ~2 GB |
| corpus-core + arxiv-radar-mcp + sentence-transformers + mcp | ~500 MB |
| lab_corpus_mcp + scripts | ~5 MB |
| **Effective shipped tag** | **~12 GB total in `docker images`** |

Pulled once to gomer; subsequent rebuilds reuse layers. The build-time
`audit_image.py` step verifies single-distribution invariants — no
duplicate torch / sentence-transformers / mineru even after MinerU's
~95 transitive pip deps install.

## Roadmap

After the 2026-05-01 boundary re-cut, the previously-planned literature
loader and on-demand full-text fetcher moved to `arxiv-radar-mcp`
(arxiv source has uniform IDs and HTML/LaTeX endpoints — fits the
public repo's "feed reader" pattern). What stays here:

1. ✅ **PDF ingest tools** — DONE (Phase 2B-1, 2026-05-09). The
   originally-planned `upload_corpus` shipped as `ingest_pdf`
   (single file) + `ingest_local_dir` (bulk by glob, optional
   recursion). Both async, both write LabPaper sidecars + figures
   + post-ingest chunk index. End-to-end verified on
   `arxiv:2512.14129` (Yin et al., (Cr,Fe)S pyrrhotite) — closes
   arxiv-radar-mcp's deferred U7.
2. ✅ **MinerU backend default = `pipeline`** (Phase 2B+, 2026-05-10).
   The 1.2B Qwen2-VL backend wedges on a 12 GB GPU; pipeline does
   2 MB / 90 sec while sharing VRAM with our embedding Qwen.
2a. ✅ **Fetch-by-URL** (U14, 2026-05-13). `ingest_url(url, paper_id?)`
    and `ingest_arxiv_pdf(arxiv_id)` MCP tools download server-side
    via `corpus_core.fetch_url`, dropping the curl + docker cp +
    ingest_local_dir workflow that surfaced in the s142 dogfood.
    arxiv hosts share the singleton 1 req / 3 sec throttle with
    arxiv-radar's HTML/LaTeX fetcher in the combined image.
3. ⏳ **PDF-content DOI extraction** — currently filename-based
   arxiv-id pattern + sha256 fallback. Adding pypdf-based DOI lookup
   to fill in `paper_id_kind="doi"` for arxiv-less PDFs.
4. ⏳ **YouTube / video → MD pipelines** — transcribe + chunk + add
   to a fulltext-style index. Probably a new MCP tool family
   `search_video_*` parallel to `search_paper_*`.
5. ⏳ **Sideloaded books** — long-form PDFs with chapter headings;
   MinerU does the parse, then the chunker from corpus-core can be
   reused (heading-based split is the same problem).
6. ⏳ **GitHub Actions CI** — pytest matrix on push/PR for Python
   3.11–3.13. Cheap (~30 LOC YAML), unblocks public traffic.

corpus-core, arxiv-radar-mcp and lab-corpus-mcp share one async
JobRegistry + chunker via [corpus-core](https://github.com/exopoiesis/corpus-core),
so no glue duplication between siblings.
