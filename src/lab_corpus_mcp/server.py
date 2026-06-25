"""MCP server for lab-corpus-mcp — built on `corpus_core.mcp_scaffold`.

Phase 2B tool surface (13 tools):

  Skeleton (Phase 2A):
    * `corpus_stats`      — coarse health: parsed / indexed / model.
    * `list_corpus`       — paper ids on disk (extended with title + kind).
    * `job_status` / `job_list` — delegate to corpus_core.JobRegistry.

  Ingest (Phase 2B-1):
    * `ingest_pdf`        — submit MinerU-driven ingest of one PDF/DOCX/...
    * `ingest_local_dir`  — bulk ingest of every matching file in a dir.

  Fetch-by-URL (Phase 2B+, U14 — 2026-05-13):
    * `ingest_url`        — download a URL into <parse.dir>/inbox/ then MinerU.
    * `ingest_arxiv_pdf`  — convenience over ingest_url for arxiv preprints.

  Index + search (Phase 2B-2):
    * `rebuild_index`     — submit corpus_core.corpus_index.reindex job.
    * `search_paper_text` — substring AND-scan over chunks (cheap).
    * `search_paper_semantic` — cosine over chunk embeddings.
    * `similar_to_paper`  — nearest neighbours by chunk-mean.
    * `paper_info`        — full LabPaper metadata + indexed status.

The handler holds a `JobRegistry` (background ingest/reindex), an
`Encoder` (lazy — only loads weights on first encode), and a
`fulltext_index: EmbeddingIndex | None` slot populated on construction
or after a successful `rebuild_index`. Tool-method bodies follow the
arxiv-radar-mcp `RadarServer` patterns; the difference is that lab
corpus comes from on-disk metadata (`<parse.dir>/sources/*.meta.json`)
instead of a fork-loaded JSON shard.
"""
from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Any

from corpus_core.corpus_index import (
    FULLTEXT_MAX_SEQ_LENGTH,
    load_chunk_texts,
    reindex,
    search_paper_semantic,
    search_paper_text,
    similar_to_paper,
)
from corpus_core.archive import PaperFiles, make_download_handler
from corpus_core.embeddings import EmbeddingIndex, Encoder
from corpus_core.jobs import JobError, JobHandle, JobRegistry
from corpus_core.mcp_scaffold import (
    BackgroundTaskFactory,
    build_mcp_app,
    make_method_dispatcher,
    serve_stdio,
)

from lab_corpus_mcp.config import Config, load
from lab_corpus_mcp.corpus import LabPaper, load_lab_papers
from lab_corpus_mcp.ingest import (
    IngestError,
    MineruRunner,
    fetch_and_ingest,
    ingest_dir,
    ingest_one,
    unload_mineru_models,
)

LOG = logging.getLogger(__name__)

# Strict arxiv-id validator for the `ingest_arxiv_pdf` tool: must be the
# canonical post-2007 form, optionally with a vN revision suffix.
_ARXIV_ID_RE = re.compile(r"^\d{4}\.\d{4,5}(?:v\d+)?$")


class LabCorpusServer:
    """Holds parsed-corpus location + EmbeddingIndex + JobRegistry.

    Tools are methods on the instance — the dispatcher routes by name.
    Encoder is lazy (only loads on first `encode_query` call), so
    constructing the server is cheap even on machines without a GPU /
    without the Qwen3 weights cached locally.

    `mineru_runner` is a test/benchmark seam — the `ingest_*` tools
    pass it through to `lab_corpus_mcp.ingest.ingest_one`.
    """

    def __init__(
        self,
        config: Config,
        *,
        encoder: Encoder | None = None,
        mineru_runner: MineruRunner | None = None,
    ) -> None:
        self.config = config
        self.parse_dir: Path = config.parse.dir
        # corpus_core.corpus_index.reindex writes embeddings.npy + index.json
        # at the SAME level as `sources/` (the dir containing the parsed
        # markdowns). For lab-corpus that's `parse.dir`. embeddings.cache_dir
        # in our config is kept around for legacy uniformity with
        # arxiv-radar-mcp's abstract-cache convention but is not where
        # the chunk index lives.
        self.index_dir: Path = self.parse_dir
        self.mineru_runner = mineru_runner

        self.papers: dict[str, LabPaper] = load_lab_papers(self.parse_dir)
        # Encoder injection point: combined-server supervisor passes one
        # shared Encoder so arxiv-radar + lab-corpus reuse a single
        # Qwen3-4B copy in VRAM (see lab_corpus_mcp.combined). Standalone
        # callers omit it and we construct our own (lazy weight load).
        self.encoder = encoder if encoder is not None else Encoder(config)
        # JobRegistry persists at <embeddings.cache_dir.parent>/jobs/ —
        # i.e. /srv/lab-corpus/cache/jobs/. Independent of the chunk
        # index dir, so reindex churn doesn't compete with job state.
        self.jobs = JobRegistry(cache_dir=config.embeddings.cache_dir.parent)

        self.fulltext_index: EmbeddingIndex | None = None
        try:
            self.fulltext_index = EmbeddingIndex.load(self.index_dir)
            LOG.info(
                f"loaded lab corpus index: {self.fulltext_index.matrix.shape} "
                f"({self.fulltext_index.model_name})"
            )
            if self.fulltext_index.model_name != config.embeddings.model:
                LOG.warning(
                    f"index built with {self.fulltext_index.model_name!r} but "
                    f"config requests {config.embeddings.model!r} — rerun "
                    f"`rebuild_index` with force_full=true to align"
                )
        except FileNotFoundError:
            LOG.info("no lab corpus index yet — search_paper_* will be unavailable "
                     "until rebuild_index runs")

    # ----- coarse health ---------------------------------------------------

    def corpus_stats(self) -> dict:
        sources_dir = self.parse_dir / "sources"
        n_parsed = sum(1 for _ in sources_dir.glob("*.md")) if sources_dir.exists() else 0

        if self.fulltext_index is None:
            n_indexed = 0
            n_chunks = 0
        else:
            chunks = (self.fulltext_index.metadata or {}).get("chunks", [])
            n_chunks = len(chunks)
            n_indexed = len({c.get("arxiv_id") for c in chunks if c.get("arxiv_id")})

        last_ingest_at: str | None = None
        if self.papers:
            last_ingest_at = max(p.ingested_at for p in self.papers.values()
                                 if p.ingested_at) or None

        return {
            "n_parsed": n_parsed,
            "n_indexed": n_indexed,
            "n_chunks": n_chunks,
            "embedding_model": self.config.embeddings.model,
            "parse_dir": str(self.parse_dir),
            "index_dir": str(self.index_dir),
            "last_ingest_at": last_ingest_at,
        }

    def list_corpus(self, limit: int | None = None) -> list[dict]:
        """Every paper id with parsed metadata on disk, with title + kind.

        Sorted by ingest time (newest first); falls back to lexicographic
        sort by paper_id when no timestamps are available.
        """
        rows = sorted(
            self.papers.values(),
            key=lambda p: (p.ingested_at or "", p.paper_id),
            reverse=True,
        )
        if limit is not None:
            rows = rows[:max(0, int(limit))]
        return [
            {
                "paper_id": p.paper_id,
                "paper_id_kind": p.paper_id_kind,
                "title": p.title,
                "source_kind": p.source_kind,
                "n_chars": p.n_chars,
                "n_chunks": p.n_chunks,
                "ingested_at": p.ingested_at,
            }
            for p in rows
        ]

    def paper_info(self, paper_id: str) -> dict:
        """Full LabPaper metadata + indexed status. Returns {error} if absent."""
        paper = self.papers.get(paper_id)
        if paper is None:
            return {"error": f"unknown paper_id: {paper_id!r}"}

        indexed = False
        n_chunks = 0
        if self.fulltext_index is not None:
            chunks = (self.fulltext_index.metadata or {}).get("chunks", [])
            paper_chunks = [c for c in chunks if c.get("arxiv_id") == paper_id]
            indexed = bool(paper_chunks)
            n_chunks = len(paper_chunks)

        return {
            "paper_id": paper.paper_id,
            "paper_id_kind": paper.paper_id_kind,
            "title": paper.title,
            "source_kind": paper.source_kind,
            "source_path": paper.source_path,
            "parsed_path": paper.parsed_path,
            "n_chars": paper.n_chars,
            "n_chunks": n_chunks if indexed else paper.n_chunks,
            "ingested_at": paper.ingested_at,
            "figures_dir": paper.figures_dir,
            "indexed": indexed,
        }

    # ----- search ----------------------------------------------------------

    def search_paper_text(self, query: str, k: int = 10,
                          snippet_chars: int = 240) -> list[dict]:
        if self.fulltext_index is None:
            return [{"error": "lab corpus index empty — run rebuild_index "
                              "after ingest_pdf / ingest_local_dir"}]
        chunk_texts = load_chunk_texts(self.parse_dir, self.fulltext_index)
        chunk_meta = (self.fulltext_index.metadata or {}).get("chunks", [])
        return search_paper_text(chunk_texts, chunk_meta, query, k=k,
                                 snippet_chars=snippet_chars)

    def search_paper_semantic(self, query: str, k: int = 10,
                              snippet_chars: int = 240) -> list[dict]:
        if self.fulltext_index is None:
            return [{"error": "lab corpus index empty — run rebuild_index"}]
        qvec = self.encoder.encode_query(query)
        chunk_texts = load_chunk_texts(self.parse_dir, self.fulltext_index)
        return search_paper_semantic(self.fulltext_index, chunk_texts, qvec,
                                     k=k, snippet_chars=snippet_chars)

    def similar_to_paper(self, paper_id: str, k: int = 10) -> list[dict]:
        if self.fulltext_index is None:
            return [{"error": "lab corpus index empty — run rebuild_index"}]
        return similar_to_paper(self.fulltext_index, paper_id, k=k)

    # ----- async admin -----------------------------------------------------

    def ingest_pdf(self, pdf_path: str, paper_id: str | None = None,
                   backend: str | None = None) -> dict:
        """Submit a background ingest job for one file. Returns {job_id}.

        `backend` selects MinerU mode (`pipeline` default — fast on
        12 GB GPU; `vlm-transformers` for higher fidelity if you have
        24 GB+ headroom). Omit to use ingest.DEFAULT_BACKEND.
        """
        p = Path(pdf_path)
        if not p.exists():
            return {"error": f"file not found: {pdf_path}"}
        from lab_corpus_mcp.ingest import DEFAULT_BACKEND
        backend_eff = backend or DEFAULT_BACKEND
        job_id = self.jobs.submit(
            kind="ingest_pdf",
            fn=lambda h: self._do_ingest_one(h, p, paper_id, backend_eff),
            args={"pdf_path": str(p), "paper_id": paper_id, "backend": backend_eff},
            n_total=1,
        )
        return {"job_id": job_id, "kind": "ingest_pdf", "backend": backend_eff}

    def ingest_local_dir(self, dir_path: str,
                         glob: str = "*.pdf",
                         recursive: bool = False,
                         backend: str | None = None) -> dict:
        """Submit a background bulk-ingest job. Returns {job_id, n_total}.

        `dir_path` is an **absolute path on the server's filesystem**
        (e.g. inside the Docker container). For the common case of
        ingesting files you've copied to the inbox directory, use
        `ingest_inbox` instead — it requires no path argument.

        `backend` is forwarded to each per-file MinerU run; see
        ingest_pdf for the trade-off.
        """
        d = Path(dir_path)
        if not d.exists() or not d.is_dir():
            return {"error": f"directory not found: {dir_path}"}
        # Pre-count so the registry can show progress out of N.
        iterator = d.rglob(glob) if recursive else d.glob(glob)
        n_total = sum(1 for p in iterator if p.is_file())
        if n_total == 0:
            return {"error": f"no files matching {glob!r} in {dir_path}"}

        from lab_corpus_mcp.ingest import DEFAULT_BACKEND
        backend_eff = backend or DEFAULT_BACKEND
        job_id = self.jobs.submit(
            kind="ingest_local_dir",
            fn=lambda h: self._do_ingest_dir(h, d, glob, recursive, backend_eff),
            args={"dir_path": str(d), "glob": glob, "recursive": recursive,
                  "backend": backend_eff},
            n_total=n_total,
        )
        return {"job_id": job_id, "kind": "ingest_local_dir",
                "n_total": n_total, "backend": backend_eff}

    def ingest_inbox(self, glob: str = "*.pdf",
                     recursive: bool = False,
                     backend: str | None = None) -> dict:
        """Submit a bulk ingest of all files in <parse.dir>/inbox/.

        Drop-box workflow for files on a remote host:
          1. Copy files to the server's inbox:
               docker cp paper.pdf lab-corpus:/srv/lab-corpus/inbox/
               # or: scp paper.pdf gomer:/srv/lab-corpus/inbox/
          2. Call ingest_inbox() — no path argument needed.

        The inbox is the same directory used by `ingest_url` /
        `ingest_arxiv_pdf` for server-side downloads, so any files
        previously fetched there are also eligible.

        `backend` is forwarded to each per-file MinerU run; see
        ingest_pdf for the trade-off.
        """
        inbox = self.parse_dir / "inbox"
        inbox.mkdir(parents=True, exist_ok=True)

        iterator = inbox.rglob(glob) if recursive else inbox.glob(glob)
        n_total = sum(1 for p in iterator if p.is_file())
        if n_total == 0:
            return {
                "error": (
                    f"no files matching {glob!r} in {inbox} — "
                    "copy files there first "
                    "(docker cp file.pdf lab-corpus:/srv/lab-corpus/inbox/ "
                    "or scp)"
                )
            }

        from lab_corpus_mcp.ingest import DEFAULT_BACKEND
        backend_eff = backend or DEFAULT_BACKEND
        job_id = self.jobs.submit(
            kind="ingest_inbox",
            fn=lambda h: self._do_ingest_dir(h, inbox, glob, recursive, backend_eff),
            args={"inbox": str(inbox), "glob": glob, "recursive": recursive,
                  "backend": backend_eff},
            n_total=n_total,
        )
        return {
            "job_id": job_id,
            "kind": "ingest_inbox",
            "inbox": str(inbox),
            "n_total": n_total,
            "backend": backend_eff,
        }

    def ingest_url(self, url: str, paper_id: str | None = None,
                   backend: str | None = None) -> dict:
        """Submit a background download-and-ingest job. Returns {job_id}.

        The URL is fetched via ``corpus_core.http_fetch.fetch_url`` —
        arxiv.org hosts share the module-global 1 req / 3 sec throttle
        with the arxiv-radar fetcher when both run in one process
        (combined image). Other hosts are not rate-limited.

        After download the file is parsed with MinerU exactly like
        ``ingest_pdf`` (same backend choice / paper_id derivation).
        """
        if not url or not isinstance(url, str) or not url.startswith(("http://", "https://")):
            return {"error": f"invalid url: {url!r}"}
        from lab_corpus_mcp.ingest import DEFAULT_BACKEND
        backend_eff = backend or DEFAULT_BACKEND
        job_id = self.jobs.submit(
            kind="ingest_url",
            fn=lambda h: self._do_ingest_url(h, url, paper_id, backend_eff),
            args={"url": url, "paper_id": paper_id, "backend": backend_eff},
            n_total=1,
        )
        return {"job_id": job_id, "kind": "ingest_url", "backend": backend_eff}

    def ingest_arxiv_pdf(self, arxiv_id: str,
                         backend: str | None = None) -> dict:
        """Convenience wrapper: download arxiv.org/pdf/<arxiv_id> and ingest.

        Equivalent to ``ingest_url(url='https://arxiv.org/pdf/<id>',
        paper_id=<id>)`` — but presents as its own job kind so dashboards
        / `job_list` make the arxiv-specific origin obvious. Closes the
        s142 dogfood gap where users had to curl + docker cp + ingest_local_dir
        for fresh PDF-only arxiv papers.
        """
        if not arxiv_id or not isinstance(arxiv_id, str) or not _ARXIV_ID_RE.match(arxiv_id):
            return {"error": f"invalid arxiv_id: {arxiv_id!r}"}
        from lab_corpus_mcp.ingest import DEFAULT_BACKEND
        backend_eff = backend or DEFAULT_BACKEND
        url = f"https://arxiv.org/pdf/{arxiv_id}"
        job_id = self.jobs.submit(
            kind="ingest_arxiv_pdf",
            fn=lambda h: self._do_ingest_url(h, url, arxiv_id, backend_eff),
            args={"arxiv_id": arxiv_id, "backend": backend_eff},
            n_total=1,
        )
        return {"job_id": job_id, "kind": "ingest_arxiv_pdf",
                "arxiv_id": arxiv_id, "backend": backend_eff}

    def rebuild_index(self, force_full: bool = False) -> dict:
        """Submit a background reindex of the parsed-corpus tree."""
        sources_dir = self.parse_dir / "sources"
        n_papers = sum(1 for _ in sources_dir.glob("*.md")) if sources_dir.exists() else 0
        if n_papers == 0:
            return {"error": "no parsed papers — run ingest_pdf / ingest_local_dir first"}

        if not self.jobs.acquire_reindex_lock():
            return {"error": "reindex already in progress (lockfile held)"}

        force = bool(force_full)
        job_id = self.jobs.submit(
            kind="rebuild_index",
            fn=lambda h: self._do_reindex(h, force_full=force),
            args={"n_papers": n_papers, "force_full": force},
            n_total=n_papers,
        )
        return {
            "job_id": job_id, "kind": "rebuild_index",
            "n_total": n_papers,
            "strategy_planned": "full" if force else "incremental",
        }

    def job_status(self, job_id: str) -> dict:
        info = self.jobs.get(job_id)
        if info is None:
            return {"error": f"unknown job_id: {job_id!r}"}
        return info

    def job_list(self, limit: int = 50) -> list[dict]:
        return self.jobs.list_recent(limit=limit)

    # ----- job workers (called by JobRegistry on a worker thread) ----------

    def _release_gpu_vram(self) -> None:
        """Drop the bi-encoder AND MinerU models from VRAM after a job.

        Lab-corpus runs on a single 12 GB RTX 4070 that's also used for
        unrelated compute on the host (DFT/MLIP runs). Two big resident
        consumers pin VRAM unless explicitly released:

          * Qwen3-Embedding-4B (~7-8 GB bf16) — loaded on first query
            or by `rebuild_index`. Encoder.unload() returns ownership.
          * MinerU pipeline singletons (~2-3 GB) — loaded on first
            ingest call when running in library mode (s153). Cleared
            via `unload_mineru_models()` (clears AtomModelSingleton +
            HybridModelSingleton dicts + cuda empty_cache).

        Both unloads are idempotent: this is safe to call from every
        job-completion path even when only one of the two was loaded.
        First call after unload pays the cold-load cost again
        (~20 sec encoder, ~30 sec MinerU pipeline).
        """
        try:
            self.encoder.unload()
        except Exception as e:  # noqa: BLE001
            LOG.warning(f"encoder.unload() failed (will keep model resident): {e}")
        try:
            unload_mineru_models()
        except Exception as e:  # noqa: BLE001
            LOG.warning(f"unload_mineru_models() failed: {e}")

    # Back-compat alias for tests / external callers written before s153.
    _release_encoder_vram = _release_gpu_vram

    def _do_ingest_one(self, handle: JobHandle, input_file: Path,
                       paper_id: str | None, backend: str) -> dict:
        try:
            paper = ingest_one(input_file, self.parse_dir,
                               paper_id=paper_id, backend=backend,
                               runner=self.mineru_runner)
        except IngestError as e:
            raise JobError(str(e)) from e
        finally:
            self._release_gpu_vram()
        # Refresh in-memory papers so corpus_stats / list_corpus see the new file.
        self.papers[paper.paper_id] = paper
        handle.update(n_done=1)
        return {"paper_id": paper.paper_id, "n_chars": paper.n_chars,
                "title": paper.title, "source_kind": paper.source_kind}

    def _do_ingest_dir(self, handle: JobHandle, dir_path: Path,
                       glob: str, recursive: bool, backend: str) -> dict:
        try:
            result = ingest_dir(
                dir_path, self.parse_dir,
                glob=glob, recursive=recursive, backend=backend,
                runner=self.mineru_runner,
                progress_cb=lambda done, total: handle.update(
                    n_done=done, n_total=total),
            )
        except IngestError as e:
            raise JobError(str(e)) from e
        finally:
            self._release_gpu_vram()
        # Reload papers map — bulk ingest may have added many.
        self.papers = load_lab_papers(self.parse_dir)
        return result

    def _do_ingest_url(self, handle: JobHandle, url: str,
                       paper_id: str | None, backend: str) -> dict:
        try:
            paper = fetch_and_ingest(
                url, self.parse_dir,
                paper_id=paper_id, backend=backend, runner=self.mineru_runner,
            )
        except IngestError as e:
            raise JobError(str(e)) from e
        finally:
            self._release_gpu_vram()
        self.papers[paper.paper_id] = paper
        handle.update(n_done=1)
        return {
            "paper_id": paper.paper_id,
            "n_chars": paper.n_chars,
            "title": paper.title,
            "source_kind": paper.source_kind,
            "url": url,
        }

    def _do_reindex(self, handle: JobHandle, *, force_full: bool = False) -> dict:
        try:
            new_index = reindex(
                self.parse_dir,
                self.encoder,
                incremental=not force_full,
                progress_cb=lambda done, total: handle.update(
                    n_done=done, n_total=total),
            )
        except FileNotFoundError as e:
            raise JobError(str(e)) from e
        finally:
            self.jobs.release_reindex_lock()
            # Release Qwen3-4B from VRAM once chunks are written and the
            # index is back on disk. The next search call will re-load
            # (≈20 sec cold). Without this, the encoder pins ~7-8 GB
            # for the server lifetime, blocking unrelated GPU work on
            # the same host.
            self._release_gpu_vram()

        self.fulltext_index = new_index
        # Sync per-paper n_chunks back into LabPaper metadata.
        chunks = (new_index.metadata or {}).get("chunks", [])
        per_paper: dict[str, int] = {}
        for c in chunks:
            pid = c.get("arxiv_id")
            if pid:
                per_paper[pid] = per_paper.get(pid, 0) + 1
        for pid, n in per_paper.items():
            if pid in self.papers:
                self.papers[pid].n_chunks = n

        return {
            "n_papers": (new_index.metadata or {}).get("n_papers", 0),
            "n_chunks": new_index.matrix.shape[0],
            "dims": new_index.dims,
            "model": new_index.model_name,
            "max_seq_length": FULLTEXT_MAX_SEQ_LENGTH,
        }


# ---------------------------------------------------------------------------
# MCP tool catalogue
# ---------------------------------------------------------------------------

LAB_TOOL_SPECS: list[dict[str, Any]] = [
    # ----- coarse health (4 tools — Phase 2A) -----------------------------
    {
        "name": "corpus_stats",
        "description": (
            "High-level health of the lab corpus: how many papers parsed, "
            "how many indexed, plus the active embedding model, on-disk "
            "paths and most-recent ingest timestamp. Cheap, deterministic."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "list_corpus",
        "description": (
            "Enumerate parsed papers (newest ingest first) with title + "
            "source_kind + chunk count. Pass `limit` to cap the response."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer", "minimum": 1,
                    "description": "Cap on the number of rows returned. Omit for all.",
                },
            },
        },
    },
    {
        "name": "paper_info",
        "description": (
            "Full metadata for one paper_id: title, source kind / path, "
            "parsed-markdown path, chunk count, indexed status, figures "
            "directory if MinerU extracted any. Returns {error} if absent."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "paper_id": {"type": "string"},
            },
            "required": ["paper_id"],
        },
    },
    {
        "name": "job_status",
        "description": (
            "Status of a background job (ingest_pdf / ingest_local_dir / "
            "rebuild_index). Same schema as arxiv-radar-mcp's job_status — "
            "both servers share corpus_core.JobRegistry."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"job_id": {"type": "string"}},
            "required": ["job_id"],
        },
    },
    {
        "name": "job_list",
        "description": "Recent background jobs, newest first. `limit` defaults to 50.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "minimum": 1, "default": 50},
            },
        },
    },
    # ----- ingest (Phase 2B-1) --------------------------------------------
    {
        "name": "ingest_pdf",
        "description": (
            "Submit a background MinerU-driven ingest of one file. Async — "
            "returns {job_id} immediately; poll with job_status. The file "
            "extension determines the parse path (PDF/DOCX/PPTX/image all "
            "supported by MinerU). `paper_id` overrides the auto-derived id "
            "(useful when you already know a DOI / arxiv id). `backend` "
            "selects MinerU mode: `pipeline` (default — fast, ~90 sec for "
            "a 2 MB PDF on RTX 4070) or `vlm-transformers` (1.2B Qwen2-VL, "
            "higher fidelity but needs 24 GB+ GPU)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "pdf_path": {
                    "type": "string",
                    "description": "Absolute path on the server's filesystem.",
                },
                "paper_id": {
                    "type": "string",
                    "description": "Optional explicit id. Defaults to "
                                   "filename-arxiv-id or sha256 prefix.",
                },
                "backend": {
                    "type": "string",
                    "enum": ["pipeline", "vlm-transformers"],
                    "description": "MinerU backend (default: pipeline).",
                },
            },
            "required": ["pdf_path"],
        },
    },
    {
        "name": "ingest_local_dir",
        "description": (
            "Submit a bulk MinerU ingest of every matching file in an "
            "arbitrary directory on the **server's** filesystem. "
            "Returns {job_id, n_total} immediately. "
            "Glob defaults to `*.pdf`; pass `*.docx` / `*.pptx` / `*` etc. "
            "for other types. `recursive=true` walks subdirectories. "
            "For the common remote-client workflow (copy PDFs via docker cp / "
            "scp then ingest), prefer `ingest_inbox` — it needs no path argument."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "dir_path": {
                    "type": "string",
                    "description": "Absolute path on the server's filesystem.",
                },
                "glob": {"type": "string", "default": "*.pdf"},
                "recursive": {"type": "boolean", "default": False},
                "backend": {
                    "type": "string",
                    "enum": ["pipeline", "vlm-transformers"],
                    "description": "MinerU backend (default: pipeline).",
                },
            },
            "required": ["dir_path"],
        },
    },
    {
        "name": "ingest_inbox",
        "description": (
            "Submit a bulk ingest of all files in the server's drop-box "
            "inbox (<parse.dir>/inbox/). Async — returns {job_id, n_total} "
            "immediately; poll with job_status. "
            "Workflow: copy PDFs to the container inbox first — "
            "`docker cp paper.pdf lab-corpus:/srv/lab-corpus/inbox/` "
            "or `scp paper.pdf gomer:/srv/lab-corpus/inbox/` — "
            "then call this tool with no required args. "
            "Glob defaults to `*.pdf`; pass `*.docx` / `*` for other types. "
            "The inbox is also where `ingest_url` / `ingest_arxiv_pdf` "
            "land their downloads, so any previously fetched files "
            "sitting there are included."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "glob": {
                    "type": "string",
                    "default": "*.pdf",
                    "description": "Filename glob (default: `*.pdf`).",
                },
                "recursive": {"type": "boolean", "default": False},
                "backend": {
                    "type": "string",
                    "enum": ["pipeline", "vlm-transformers"],
                    "description": "MinerU backend (default: pipeline).",
                },
            },
        },
    },
    # ----- fetch-by-URL (Phase 2B+, U14) -----------------------------------
    {
        "name": "ingest_url",
        "description": (
            "Download a remote document and ingest it. Async — returns "
            "{job_id} immediately; poll with job_status. Works for any "
            "http(s) URL (preprint server, journal supplement, OSF, "
            "personal page). The download lands in `<parse.dir>/inbox/` "
            "and is then parsed by MinerU exactly like `ingest_pdf`. "
            "arxiv.org URLs share the module-global 1 req / 3 sec "
            "throttle with the arxiv-radar fetcher (combined image). "
            "Pass `paper_id` to override the auto-derived id "
            "(useful when you already know a DOI / arxiv id)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Absolute http(s) URL to fetch.",
                },
                "paper_id": {
                    "type": "string",
                    "description": "Optional explicit id. Defaults to "
                                   "filename-arxiv-id or sha256 prefix.",
                },
                "backend": {
                    "type": "string",
                    "enum": ["pipeline", "vlm-transformers"],
                    "description": "MinerU backend (default: pipeline).",
                },
            },
            "required": ["url"],
        },
    },
    {
        "name": "ingest_arxiv_pdf",
        "description": (
            "Convenience wrapper over `ingest_url` for arxiv preprints. "
            "Downloads `https://arxiv.org/pdf/<arxiv_id>` and ingests "
            "with `paper_id=<arxiv_id>`. Closes the s142 dogfood gap "
            "where PDF-only arxiv papers required manual curl + "
            "docker cp + ingest_local_dir. Async — returns {job_id}. "
            "TO READ THE WHOLE PARSED PAPER (markdown + figures) once done, "
            "pull the bundle zip via the HTTP side-channel GET /download?id="
            "<paper_id> on this server's host:port (see server instructions); "
            "search_paper_* only returns chunks, not the full document."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "arxiv_id": {
                    "type": "string",
                    "description": "Canonical post-2007 arxiv id "
                                   "(e.g. `2512.14129`). Optional `vN` "
                                   "revision suffix allowed.",
                    "pattern": r"^\d{4}\.\d{4,5}(v\d+)?$",
                },
                "backend": {
                    "type": "string",
                    "enum": ["pipeline", "vlm-transformers"],
                    "description": "MinerU backend (default: pipeline).",
                },
            },
            "required": ["arxiv_id"],
        },
    },
    # ----- index + search (Phase 2B-2) ------------------------------------
    {
        "name": "rebuild_index",
        "description": (
            "Submit a background reindex of the parsed corpus. Incremental "
            "by default — only papers added or changed since the last index "
            "get re-encoded. Pass `force_full=true` after model swaps or "
            "manual cache surgery. Falls back to full automatically when "
            "the existing index was built with a different embedding model."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "force_full": {"type": "boolean", "default": False},
            },
        },
    },
    {
        "name": "search_paper_text",
        "description": (
            "Substring AND-scan over chunked corpus, returning top-k chunks "
            "with section + snippet. Cheap and deterministic; best when the "
            "query uses exact terminology."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer", "default": 10, "minimum": 1, "maximum": 200},
                "snippet_chars": {"type": "integer", "default": 240, "minimum": 40},
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_paper_semantic",
        "description": (
            "Cosine-similarity search over chunk embeddings (Qwen3-4B-native "
            "by default). Robust to terminology drift. Requires a populated "
            "embedding index (`rebuild_index`)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer", "default": 10, "minimum": 1, "maximum": 200},
                "snippet_chars": {"type": "integer", "default": 240, "minimum": 40},
            },
            "required": ["query"],
        },
    },
    {
        "name": "similar_to_paper",
        "description": (
            "Nearest-neighbour papers by mean-of-chunks cosine similarity to "
            "a known paper_id. Self-match excluded."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "paper_id": {"type": "string"},
                "k": {"type": "integer", "default": 10, "minimum": 1, "maximum": 200},
            },
            "required": ["paper_id"],
        },
    },
]


def _tool_names() -> list[str]:
    return [spec["name"] for spec in LAB_TOOL_SPECS]


def _make_upload_handler(server: "LabCorpusServer"):
    """Starlette async endpoint: POST /upload (multipart) → <parse.dir>/inbox/.

    Accepts one or more files per request. Filename is taken from the
    Content-Disposition header and sanitised (basename only, no traversal).

    Query params:
      ?ingest=true  — after saving files, submit an ingest_inbox() background
                      job. Returns the job_id in the response. Useful for
                      one-shot upload+ingest from the companion CLI script.

    Response JSON: {saved, n_saved, errors, inbox, job_id}

    Requires ``python-multipart`` to be installed (Starlette's multipart
    parser). Already declared in pyproject.toml dependencies.
    """
    async def upload(request):
        from starlette.datastructures import UploadFile   # noqa: PLC0415
        from starlette.responses import JSONResponse       # noqa: PLC0415

        inbox = server.parse_dir / "inbox"
        inbox.mkdir(parents=True, exist_ok=True)

        try:
            form = await request.form()
        except Exception as e:  # noqa: BLE001
            from starlette.responses import JSONResponse as _JR  # noqa: PLC0415
            return _JR({"error": f"multipart parse failed: {e}"}, status_code=400)

        saved: list[str] = []
        errors: list[dict] = []
        for _key, field in form.multi_items():
            if not isinstance(field, UploadFile):
                continue
            raw_name = field.filename or ""
            safe_name = Path(raw_name).name  # strip any path prefix
            if not safe_name or safe_name in (".", ".."):
                errors.append({"filename": raw_name, "error": "invalid filename"})
                continue
            dest = inbox / safe_name
            tmp = dest.with_suffix(dest.suffix + ".tmp")
            try:
                content = await field.read()
                tmp.write_bytes(content)
                tmp.replace(dest)
                saved.append(safe_name)
                LOG.info(f"upload: saved {safe_name} ({len(content):,} bytes)")
            except Exception as e:  # noqa: BLE001
                errors.append({"filename": safe_name, "error": str(e)})
            finally:
                await field.close()

        job_id = None
        if saved and request.query_params.get("ingest") in ("1", "true", "yes"):
            result = server.ingest_inbox()
            job_id = result.get("job_id")

        status = 200 if saved else (400 if errors else 422)
        return JSONResponse({
            "saved": saved,
            "n_saved": len(saved),
            "errors": errors,
            "inbox": str(inbox),
            "job_id": job_id,
        }, status_code=status)

    return upload


def _lab_paper_files(server: "LabCorpusServer", paper_id: str) -> PaperFiles:
    """Locate an ingested paper's pieces for the download archive.

    lab-corpus keeps the markdown at `sources/<id>.md` (+ `<id>.meta.json`)
    but copies MinerU's figures to `figures/<id>/`, while the markdown keeps
    MinerU's `![](images/<name>)` refs — so the in-archive subdir is named
    `images` to make those refs resolve after unzip.
    """
    sources_dir = server.parse_dir / "sources"
    return PaperFiles(
        markdown_path=sources_dir / f"{paper_id}.md",
        media_dir=server.parse_dir / "figures" / paper_id,
        media_arcname="images",
        meta_path=sources_dir / f"{paper_id}.meta.json",
    )


def _dispatch(server: LabCorpusServer, name: str, arguments: dict[str, Any] | None) -> Any:
    """Route an MCP tool-call to the matching LabCorpusServer method.

    Thin wrapper over `corpus_core.mcp_scaffold.make_method_dispatcher`.
    """
    return make_method_dispatcher(server, _tool_names())(name, arguments)


# Server-level metadata (MCP `initialize.instructions`) — documents the two
# binary HTTP side-channels (/upload, /download) the tool catalogue can't
# convey. Travels with the server to every client, not one local memory.
SERVER_INSTRUCTIONS = (
    "lab-corpus: semantic + full-text search over non-arXiv documents "
    "(PDF/DOCX/PPTX) parsed by MinerU. Ingest with ingest_arxiv_pdf / "
    "ingest_url / ingest_local_dir / ingest_pdf, search with search_paper_*.\n\n"
    "TWO BINARY SIDE-CHANNELS (NOT MCP tools — MCP JSON-RPC can't carry "
    "files), both on the SAME host:port as this server:\n"
    "1. UPLOAD local files to ingest:  POST /upload  (multipart; add "
    "?ingest=true to parse immediately). The `lab-corpus-upload` CLI wraps "
    "this: `lab-corpus-upload ~/papers/ http://<host>:<port>`.\n"
    "2. DOWNLOAD a parsed paper:  GET /download?id=<paper_id>  -> "
    "application/zip, after the paper is ingested. The zip is a single "
    "<id>/ folder: <id>.md (markdown) + images/<name> figures (MinerU refs "
    "resolve in place) + <id>.meta.json. Responses: 200 zip, 400 missing "
    "?id=, 404 not ingested. Example:\n"
    "    curl 'http://<host>:<port>/download?id=<paper_id>' -o paper.zip && unzip paper.zip\n"
    "(`paper_id` is what list_corpus / ingest_* report.)"
)


def _build_mcp_app(server: LabCorpusServer):
    return build_mcp_app(
        server_name="lab-corpus",
        tool_specs=LAB_TOOL_SPECS,
        dispatcher=make_method_dispatcher(server, _tool_names()),
        instructions=SERVER_INSTRUCTIONS,
    )


async def _warmup_encoder(server: LabCorpusServer) -> None:
    """Force a single dummy encode so the user's first real query doesn't
    pay the lazy-load cost (~20 sec for Qwen3-4B on RTX 4070, longer if
    the model has to download from HF Hub first). Failures don't propagate;
    cold-start search will just be slow but functional.
    """
    LOG.info("encoder warm-up: starting (so first query doesn't cold-load)")
    try:
        await asyncio.to_thread(server.encoder.encode_query, "warmup")
        LOG.info("encoder warm-up: ready")
    except Exception as e:  # noqa: BLE001
        LOG.warning(f"encoder warm-up failed (will retry on first query): {e}")


def _lab_background_tasks(server: LabCorpusServer) -> list[BackgroundTaskFactory]:
    """Background tasks the lab shell wants alongside the MCP transport.

    Phase 2B: encoder warm-up only. No periodic refresh — the lab corpus
    grows by user-driven `ingest_*` calls, not a scheduled feed.
    """
    return [lambda: _warmup_encoder(server)]


async def _run_stdio(server: LabCorpusServer) -> None:
    await serve_stdio(
        server_name="lab-corpus",
        tool_specs=LAB_TOOL_SPECS,
        dispatcher=make_method_dispatcher(server, _tool_names()),
        background_tasks=_lab_background_tasks(server),
        instructions=SERVER_INSTRUCTIONS,
    )


async def _run_streamable_http(server: LabCorpusServer, host: str, port: int) -> None:
    """Streamable-HTTP MCP loop with lab-specific /upload sidecar.

    Builds the Starlette app directly (not via corpus_core.serve_streamable_http)
    so we can mount /upload alongside /mcp on the same port without
    pushing lab-specific routing concerns into corpus-core.
    """
    import uvicorn                                                     # noqa: PLC0415
    from mcp.server.streamable_http_manager import (                  # noqa: PLC0415
        StreamableHTTPSessionManager,
    )
    from starlette.applications import Starlette                       # noqa: PLC0415
    from starlette.routing import Mount, Route                         # noqa: PLC0415

    from corpus_core.mcp_scaffold import build_mcp_app                # noqa: PLC0415

    mcp_app = build_mcp_app(
        server_name="lab-corpus",
        tool_specs=LAB_TOOL_SPECS,
        dispatcher=make_method_dispatcher(server, _tool_names()),
        instructions=SERVER_INSTRUCTIONS,
    )
    session_manager = StreamableHTTPSessionManager(
        app=mcp_app, json_response=True, stateless=False,
    )

    async def _handle_mcp(scope, receive, send):
        await session_manager.handle_request(scope, receive, send)

    starlette_app = Starlette(routes=[
        Mount("/mcp", app=_handle_mcp),
        Route("/upload", endpoint=_make_upload_handler(server), methods=["POST"]),
        Route("/download",
              endpoint=make_download_handler(
                  lambda pid: _lab_paper_files(server, pid)),
              methods=["GET"]),
    ])

    bg = [asyncio.create_task(make()) for make in _lab_background_tasks(server)]
    try:
        async with session_manager.run():
            uv_config = uvicorn.Config(
                starlette_app, host=host, port=port,
                log_level="info", access_log=False,
            )
            await uvicorn.Server(uv_config).serve()
    finally:
        for t in bg:
            t.cancel()


def serve(config_path: Path | None = None) -> None:
    """Entry point: stdio MCP server."""
    config = load(config_path)
    server = LabCorpusServer(config)
    asyncio.run(_run_stdio(server))


def serve_http(host: str, port: int, config_path: Path | None = None) -> None:
    """Entry point: streamable-HTTP MCP server (long-running backend mode)."""
    config = load(config_path)
    server = LabCorpusServer(config)
    LOG.info(f"lab-corpus streamable-HTTP server listening on {host}:{port}")
    LOG.info(f"  embedding model: {config.embeddings.model}")
    LOG.info(f"  parse_dir:       {config.parse.dir}")
    LOG.info(f"  index_dir:       {config.embeddings.cache_dir}")
    asyncio.run(_run_streamable_http(server, host, port))
