"""MCP server for lab-corpus-mcp — built on `corpus_core.mcp_scaffold`.

Phase 2A skeleton. Today it exposes a four-tool surface that proves the
dispatcher + transport wiring without depending on `arxiv_radar_mcp`:

  * `corpus_stats`   — coarse health: how many papers parsed, how many
                       indexed, what the embedding model is.
  * `list_corpus`    — list every paper id that has a parsed markdown
                       source (`<parse.dir>/sources/<id>.md`).
  * `job_status`     — delegate to `corpus_core.JobRegistry.get`.
  * `job_list`       — delegate to `corpus_core.JobRegistry.list_recent`.

The handler holds a `JobRegistry` so future tools (`ingest_pdf`,
`rebuild_index`, `upload_corpus`) can submit long-running work in the
same async pattern arxiv-radar-mcp uses for `fetch_papers` and
`reindex`.

Phase 2B will add MinerU-driven `ingest_pdf` + `ingest_local_dir` tools
and wire `corpus_core.corpus_index.reindex` against the parsed tree.
Phase 2B+ adds `search_paper_*` once the index is healthy.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from corpus_core.embeddings import EmbeddingIndex
from corpus_core.jobs import JobRegistry
from corpus_core.mcp_scaffold import (
    BackgroundTaskFactory,
    build_mcp_app,
    make_method_dispatcher,
    serve_stdio,
    serve_streamable_http,
)

from lab_corpus_mcp.config import Config, load

LOG = logging.getLogger(__name__)


class LabCorpusServer:
    """Holds the parsed-corpus location + JobRegistry. Tools are methods.

    Skeleton — no Encoder/EmbeddingIndex loaded yet. The `index_dir` slot
    is reserved for Phase 2B when reindex lands; today we only inspect
    the parsed-source tree on disk.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.parse_dir: Path = config.parse.dir
        self.index_dir: Path = config.embeddings.cache_dir
        self.jobs = JobRegistry(cache_dir=self.index_dir.parent)

        # Optional: load the chunk-level index if it already exists
        # (e.g. built out-of-band by a `lab-corpus-mcp --build-index`
        # CLI in a future phase). Today this stays None.
        self.corpus_index: EmbeddingIndex | None = None
        try:
            self.corpus_index = EmbeddingIndex.load(self.index_dir)
            LOG.info(f"loaded lab corpus index: {self.corpus_index.matrix.shape} "
                     f"({self.corpus_index.model_name})")
        except FileNotFoundError:
            LOG.info("no lab corpus index yet — corpus_stats will report n_indexed=0")

    # ----- tool methods ----------------------------------------------------

    def corpus_stats(self) -> dict:
        """High-level corpus health.

        Cheap — touches only directory listings + index metadata.
        """
        sources_dir = self.parse_dir / "sources"
        n_parsed = 0
        if sources_dir.exists():
            n_parsed = sum(1 for _ in sources_dir.glob("*.md"))

        if self.corpus_index is None:
            n_indexed = 0
            n_chunks = 0
        else:
            chunks = (self.corpus_index.metadata or {}).get("chunks", [])
            n_chunks = len(chunks)
            n_indexed = len({c.get("arxiv_id") for c in chunks if c.get("arxiv_id")})

        return {
            "n_parsed": n_parsed,
            "n_indexed": n_indexed,
            "n_chunks": n_chunks,
            "embedding_model": self.config.embeddings.model,
            "parse_dir": str(self.parse_dir),
            "index_dir": str(self.index_dir),
        }

    def list_corpus(self, limit: int | None = None) -> list[str]:
        """Every paper id with a parsed markdown source on disk.

        Sorted lexicographically. Pass `limit` to cap the result (the
        underlying directory may hold thousands of files at scale).
        """
        sources_dir = self.parse_dir / "sources"
        if not sources_dir.exists():
            return []
        ids = sorted(p.stem for p in sources_dir.glob("*.md"))
        if limit is not None:
            ids = ids[:max(0, int(limit))]
        return ids

    def job_status(self, job_id: str) -> dict:
        info = self.jobs.get(job_id)
        if info is None:
            return {"error": f"unknown job_id: {job_id!r}"}
        return info

    def job_list(self, limit: int = 50) -> list[dict]:
        return self.jobs.list_recent(limit=limit)


# ---------------------------------------------------------------------------
# MCP tool catalogue
# ---------------------------------------------------------------------------

LAB_TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "corpus_stats",
        "description": (
            "High-level health of the lab corpus: how many papers have a "
            "parsed markdown source, how many are indexed, plus the active "
            "embedding model and on-disk paths. Cheap, deterministic."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "list_corpus",
        "description": (
            "Enumerate every paper id with a parsed markdown source on disk "
            "(`<parse.dir>/sources/<id>.md`). Sorted lexicographically. Pass "
            "`limit` to cap the response when the corpus is large."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Cap on the number of ids returned. Omit for all.",
                },
            },
        },
    },
    {
        "name": "job_status",
        "description": (
            "Status of a background job submitted earlier (e.g. ingest_pdf, "
            "reindex). Returns {state, progress, n_total, n_done, started_at, "
            "finished_at, result, error}. Same schema as arxiv-radar-mcp's "
            "job_status — both servers share corpus_core.JobRegistry."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "job_id": {"type": "string"},
            },
            "required": ["job_id"],
        },
    },
    {
        "name": "job_list",
        "description": (
            "Recent background jobs, newest first. `limit` defaults to 50."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "minimum": 1, "default": 50},
            },
        },
    },
]


def _tool_names() -> list[str]:
    return [spec["name"] for spec in LAB_TOOL_SPECS]


def _dispatch(server: LabCorpusServer, name: str, arguments: dict[str, Any] | None) -> Any:
    """Route an MCP tool-call to the matching LabCorpusServer method.

    Thin wrapper over `corpus_core.mcp_scaffold.make_method_dispatcher`.
    Kept here so unit tests can call it with a stub `LabCorpusServer`.
    """
    return make_method_dispatcher(server, _tool_names())(name, arguments)


def _build_mcp_app(server: LabCorpusServer):
    """Construct the MCP app for `server`, wired with our LAB_TOOL_SPECS."""
    return build_mcp_app(
        server_name="lab-corpus",
        tool_specs=LAB_TOOL_SPECS,
        dispatcher=make_method_dispatcher(server, _tool_names()),
    )


def _lab_background_tasks(server: LabCorpusServer) -> list[BackgroundTaskFactory]:
    """Background tasks the lab shell needs alongside the MCP transport.

    Empty for the Phase 2A skeleton — no encoder warm-up (no encoder yet)
    and no refresh loop (the parsed corpus is updated by user-driven
    `ingest_*` calls, not a scheduled feed). Phase 2B will add an
    encoder warm-up factory once `EmbeddingIndex` is live.
    """
    return []


async def _run_stdio(server: LabCorpusServer) -> None:
    await serve_stdio(
        server_name="lab-corpus",
        tool_specs=LAB_TOOL_SPECS,
        dispatcher=make_method_dispatcher(server, _tool_names()),
        background_tasks=_lab_background_tasks(server),
    )


async def _run_streamable_http(server: LabCorpusServer, host: str, port: int) -> None:
    await serve_streamable_http(
        server_name="lab-corpus",
        tool_specs=LAB_TOOL_SPECS,
        dispatcher=make_method_dispatcher(server, _tool_names()),
        host=host,
        port=port,
        background_tasks=_lab_background_tasks(server),
    )


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
