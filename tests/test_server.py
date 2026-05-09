"""Tool catalogue + dispatcher + LabCorpusServer tests.

Adapted from arxiv-radar-mcp/tests/test_server.py. Exercises the parts
of `lab_corpus_mcp.server` that are pure / sync — no MCP SDK runtime,
no stdio, no encoder loading. The async transport is left to integration
testing once the server runs against a real client.
"""
from __future__ import annotations

import inspect
import time

import pytest

from lab_corpus_mcp.server import (
    LAB_TOOL_SPECS,
    LabCorpusServer,
    _build_mcp_app,
    _dispatch,
    _lab_background_tasks,
    _tool_names,
)


# ----- LAB_TOOL_SPECS shape --------------------------------------------------

EXPECTED_TOOLS = {
    "corpus_stats",
    "list_corpus",
    "job_status",
    "job_list",
}


def test_tool_specs_cover_all_expected_tools():
    listed = {s["name"] for s in LAB_TOOL_SPECS}
    assert listed == EXPECTED_TOOLS, (
        f"missing: {EXPECTED_TOOLS - listed}, extra: {listed - EXPECTED_TOOLS}")


def test_tool_specs_match_method_signatures():
    """Every tool's required-args + declared properties must be real method params."""
    for spec in LAB_TOOL_SPECS:
        method = getattr(LabCorpusServer, spec["name"])
        sig = inspect.signature(method)
        params = set(sig.parameters) - {"self"}
        declared = set(spec["inputSchema"].get("properties", {}))
        assert declared <= params, (
            f"{spec['name']}: schema declares {declared - params} "
            f"that aren't method params {params}")
        required = set(spec["inputSchema"].get("required", []))
        assert required <= params, (
            f"{spec['name']}: missing required {required - params}")


def test_tool_specs_have_descriptions_and_object_schema():
    for spec in LAB_TOOL_SPECS:
        assert spec["description"].strip(), f"{spec['name']}: empty description"
        assert spec["inputSchema"]["type"] == "object"


def test_no_arxiv_radar_tools_leak_through():
    """Lab catalogue must not silently inherit arxiv-radar's tool surface."""
    listed = {s["name"] for s in LAB_TOOL_SPECS}
    forbidden = {
        "search_abstract_text", "search_abstract_semantic", "similar_to_abstract",
        "paper_info", "list_tags", "list_domains",
        "search_paper_text", "search_paper_semantic", "similar_to_paper",
        "fetch_papers", "reindex", "refresh_abstracts", "validate_arxiv_ids",
        "list_enriched",
    }
    assert listed.isdisjoint(forbidden), (
        f"arxiv-radar tools resurrected: {listed & forbidden}")


def test_tool_names_helper_matches_specs():
    assert set(_tool_names()) == EXPECTED_TOOLS
    assert len(_tool_names()) == len(LAB_TOOL_SPECS)


# ----- _dispatch -------------------------------------------------------------

class _StubLab:
    """Mimics the subset of LabCorpusServer surface that _dispatch routes to."""

    def corpus_stats(self):
        return {"called": "corpus_stats", "n_parsed": 0}

    def list_corpus(self, limit=None):
        return ["a", "b"] if limit is None else ["a", "b"][:limit]

    def job_status(self, job_id):
        return {"called": "job_status", "job_id": job_id}

    def job_list(self, limit=50):
        return [{"limit": limit}]


def test_dispatch_routes_to_method():
    out = _dispatch(_StubLab(), "corpus_stats", {})
    assert out["called"] == "corpus_stats"


def test_dispatch_handles_none_arguments():
    out = _dispatch(_StubLab(), "job_list", None)
    assert out == [{"limit": 50}]


def test_dispatch_passes_kwargs_through():
    out = _dispatch(_StubLab(), "list_corpus", {"limit": 1})
    assert out == ["a"]


def test_dispatch_unknown_tool_returns_error():
    out = _dispatch(_StubLab(), "drop_database", {})
    assert "error" in out and "drop_database" in out["error"]


def test_dispatch_rejects_dunder_or_private_names():
    out = _dispatch(_StubLab(), "__init__", {})
    assert "error" in out


def test_dispatch_bad_arguments_returns_error():
    out = _dispatch(_StubLab(), "job_status", {"wrong_kw": "x"})
    assert "error" in out and "job_status" in out["error"]


def test_dispatch_rejects_arxiv_only_tool_names():
    """Arxiv-radar's tool names must not silently pass through dispatch."""
    for name in ("search_abstract_text", "fetch_papers", "paper_info"):
        out = _dispatch(_StubLab(), name, {})
        assert "error" in out, f"{name} unexpectedly accepted"


# ----- LabCorpusServer end-to-end --------------------------------------------

def test_init_without_index_sets_corpus_index_none(lab_config):
    srv = LabCorpusServer(lab_config)
    try:
        assert srv.corpus_index is None
        assert srv.parse_dir == lab_config.parse.dir
        assert srv.index_dir == lab_config.embeddings.cache_dir
    finally:
        srv.jobs.shutdown()


class _FakeIndex:
    """Stand-in for EmbeddingIndex used to exercise the loaded-index path."""

    def __init__(self, *, n_chunks: int, n_papers: int, model_name: str):
        # Minimum surface LabCorpusServer touches: matrix.shape, model_name, metadata.
        class _M:
            shape = (n_chunks, 1024)
        self.matrix = _M()
        self.model_name = model_name
        self.metadata = {
            "chunks": [
                {"arxiv_id": f"paper-{i % n_papers:03d}", "chunk_idx": i}
                for i in range(n_chunks)
            ],
        }


def test_init_with_existing_index_loads_metadata(monkeypatch, lab_config):
    """When an EmbeddingIndex is on disk, LabCorpusServer should load it
    and corpus_stats should reflect its metadata. Mocks EmbeddingIndex.load
    to avoid building a real on-disk index."""
    fake = _FakeIndex(n_chunks=12, n_papers=4, model_name="test/dummy")
    monkeypatch.setattr(
        "lab_corpus_mcp.server.EmbeddingIndex.load",
        classmethod(lambda cls, _path: fake),
    )

    srv = LabCorpusServer(lab_config)
    try:
        assert srv.corpus_index is fake
        out = srv.corpus_stats()
        assert out["n_chunks"] == 12
        assert out["n_indexed"] == 4   # distinct arxiv_ids in metadata
    finally:
        srv.jobs.shutdown()


def test_corpus_stats_empty_corpus(lab_config):
    srv = LabCorpusServer(lab_config)
    try:
        out = srv.corpus_stats()
        assert out["n_parsed"] == 0
        assert out["n_indexed"] == 0
        assert out["n_chunks"] == 0
        assert out["embedding_model"] == lab_config.embeddings.model
        assert out["parse_dir"] == str(lab_config.parse.dir)
    finally:
        srv.jobs.shutdown()


def test_corpus_stats_counts_parsed_files(lab_config, populated_parse_dir):
    srv = LabCorpusServer(lab_config)
    try:
        out = srv.corpus_stats()
        assert out["n_parsed"] == 3
        # No index built → still 0 indexed/0 chunks.
        assert out["n_indexed"] == 0
        assert out["n_chunks"] == 0
    finally:
        srv.jobs.shutdown()


def test_list_corpus_empty(lab_config):
    srv = LabCorpusServer(lab_config)
    try:
        assert srv.list_corpus() == []
    finally:
        srv.jobs.shutdown()


def test_list_corpus_sorted_by_id(lab_config, populated_parse_dir):
    srv = LabCorpusServer(lab_config)
    try:
        ids = srv.list_corpus()
        # Lex-sorted: "doi-..." < "paper-001" < "paper-002"
        assert ids == ["doi-10.1000-zzz", "paper-001", "paper-002"]
    finally:
        srv.jobs.shutdown()


def test_list_corpus_respects_limit(lab_config, populated_parse_dir):
    srv = LabCorpusServer(lab_config)
    try:
        assert srv.list_corpus(limit=2) == ["doi-10.1000-zzz", "paper-001"]
        assert srv.list_corpus(limit=0) == []
    finally:
        srv.jobs.shutdown()


def test_list_corpus_via_dispatch(lab_config, populated_parse_dir):
    srv = LabCorpusServer(lab_config)
    try:
        out = _dispatch(srv, "list_corpus", {"limit": 1})
        assert out == ["doi-10.1000-zzz"]
    finally:
        srv.jobs.shutdown()


def test_corpus_stats_via_dispatch(lab_config, populated_parse_dir):
    srv = LabCorpusServer(lab_config)
    try:
        out = _dispatch(srv, "corpus_stats", {})
        assert out["n_parsed"] == 3
    finally:
        srv.jobs.shutdown()


def test_job_status_unknown_id(lab_config):
    srv = LabCorpusServer(lab_config)
    try:
        out = srv.job_status("nonexistent")
        assert "error" in out and "nonexistent" in out["error"]
    finally:
        srv.jobs.shutdown()


def test_job_list_empty(lab_config):
    srv = LabCorpusServer(lab_config)
    try:
        assert srv.job_list() == []
    finally:
        srv.jobs.shutdown()


def test_job_list_returns_recent_jobs(lab_config):
    """Submit a no-op job, wait briefly, verify it shows up in job_list."""
    srv = LabCorpusServer(lab_config)
    try:
        job_id = srv.jobs.submit(
            kind="smoke",
            fn=lambda h: {"ok": True},
            args={},
            n_total=1,
        )
        # Tiny poll loop (≤1 s) — the worker pool runs the lambda promptly.
        for _ in range(50):
            info = srv.job_status(job_id)
            if info.get("state") in ("done", "failed"):
                break
            time.sleep(0.02)

        info = srv.job_status(job_id)
        assert info["state"] == "done"
        assert info["result"] == {"ok": True}

        recent = srv.job_list(limit=10)
        assert any(j["job_id"] == job_id for j in recent)
    finally:
        srv.jobs.shutdown()


def test_job_list_respects_limit(lab_config):
    srv = LabCorpusServer(lab_config)
    try:
        for i in range(3):
            srv.jobs.submit(
                kind="smoke",
                fn=lambda h, i=i: {"i": i},
                n_total=1,
            )
        # Wait for everything to settle.
        for _ in range(50):
            recent = srv.job_list()
            if all(j.get("state") in ("done", "failed") for j in recent):
                break
            time.sleep(0.02)

        capped = srv.job_list(limit=2)
        assert len(capped) <= 2
    finally:
        srv.jobs.shutdown()


# ----- _build_mcp_app + background tasks -------------------------------------

def test_build_mcp_app_constructs_server(lab_config):
    srv = LabCorpusServer(lab_config)
    try:
        app = _build_mcp_app(srv)
        # We can't easily call the SDK handler without a session, but the
        # constructor returning an object guards against signature drift.
        assert app is not None
        assert len(LAB_TOOL_SPECS) == 4
    finally:
        srv.jobs.shutdown()


def test_lab_background_tasks_empty_for_skeleton(lab_config):
    """Phase 2A intentionally has no warm-up / refresh — verify so we
    catch surprise additions before the 2B encoder warm-up lands."""
    srv = LabCorpusServer(lab_config)
    try:
        assert _lab_background_tasks(srv) == []
    finally:
        srv.jobs.shutdown()
