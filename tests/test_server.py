"""Tool catalogue + dispatcher + LabCorpusServer tests.

Adapted from arxiv-radar-mcp/tests/test_server.py; covers Phase 2A
skeleton + Phase 2B-1 ingest + Phase 2B-2 search/index plumbing. The
heavy paths (real MinerU subprocess, real Encoder weights) are stubbed
out via fixtures so the suite stays fast and dependency-light.
"""
from __future__ import annotations

import inspect
import time
from pathlib import Path

import pytest

from lab_corpus_mcp.server import (
    LAB_TOOL_SPECS,
    LabCorpusServer,
    _build_mcp_app,
    _dispatch,
    _lab_background_tasks,
    _tool_names,
    _warmup_encoder,
)


# ----- LAB_TOOL_SPECS shape --------------------------------------------------

EXPECTED_TOOLS = {
    # Phase 2A skeleton + paper_info
    "corpus_stats", "list_corpus", "paper_info", "job_status", "job_list",
    # Phase 2B-1 ingest
    "ingest_pdf", "ingest_local_dir", "ingest_inbox",
    # Phase 2B+ U14 fetch-by-URL
    "ingest_url", "ingest_arxiv_pdf",
    # Phase 2B-2 index + search
    "rebuild_index", "search_paper_text", "search_paper_semantic", "similar_to_paper",
}


def test_tool_specs_cover_all_expected_tools():
    listed = {s["name"] for s in LAB_TOOL_SPECS}
    assert listed == EXPECTED_TOOLS, (
        f"missing: {EXPECTED_TOOLS - listed}, extra: {listed - EXPECTED_TOOLS}")


def test_tool_specs_match_method_signatures():
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
    listed = {s["name"] for s in LAB_TOOL_SPECS}
    forbidden = {
        "search_abstract_text", "search_abstract_semantic", "similar_to_abstract",
        "list_tags", "list_domains",
        "fetch_papers", "reindex", "refresh_abstracts", "validate_arxiv_ids",
        "list_enriched",
    }
    assert listed.isdisjoint(forbidden), (
        f"arxiv-radar tools resurrected: {listed & forbidden}")


def test_tool_names_helper_matches_specs():
    assert set(_tool_names()) == EXPECTED_TOOLS
    assert len(_tool_names()) == len(LAB_TOOL_SPECS) == 14


# ----- _dispatch -------------------------------------------------------------

class _StubLab:
    """Mimics the subset of LabCorpusServer surface that _dispatch routes to."""

    def corpus_stats(self):
        return {"called": "corpus_stats", "n_parsed": 0}

    def list_corpus(self, limit=None):
        return [{"paper_id": "x"}] if limit != 0 else []

    def paper_info(self, paper_id):
        return {"paper_id": paper_id}

    def job_status(self, job_id):
        return {"job_id": job_id}

    def job_list(self, limit=50):
        return [{"limit": limit}]

    def ingest_pdf(self, pdf_path, paper_id=None):
        return {"called": "ingest_pdf", "pdf": pdf_path, "id": paper_id}

    def ingest_local_dir(self, dir_path, glob="*.pdf", recursive=False):
        return {"called": "ingest_local_dir", "dir": dir_path, "glob": glob,
                "recursive": recursive}

    def ingest_inbox(self, glob="*.pdf", recursive=False, backend=None):
        return {"called": "ingest_inbox", "glob": glob, "recursive": recursive}

    def ingest_url(self, url, paper_id=None, backend=None):
        return {"called": "ingest_url", "url": url,
                "paper_id": paper_id, "backend": backend}

    def ingest_arxiv_pdf(self, arxiv_id, backend=None):
        return {"called": "ingest_arxiv_pdf", "arxiv_id": arxiv_id,
                "backend": backend}

    def rebuild_index(self, force_full=False):
        return {"called": "rebuild_index", "force_full": force_full}

    def search_paper_text(self, query, k=10, snippet_chars=240):
        return [{"q": query, "k": k}]

    def search_paper_semantic(self, query, k=10, snippet_chars=240):
        return [{"q": query, "k": k, "kind": "semantic"}]

    def similar_to_paper(self, paper_id, k=10):
        return [{"to": paper_id, "k": k}]


def test_dispatch_routes_to_method():
    out = _dispatch(_StubLab(), "corpus_stats", {})
    assert out["called"] == "corpus_stats"


def test_dispatch_handles_none_arguments():
    out = _dispatch(_StubLab(), "job_list", None)
    assert out == [{"limit": 50}]


def test_dispatch_passes_kwargs_through():
    out = _dispatch(_StubLab(), "ingest_local_dir",
                    {"dir_path": "/x", "glob": "*.pdf", "recursive": True})
    assert out == {"called": "ingest_local_dir", "dir": "/x",
                   "glob": "*.pdf", "recursive": True}


def test_dispatch_unknown_tool_returns_error():
    out = _dispatch(_StubLab(), "drop_database", {})
    assert "error" in out and "drop_database" in out["error"]


def test_dispatch_rejects_dunder_or_private_names():
    out = _dispatch(_StubLab(), "__init__", {})
    assert "error" in out


def test_dispatch_bad_arguments_returns_error():
    out = _dispatch(_StubLab(), "paper_info", {"wrong_kw": "x"})
    assert "error" in out and "paper_info" in out["error"]


def test_dispatch_rejects_arxiv_only_tool_names():
    for name in ("search_abstract_text", "fetch_papers", "list_domains"):
        out = _dispatch(_StubLab(), name, {})
        assert "error" in out, f"{name} unexpectedly accepted"


def test_dispatch_routes_to_search_methods():
    out = _dispatch(_StubLab(), "search_paper_semantic", {"query": "dft"})
    assert out == [{"q": "dft", "k": 10, "kind": "semantic"}]


# ----- LabCorpusServer end-to-end --------------------------------------------

def test_init_without_index_sets_fulltext_index_none(lab_config):
    srv = LabCorpusServer(lab_config)
    try:
        assert srv.fulltext_index is None
        assert srv.parse_dir == lab_config.parse.dir
        # index_dir == parse_dir — corpus_core.corpus_index.reindex
        # writes embeddings.npy + index.json at the same level as
        # `sources/`, not under the legacy embeddings.cache_dir.
        assert srv.index_dir == lab_config.parse.dir
        assert srv.papers == {}
    finally:
        srv.jobs.shutdown()


class _FakeIndex:
    """Stand-in for EmbeddingIndex used to exercise the loaded-index path."""

    def __init__(self, *, n_chunks: int, n_papers: int, model_name: str):
        class _M:
            shape = (n_chunks, 1024)
        self.matrix = _M()
        self.model_name = model_name
        self.dims = 1024
        self.metadata = {
            "n_papers": n_papers,
            "chunks": [
                {"arxiv_id": f"paper-{i % n_papers:03d}",
                 "section": f"sec-{i}",
                 "chunk_idx": i}
                for i in range(n_chunks)
            ],
        }


def test_init_with_existing_index_loads_metadata(monkeypatch, lab_config):
    fake = _FakeIndex(n_chunks=12, n_papers=4, model_name="test/dummy")
    monkeypatch.setattr(
        "lab_corpus_mcp.server.EmbeddingIndex.load",
        classmethod(lambda cls, _path: fake),
    )

    srv = LabCorpusServer(lab_config)
    try:
        assert srv.fulltext_index is fake
        out = srv.corpus_stats()
        assert out["n_chunks"] == 12
        assert out["n_indexed"] == 4
    finally:
        srv.jobs.shutdown()


def test_init_warns_on_model_mismatch(monkeypatch, lab_config, caplog):
    fake = _FakeIndex(n_chunks=2, n_papers=1, model_name="other/model")
    monkeypatch.setattr(
        "lab_corpus_mcp.server.EmbeddingIndex.load",
        classmethod(lambda cls, _path: fake),
    )
    with caplog.at_level("WARNING"):
        srv = LabCorpusServer(lab_config)
        try:
            assert any("force_full=true" in rec.message for rec in caplog.records)
        finally:
            srv.jobs.shutdown()


# ----- corpus_stats / list_corpus / paper_info -------------------------------

def test_corpus_stats_empty_corpus(lab_config):
    srv = LabCorpusServer(lab_config)
    try:
        out = srv.corpus_stats()
        assert out["n_parsed"] == 0
        assert out["n_indexed"] == 0
        assert out["last_ingest_at"] is None
        assert out["embedding_model"] == lab_config.embeddings.model
    finally:
        srv.jobs.shutdown()


def test_corpus_stats_counts_parsed_files(lab_config, populated_parse_dir):
    srv = LabCorpusServer(lab_config)
    try:
        out = srv.corpus_stats()
        assert out["n_parsed"] == 3
        assert out["n_indexed"] == 0
        assert out["last_ingest_at"] == "2026-05-09T10:02:00+00:00"
    finally:
        srv.jobs.shutdown()


def test_list_corpus_empty(lab_config):
    srv = LabCorpusServer(lab_config)
    try:
        assert srv.list_corpus() == []
    finally:
        srv.jobs.shutdown()


def test_list_corpus_newest_first_with_metadata(lab_config, populated_parse_dir):
    srv = LabCorpusServer(lab_config)
    try:
        rows = srv.list_corpus()
        ids = [r["paper_id"] for r in rows]
        # ingested_at desc → paper-002 (10:02) > paper-001 (10:01) > doi-... (10:00)
        assert ids == ["paper-002", "paper-001", "doi-10.1000-zzz"]
        assert all("title" in r and "source_kind" in r for r in rows)
    finally:
        srv.jobs.shutdown()


def test_list_corpus_respects_limit(lab_config, populated_parse_dir):
    srv = LabCorpusServer(lab_config)
    try:
        assert len(srv.list_corpus(limit=2)) == 2
        assert srv.list_corpus(limit=0) == []
    finally:
        srv.jobs.shutdown()


def test_paper_info_unknown(lab_config):
    srv = LabCorpusServer(lab_config)
    try:
        out = srv.paper_info("nonexistent")
        assert "error" in out and "nonexistent" in out["error"]
    finally:
        srv.jobs.shutdown()


def test_paper_info_returns_metadata(lab_config, populated_parse_dir):
    srv = LabCorpusServer(lab_config)
    try:
        out = srv.paper_info("paper-001")
        assert out["paper_id"] == "paper-001"
        assert out["title"] == "First"
        assert out["source_kind"] == "pdf"
        assert out["indexed"] is False  # no index built
    finally:
        srv.jobs.shutdown()


def test_paper_info_reflects_index_when_loaded(monkeypatch, lab_config, populated_parse_dir):
    fake = _FakeIndex(n_chunks=6, n_papers=3, model_name=lab_config.embeddings.model)
    # Wire the fake's chunks to use one of our populated paper ids.
    fake.metadata["chunks"] = [
        {"arxiv_id": "paper-001", "section": "Introduction", "chunk_idx": 0},
        {"arxiv_id": "paper-001", "section": "Methods", "chunk_idx": 1},
        {"arxiv_id": "paper-002", "section": "Results", "chunk_idx": 0},
    ]
    monkeypatch.setattr(
        "lab_corpus_mcp.server.EmbeddingIndex.load",
        classmethod(lambda cls, _path: fake),
    )
    srv = LabCorpusServer(lab_config)
    try:
        out = srv.paper_info("paper-001")
        assert out["indexed"] is True
        assert out["n_chunks"] == 2
    finally:
        srv.jobs.shutdown()


# ----- ingest_pdf / ingest_local_dir -----------------------------------------

def _wait_for_terminal(srv, job_id, timeout=2.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        info = srv.job_status(job_id)
        if info.get("state") in ("done", "failed"):
            return info
        time.sleep(0.02)
    return srv.job_status(job_id)


def test_ingest_pdf_missing_file(lab_config):
    srv = LabCorpusServer(lab_config)
    try:
        out = srv.ingest_pdf(pdf_path="C:/no/such/file.pdf")
        assert "error" in out and "not found" in out["error"]
    finally:
        srv.jobs.shutdown()


def test_ingest_pdf_runs_via_fake_runner(lab_config, fake_pdf, fake_mineru_runner):
    srv = LabCorpusServer(lab_config, mineru_runner=fake_mineru_runner)
    try:
        result = srv.ingest_pdf(str(fake_pdf))
        assert "job_id" in result
        info = _wait_for_terminal(srv, result["job_id"])
        assert info["state"] == "done", info
        assert info["result"]["paper_id"] == "2503.99999"   # arxiv_id derivation
        assert info["result"]["source_kind"] == "pdf"
        # paper now visible to corpus_stats
        assert srv.corpus_stats()["n_parsed"] == 1
        # markdown landed on disk
        out_md = lab_config.parse.dir / "sources" / "2503.99999.md"
        assert out_md.exists() and out_md.read_text(encoding="utf-8")
        # figures copied
        figures = lab_config.parse.dir / "figures" / "2503.99999"
        assert (figures / "fig1.png").exists()
    finally:
        srv.jobs.shutdown()


def test_ingest_pdf_with_explicit_paper_id(lab_config, fake_pdf, fake_mineru_runner):
    srv = LabCorpusServer(lab_config, mineru_runner=fake_mineru_runner)
    try:
        result = srv.ingest_pdf(str(fake_pdf), paper_id="my-custom-id")
        info = _wait_for_terminal(srv, result["job_id"])
        assert info["state"] == "done"
        assert info["result"]["paper_id"] == "my-custom-id"
    finally:
        srv.jobs.shutdown()


def test_ingest_local_dir_missing_path(lab_config):
    srv = LabCorpusServer(lab_config)
    try:
        out = srv.ingest_local_dir(dir_path="C:/no/such/dir")
        assert "error" in out and "not found" in out["error"]
    finally:
        srv.jobs.shutdown()


def test_ingest_local_dir_no_matches(lab_config, tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    srv = LabCorpusServer(lab_config)
    try:
        out = srv.ingest_local_dir(dir_path=str(empty_dir))
        assert "error" in out and "no files matching" in out["error"]
    finally:
        srv.jobs.shutdown()


def test_ingest_local_dir_bulk(lab_config, tmp_path, fake_mineru_runner):
    bulk = tmp_path / "bulk"
    bulk.mkdir()
    # Distinct contents → distinct sha256 → three independent paper_ids.
    for i in range(3):
        (bulk / f"paper-{i:03d}.pdf").write_bytes(f"%PDF-1.4\n{i}".encode())

    srv = LabCorpusServer(lab_config, mineru_runner=fake_mineru_runner)
    try:
        result = srv.ingest_local_dir(dir_path=str(bulk))
        assert result["n_total"] == 3
        info = _wait_for_terminal(srv, result["job_id"], timeout=4.0)
        assert info["state"] == "done", info
        assert info["result"]["n_ok"] == 3
        assert info["result"]["n_failed"] == 0
        # Three new papers picked up by the in-memory map.
        assert len(srv.papers) == 3
    finally:
        srv.jobs.shutdown()


# ----- ingest_inbox -----------------------------------------------------------

def test_ingest_inbox_dispatch_routes():
    out = _dispatch(_StubLab(), "ingest_inbox", {})
    assert out["called"] == "ingest_inbox"
    assert out["glob"] == "*.pdf"


def test_ingest_inbox_dispatch_with_args():
    out = _dispatch(_StubLab(), "ingest_inbox",
                    {"glob": "*.docx", "recursive": True})
    assert out == {"called": "ingest_inbox", "glob": "*.docx", "recursive": True}


def test_ingest_inbox_empty_inbox_returns_error(lab_config):
    # inbox dir does not exist yet — should be created, then return an error
    # because there are no files.
    srv = LabCorpusServer(lab_config)
    try:
        out = srv.ingest_inbox()
        assert "error" in out
        assert "no files" in out["error"].lower() or "copy files" in out["error"].lower()
        # Inbox dir was created as a side effect.
        assert (lab_config.parse.dir / "inbox").is_dir()
    finally:
        srv.jobs.shutdown()


def test_ingest_inbox_creates_inbox_dir_if_missing(lab_config):
    srv = LabCorpusServer(lab_config)
    try:
        assert not (lab_config.parse.dir / "inbox").exists()
        srv.ingest_inbox()  # empty — returns error, but dir should exist
        assert (lab_config.parse.dir / "inbox").is_dir()
    finally:
        srv.jobs.shutdown()


def test_ingest_inbox_submits_job_and_processes(lab_config, fake_mineru_runner):
    inbox = lab_config.parse.dir / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    for i in range(2):
        (inbox / f"paper-{i:02d}.pdf").write_bytes(f"%PDF-1.4\n{i}".encode())

    srv = LabCorpusServer(lab_config, mineru_runner=fake_mineru_runner)
    try:
        result = srv.ingest_inbox()
        assert "job_id" in result, result
        assert result["kind"] == "ingest_inbox"
        assert result["n_total"] == 2
        assert str(inbox) == result["inbox"]
        info = _wait_for_terminal(srv, result["job_id"], timeout=4.0)
        assert info["state"] == "done", info
        assert info["result"]["n_ok"] == 2
        assert info["result"]["n_failed"] == 0
    finally:
        srv.jobs.shutdown()


def test_ingest_inbox_glob_filters_files(lab_config, fake_mineru_runner):
    inbox = lab_config.parse.dir / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    (inbox / "doc.pdf").write_bytes(b"%PDF-1.4")
    (inbox / "slide.pptx").write_bytes(b"PK fake pptx")

    srv = LabCorpusServer(lab_config, mineru_runner=fake_mineru_runner)
    try:
        result = srv.ingest_inbox(glob="*.pptx")
        assert result["n_total"] == 1
    finally:
        srv.jobs.shutdown()


# ----- ingest_url / ingest_arxiv_pdf (U14) -----------------------------------

def _make_fake_fetcher(body: bytes = b"%PDF-1.4 fake remote bytes"):
    """Return (fetcher, captured) — fetcher writes `body` to dest_path,
    returns a successful FetchResult, and records its kwargs."""
    from corpus_core.http_fetch import FetchResult
    captured: dict = {}

    def _fetcher(url, dest_path, *, throttle, timeout_s):
        captured["url"] = url
        captured["dest_path"] = Path(dest_path)
        captured["throttle"] = throttle
        captured["timeout_s"] = timeout_s
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(body)
        return FetchResult(
            url=url, dest_path=Path(dest_path),
            ok=True, status=200, n_bytes=len(body), error=None,
        )

    return _fetcher, captured


def _patch_fetcher(monkeypatch, fetcher):
    """Replace `corpus_core.http_fetch.fetch_url` for the test."""
    monkeypatch.setattr("lab_corpus_mcp.ingest.fetch_url", fetcher)


def test_ingest_url_rejects_bad_scheme(lab_config):
    srv = LabCorpusServer(lab_config)
    try:
        out = srv.ingest_url(url="ftp://example.com/x.pdf")
        assert "error" in out and "invalid url" in out["error"]
    finally:
        srv.jobs.shutdown()


def test_ingest_url_rejects_empty_url(lab_config):
    srv = LabCorpusServer(lab_config)
    try:
        assert "error" in srv.ingest_url(url="")
    finally:
        srv.jobs.shutdown()


def test_ingest_url_happy_path(monkeypatch, lab_config, fake_mineru_runner):
    """Generic non-arxiv URL → no arxiv throttle, file lands in inbox,
    MinerU parses, paper persisted."""
    fetcher, captured = _make_fake_fetcher()
    _patch_fetcher(monkeypatch, fetcher)

    srv = LabCorpusServer(lab_config, mineru_runner=fake_mineru_runner)
    try:
        result = srv.ingest_url(url="https://example.com/preprint.pdf")
        assert "job_id" in result
        info = _wait_for_terminal(srv, result["job_id"])
        assert info["state"] == "done", info
        assert info["result"]["url"] == "https://example.com/preprint.pdf"
        # filename derived from URL last segment (has .pdf extension)
        assert captured["dest_path"].name == "preprint.pdf"
        # non-arxiv host → no rate limit
        assert captured["throttle"] is None
        # File landed in <parse.dir>/inbox/
        assert captured["dest_path"].parent == lab_config.parse.dir / "inbox"
        # And ingest_one then wrote a parsed markdown under sources/
        assert srv.corpus_stats()["n_parsed"] == 1
    finally:
        srv.jobs.shutdown()


def test_ingest_url_explicit_paper_id_dictates_filename(
    monkeypatch, lab_config, fake_mineru_runner,
):
    fetcher, captured = _make_fake_fetcher()
    _patch_fetcher(monkeypatch, fetcher)

    srv = LabCorpusServer(lab_config, mineru_runner=fake_mineru_runner)
    try:
        result = srv.ingest_url(
            url="https://example.com/messy/?id=42",
            paper_id="my-paper",
        )
        info = _wait_for_terminal(srv, result["job_id"])
        assert info["state"] == "done", info
        assert info["result"]["paper_id"] == "my-paper"
        # filename = `<paper_id>.pdf`
        assert captured["dest_path"].name == "my-paper.pdf"
    finally:
        srv.jobs.shutdown()


def test_ingest_url_fetch_failure_marks_job_failed(monkeypatch, lab_config):
    """Non-2xx response → IngestError → JobError → state == failed."""
    from corpus_core.http_fetch import FetchResult

    def _fetcher(url, dest_path, *, throttle, timeout_s):
        return FetchResult(
            url=url, dest_path=None,
            ok=False, status=404, n_bytes=0, error="http 404",
        )

    _patch_fetcher(monkeypatch, _fetcher)

    srv = LabCorpusServer(lab_config)
    try:
        result = srv.ingest_url(url="https://example.com/missing.pdf")
        info = _wait_for_terminal(srv, result["job_id"])
        assert info["state"] == "failed"
        assert "404" in info["error"]
    finally:
        srv.jobs.shutdown()


def test_ingest_arxiv_pdf_rejects_invalid_id(lab_config):
    srv = LabCorpusServer(lab_config)
    try:
        for bad in ("not-an-id", "1234", "12345.6789", ""):
            out = srv.ingest_arxiv_pdf(arxiv_id=bad)
            assert "error" in out, f"unexpectedly accepted: {bad!r}"
            assert "invalid arxiv_id" in out["error"]
    finally:
        srv.jobs.shutdown()


def test_ingest_arxiv_pdf_accepts_with_revision(lab_config):
    """Canonical `<id>` and `<id>vN` both validate."""
    from corpus_core.http_fetch import FetchResult

    def _fetcher(url, dest_path, *, throttle, timeout_s):
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(b"%PDF-1.4 dummy")
        return FetchResult(
            url=url, dest_path=Path(dest_path),
            ok=True, status=200, n_bytes=10, error=None,
        )

    import lab_corpus_mcp.ingest as ing
    ing_real = ing.fetch_url
    ing.fetch_url = _fetcher
    try:
        srv = LabCorpusServer(lab_config)
        try:
            out = srv.ingest_arxiv_pdf(arxiv_id="2512.14129v2")
            assert "job_id" in out
            assert out["arxiv_id"] == "2512.14129v2"
        finally:
            srv.jobs.shutdown()
    finally:
        ing.fetch_url = ing_real


def test_ingest_arxiv_pdf_uses_arxiv_throttle_and_url(
    monkeypatch, lab_config, fake_mineru_runner,
):
    """arxiv URL → arxiv throttle wired; filename = <id>.pdf;
    paper_id == arxiv_id."""
    from corpus_core.http_fetch import get_arxiv_throttle
    fetcher, captured = _make_fake_fetcher()
    _patch_fetcher(monkeypatch, fetcher)

    srv = LabCorpusServer(lab_config, mineru_runner=fake_mineru_runner)
    try:
        result = srv.ingest_arxiv_pdf(arxiv_id="2512.14129")
        assert result["arxiv_id"] == "2512.14129"
        assert result["kind"] == "ingest_arxiv_pdf"
        info = _wait_for_terminal(srv, result["job_id"])
        assert info["state"] == "done", info
        # paper_id forced to the arxiv id, not the sha256 fallback
        assert info["result"]["paper_id"] == "2512.14129"
        # URL composed correctly
        assert captured["url"] == "https://arxiv.org/pdf/2512.14129"
        # arxiv host → shared singleton throttle is wired in
        assert captured["throttle"] is get_arxiv_throttle()
        # Filename matches the paper_id rule
        assert captured["dest_path"].name == "2512.14129.pdf"
    finally:
        srv.jobs.shutdown()


# ----- rebuild_index ---------------------------------------------------------

def test_rebuild_index_no_papers(lab_config):
    srv = LabCorpusServer(lab_config)
    try:
        out = srv.rebuild_index()
        assert "error" in out and "ingest_pdf" in out["error"]
    finally:
        srv.jobs.shutdown()


def test_rebuild_index_lock_already_held(lab_config, populated_parse_dir):
    srv = LabCorpusServer(lab_config)
    try:
        # Steal the lock — second rebuild_index should refuse.
        assert srv.jobs.acquire_reindex_lock()
        out = srv.rebuild_index()
        assert "error" in out and "lockfile held" in out["error"]
    finally:
        srv.jobs.release_reindex_lock()
        srv.jobs.shutdown()


def test_rebuild_index_delegates_to_corpus_core(monkeypatch, lab_config,
                                                populated_parse_dir):
    """Stub `corpus_core.corpus_index.reindex` so we don't need a real Encoder."""
    captured = {}
    fake_index = _FakeIndex(n_chunks=4, n_papers=2,
                            model_name=lab_config.embeddings.model)

    def _fake_reindex(parse_dir, encoder, *, incremental, progress_cb=None):
        captured["parse_dir"] = parse_dir
        captured["incremental"] = incremental
        if progress_cb is not None:
            progress_cb(2, 3)
        return fake_index

    monkeypatch.setattr("lab_corpus_mcp.server.reindex", _fake_reindex)

    srv = LabCorpusServer(lab_config)
    try:
        out = srv.rebuild_index(force_full=True)
        assert "job_id" in out
        info = _wait_for_terminal(srv, out["job_id"])
        assert info["state"] == "done", info
        assert captured["incremental"] is False
        assert captured["parse_dir"] == lab_config.parse.dir
        assert srv.fulltext_index is fake_index
    finally:
        srv.jobs.shutdown()


# ----- search paths ----------------------------------------------------------

def test_search_paper_text_no_index(lab_config):
    srv = LabCorpusServer(lab_config)
    try:
        out = srv.search_paper_text("anything")
        assert isinstance(out, list) and "error" in out[0]
    finally:
        srv.jobs.shutdown()


def test_search_paper_semantic_no_index(lab_config):
    srv = LabCorpusServer(lab_config)
    try:
        out = srv.search_paper_semantic("anything")
        assert "error" in out[0]
    finally:
        srv.jobs.shutdown()


def test_similar_to_paper_no_index(lab_config):
    srv = LabCorpusServer(lab_config)
    try:
        out = srv.similar_to_paper("paper-001")
        assert "error" in out[0]
    finally:
        srv.jobs.shutdown()


def test_search_paper_text_via_corpus_core(monkeypatch, lab_config, populated_parse_dir):
    """Wire fulltext_index manually and stub corpus_core.search_paper_text."""
    fake = _FakeIndex(n_chunks=2, n_papers=1, model_name=lab_config.embeddings.model)
    captured: dict = {}

    def _fake_search(chunk_texts, chunk_meta, query, k=10, snippet_chars=240):
        captured["query"] = query
        captured["k"] = k
        captured["snippet_chars"] = snippet_chars
        return [{"hit": query}]

    monkeypatch.setattr("lab_corpus_mcp.server.load_chunk_texts",
                        lambda parse_dir, idx: ["t1", "t2"])
    monkeypatch.setattr("lab_corpus_mcp.server.search_paper_text", _fake_search)

    srv = LabCorpusServer(lab_config)
    try:
        srv.fulltext_index = fake
        out = srv.search_paper_text("dft", k=3, snippet_chars=120)
        assert out == [{"hit": "dft"}]
        assert captured == {"query": "dft", "k": 3, "snippet_chars": 120}
    finally:
        srv.jobs.shutdown()


def test_search_paper_semantic_via_corpus_core(monkeypatch, lab_config):
    fake = _FakeIndex(n_chunks=2, n_papers=1, model_name=lab_config.embeddings.model)

    monkeypatch.setattr("lab_corpus_mcp.server.load_chunk_texts",
                        lambda parse_dir, idx: ["t1"])
    monkeypatch.setattr("lab_corpus_mcp.server.search_paper_semantic",
                        lambda idx, texts, qvec, k=10, snippet_chars=240: [
                            {"score": float(qvec.sum()), "k": k}
                        ])

    srv = LabCorpusServer(lab_config)
    try:
        # Stub encoder so we don't load real Qwen weights.
        import numpy as np
        srv.encoder.encode_query = lambda q: np.array([0.5, 0.5], dtype=np.float32)
        srv.fulltext_index = fake
        out = srv.search_paper_semantic("query", k=2)
        assert out[0]["score"] == pytest.approx(1.0)
        assert out[0]["k"] == 2
    finally:
        srv.jobs.shutdown()


def test_similar_to_paper_via_corpus_core(monkeypatch, lab_config):
    fake = _FakeIndex(n_chunks=2, n_papers=1, model_name=lab_config.embeddings.model)
    monkeypatch.setattr("lab_corpus_mcp.server.similar_to_paper",
                        lambda idx, paper_id, k=10: [{"id": paper_id, "k": k}])

    srv = LabCorpusServer(lab_config)
    try:
        srv.fulltext_index = fake
        out = srv.similar_to_paper("paper-001", k=4)
        assert out == [{"id": "paper-001", "k": 4}]
    finally:
        srv.jobs.shutdown()


# ----- background tasks + warmup --------------------------------------------

def test_lab_background_tasks_includes_warmup(lab_config):
    srv = LabCorpusServer(lab_config)
    try:
        factories = _lab_background_tasks(srv)
        assert len(factories) == 1
    finally:
        srv.jobs.shutdown()


def test_warmup_encoder_logs_and_swallows_errors(lab_config, caplog):
    """Warm-up must not crash even when the encoder blows up — first
    real query will retry."""
    srv = LabCorpusServer(lab_config)
    try:
        def _boom(_q):
            raise RuntimeError("simulated cuda OOM")
        srv.encoder.encode_query = _boom

        import asyncio
        with caplog.at_level("WARNING"):
            asyncio.run(_warmup_encoder(srv))
        assert any("warm-up failed" in rec.message for rec in caplog.records)
    finally:
        srv.jobs.shutdown()


def test_warmup_encoder_happy_path(lab_config, caplog):
    """Successful warm-up logs 'ready' and returns cleanly."""
    srv = LabCorpusServer(lab_config)
    try:
        import numpy as np
        srv.encoder.encode_query = lambda q: np.zeros(4, dtype=np.float32)
        import asyncio
        with caplog.at_level("INFO"):
            asyncio.run(_warmup_encoder(srv))
        assert any("ready" in rec.message for rec in caplog.records)
    finally:
        srv.jobs.shutdown()


# ----- job worker error paths (LabCorpusServer end-to-end) -------------------

def test_job_status_real_server_unknown(lab_config):
    srv = LabCorpusServer(lab_config)
    try:
        out = srv.job_status("no-such-id")
        assert "error" in out and "no-such-id" in out["error"]
    finally:
        srv.jobs.shutdown()


def test_job_list_real_server_returns_recent(lab_config, fake_pdf, fake_mineru_runner):
    srv = LabCorpusServer(lab_config, mineru_runner=fake_mineru_runner)
    try:
        result = srv.ingest_pdf(str(fake_pdf))
        _wait_for_terminal(srv, result["job_id"])
        recent = srv.job_list(limit=10)
        assert any(j["job_id"] == result["job_id"] for j in recent)
    finally:
        srv.jobs.shutdown()


def test_ingest_pdf_runner_failure_marks_job_failed(lab_config, fake_pdf):
    """Runner raising IngestError should make _do_ingest_one re-raise as
    JobError so the registry records it as `failed`, not crashed."""
    from lab_corpus_mcp.ingest import IngestError

    def _boom(*a, **kw):
        raise IngestError("simulated parse failure")

    srv = LabCorpusServer(lab_config, mineru_runner=_boom)
    try:
        result = srv.ingest_pdf(str(fake_pdf))
        info = _wait_for_terminal(srv, result["job_id"])
        assert info["state"] == "failed"
        assert "simulated parse failure" in info["error"]
    finally:
        srv.jobs.shutdown()


def test_ingest_local_dir_runner_failure_marks_job_failed(lab_config, tmp_path):
    """`ingest_dir` raising IngestError must surface through _do_ingest_dir
    as a clean JobError (not a crash)."""
    from lab_corpus_mcp import ingest as ing

    bulk = tmp_path / "bulk"
    bulk.mkdir()
    (bulk / "x.pdf").write_bytes(b"%PDF-1.4")

    def _ingest_dir_raises(*a, **kw):
        raise ing.IngestError("ingest_dir blew up")

    # Patch the symbol that server.py imported.
    import lab_corpus_mcp.server as srv_mod

    srv = LabCorpusServer(lab_config)
    try:
        original = srv_mod.ingest_dir
        srv_mod.ingest_dir = _ingest_dir_raises
        try:
            result = srv.ingest_local_dir(str(bulk))
            info = _wait_for_terminal(srv, result["job_id"])
            assert info["state"] == "failed"
            assert "ingest_dir blew up" in info["error"]
        finally:
            srv_mod.ingest_dir = original
    finally:
        srv.jobs.shutdown()


# ----- VRAM release after heavy jobs -----------------------------------------

def test_release_gpu_vram_calls_encoder_and_mineru_unload(lab_config, monkeypatch):
    """The helper proxies to encoder.unload() AND unload_mineru_models()."""
    srv = LabCorpusServer(lab_config)
    try:
        calls = {"encoder": 0, "mineru": 0}

        def _fake_encoder_unload():
            calls["encoder"] += 1
            return True

        def _fake_mineru_unload():
            calls["mineru"] += 1
            return True

        srv.encoder.unload = _fake_encoder_unload  # type: ignore[method-assign]
        monkeypatch.setattr("lab_corpus_mcp.server.unload_mineru_models",
                            _fake_mineru_unload)
        srv._release_gpu_vram()
        assert calls == {"encoder": 1, "mineru": 1}
    finally:
        srv.jobs.shutdown()


def test_release_gpu_vram_swallows_encoder_exceptions(lab_config, caplog):
    srv = LabCorpusServer(lab_config)
    try:
        def _boom():
            raise RuntimeError("simulated cuda issue")
        srv.encoder.unload = _boom  # type: ignore[method-assign]
        with caplog.at_level("WARNING"):
            srv._release_gpu_vram()  # must not raise
        assert any("encoder.unload() failed" in rec.message
                   for rec in caplog.records)
    finally:
        srv.jobs.shutdown()


def test_release_gpu_vram_swallows_mineru_exceptions(lab_config, caplog, monkeypatch):
    srv = LabCorpusServer(lab_config)
    try:
        def _boom():
            raise RuntimeError("simulated mineru issue")
        monkeypatch.setattr("lab_corpus_mcp.server.unload_mineru_models", _boom)
        with caplog.at_level("WARNING"):
            srv._release_gpu_vram()  # must not raise
        assert any("unload_mineru_models() failed" in rec.message
                   for rec in caplog.records)
    finally:
        srv.jobs.shutdown()


def test_release_encoder_vram_alias_still_works(lab_config):
    """Back-compat alias for code/tests written before s153 rename."""
    srv = LabCorpusServer(lab_config)
    try:
        # The alias must be a real attribute, not a typo.
        assert srv._release_encoder_vram is not None
        # And calling it should not raise.
        srv._release_encoder_vram()
    finally:
        srv.jobs.shutdown()


def test_rebuild_index_unloads_encoder_on_success(monkeypatch, lab_config,
                                                   populated_parse_dir):
    """Successful reindex must release VRAM before returning."""
    fake_index = _FakeIndex(n_chunks=2, n_papers=1,
                            model_name=lab_config.embeddings.model)
    monkeypatch.setattr("lab_corpus_mcp.server.reindex",
                        lambda *a, **kw: fake_index)

    srv = LabCorpusServer(lab_config)
    try:
        calls = {"n": 0}
        srv.encoder.unload = lambda: (calls.__setitem__("n", calls["n"] + 1) or True)  # type: ignore[method-assign]
        result = srv.rebuild_index()
        _wait_for_terminal(srv, result["job_id"])
        assert calls["n"] == 1
    finally:
        srv.jobs.shutdown()


def test_rebuild_index_unloads_encoder_on_failure(monkeypatch, lab_config,
                                                   populated_parse_dir):
    """Even when reindex blows up, the finally block must release VRAM."""
    def _missing(*a, **kw):
        raise FileNotFoundError("phantom cache")
    monkeypatch.setattr("lab_corpus_mcp.server.reindex", _missing)

    srv = LabCorpusServer(lab_config)
    try:
        calls = {"n": 0}
        srv.encoder.unload = lambda: (calls.__setitem__("n", calls["n"] + 1) or True)  # type: ignore[method-assign]
        result = srv.rebuild_index()
        _wait_for_terminal(srv, result["job_id"])
        assert calls["n"] == 1
    finally:
        srv.jobs.shutdown()


def test_ingest_pdf_unloads_encoder_after_job(lab_config, fake_pdf,
                                               fake_mineru_runner):
    """ingest_pdf success path must hit _release_encoder_vram."""
    srv = LabCorpusServer(lab_config, mineru_runner=fake_mineru_runner)
    try:
        calls = {"n": 0}
        srv.encoder.unload = lambda: (calls.__setitem__("n", calls["n"] + 1) or True)  # type: ignore[method-assign]
        result = srv.ingest_pdf(str(fake_pdf))
        _wait_for_terminal(srv, result["job_id"])
        assert calls["n"] == 1
    finally:
        srv.jobs.shutdown()


def test_rebuild_index_filenotfound_becomes_joberror(monkeypatch, lab_config,
                                                     populated_parse_dir):
    """corpus_core.corpus_index.reindex raising FileNotFoundError must be
    converted to JobError so the registry sees `failed`, not crashed."""
    def _missing(*a, **kw):
        raise FileNotFoundError("phantom cache file")

    monkeypatch.setattr("lab_corpus_mcp.server.reindex", _missing)

    srv = LabCorpusServer(lab_config)
    try:
        result = srv.rebuild_index()
        info = _wait_for_terminal(srv, result["job_id"])
        assert info["state"] == "failed"
        assert "phantom cache file" in info["error"]
    finally:
        srv.jobs.shutdown()


# ----- _build_mcp_app --------------------------------------------------------

def test_build_mcp_app_constructs_server(lab_config):
    srv = LabCorpusServer(lab_config)
    try:
        app = _build_mcp_app(srv)
        assert app is not None
        assert len(LAB_TOOL_SPECS) == 14
    finally:
        srv.jobs.shutdown()


def test_dispatch_via_real_server(lab_config, populated_parse_dir):
    srv = LabCorpusServer(lab_config)
    try:
        out = _dispatch(srv, "list_corpus", {"limit": 1})
        assert isinstance(out, list) and len(out) == 1
        assert out[0]["paper_id"] == "paper-002"
    finally:
        srv.jobs.shutdown()
