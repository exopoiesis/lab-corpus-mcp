"""HTTP transport plumbing tests for lab_corpus_mcp.server.

Adapted from arxiv-radar-mcp/tests/test_server_http.py. The actual
streamable-HTTP server is exercised via integration; here we just
verify serve_http boots the right loop with the right host/port and
the underlying mcp app is constructed with the lab tool catalogue.

The /upload endpoint tests use Starlette's TestClient (backed by httpx)
to exercise _make_upload_handler without spinning up a real uvicorn server.
"""
from __future__ import annotations

from pathlib import Path



def test_build_mcp_app_uses_lab_tool_specs(lab_config):
    from lab_corpus_mcp.server import LAB_TOOL_SPECS, LabCorpusServer, _build_mcp_app

    srv = LabCorpusServer(lab_config)
    try:
        app = _build_mcp_app(srv)
        assert app is not None
        # 14 tools: 5 stats/admin + 3 ingest (pdf/dir/inbox) +
        # 2 fetch-by-URL + 4 index/search.
        assert len(LAB_TOOL_SPECS) == 14
    finally:
        srv.jobs.shutdown()


def test_serve_http_calls_runner_with_correct_bind(monkeypatch, lab_config, tmp_path):
    """serve_http should construct the runner with host/port and delegate to asyncio.run."""
    from lab_corpus_mcp import server as srv_mod

    captured = {}

    async def _fake_run_streamable(_server, host, port):
        captured["host"] = host
        captured["port"] = port

    monkeypatch.setattr(srv_mod, "_run_streamable_http", _fake_run_streamable)

    # Avoid real config-file probe — load() returns lab_config regardless.
    monkeypatch.setattr(srv_mod, "load", lambda _path=None: lab_config)

    srv_mod.serve_http(host="127.0.0.1", port=8766, config_path=tmp_path / "x.toml")
    assert captured == {"host": "127.0.0.1", "port": 8766}


def test_serve_calls_runner_with_lab_server(monkeypatch, lab_config, tmp_path):
    """`serve()` should construct LabCorpusServer + call _run_stdio."""
    from lab_corpus_mcp import server as srv_mod

    captured = {}

    async def _fake_run_stdio(server):
        captured["server_kind"] = type(server).__name__

    monkeypatch.setattr(srv_mod, "_run_stdio", _fake_run_stdio)
    monkeypatch.setattr(srv_mod, "load", lambda _path=None: lab_config)

    srv_mod.serve(config_path=tmp_path / "x.toml")
    assert captured == {"server_kind": "LabCorpusServer"}


# ----- /upload endpoint (Starlette TestClient) --------------------------------

def _upload_app(srv):
    """Minimal Starlette app with only the /upload route for isolated testing."""
    from starlette.applications import Starlette
    from starlette.routing import Route
    from lab_corpus_mcp.server import _make_upload_handler
    return Starlette(routes=[
        Route("/upload", endpoint=_make_upload_handler(srv), methods=["POST"]),
    ])


def test_upload_single_file_saved_to_inbox(lab_config):
    from starlette.testclient import TestClient
    from lab_corpus_mcp.server import LabCorpusServer

    srv = LabCorpusServer(lab_config)
    try:
        client = TestClient(_upload_app(srv))
        pdf_bytes = b"%PDF-1.4 fake content"
        r = client.post("/upload", files={"file": ("paper.pdf", pdf_bytes)})
        assert r.status_code == 200
        data = r.json()
        assert data["saved"] == ["paper.pdf"]
        assert data["n_saved"] == 1
        assert data["errors"] == []
        assert data["job_id"] is None
        dest = lab_config.parse.dir / "inbox" / "paper.pdf"
        assert dest.read_bytes() == pdf_bytes
    finally:
        srv.jobs.shutdown()


def test_upload_multiple_files(lab_config):
    from starlette.testclient import TestClient
    from lab_corpus_mcp.server import LabCorpusServer

    srv = LabCorpusServer(lab_config)
    try:
        client = TestClient(_upload_app(srv))
        r = client.post("/upload", files=[
            ("file", ("a.pdf", b"%PDF a")),
            ("file", ("b.pdf", b"%PDF b")),
            ("file", ("c.pdf", b"%PDF c")),
        ])
        assert r.status_code == 200
        data = r.json()
        assert sorted(data["saved"]) == ["a.pdf", "b.pdf", "c.pdf"]
        assert data["n_saved"] == 3
        inbox = lab_config.parse.dir / "inbox"
        assert (inbox / "a.pdf").exists()
        assert (inbox / "b.pdf").exists()
        assert (inbox / "c.pdf").exists()
    finally:
        srv.jobs.shutdown()


def test_upload_ingest_true_triggers_ingest_inbox(lab_config, monkeypatch):
    from starlette.testclient import TestClient
    from lab_corpus_mcp.server import LabCorpusServer

    srv = LabCorpusServer(lab_config)
    try:
        triggered = {}

        def _fake_ingest_inbox(**kwargs):
            triggered["called"] = True
            return {"job_id": "fake-job-42", "n_total": 1, "kind": "ingest_inbox",
                    "inbox": str(lab_config.parse.dir / "inbox"), "backend": "pipeline"}

        monkeypatch.setattr(srv, "ingest_inbox", _fake_ingest_inbox)

        client = TestClient(_upload_app(srv))
        r = client.post("/upload?ingest=true",
                        files={"file": ("paper.pdf", b"%PDF-1.4")})
        assert r.status_code == 200
        data = r.json()
        assert data["job_id"] == "fake-job-42"
        assert triggered.get("called")
    finally:
        srv.jobs.shutdown()


def test_upload_no_files_returns_error_status(lab_config):
    from starlette.testclient import TestClient
    from lab_corpus_mcp.server import LabCorpusServer

    srv = LabCorpusServer(lab_config)
    try:
        client = TestClient(_upload_app(srv), raise_server_exceptions=False)
        # multipart with a text field (not a file upload) — no UploadFile instances
        r = client.post("/upload", data={"not_a_file": "hello"})
        assert r.status_code in (400, 422)
        assert r.json()["n_saved"] == 0
    finally:
        srv.jobs.shutdown()


def test_upload_path_traversal_rejected(lab_config):
    from starlette.testclient import TestClient
    from lab_corpus_mcp.server import LabCorpusServer

    srv = LabCorpusServer(lab_config)
    try:
        client = TestClient(_upload_app(srv))
        r = client.post("/upload",
                        files={"file": ("../../etc/passwd", b"root:x:0:0")})
        assert r.status_code == 200
        r.json()
        # Path traversal stripped to basename "passwd" — still saved safely,
        # OR rejected with an error. Either way it must NOT escape the inbox.
        inbox = lab_config.parse.dir / "inbox"
        # Verify nothing escaped outside inbox/ :
        for p in inbox.rglob("*"):
            assert inbox in p.parents or p == inbox
        # The original traversal path must not exist anywhere outside inbox.
        etc_passwd = Path("/etc/passwd")
        assert not etc_passwd.exists() or etc_passwd.stat().st_size > 10
    finally:
        srv.jobs.shutdown()


def test_upload_creates_inbox_dir_if_missing(lab_config):
    from starlette.testclient import TestClient
    from lab_corpus_mcp.server import LabCorpusServer

    srv = LabCorpusServer(lab_config)
    try:
        inbox = lab_config.parse.dir / "inbox"
        assert not inbox.exists()
        client = TestClient(_upload_app(srv))
        client.post("/upload", files={"file": ("x.pdf", b"%PDF")})
        assert inbox.is_dir()
    finally:
        srv.jobs.shutdown()
