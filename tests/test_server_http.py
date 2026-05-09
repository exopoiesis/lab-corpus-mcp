"""HTTP transport plumbing tests for lab_corpus_mcp.server.

Adapted from arxiv-radar-mcp/tests/test_server_http.py. The actual
streamable-HTTP server is exercised via integration; here we just
verify serve_http boots the right loop with the right host/port and
the underlying mcp app is constructed with the lab tool catalogue.
"""
from __future__ import annotations

from pathlib import Path


def test_build_mcp_app_uses_lab_tool_specs(lab_config):
    from lab_corpus_mcp.server import LAB_TOOL_SPECS, LabCorpusServer, _build_mcp_app

    srv = LabCorpusServer(lab_config)
    try:
        app = _build_mcp_app(srv)
        assert app is not None
        # 11 tools after Phase 2B: 5 stats/admin + 2 ingest + 4 index/search.
        assert len(LAB_TOOL_SPECS) == 11
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
