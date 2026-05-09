"""CLI-level argparse tests for lab_corpus_mcp.__main__.

Adapted from arxiv-radar-mcp/tests/test_cli.py. Covers transport
selection, mutual exclusion, default mode dispatch, custom port. We
don't actually start any server — entry points are monkeypatched and
we check what got called with what args.
"""
from __future__ import annotations

import sys

import pytest

from lab_corpus_mcp import __main__ as cli


def _run(monkeypatch, argv, **patches):
    monkeypatch.setattr(sys, "argv", ["lab-corpus-mcp", *argv])
    record: dict = {}

    def _make_recorder(name):
        def _rec(*args, **kwargs):
            record[name] = {"args": args, "kwargs": kwargs}
            return 0 if name == "run_proxy" else None
        return _rec

    monkeypatch.setattr("lab_corpus_mcp.server.serve",
                        _make_recorder("serve"))
    monkeypatch.setattr("lab_corpus_mcp.server.serve_http",
                        _make_recorder("serve_http"))
    monkeypatch.setattr("corpus_core.proxy.run_proxy",
                        _make_recorder("run_proxy"))

    rc = cli.main()
    return rc, record


def test_default_mode_runs_stdio_server(monkeypatch):
    rc, rec = _run(monkeypatch, [])
    assert rc == 0
    assert "serve" in rec
    assert "serve_http" not in rec
    assert "run_proxy" not in rec


def test_transport_http_runs_http_server(monkeypatch):
    rc, rec = _run(monkeypatch, ["--transport", "http"])
    assert rc == 0
    assert "serve_http" in rec
    assert rec["serve_http"]["kwargs"]["host"] == "127.0.0.1"
    # Lab default differs from arxiv-radar (8765) so two backends co-exist.
    assert rec["serve_http"]["kwargs"]["port"] == 8766


def test_transport_http_custom_bind_and_port(monkeypatch):
    rc, rec = _run(monkeypatch, [
        "--transport", "http", "--bind", "0.0.0.0", "--port", "9100",
    ])
    assert rc == 0
    assert rec["serve_http"]["kwargs"]["host"] == "0.0.0.0"
    assert rec["serve_http"]["kwargs"]["port"] == 9100


def test_remote_mode_runs_proxy(monkeypatch):
    rc, rec = _run(monkeypatch, ["--remote", "user@gomer"])
    assert rc == 0
    assert "run_proxy" in rec
    assert rec["run_proxy"]["kwargs"]["target"] == "user@gomer"
    assert rec["run_proxy"]["kwargs"]["remote_port"] == 8766


def test_remote_with_custom_port(monkeypatch):
    rc, rec = _run(monkeypatch, [
        "--remote", "user@gomer", "--remote-port", "9999",
    ])
    assert rc == 0
    assert rec["run_proxy"]["kwargs"]["remote_port"] == 9999


def test_remote_and_transport_http_are_mutually_exclusive(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "lab-corpus-mcp", "--remote", "user@gomer", "--transport", "http",
    ])
    with pytest.raises(SystemExit):
        cli.main()


def test_unknown_log_level_rejected(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "lab-corpus-mcp", "--log-level", "INVALID",
    ])
    with pytest.raises(SystemExit):
        cli.main()
