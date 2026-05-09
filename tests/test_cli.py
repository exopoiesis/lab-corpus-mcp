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
    monkeypatch.setattr("lab_corpus_mcp.combined.serve_combined",
                        _make_recorder("serve_combined"))

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


# ----- combined mode --------------------------------------------------------

def test_combined_mode_routes_to_supervisor(monkeypatch):
    rc, rec = _run(monkeypatch, ["--mode", "combined"])
    assert rc == 0
    assert "serve_combined" in rec
    kw = rec["serve_combined"]["kwargs"]
    # 0.0.0.0 default in combined mode (auto-override of 127.0.0.1).
    assert kw["host"] == "0.0.0.0"
    assert kw["arxiv_port"] == 8765
    assert kw["lab_port"] == 8766
    assert kw["encoder_lock"] is True


def test_combined_mode_custom_ports_and_configs(monkeypatch, tmp_path):
    arxiv_cfg = tmp_path / "a.toml"
    lab_cfg = tmp_path / "l.toml"
    rc, rec = _run(monkeypatch, [
        "--mode", "combined",
        "--arxiv-config", str(arxiv_cfg),
        "--lab-config", str(lab_cfg),
        "--arxiv-port", "9101",
        "--lab-port", "9102",
        "--bind", "10.0.0.5",
    ])
    assert rc == 0
    kw = rec["serve_combined"]["kwargs"]
    assert kw["arxiv_config_path"] == arxiv_cfg
    assert kw["lab_config_path"] == lab_cfg
    assert kw["arxiv_port"] == 9101
    assert kw["lab_port"] == 9102
    assert kw["host"] == "10.0.0.5"


def test_combined_mode_no_encoder_lock(monkeypatch):
    rc, rec = _run(monkeypatch, ["--mode", "combined", "--no-encoder-lock"])
    assert rc == 0
    assert rec["serve_combined"]["kwargs"]["encoder_lock"] is False


def test_combined_mode_rejects_remote(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "lab-corpus-mcp", "--mode", "combined", "--remote", "user@host",
    ])
    with pytest.raises(SystemExit):
        cli.main()
