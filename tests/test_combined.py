"""Combined-supervisor tests.

Cover the supervisor's contracts without booting a real Encoder or
HTTP transport:
  * `_LockedEncoder` proxies + serializes encode calls.
  * `_assert_models_agree` rejects model / target_dim mismatch.
  * `build_servers` wires the SAME encoder into both servers.
  * `serve_combined` (CLI integration) routes to `_serve_both` with
    correct host / port / lock-flag.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from lab_corpus_mcp.combined import (
    _LockedEncoder,
    _assert_models_agree,
    build_servers,
    serve_combined,
)


# ----- _LockedEncoder --------------------------------------------------------

class _RecordingEncoder:
    """Stand-in Encoder that records call order and overlaps."""

    def __init__(self):
        self.model_name = "test/dummy"
        self.in_flight = 0
        self.peak_in_flight = 0
        self.calls: list[str] = []
        self._call_lock = threading.Lock()

    def _enter(self, label: str) -> None:
        with self._call_lock:
            self.in_flight += 1
            self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
            self.calls.append(label)

    def _leave(self) -> None:
        with self._call_lock:
            self.in_flight -= 1

    def encode_query(self, text: str, max_seq_length: int = 512) -> np.ndarray:
        self._enter(f"q:{text}:{max_seq_length}")
        time.sleep(0.05)
        self._leave()
        return np.zeros(4, dtype=np.float32)

    def encode_passages(self, texts, show_progress=True,
                        max_seq_length=512, batch_size=None):
        self._enter(f"p:{len(texts)}:{batch_size}")
        time.sleep(0.05)
        self._leave()
        return np.zeros((len(texts), 4), dtype=np.float32)


def test_locked_encoder_proxies_model_name():
    inner = _RecordingEncoder()
    locked = _LockedEncoder(inner)
    assert locked.model_name == "test/dummy"
    assert locked.inner is inner


def test_locked_encoder_passes_args_through():
    inner = _RecordingEncoder()
    locked = _LockedEncoder(inner)

    locked.encode_query("dft", max_seq_length=128)
    locked.encode_passages(["a", "b"], batch_size=2, max_seq_length=256,
                           show_progress=False)
    assert inner.calls == ["q:dft:128", "p:2:2"]


def test_locked_encoder_serializes_concurrent_calls():
    """Two threads hitting encode_query simultaneously must NOT overlap."""
    inner = _RecordingEncoder()
    locked = _LockedEncoder(inner)

    def _worker(label):
        locked.encode_query(label)

    threads = [threading.Thread(target=_worker, args=(f"q{i}",)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert inner.peak_in_flight == 1, (
        f"expected 1 in-flight at a time, peaked at {inner.peak_in_flight}")
    # All four calls executed.
    assert len(inner.calls) == 4


def test_locked_encoder_attribute_passthrough():
    inner = _RecordingEncoder()
    inner.custom_attr = "yes"  # type: ignore[attr-defined]
    locked = _LockedEncoder(inner)
    # __getattr__ proxies anything not explicitly mirrored.
    assert locked.custom_attr == "yes"


def test_locked_encoder_returns_inner_result():
    """encode_* must return whatever the inner encoder returned."""
    inner = _RecordingEncoder()
    locked = _LockedEncoder(inner)
    out = locked.encode_query("x")
    assert isinstance(out, np.ndarray)
    assert out.shape == (4,)


def test_locked_encoder_unload_proxies_to_inner():
    """unload() must call through to the wrapped encoder."""
    class _UnloadableInner(_RecordingEncoder):
        def __init__(self):
            super().__init__()
            self.unload_calls = 0

        def unload(self) -> bool:
            self.unload_calls += 1
            return True

    inner = _UnloadableInner()
    locked = _LockedEncoder(inner)
    assert locked.unload() is True
    assert inner.unload_calls == 1


def test_locked_encoder_unload_serializes_with_encode():
    """A pending encode must complete before unload runs (same lock).

    We start an encode_query in a worker thread, then on the main thread
    call unload(). The unload must wait for the encode to finish — once
    it returns, the inner.unload_calls counter should be 1 AND the
    encode should already have left in_flight (i.e. no overlap).
    """
    overlap_seen = {"flag": False}

    class _SlowUnloadable(_RecordingEncoder):
        def __init__(self):
            super().__init__()
            self.unload_calls = 0

        def encode_query(self, text, max_seq_length=512):
            self._enter(f"q:{text}")
            # Hold the lock long enough for the unload thread to queue.
            time.sleep(0.1)
            self._leave()
            return np.zeros(4, dtype=np.float32)

        def unload(self) -> bool:
            # If we were called while an encode was in flight that's a
            # serialization bug — record it.
            if self.in_flight > 0:
                overlap_seen["flag"] = True
            self.unload_calls += 1
            return True

    inner = _SlowUnloadable()
    locked = _LockedEncoder(inner)

    encoder_thread = threading.Thread(target=lambda: locked.encode_query("x"))
    encoder_thread.start()
    # Give the encode time to start so unload() actually has to queue.
    time.sleep(0.02)
    locked.unload()
    encoder_thread.join()

    assert inner.unload_calls == 1
    assert overlap_seen["flag"] is False, (
        "unload ran while encode was still in flight — lock not held")


# ----- _assert_models_agree --------------------------------------------------

class _FakeEmbeddingsCfg:
    def __init__(self, model: str, target_dim: int | None = None,
                 cache_dir: Path | None = None):
        self.model = model
        self.target_dim = target_dim
        self.cache_dir = cache_dir or Path("/fake/cache")


class _FakeCfg:
    def __init__(self, model: str, target_dim: int | None = None):
        self.embeddings = _FakeEmbeddingsCfg(model, target_dim)


def test_assert_models_agree_passes_when_equal():
    a = _FakeCfg("Qwen/Qwen3-Embedding-4B")
    b = _FakeCfg("Qwen/Qwen3-Embedding-4B")
    _assert_models_agree(a, b)  # no exception


def test_assert_models_agree_rejects_model_mismatch():
    a = _FakeCfg("Qwen/Qwen3-Embedding-4B")
    b = _FakeCfg("BAAI/bge-large-en-v1.5")
    with pytest.raises(ValueError, match="embedding models to match"):
        _assert_models_agree(a, b)


def test_assert_models_agree_rejects_target_dim_mismatch():
    a = _FakeCfg("Qwen/Qwen3-Embedding-4B", target_dim=1024)
    b = _FakeCfg("Qwen/Qwen3-Embedding-4B", target_dim=2560)
    with pytest.raises(ValueError, match="target_dim to match"):
        _assert_models_agree(a, b)


# ----- build_servers + serve_combined ---------------------------------------

@pytest.fixture
def shared_combined_env(monkeypatch, tmp_path):
    """Patch arxiv_load + lab_load + Encoder + RadarServer + LabCorpusServer
    so build_servers can run on this Windows host without Qwen weights."""
    arxiv_cfg = _FakeCfg("Qwen/Qwen3-Embedding-4B")
    lab_cfg = _FakeCfg("Qwen/Qwen3-Embedding-4B")

    captured: dict[str, Any] = {}

    def _fake_arxiv_load(_path):
        captured["arxiv_path"] = _path
        return arxiv_cfg

    def _fake_lab_load(_path):
        captured["lab_path"] = _path
        return lab_cfg

    fake_real_encoder_calls: list[Any] = []

    class _FakeRealEncoder:
        def __init__(self, cfg):
            fake_real_encoder_calls.append(cfg)
            self.config = cfg

        @property
        def model_name(self):
            return self.config.embeddings.model

    class _FakeRadarServer:
        def __init__(self, cfg, *, encoder=None):
            captured["radar_cfg"] = cfg
            captured["radar_encoder"] = encoder
            self.config = cfg

    class _FakeLabServer:
        def __init__(self, cfg, *, encoder=None, mineru_runner=None):
            captured["lab_cfg"] = cfg
            captured["lab_encoder"] = encoder
            self.config = cfg
            self.parse_dir = Path("/fake/parse")

    # build_servers does its imports lazily inside the function body so
    # single-server callers don't pay the arxiv-radar import cost.
    # That means we must patch the *source* modules (where the names
    # actually live), not lab_corpus_mcp.combined.
    monkeypatch.setattr("arxiv_radar_mcp.config.load", _fake_arxiv_load)
    monkeypatch.setattr("arxiv_radar_mcp.server.RadarServer", _FakeRadarServer)
    monkeypatch.setattr("lab_corpus_mcp.config.load", _fake_lab_load)
    monkeypatch.setattr("lab_corpus_mcp.server.LabCorpusServer", _FakeLabServer)
    monkeypatch.setattr("lab_corpus_mcp.combined.Encoder", _FakeRealEncoder)
    return captured


def test_build_servers_shares_encoder(shared_combined_env):
    radar, lab, encoder = build_servers(
        Path("/x/arxiv.toml"), Path("/x/lab.toml"), encoder_lock=True,
    )
    assert isinstance(encoder, _LockedEncoder)
    assert shared_combined_env["radar_encoder"] is encoder
    assert shared_combined_env["lab_encoder"] is encoder
    assert encoder.model_name == "Qwen/Qwen3-Embedding-4B"


def test_build_servers_without_lock(shared_combined_env):
    radar, lab, encoder = build_servers(
        Path("/x/arxiv.toml"), Path("/x/lab.toml"), encoder_lock=False,
    )
    # Raw _FakeRealEncoder, not the wrapper.
    assert not isinstance(encoder, _LockedEncoder)
    assert shared_combined_env["radar_encoder"] is encoder
    assert shared_combined_env["lab_encoder"] is encoder


def test_build_servers_passes_config_paths(shared_combined_env):
    build_servers(Path("/a/x.toml"), Path("/b/y.toml"))
    assert shared_combined_env["arxiv_path"] == Path("/a/x.toml")
    assert shared_combined_env["lab_path"] == Path("/b/y.toml")


def test_build_servers_rejects_model_mismatch(monkeypatch):
    """If the two configs disagree, supervisor refuses cleanly."""
    monkeypatch.setattr("arxiv_radar_mcp.config.load",
                        lambda _p: _FakeCfg("Qwen/Qwen3-Embedding-4B"))
    monkeypatch.setattr("lab_corpus_mcp.config.load",
                        lambda _p: _FakeCfg("BAAI/bge-large-en-v1.5"))

    with pytest.raises(ValueError, match="embedding models to match"):
        build_servers(None, None)


def test_serve_combined_routes_to_serve_both(monkeypatch, shared_combined_env):
    """End-to-end: serve_combined should call _serve_both with the right args."""
    captured = {}

    def _fake_serve_both(radar, lab, *, host, arxiv_port, lab_port):
        captured["host"] = host
        captured["arxiv_port"] = arxiv_port
        captured["lab_port"] = lab_port
        captured["radar"] = radar
        captured["lab"] = lab

    async def _fake_async(radar, lab, *, host, arxiv_port, lab_port):
        _fake_serve_both(radar, lab, host=host,
                         arxiv_port=arxiv_port, lab_port=lab_port)

    monkeypatch.setattr("lab_corpus_mcp.combined._serve_both", _fake_async)

    serve_combined(
        arxiv_config_path=Path("/a"), lab_config_path=Path("/b"),
        host="127.0.0.1", arxiv_port=9000, lab_port=9001,
        encoder_lock=True,
    )
    assert captured == {
        "host": "127.0.0.1",
        "arxiv_port": 9000,
        "lab_port": 9001,
        "radar": captured["radar"],   # constructed by build_servers
        "lab": captured["lab"],
    }
