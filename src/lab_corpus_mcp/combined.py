"""Combined arxiv-radar + lab-corpus MCP supervisor — one container,
two HTTP backends, one Qwen3 model in VRAM.

Each backend owns its own corpus, config, JobRegistry and tool surface.
What they share is the heavyweight bi-encoder: ~8 GB of Qwen3-4B
weights in bf16 won't fit twice on a 12 GB RTX 4070, so the supervisor
constructs ONE `corpus_core.embeddings.Encoder` and hands the same
reference to both `RadarServer` and `LabCorpusServer`.

A `_LockedEncoder` wrapper serializes encode calls with a
`threading.Lock`, guaranteeing peak VRAM stays at "weights + one
batch's activations" instead of doubling when both reindexes happen
to overlap. The lock is enabled by default; disable with
`encoder_lock=False` if you have headroom and want concurrency.

This module imports `arxiv_radar_mcp.server` lazily so single-server
deployments of lab-corpus-mcp don't pull in the arxiv-radar shell at
import time.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np

from corpus_core.embeddings import Encoder

LOG = logging.getLogger(__name__)


class _LockedEncoder:
    """Thread-safe wrapper around `corpus_core.embeddings.Encoder`.

    Both servers run their encoder calls on `asyncio.to_thread`. With
    the default CUDA stream they'd queue at the driver level anyway,
    but the activation tensors for two in-flight forward passes can
    co-exist in VRAM briefly — enough to OOM a 12 GB card on Qwen3-4B
    + bf16 + max_seq_length=8192. A Python-level lock pulls the
    serialization point earlier and keeps peak VRAM = weights + one
    activation set ≈ 10 GB.

    Encode methods are mirrored explicitly so type-checkers see them.
    Other Encoder attributes (e.g. `model_name`, `config`) tunnel
    through `__getattr__`.
    """

    def __init__(self, inner: Encoder) -> None:
        self._inner = inner
        self._lock = threading.Lock()

    @property
    def inner(self) -> Encoder:
        """Underlying Encoder. Useful for tests + introspection."""
        return self._inner

    @property
    def model_name(self) -> str:
        return self._inner.model_name

    def encode_query(self, text: str, max_seq_length: int = 512) -> np.ndarray:
        with self._lock:
            return self._inner.encode_query(text, max_seq_length=max_seq_length)

    def encode_passages(
        self,
        texts: list[str],
        show_progress: bool = True,
        max_seq_length: int = 512,
        batch_size: int | None = None,
    ) -> np.ndarray:
        with self._lock:
            return self._inner.encode_passages(
                texts,
                show_progress=show_progress,
                max_seq_length=max_seq_length,
                batch_size=batch_size,
            )

    def unload(self) -> bool:
        """Drop the shared inner Encoder's model + free VRAM.

        Taken from the same lock as `encode_*` so a concurrent encode
        either runs first or, after this call, triggers a lazy re-load.
        Use after rebuild_index / bulk ingest finishes when the host's
        GPU is also being used for unrelated compute.
        """
        with self._lock:
            return self._inner.unload()

    def __getattr__(self, name: str) -> Any:
        # Last-resort attribute proxy for anything we forgot to mirror
        # (e.g. `config`). Bypassed when the attribute is set directly
        # on the wrapper itself; keep the wrapper's own surface tight
        # so this fallback rarely fires in practice.
        return getattr(self._inner, name)


def _assert_models_agree(arxiv_cfg: Any, lab_cfg: Any) -> None:
    """Both configs must agree on the embedding model — they share
    one in-memory copy of it."""
    arxiv_model = arxiv_cfg.embeddings.model
    lab_model = lab_cfg.embeddings.model
    if arxiv_model != lab_model:
        raise ValueError(
            f"combined supervisor requires both servers' embedding "
            f"models to match (one Qwen instance is shared). "
            f"arxiv-radar wants {arxiv_model!r}, lab-corpus wants "
            f"{lab_model!r}. Align the [embeddings] sections in both "
            f"radar.toml files and retry."
        )
    arxiv_dim = arxiv_cfg.embeddings.target_dim
    lab_dim = lab_cfg.embeddings.target_dim
    if arxiv_dim != lab_dim:
        raise ValueError(
            f"combined supervisor requires both servers' target_dim "
            f"to match (matryoshka truncation is per-encoder). "
            f"arxiv-radar wants {arxiv_dim!r}, lab-corpus wants "
            f"{lab_dim!r}."
        )


def build_servers(
    arxiv_config_path: Path | None,
    lab_config_path: Path | None,
    *,
    encoder_lock: bool = True,
):
    """Construct the shared encoder + both servers without starting any
    transport. Useful for tests and for callers that want to drive the
    asyncio loop themselves.

    Returns `(radar, lab, encoder)` — `encoder` is the wrapper if
    `encoder_lock=True`, else the raw Encoder.
    """
    # Lazy imports so the lab-corpus-mcp package can be imported on a
    # machine that doesn't have arxiv-radar-mcp installed (single-server
    # deployments). Combined mode obviously requires both.
    from arxiv_radar_mcp.config import load as arxiv_load
    from arxiv_radar_mcp.server import RadarServer

    from lab_corpus_mcp.config import load as lab_load
    from lab_corpus_mcp.server import LabCorpusServer

    arxiv_cfg = arxiv_load(arxiv_config_path)
    lab_cfg = lab_load(lab_config_path)
    _assert_models_agree(arxiv_cfg, lab_cfg)

    real = Encoder(arxiv_cfg)
    encoder = _LockedEncoder(real) if encoder_lock else real

    radar = RadarServer(arxiv_cfg, encoder=encoder)
    lab = LabCorpusServer(lab_cfg, encoder=encoder)
    return radar, lab, encoder


async def _serve_both(
    radar,
    lab,
    *,
    host: str,
    arxiv_port: int,
    lab_port: int,
) -> None:
    """Run both `_run_streamable_http` coroutines concurrently."""
    from arxiv_radar_mcp.server import _run_streamable_http as run_arxiv_http
    from lab_corpus_mcp.server import _run_streamable_http as run_lab_http

    LOG.info(f"combined: arxiv-radar on {host}:{arxiv_port}, "
             f"lab-corpus on {host}:{lab_port}")
    await asyncio.gather(
        run_arxiv_http(radar, host, arxiv_port),
        run_lab_http(lab, host, lab_port),
    )


def serve_combined(
    *,
    arxiv_config_path: Path | None = None,
    lab_config_path: Path | None = None,
    host: str = "0.0.0.0",
    arxiv_port: int = 8765,
    lab_port: int = 8766,
    encoder_lock: bool = True,
) -> None:
    """Entry point: run BOTH servers in one process, sharing one Encoder.

    Bind defaults to 0.0.0.0 so two MCP proxies on the host can connect;
    flip to 127.0.0.1 if the container itself is the perimeter (SSH
    tunnel from outside).
    """
    radar, lab, encoder = build_servers(
        arxiv_config_path, lab_config_path, encoder_lock=encoder_lock,
    )
    LOG.info(f"combined: shared encoder = {encoder.model_name} "
             f"(lock={'on' if encoder_lock else 'off'})")
    LOG.info(f"  arxiv cache_dir: {radar.config.embeddings.cache_dir}")
    LOG.info(f"  lab parse_dir:   {lab.parse_dir}")
    asyncio.run(_serve_both(
        radar, lab,
        host=host, arxiv_port=arxiv_port, lab_port=lab_port,
    ))
