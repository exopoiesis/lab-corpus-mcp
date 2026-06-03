"""Configuration loader for lab-corpus-mcp.

Minimal Phase 2A schema — only enough to wire the MCP scaffold. The
PDF-parser / ingest pipeline will extend this in Phase 2B.

Resolution order for the config file (mirrors arxiv-radar-mcp so users
can keep both servers' configs in the same place):
  1. explicit --config <path> CLI arg
  2. $LAB_CORPUS_CONFIG env var
  3. platformdirs user config (~/.config/lab-corpus-mcp/radar.toml on Linux)
  4. ./radar.toml in CWD
  5. built-in defaults
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - 3.10 fallback for the published wheel
    import tomli as tomllib  # type: ignore[no-redef]

from platformdirs import user_cache_dir, user_config_dir


def _default_config_path() -> Path:
    return Path(user_config_dir("lab-corpus-mcp", appauthor="exopoiesis")) / "radar.toml"


def _default_cache_dir() -> Path:
    return Path(user_cache_dir("lab-corpus-mcp", appauthor="exopoiesis"))


@dataclass
class EmbeddingsConfig:
    """Bi-encoder used for the dense retrieval index.

    Default matches arxiv-radar-mcp's empirically-validated Qwen3-4B native
    setting (see arxiv-radar-mcp/docs/MODEL_BENCHMARKS.md).
    """
    model: str = "Qwen/Qwen3-Embedding-4B"
    cache_dir: Path = field(default_factory=lambda: _default_cache_dir() / "embeddings")
    batch_size: int = 32
    target_dim: int | None = None
    # Drop the bi-encoder from VRAM after this many seconds with no encode
    # activity (search query / reindex). Explicit unload() only fires on job
    # completion, so without this a single search would pin ~7-8 GB for the
    # server's lifetime. 0 disables (model stays resident once warmed).
    idle_unload_s: int = 600


@dataclass
class ParseConfig:
    """Where the MinerU-parsed corpus lives.

    Phase 2B (ingest_pdf, ingest_local_dir) will write into
    `<dir>/sources/<paper_id>.md` and `<dir>/figures/<paper_id>/...`,
    mirroring arxiv-radar-mcp's `<fulltext_dir>/sources/<arxiv_id>.md`
    layout so corpus_core.corpus_index.reindex can index either tree.
    """
    dir: Path = field(default_factory=lambda: _default_cache_dir() / "parsed")


@dataclass
class ServerConfig:
    default_k: int = 10


@dataclass
class Config:
    embeddings: EmbeddingsConfig
    parse: ParseConfig
    server: ServerConfig

    @classmethod
    def defaults(cls) -> "Config":
        return cls(
            embeddings=EmbeddingsConfig(),
            parse=ParseConfig(),
            server=ServerConfig(),
        )


def load(config_path: Path | None = None) -> Config:
    """Load config from the first existing path in the resolution order, else defaults."""
    candidate: Path | None = config_path or _env_path()
    if candidate is None:
        default = _default_config_path()
        if default.exists():
            candidate = default
    if candidate is None:
        cwd_cfg = Path.cwd() / "radar.toml"
        if cwd_cfg.exists():
            candidate = cwd_cfg

    if candidate is None or not candidate.exists():
        return Config.defaults()

    with open(candidate, "rb") as f:
        data = tomllib.load(f)
    return _from_dict(data)


def _env_path() -> Path | None:
    p = os.environ.get("LAB_CORPUS_CONFIG")
    return Path(p) if p else None


def _from_dict(data: dict) -> Config:
    embeddings_raw = data.get("embeddings", {})
    cache_dir = embeddings_raw.get("cache_dir")
    embeddings = EmbeddingsConfig(
        model=embeddings_raw.get("model", "Qwen/Qwen3-Embedding-4B"),
        cache_dir=Path(cache_dir).expanduser() if cache_dir else _default_cache_dir() / "embeddings",
        batch_size=int(embeddings_raw.get("batch_size", 32)),
        target_dim=embeddings_raw.get("target_dim"),
        idle_unload_s=int(embeddings_raw.get("idle_unload_s", 600)),
    )

    parse_raw = data.get("parse", {})
    parse_dir = parse_raw.get("dir")
    parse = ParseConfig(
        dir=Path(parse_dir).expanduser() if parse_dir else (embeddings.cache_dir.parent / "parsed"),
    )

    server_raw = data.get("server", {})
    server = ServerConfig(
        default_k=int(server_raw.get("default_k", 10)),
    )

    return Config(embeddings=embeddings, parse=parse, server=server)
