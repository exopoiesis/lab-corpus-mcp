"""Shared fixtures for lab-corpus-mcp tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from lab_corpus_mcp.config import Config, EmbeddingsConfig, ParseConfig, ServerConfig


@pytest.fixture
def lab_config(tmp_path: Path) -> Config:
    """Lab config rooted in a tmp dir so tests stay sandboxed.

    Mirrors the production layout: `cache_dir/embeddings/...` for the
    embedding index, `cache_dir/parsed/sources/<id>.md` for the
    MinerU output. JobRegistry persists alongside under `cache_dir/jobs/`.
    """
    cache_dir = tmp_path / "cache"
    return Config(
        embeddings=EmbeddingsConfig(
            model="test/dummy",
            cache_dir=cache_dir / "embeddings",
            batch_size=4,
            target_dim=None,
        ),
        parse=ParseConfig(dir=cache_dir / "parsed"),
        server=ServerConfig(default_k=5),
    )


@pytest.fixture
def populated_parse_dir(lab_config: Config) -> Path:
    """`parse.dir/sources/` populated with three deterministic markdown stubs.

    Used by list_corpus / corpus_stats tests so they exercise the
    real disk-walk path instead of mocking it.
    """
    sources = lab_config.parse.dir / "sources"
    sources.mkdir(parents=True, exist_ok=True)
    for stem, body in [
        ("paper-002", "# Second\nbody two\n"),
        ("paper-001", "# First\nbody one\n"),
        ("doi-10.1000-zzz", "# DOI-style id\nbody three\n"),
    ]:
        (sources / f"{stem}.md").write_text(body, encoding="utf-8")
    return sources
