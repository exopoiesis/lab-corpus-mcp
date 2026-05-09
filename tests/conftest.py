"""Shared fixtures for lab-corpus-mcp tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from lab_corpus_mcp.config import Config, EmbeddingsConfig, ParseConfig, ServerConfig
from lab_corpus_mcp.corpus import LabPaper, write_meta


@pytest.fixture
def lab_config(tmp_path: Path) -> Config:
    """Lab config rooted in a tmp dir so tests stay sandboxed.

    Mirrors the production layout: `cache_dir/embeddings/...` for the
    embedding index, `cache_dir/parsed/sources/<id>.{md,meta.json}` for
    the MinerU output. JobRegistry persists at `cache_dir/jobs/`.
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
    """`parse.dir/sources/` populated with three deterministic ingest results
    (markdown + meta.json), as if `ingest_one()` had run on them.
    """
    sources = lab_config.parse.dir / "sources"
    sources.mkdir(parents=True, exist_ok=True)

    fixtures = [
        # (paper_id, kind, body, ingested_at)
        ("paper-002", "sha256", "# Second\n\nBody of the second paper.\n", "2026-05-09T10:02:00+00:00"),
        ("paper-001", "sha256", "# First\n\nBody of the first paper.\n", "2026-05-09T10:01:00+00:00"),
        ("doi-10.1000-zzz", "doi", "# DOI-style id\n\nBody three.\n", "2026-05-09T10:00:00+00:00"),
    ]
    for paper_id, kind, body, ts in fixtures:
        md_path = sources / f"{paper_id}.md"
        md_path.write_text(body, encoding="utf-8")
        meta_path = sources / f"{paper_id}.meta.json"
        write_meta(meta_path, LabPaper(
            paper_id=paper_id,
            paper_id_kind=kind,
            title=body.splitlines()[0].lstrip("# "),
            source_kind="pdf",
            source_path=str(sources / f"{paper_id}.original.pdf"),
            parsed_path=str(md_path.resolve()),
            n_chars=len(body),
            n_chunks=0,
            ingested_at=ts,
            figures_dir=None,
        ))
    return sources


@pytest.fixture
def fake_pdf(tmp_path: Path) -> Path:
    """A path that looks like a PDF but is just a tiny binary blob.

    Tests pair this with a fake MinerU runner that produces deterministic
    markdown output, so the orchestration code is exercised end-to-end
    without the real 2 GB MinerU install.
    """
    p = tmp_path / "input" / "sample-2503.99999.pdf"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"%PDF-1.4\n%fake bytes\n")
    return p


@pytest.fixture
def fake_mineru_runner():
    """Returns a runner that pretends MinerU produced a markdown file.

    Signature matches `lab_corpus_mcp.ingest.MineruRunner`. Writes a
    deterministic three-section markdown so chunker tests downstream
    have something predictable to operate on.
    """
    def _runner(input_file: Path, output_dir: Path, timeout: int) -> Path:
        stem = input_file.stem
        target = output_dir / stem / "auto"
        target.mkdir(parents=True, exist_ok=True)
        md = target / f"{stem}.md"
        md.write_text(
            f"# {stem.replace('-', ' ').title()}\n\n"
            "## Introduction\n\nIntroductory text body.\n\n"
            "## Methods\n\nMethods text body.\n\n"
            "## Conclusion\n\nConcluding remarks.\n",
            encoding="utf-8",
        )
        # Drop a fake figure so _copy_figures has something to lift.
        figures = target / "images"
        figures.mkdir(exist_ok=True)
        (figures / "fig1.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        return md
    return _runner
