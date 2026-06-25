"""Tests for the lab-corpus download side-channel (GET /download).

Only the lab-specific file-resolver is tested here; the zip building and
HTTP handler live in (and are tested by) corpus_core.archive.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path
from types import SimpleNamespace

from corpus_core.archive import build_paper_archive
from lab_corpus_mcp.server import _lab_paper_files

_PNG = b"\x89PNG\r\n\x1a\n" + b"fig" * 8


def test_lab_paper_files_maps_figures_to_images_arcname(tmp_path: Path):
    # Mirror lab-corpus's on-disk layout after a MinerU ingest.
    sources = tmp_path / "sources"
    sources.mkdir(parents=True)
    (sources / "DOC.md").write_text("# Doc\n\n![](images/fig1.png)\n",
                                    encoding="utf-8")
    (sources / "DOC.meta.json").write_text('{"source_kind":"pdf"}',
                                            encoding="utf-8")
    figdir = tmp_path / "figures" / "DOC"
    figdir.mkdir(parents=True)
    (figdir / "fig1.png").write_bytes(_PNG)

    server = SimpleNamespace(parse_dir=tmp_path)
    files = _lab_paper_files(server, "DOC")

    # figures/<id>/ on disk, but the archive subdir matches the md ref prefix.
    assert files.media_dir == figdir
    assert files.media_arcname == "images"

    data = build_paper_archive("DOC", files)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = set(zf.namelist())
        assert names == {"DOC/DOC.md", "DOC/DOC.meta.json", "DOC/images/fig1.png"}
        assert zf.read("DOC/images/fig1.png") == _PNG
        # MinerU's `images/fig1.png` ref now resolves inside the unzipped tree.
        assert "![](images/fig1.png)" in zf.read("DOC/DOC.md").decode()
