"""MinerU ingest-pipeline tests.

The MinerU subprocess is replaced with a fake `MineruRunner` from
`conftest.fake_mineru_runner`, so these tests cover the orchestration
(figures copy, atomic markdown write, meta.json sidecar, error
propagation) without needing the 2 GB MinerU install.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from lab_corpus_mcp.ingest import (
    IngestError,
    _default_mineru_runner,
    _copy_figures,
    ingest_dir,
    ingest_one,
    run_mineru,
)


def test_ingest_one_writes_markdown_and_meta(tmp_path, fake_pdf, fake_mineru_runner):
    parse_dir = tmp_path / "parsed"
    paper = ingest_one(fake_pdf, parse_dir, runner=fake_mineru_runner)

    md = parse_dir / "sources" / f"{paper.paper_id}.md"
    meta = parse_dir / "sources" / f"{paper.paper_id}.meta.json"
    assert md.exists() and meta.exists()
    body = md.read_text(encoding="utf-8")
    assert body.startswith("# ")
    on_disk = json.loads(meta.read_text(encoding="utf-8"))
    assert on_disk["paper_id"] == paper.paper_id
    assert on_disk["n_chars"] == paper.n_chars == len(body)


def test_ingest_one_extracts_arxiv_id_from_filename(tmp_path, fake_pdf, fake_mineru_runner):
    """fake_pdf fixture is named `sample-2503.99999.pdf` → arxiv_id derivation."""
    parse_dir = tmp_path / "parsed"
    paper = ingest_one(fake_pdf, parse_dir, runner=fake_mineru_runner)
    assert paper.paper_id == "2503.99999"
    assert paper.paper_id_kind == "arxiv_id"


def test_ingest_one_explicit_paper_id_overrides_derivation(tmp_path, fake_pdf,
                                                           fake_mineru_runner):
    parse_dir = tmp_path / "parsed"
    paper = ingest_one(fake_pdf, parse_dir, paper_id="custom",
                       runner=fake_mineru_runner)
    assert paper.paper_id == "custom"
    assert paper.paper_id_kind == "user_supplied"


def test_ingest_one_copies_figures(tmp_path, fake_pdf, fake_mineru_runner):
    parse_dir = tmp_path / "parsed"
    paper = ingest_one(fake_pdf, parse_dir, runner=fake_mineru_runner)
    figure = parse_dir / "figures" / paper.paper_id / "fig1.png"
    assert figure.exists()
    assert paper.figures_dir is not None


def test_ingest_one_no_figures_yields_none_dir(tmp_path):
    """Custom runner that emits markdown without an `images/` sibling."""
    pdf = tmp_path / "in.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    def _no_fig_runner(input_file: Path, output_dir: Path, timeout: int) -> Path:
        target = output_dir / "auto"
        target.mkdir(parents=True, exist_ok=True)
        md = target / "x.md"
        md.write_text("# Title\nbody\n", encoding="utf-8")
        return md

    parse_dir = tmp_path / "parsed"
    paper = ingest_one(pdf, parse_dir, runner=_no_fig_runner)
    assert paper.figures_dir is None
    assert not (parse_dir / "figures" / paper.paper_id).exists()


def test_ingest_one_missing_input_raises(tmp_path):
    with pytest.raises(IngestError, match="not found"):
        ingest_one(tmp_path / "missing.pdf", tmp_path / "parsed",
                   runner=lambda *a, **kw: None)


def test_ingest_one_propagates_runner_failure(tmp_path, fake_pdf):
    def _boom(*_a, **_kw):
        raise IngestError("simulated mineru crash")
    with pytest.raises(IngestError, match="simulated"):
        ingest_one(fake_pdf, tmp_path / "parsed", runner=_boom)


def test_ingest_one_atomic_write(tmp_path, fake_pdf, fake_mineru_runner):
    """No `.tmp` leftover after a successful ingest."""
    parse_dir = tmp_path / "parsed"
    paper = ingest_one(fake_pdf, parse_dir, runner=fake_mineru_runner)
    leftovers = list((parse_dir / "sources").glob("*.tmp"))
    assert leftovers == []
    assert (parse_dir / "sources" / f"{paper.paper_id}.md").exists()


def test_extract_title_from_runner_output(tmp_path, fake_pdf, fake_mineru_runner):
    parse_dir = tmp_path / "parsed"
    paper = ingest_one(fake_pdf, parse_dir, runner=fake_mineru_runner)
    # Fake runner emits "# Sample 2503.99999" as first heading.
    assert paper.title == "Sample 2503.99999"


# ----- bulk ingest_dir -------------------------------------------------------

def test_ingest_dir_missing(tmp_path):
    with pytest.raises(IngestError, match="not found"):
        ingest_dir(tmp_path / "no", tmp_path / "parsed",
                   runner=lambda *a, **kw: None)


def test_ingest_dir_empty(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    out = ingest_dir(src, tmp_path / "parsed", runner=lambda *a, **kw: None)
    assert out == {"n_total": 0, "n_ok": 0, "n_failed": 0, "ok": [], "failed": []}


def test_ingest_dir_processes_all_matches(tmp_path, fake_mineru_runner):
    src = tmp_path / "src"
    src.mkdir()
    for i in range(3):
        (src / f"sample-2503.{10000 + i}.pdf").write_bytes(b"%PDF-1.4")

    progress: list[tuple[int, int]] = []
    out = ingest_dir(src, tmp_path / "parsed",
                     runner=fake_mineru_runner,
                     progress_cb=lambda d, t: progress.append((d, t)))
    assert out["n_total"] == 3
    assert out["n_ok"] == 3
    assert out["n_failed"] == 0
    assert progress == [(1, 3), (2, 3), (3, 3)]


def test_ingest_dir_recursive_glob(tmp_path, fake_mineru_runner):
    src = tmp_path / "src"
    (src / "deep").mkdir(parents=True)
    (src / "top.pdf").write_bytes(b"%PDF-1.4")
    (src / "deep" / "buried.pdf").write_bytes(b"%PDF-1.4")

    flat = ingest_dir(src, tmp_path / "parsed", runner=fake_mineru_runner)
    deep = ingest_dir(src, tmp_path / "parsed", recursive=True,
                      runner=fake_mineru_runner)
    assert flat["n_total"] == 1
    assert deep["n_total"] == 2


def test_ingest_dir_collects_failures(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "good.pdf").write_bytes(b"%PDF-1.4")
    (src / "bad.pdf").write_bytes(b"%PDF-1.4")

    def _selective(input_file: Path, output_dir: Path, timeout: int) -> Path:
        if "bad" in input_file.name:
            raise IngestError("bad input")
        target = output_dir / "auto"
        target.mkdir(parents=True, exist_ok=True)
        md = target / "x.md"
        md.write_text("# Body\n", encoding="utf-8")
        return md

    out = ingest_dir(src, tmp_path / "parsed", runner=_selective)
    assert out["n_ok"] == 1
    assert out["n_failed"] == 1
    assert out["failed"][0]["input"].endswith("bad.pdf")


# ----- run_mineru / _default_mineru_runner ----------------------------------

def test_run_mineru_uses_injected_runner(tmp_path):
    captured = {}

    def _runner(input_file, output_dir, timeout):
        captured["args"] = (input_file, output_dir, timeout)
        md = output_dir / "x.md"
        md.write_text("# h\n", encoding="utf-8")
        return md

    src = tmp_path / "x.pdf"
    src.write_bytes(b"%PDF-1.4")
    out = run_mineru(src, tmp_path / "out", timeout=42, runner=_runner)
    assert out.exists()
    assert captured["args"][2] == 42


def test_default_runner_raises_when_subprocess_fails(monkeypatch, tmp_path):
    src = tmp_path / "x.pdf"
    src.write_bytes(b"%PDF-1.4")

    class _Fake:
        returncode = 7
        stderr = "boom"
        stdout = ""

    monkeypatch.setattr("lab_corpus_mcp.ingest.subprocess.run",
                        lambda *a, **kw: _Fake())

    with pytest.raises(IngestError, match="rc=7"):
        _default_mineru_runner(src, tmp_path / "out", timeout=10)


def test_default_runner_canonical_layout(monkeypatch, tmp_path):
    """When MinerU writes to `<out>/<stem>/auto/<stem>.md` (canonical 2.x
    path), the runner returns it directly — no rglob fallback needed."""
    src = tmp_path / "x.pdf"
    src.write_bytes(b"%PDF-1.4")
    out_dir = tmp_path / "out"
    canonical = out_dir / "x" / "auto" / "x.md"
    canonical.parent.mkdir(parents=True)
    canonical.write_text("# h\n", encoding="utf-8")

    class _Ok:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr("lab_corpus_mcp.ingest.subprocess.run",
                        lambda *a, **kw: _Ok())

    md = _default_mineru_runner(src, out_dir, timeout=10)
    assert md == canonical


def test_default_runner_recovers_with_fallback_md(monkeypatch, tmp_path):
    """If MinerU lays the markdown somewhere unexpected, runner finds it."""
    src = tmp_path / "x.pdf"
    src.write_bytes(b"%PDF-1.4")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "weird-layout").mkdir()
    fallback = out_dir / "weird-layout" / "produced.md"
    fallback.write_text("# h\n", encoding="utf-8")

    class _Ok:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr("lab_corpus_mcp.ingest.subprocess.run",
                        lambda *a, **kw: _Ok())

    md = _default_mineru_runner(src, out_dir, timeout=10)
    assert md == fallback


def test_default_runner_raises_when_no_markdown(monkeypatch, tmp_path):
    src = tmp_path / "x.pdf"
    src.write_bytes(b"%PDF-1.4")
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    class _Ok:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr("lab_corpus_mcp.ingest.subprocess.run",
                        lambda *a, **kw: _Ok())

    with pytest.raises(IngestError, match="no markdown"):
        _default_mineru_runner(src, out_dir, timeout=10)


def test_copy_figures_no_source(tmp_path):
    """No `images/` sibling → returns False, target is not created."""
    md = tmp_path / "auto" / "x.md"
    md.parent.mkdir(parents=True)
    md.write_text("# h\n", encoding="utf-8")
    target = tmp_path / "figs"
    assert _copy_figures(md, target) is False
    assert not target.exists()


def test_copy_figures_empty_dir(tmp_path):
    """Empty `images/` dir is treated as "no figures"."""
    md = tmp_path / "auto" / "x.md"
    md.parent.mkdir(parents=True)
    md.write_text("# h\n", encoding="utf-8")
    (md.parent / "images").mkdir()
    target = tmp_path / "figs"
    assert _copy_figures(md, target) is False
