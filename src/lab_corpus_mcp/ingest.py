"""MinerU-driven ingest pipeline.

Phase 2B-1 surface:
  * `run_mineru(pdf_path, out_dir, *, timeout)` — subprocess wrapper.
  * `ingest_one(file_path, parse_dir, *, paper_id=None)` — full ingest
    of a single file: run MinerU, normalize output, write
    `<parse.dir>/sources/<paper_id>.md` + `<paper_id>.meta.json`,
    optionally copy figures into `<parse.dir>/figures/<paper_id>/`.
  * `ingest_dir(dir_path, parse_dir, *, glob, recursive)` — bulk variant
    used by the `ingest_local_dir` MCP tool.

The MinerU CLI (`mineru -p <input> -o <out>`) is the only external
dependency; tests stub it with a fake that just writes a markdown
stub, so the orchestration logic is fully covered without a 2 GB
install. Real ingest happens on the gomer GPU host inside the Docker
image (see `lab-corpus-mcp/Dockerfile`).
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable

from lab_corpus_mcp.corpus import (
    LabPaper,
    derive_paper_id,
    extract_title_from_markdown,
    source_kind_for,
    utcnow_iso,
    write_meta,
)

LOG = logging.getLogger(__name__)


class IngestError(RuntimeError):
    """Raised when MinerU fails or produces no usable markdown."""


# Allow tests to inject a fake MinerU runner without monkeypatching subprocess.
MineruRunner = Callable[[Path, Path, int], Path]
"""Signature: (input_file, output_dir, timeout_seconds) -> path-to-produced-markdown."""


def _default_mineru_runner(input_file: Path, output_dir: Path, timeout: int) -> Path:
    """Real MinerU subprocess wrapper. Returns the produced markdown path.

    MinerU 2.x writes to `<output_dir>/<stem>/auto/<stem>.md` plus
    `<output_dir>/<stem>/auto/images/`. We probe that path first, then
    fall back to a recursive search to absorb minor layout drifts.
    """
    cmd = ["mineru", "-p", str(input_file), "-o", str(output_dir)]
    LOG.info(f"mineru: {' '.join(cmd)}")
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, check=False,
    )
    if proc.returncode != 0:
        raise IngestError(
            f"mineru failed (rc={proc.returncode}): "
            f"{proc.stderr.strip() or proc.stdout.strip() or 'no output'}"
        )

    stem = input_file.stem
    candidate = output_dir / stem / "auto" / f"{stem}.md"
    if candidate.exists():
        return candidate
    # Layout drift fallback — pick the first .md MinerU produced under out_dir.
    fallbacks = list(output_dir.rglob("*.md"))
    if not fallbacks:
        raise IngestError(f"mineru produced no markdown under {output_dir}")
    return fallbacks[0]


def run_mineru(
    input_file: Path, output_dir: Path, *, timeout: int = 600,
    runner: MineruRunner | None = None,
) -> Path:
    """Public entry point — uses `_default_mineru_runner` unless overridden.

    `runner` injection point is used by tests + the future
    "swap MinerU for marker" benchmark in the U7 deferred work.
    """
    runner = runner or _default_mineru_runner
    output_dir.mkdir(parents=True, exist_ok=True)
    return runner(input_file, output_dir, timeout)


def _copy_figures(produced_md: Path, target: Path) -> bool:
    """Copy MinerU's `images/` sibling dir to `<parse.dir>/figures/<paper_id>/`.

    Returns True if any figures were copied.
    """
    src = produced_md.parent / "images"
    if not src.exists() or not any(src.iterdir()):
        return False
    target.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, target, dirs_exist_ok=True)
    return True


def ingest_one(
    input_file: Path,
    parse_dir: Path,
    *,
    paper_id: str | None = None,
    timeout: int = 600,
    runner: MineruRunner | None = None,
) -> LabPaper:
    """Ingest one document. Returns the persisted `LabPaper` metadata."""
    if not input_file.exists():
        raise IngestError(f"input not found: {input_file}")

    if paper_id is None:
        paper_id, kind = derive_paper_id(input_file)
    else:
        kind = "user_supplied"

    sources_dir = parse_dir / "sources"
    figures_root = parse_dir / "figures"
    sources_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        tmp_out = Path(td)
        produced_md = run_mineru(input_file, tmp_out, timeout=timeout, runner=runner)
        markdown = produced_md.read_text(encoding="utf-8")

        figures_dir = figures_root / paper_id
        had_figures = _copy_figures(produced_md, figures_dir)

    # Atomic write of the markdown source.
    out_md = sources_dir / f"{paper_id}.md"
    tmp_md = out_md.with_suffix(out_md.suffix + ".tmp")
    tmp_md.write_text(markdown, encoding="utf-8")
    os.replace(tmp_md, out_md)

    paper = LabPaper(
        paper_id=paper_id,
        paper_id_kind=kind,
        title=extract_title_from_markdown(markdown),
        source_kind=source_kind_for(input_file),
        source_path=str(input_file.resolve()),
        parsed_path=str(out_md.resolve()),
        n_chars=len(markdown),
        n_chunks=0,
        ingested_at=utcnow_iso(),
        figures_dir=str((figures_root / paper_id).resolve()) if had_figures else None,
    )
    meta_path = sources_dir / f"{paper_id}.meta.json"
    write_meta(meta_path, paper)
    return paper


def ingest_dir(
    dir_path: Path,
    parse_dir: Path,
    *,
    glob: str = "*.pdf",
    recursive: bool = False,
    timeout: int = 600,
    runner: MineruRunner | None = None,
    progress_cb: Callable[[int, int], None] | None = None,
) -> dict:
    """Bulk ingest. Returns aggregate `{n_total, n_ok, n_failed, ok, failed}`."""
    if not dir_path.exists() or not dir_path.is_dir():
        raise IngestError(f"directory not found: {dir_path}")

    iterator = dir_path.rglob(glob) if recursive else dir_path.glob(glob)
    inputs = sorted(p for p in iterator if p.is_file())

    ok: list[dict] = []
    failed: list[dict] = []
    for i, inp in enumerate(inputs):
        try:
            paper = ingest_one(inp, parse_dir, timeout=timeout, runner=runner)
            ok.append({
                "input": str(inp),
                "paper_id": paper.paper_id,
                "n_chars": paper.n_chars,
            })
        except (IngestError, subprocess.TimeoutExpired) as e:
            failed.append({"input": str(inp), "error": str(e)})
        if progress_cb is not None:
            progress_cb(i + 1, len(inputs))

    return {
        "n_total": len(inputs),
        "n_ok": len(ok),
        "n_failed": len(failed),
        "ok": ok,
        "failed": failed,
    }
