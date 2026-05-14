"""MinerU-driven ingest pipeline.

Phase 2B-1 surface:
  * `run_mineru(pdf_path, out_dir, *, timeout)` — subprocess wrapper.
  * `ingest_one(file_path, parse_dir, *, paper_id=None)` — full ingest
    of a single file: run MinerU, normalize output, write
    `<parse.dir>/sources/<paper_id>.md` + `<paper_id>.meta.json`,
    optionally copy figures into `<parse.dir>/figures/<paper_id>/`.
  * `ingest_dir(dir_path, parse_dir, *, glob, recursive)` — bulk variant
    used by the `ingest_local_dir` MCP tool.

Phase 2B+ U14 surface (2026-05-13):
  * `fetch_and_ingest(url, parse_dir, *, paper_id=None, ...)` — download
    a URL to `<parse.dir>/inbox/<filename>` via `corpus_core.fetch_url`,
    then `ingest_one()` on the result. Used by the `ingest_url` and
    `ingest_arxiv_pdf` MCP tools to close the s142 dogfood gap (no
    server-side fetch-by-URL).

The MinerU CLI (`mineru -p <input> -o <out>`) is the only external
dependency; tests stub it with a fake that just writes a markdown
stub, so the orchestration logic is fully covered without a 2 GB
install. Real ingest happens on the gomer GPU host inside the Docker
image (see `lab-corpus-mcp/Dockerfile`).
"""
from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable, Protocol
from urllib.parse import urlparse

from corpus_core.http_fetch import (
    FetchResult,
    Throttle,
    fetch_url,
    get_arxiv_throttle,
)

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

# Default MinerU backend. `pipeline` (layout CNN + OCR + table) is what
# we actually want on a 12 GB GPU — finishes a 2 MB PDF in ~90 sec on
# RTX 4070. The alternative `vlm-transformers` runs MinerU's 1.2B Qwen2-VL
# without vllm, which OOMs / wedges next to our shared Qwen3-Embedding-4B
# (>15 min for the same input, no completion seen in a 15 min smoke).
# Override per-call by passing `backend="vlm-transformers"` if you have
# a 24 GB+ GPU and want the higher-fidelity output.
DEFAULT_BACKEND = "pipeline"


def _default_mineru_runner(
    input_file: Path, output_dir: Path, timeout: int,
    *, backend: str = DEFAULT_BACKEND,
) -> Path:
    """Real MinerU subprocess wrapper. Returns the produced markdown path.

    MinerU 2.x writes to `<output_dir>/<stem>/auto/<stem>.md` plus
    `<output_dir>/<stem>/auto/images/`. We probe that path first, then
    fall back to a recursive search to absorb minor layout drifts.
    """
    cmd = ["mineru", "-p", str(input_file), "-o", str(output_dir),
           "-b", backend]
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
    backend: str = DEFAULT_BACKEND,
    runner: MineruRunner | None = None,
) -> Path:
    """Public entry point. When `runner=None` (the common case), invokes
    `_default_mineru_runner` with the chosen `backend` flag.

    `runner` injection point is used by tests (no real MinerU subprocess)
    and the future "swap MinerU for marker" benchmark in the U7 deferred
    work. When a custom runner is provided, `backend` is ignored — the
    caller's runner controls flag selection.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if runner is not None:
        return runner(input_file, output_dir, timeout)
    return _default_mineru_runner(input_file, output_dir, timeout,
                                  backend=backend)


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
    backend: str = DEFAULT_BACKEND,
    runner: MineruRunner | None = None,
) -> LabPaper:
    """Ingest one document. Returns the persisted `LabPaper` metadata.

    `backend` selects the MinerU pipeline (`pipeline` for layout-CNN +
    OCR, `vlm-transformers` for the 1.2B Qwen2-VL backend). Defaults to
    `pipeline` because the VLM backend OOMs / wedges on a 12 GB GPU
    when sharing VRAM with our Qwen3-Embedding-4B encoder. Ignored if
    `runner` is supplied.
    """
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
        produced_md = run_mineru(input_file, tmp_out, timeout=timeout,
                                 backend=backend, runner=runner)
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
    backend: str = DEFAULT_BACKEND,
    runner: MineruRunner | None = None,
    progress_cb: Callable[[int, int], None] | None = None,
) -> dict:
    """Bulk ingest. Returns aggregate `{n_total, n_ok, n_failed, ok, failed}`.

    `backend` is forwarded to each per-file ingest_one call; see that
    function's docstring for the pipeline-vs-VLM trade-off.
    """
    if not dir_path.exists() or not dir_path.is_dir():
        raise IngestError(f"directory not found: {dir_path}")

    iterator = dir_path.rglob(glob) if recursive else dir_path.glob(glob)
    inputs = sorted(p for p in iterator if p.is_file())

    ok: list[dict] = []
    failed: list[dict] = []
    for i, inp in enumerate(inputs):
        try:
            paper = ingest_one(inp, parse_dir, timeout=timeout,
                               backend=backend, runner=runner)
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


# ---------------------------------------------------------------------------
# U14 — fetch-by-URL (PHASE 2B+, 2026-05-13)
# ---------------------------------------------------------------------------
#
# Closes the s142 dogfood gap: lab-corpus's pre-U14 surface required a
# server-side filesystem path for ingest_pdf / ingest_local_dir, so the
# user had to curl + docker cp + ingest_local_dir for every fresh PDF.
# `fetch_and_ingest` collapses that to one call: download via
# `corpus_core.fetch_url` into `<parse_dir>/inbox/`, then `ingest_one`.
#
# arxiv.org hosts get the shared module-global arxiv throttle for free
# (1 req / 3 sec), so in the combined image both servers share one
# budget and never double-spam.


INBOX_SUBDIR = "inbox"


class _Fetcher(Protocol):
    """Test seam — matches the subset of corpus_core.fetch_url we use."""

    def __call__(
        self,
        url: str,
        dest_path: Path,
        *,
        throttle: Throttle | None,
        timeout_s: float,
    ) -> FetchResult: ...


def _is_arxiv_url(url: str) -> bool:
    """True if the URL host is arxiv.org or a subdomain thereof."""
    netloc = urlparse(url).netloc.lower()
    return netloc == "arxiv.org" or netloc.endswith(".arxiv.org")


_KNOWN_DOC_EXTS = {".pdf", ".docx", ".pptx", ".png", ".jpg", ".jpeg"}


def _filename_from_url(url: str, paper_id: str | None) -> str:
    """Derive a stable filename for the inbox.

    Resolution order:
      1. Explicit ``paper_id`` → ``<paper_id>.pdf``.
      2. Last URL path segment if it carries a known extension.
      3. arxiv.org/{pdf,abs}/<id> → ``<id>.pdf`` (so ``derive_paper_id``
         picks up the arxiv pattern from the filename downstream).
      4. URL-hash fallback ``url-<sha1[:12]>.pdf`` — deterministic but
         opaque; ingest_one will fall back to sha256 of file contents.
    """
    if paper_id:
        return f"{paper_id}.pdf"

    parsed = urlparse(url)
    last = parsed.path.rsplit("/", 1)[-1] if parsed.path else ""

    if last and Path(last).suffix.lower() in _KNOWN_DOC_EXTS:
        return last

    if _is_arxiv_url(url) and last:
        # arxiv.org/pdf/2410.04059 → last="2410.04059", no extension. Add .pdf
        # so derive_paper_id matches the arxiv_id pattern on the stem.
        return f"{last}.pdf"

    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    return f"url-{digest}.pdf"


def fetch_and_ingest(
    url: str,
    parse_dir: Path,
    *,
    paper_id: str | None = None,
    inbox_subdir: str = INBOX_SUBDIR,
    timeout: int = 600,
    fetch_timeout_s: float = 60.0,
    backend: str = DEFAULT_BACKEND,
    runner: MineruRunner | None = None,
    fetcher: _Fetcher | None = None,
) -> LabPaper:
    """Download ``url`` then ``ingest_one`` the result. Returns persisted ``LabPaper``.

    Throttle is auto-selected: arxiv.org hosts share the module-global
    ``corpus_core.http_fetch.get_arxiv_throttle()`` (1 req / 3 sec, ToS-
    compliant); other hosts get no rate limit by default.

    Failures bubble out as ``IngestError`` so the job-registry path
    surfaces them uniformly with MinerU failures.

    Args:
        url: Absolute http(s) URL to GET.
        parse_dir: Same ``parse.dir`` used by ``ingest_one``. The
            download lands in ``<parse_dir>/<inbox_subdir>/<filename>``.
        paper_id: Optional explicit id — also dictates the filename
            (``<paper_id>.pdf``) so the meta path resolves
            deterministically.
        inbox_subdir: Where to land downloads. Default ``inbox``.
        timeout: MinerU subprocess timeout (sec) — same semantics as
            ``ingest_one``.
        fetch_timeout_s: Per-request HTTP timeout (sec).
        backend: MinerU backend forwarded to ``ingest_one``.
        runner: MinerU runner override (test injection).
        fetcher: HTTP fetcher override (test injection); defaults to
            ``corpus_core.http_fetch.fetch_url``.

    Returns:
        Persisted ``LabPaper``. The downloaded PDF stays under
        ``<parse_dir>/<inbox_subdir>/`` so ``source_path`` remains
        valid for re-ingest / debugging.
    """
    if not url or not url.startswith(("http://", "https://")):
        raise IngestError(f"invalid url: {url!r}")

    inbox = parse_dir / inbox_subdir
    inbox.mkdir(parents=True, exist_ok=True)
    filename = _filename_from_url(url, paper_id)
    dest = inbox / filename

    throttle = get_arxiv_throttle() if _is_arxiv_url(url) else None
    fetch = fetcher if fetcher is not None else fetch_url
    result = fetch(url, dest, throttle=throttle, timeout_s=fetch_timeout_s)
    if not result.ok:
        raise IngestError(
            f"fetch failed: {url}: {result.error or f'status={result.status}'}"
        )

    return ingest_one(
        dest, parse_dir,
        paper_id=paper_id, timeout=timeout, backend=backend, runner=runner,
    )
