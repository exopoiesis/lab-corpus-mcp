"""MinerU-driven ingest pipeline.

Phase 2B-1 surface:
  * `run_mineru(pdf_path, out_dir, *, timeout)` -- library wrapper.
  * `ingest_one(file_path, parse_dir, *, paper_id=None)` -- full ingest
    of a single file: run MinerU, normalize output, write
    `<parse.dir>/sources/<paper_id>.md` + `<paper_id>.meta.json`,
    optionally copy figures into `<parse.dir>/figures/<paper_id>/`.
  * `ingest_dir(dir_path, parse_dir, *, glob, recursive)` -- bulk variant
    used by the `ingest_local_dir` MCP tool.

Phase 2B+ U14 surface (2026-05-13):
  * `fetch_and_ingest(url, parse_dir, *, paper_id=None, ...)` -- download
    a URL to `<parse.dir>/inbox/<filename>` via `corpus_core.fetch_url`,
    then `ingest_one()` on the result. Used by the `ingest_url` and
    `ingest_arxiv_pdf` MCP tools to close the s142 dogfood gap (no
    server-side fetch-by-URL).

s153 (2026-05-24): Switched from `subprocess.run(["mineru", ...])` to
direct Python library calls (`mineru.cli.common.do_parse`). The old
subprocess path had two costs: (1) ~30 sec model cold-load on every
PDF because the granchild `LocalAPIServer` died at the end of each CLI
call; (2) HTTP overhead between CLI and LocalAPIServer. Library mode
loads MinerU's `AtomModelSingleton` once into the lab-corpus-mcp process
and reuses it for the whole batch. The trade-off -- MinerU's GPU weights
now share VRAM with our Qwen3 encoder -- is mitigated by
`unload_mineru_models()` called from the same `_release_gpu_vram` path
that handles `Encoder.unload()`.

U7 (2026-06-25): The parse mechanics (MineruRunner seam, _default_mineru_runner,
unload_mineru_models, DEFAULT_BACKEND) have been lifted into
`corpus_core.pdf` so that arxiv-radar-mcp can use them without depending
on lab-corpus-mcp. This module re-exports those names as aliases so
existing test imports continue to work without change.

The `MineruRunner` injection point stays compatible: tests can still
pass a fake runner that just writes a stub markdown without ever
importing MinerU.
"""
from __future__ import annotations

import hashlib
import logging
import os
import shutil
from pathlib import Path
from typing import Callable, Protocol
from urllib.parse import urlparse

from corpus_core.http_fetch import (
    FetchResult,
    Throttle,
    fetch_url,
    get_arxiv_throttle,
)

# U7: import parse mechanics from corpus_core.pdf (the canonical home).
# We re-export them under their original names so test imports and
# server.py imports in lab-corpus continue to work unchanged.
import corpus_core.pdf as _pdf_mod
from corpus_core.pdf import (
    DEFAULT_BACKEND,
    MineruRunner,
    PdfParseError as _PdfParseError,
    parse_pdf as _parse_pdf,
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


# ---- Backward-compat aliases (names tests + server.py import from here) ----

def _default_mineru_runner(
    input_file: Path, output_dir: Path, timeout: int,
    *, backend: str = DEFAULT_BACKEND,
) -> Path:
    """Backward-compat alias -> corpus_core.pdf._default_mineru_runner.

    All real logic lives there; this shim preserves the public name that
    lab-corpus tests (test_ingest.py) import directly from this module.

    PdfParseError is re-raised as IngestError to preserve the error type
    that lab-corpus tests and the JobRegistry expect from this module.
    """
    try:
        return _pdf_mod._default_mineru_runner(
            input_file, output_dir, timeout, backend=backend
        )
    except _PdfParseError as exc:
        raise IngestError(str(exc)) from exc


def unload_mineru_models() -> bool:
    """Backward-compat alias -> corpus_core.pdf.unload_pdf_models().

    Releases MinerU's cached pipeline / hybrid model singletons.
    Idempotent; returns True if anything was released.
    """
    return _pdf_mod.unload_pdf_models()


def run_mineru(
    input_file: Path, output_dir: Path, *, timeout: int = 600,
    backend: str = DEFAULT_BACKEND,
    runner: MineruRunner | None = None,
) -> Path:
    """Public entry point. When `runner=None` (the common case), invokes
    `_default_mineru_runner` (via corpus_core.pdf) with the chosen `backend`.

    `runner` injection point is used by tests (no real MinerU subprocess)
    and the future "swap MinerU for marker" benchmark in the U7 deferred
    work. When a custom runner is provided, `backend` is ignored -- the
    caller's runner controls flag selection.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if runner is not None:
        return runner(input_file, output_dir, timeout)
    return _pdf_mod._default_mineru_runner(
        input_file, output_dir, timeout, backend=backend
    )


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

    U7 (2026-06-25): internally delegates to corpus_core.pdf.parse_pdf
    for the actual parse step, then maps the result into the lab-corpus
    LabPaper / figures-dir format.  The public signature and output format
    are unchanged.
    """
    if not input_file.exists():
        raise IngestError(f"input not found: {input_file}")

    if paper_id is None:
        paper_id, kind = derive_paper_id(input_file)
    else:
        kind = "user_supplied"

    sources_dir = parse_dir / "sources"
    figures_root = parse_dir / "figures"
    figures_dir = figures_root / paper_id
    sources_dir.mkdir(parents=True, exist_ok=True)

    try:
        pdf_result = _parse_pdf(
            input_file,
            media_out_dir=figures_dir,
            backend=backend,
            runner=runner,
        )
    except _PdfParseError as exc:
        raise IngestError(str(exc)) from exc

    markdown = pdf_result.markdown
    had_figures = bool(pdf_result.images)

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
        figures_dir=str(figures_dir.resolve()) if had_figures else None,
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
        except IngestError as e:
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
