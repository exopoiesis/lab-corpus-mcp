"""MinerU ingest-pipeline tests.

The MinerU library call (`mineru.cli.common.do_parse`) is replaced with
either a fake `MineruRunner` from `conftest.fake_mineru_runner` (for
high-level tests) or a sys.modules injection of a stub `mineru.cli.common`
(for `_default_mineru_runner` low-level tests). Both cover the
orchestration logic without needing the 2 GB MinerU install.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from lab_corpus_mcp.ingest import (
    IngestError,
    _copy_figures,
    _default_mineru_runner,
    _filename_from_url,
    _is_arxiv_url,
    fetch_and_ingest,
    ingest_dir,
    ingest_one,
    run_mineru,
    unload_mineru_models,
)


def _install_fake_mineru(monkeypatch, do_parse_impl):
    """Inject a stand-in for `mineru.cli.common.do_parse` into sys.modules.

    The real mineru package is a 2 GB install we don't want in the test
    venv. `_default_mineru_runner` lazy-imports `do_parse` only on call,
    so swapping the module entry before that import gives the runner
    our fake without ever touching real MinerU.
    """
    # Build the parent chain so the relative import resolves cleanly.
    pkg_mineru = types.ModuleType("mineru")
    pkg_mineru_cli = types.ModuleType("mineru.cli")
    mod_common = types.ModuleType("mineru.cli.common")
    mod_common.do_parse = do_parse_impl
    pkg_mineru.cli = pkg_mineru_cli
    pkg_mineru_cli.common = mod_common

    monkeypatch.setitem(sys.modules, "mineru", pkg_mineru)
    monkeypatch.setitem(sys.modules, "mineru.cli", pkg_mineru_cli)
    monkeypatch.setitem(sys.modules, "mineru.cli.common", mod_common)


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


def test_default_runner_raises_when_library_call_fails(monkeypatch, tmp_path):
    """`do_parse` raising must surface as IngestError so the JobRegistry
    records a clean `failed` state instead of a crash."""
    src = tmp_path / "x.pdf"
    src.write_bytes(b"%PDF-1.4")

    def _boom(**kwargs):
        raise RuntimeError("simulated mineru failure")

    _install_fake_mineru(monkeypatch, _boom)

    with pytest.raises(IngestError, match="mineru library call failed"):
        _default_mineru_runner(src, tmp_path / "out", timeout=10)


def test_default_runner_canonical_layout(monkeypatch, tmp_path):
    """When MinerU writes to `<out>/<stem>/<parse_method>/<stem>.md` (canonical
    library output), the runner returns it directly — no rglob fallback."""
    src = tmp_path / "x.pdf"
    src.write_bytes(b"%PDF-1.4")
    out_dir = tmp_path / "out"
    canonical = out_dir / "x" / "auto" / "x.md"
    canonical.parent.mkdir(parents=True)
    canonical.write_text("# h\n", encoding="utf-8")

    def _stub_do_parse(**kwargs):
        # Real do_parse already wrote the markdown — emulate by being a no-op.
        return None

    _install_fake_mineru(monkeypatch, _stub_do_parse)

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

    def _stub_do_parse(**kwargs):
        return None

    _install_fake_mineru(monkeypatch, _stub_do_parse)

    md = _default_mineru_runner(src, out_dir, timeout=10)
    assert md == fallback


def test_default_runner_raises_when_no_markdown(monkeypatch, tmp_path):
    src = tmp_path / "x.pdf"
    src.write_bytes(b"%PDF-1.4")
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    def _stub_do_parse(**kwargs):
        return None  # produces no markdown anywhere under out_dir.

    _install_fake_mineru(monkeypatch, _stub_do_parse)

    with pytest.raises(IngestError, match="no markdown"):
        _default_mineru_runner(src, out_dir, timeout=10)


def test_default_runner_passes_pdf_bytes_to_do_parse(monkeypatch, tmp_path):
    """Confirm the library call gets the file's bytes (not its path) — this is
    the contract `mineru.cli.common.do_parse` expects."""
    src = tmp_path / "x.pdf"
    src.write_bytes(b"%PDF-1.4 minimal payload")
    out_dir = tmp_path / "out"
    # Pre-create the canonical output so the runner can return cleanly.
    canonical = out_dir / "x" / "auto" / "x.md"
    canonical.parent.mkdir(parents=True)
    canonical.write_text("# h\n", encoding="utf-8")

    captured: dict = {}

    def _stub_do_parse(**kwargs):
        captured.update(kwargs)
        return None

    _install_fake_mineru(monkeypatch, _stub_do_parse)
    _default_mineru_runner(src, out_dir, timeout=10, backend="pipeline")

    assert captured["pdf_file_names"] == ["x"]
    assert captured["pdf_bytes_list"] == [b"%PDF-1.4 minimal payload"]
    assert captured["p_lang_list"] == ["ch"]
    assert captured["backend"] == "pipeline"
    assert captured["parse_method"] == "auto"


# ----- unload_mineru_models -------------------------------------------------

def test_unload_mineru_models_returns_false_when_mineru_absent(monkeypatch):
    """On a host without MinerU installed (laptop / unit tests), the helper
    must return False instead of raising ImportError."""
    # Sabotage the import by removing any cached module + blocking the parent.
    for name in ("mineru.backend.pipeline.model_init",
                 "mineru.backend.pipeline",
                 "mineru.backend",
                 "mineru"):
        monkeypatch.delitem(sys.modules, name, raising=False)

    # Inject a finder that raises ImportError for the mineru.* path.
    class _BlockMineru:
        def find_module(self, name, path=None):
            return self if name.startswith("mineru") else None

        def find_spec(self, name, path=None, target=None):
            if name.startswith("mineru"):
                raise ImportError(f"blocked for test: {name}")
            return None

    monkeypatch.setattr("sys.meta_path", [_BlockMineru()] + sys.meta_path)
    assert unload_mineru_models() is False


def test_unload_mineru_models_clears_singleton_dicts(monkeypatch):
    """When MinerU's singletons hold model instances, unload clears them
    and reports True."""
    # Build minimal fake singleton classes with a `_models` dict each.
    class _FakeAtom:
        _models = {"layout-en": object(), "ocr-ch": object()}

    class _FakeHybrid:
        _models = {"hybrid-en": object()}

    # Inject fake `mineru.backend.pipeline.model_init` so the lazy import
    # inside unload_mineru_models picks them up.
    pkg_mineru = types.ModuleType("mineru")
    pkg_backend = types.ModuleType("mineru.backend")
    pkg_pipeline = types.ModuleType("mineru.backend.pipeline")
    mod_init = types.ModuleType("mineru.backend.pipeline.model_init")
    mod_init.AtomModelSingleton = _FakeAtom
    mod_init.HybridModelSingleton = _FakeHybrid

    monkeypatch.setitem(sys.modules, "mineru", pkg_mineru)
    monkeypatch.setitem(sys.modules, "mineru.backend", pkg_backend)
    monkeypatch.setitem(sys.modules, "mineru.backend.pipeline", pkg_pipeline)
    monkeypatch.setitem(sys.modules, "mineru.backend.pipeline.model_init", mod_init)

    assert _FakeAtom._models and _FakeHybrid._models
    released = unload_mineru_models()
    assert released is True
    assert _FakeAtom._models == {}
    assert _FakeHybrid._models == {}


def test_unload_mineru_models_idempotent_when_already_empty(monkeypatch):
    """When the singletons hold no models, unload returns False — the cuda
    cache clear path is skipped to avoid touching the GPU for nothing."""
    class _FakeAtom:
        _models = {}

    class _FakeHybrid:
        _models = {}

    pkg_mineru = types.ModuleType("mineru")
    pkg_backend = types.ModuleType("mineru.backend")
    pkg_pipeline = types.ModuleType("mineru.backend.pipeline")
    mod_init = types.ModuleType("mineru.backend.pipeline.model_init")
    mod_init.AtomModelSingleton = _FakeAtom
    mod_init.HybridModelSingleton = _FakeHybrid

    monkeypatch.setitem(sys.modules, "mineru", pkg_mineru)
    monkeypatch.setitem(sys.modules, "mineru.backend", pkg_backend)
    monkeypatch.setitem(sys.modules, "mineru.backend.pipeline", pkg_pipeline)
    monkeypatch.setitem(sys.modules, "mineru.backend.pipeline.model_init", mod_init)

    assert unload_mineru_models() is False


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


# ----- U14: fetch_and_ingest + helpers ---------------------------------------

def test_is_arxiv_url_positive():
    assert _is_arxiv_url("https://arxiv.org/pdf/2410.04059")
    assert _is_arxiv_url("https://arxiv.org/abs/2410.04059v2")
    # Case-insensitive netloc
    assert _is_arxiv_url("https://ARXIV.ORG/pdf/2410.04059")
    # Future subdomain (e.g. export.arxiv.org)
    assert _is_arxiv_url("https://export.arxiv.org/pdf/2410.04059")


def test_is_arxiv_url_negative():
    assert not _is_arxiv_url("https://example.com/pdf/2410.04059")
    # Path-only mention of "arxiv.org" must NOT trigger the throttle.
    assert not _is_arxiv_url("https://example.com/mirror/arxiv.org/x.pdf")
    assert not _is_arxiv_url("https://notarxiv.org/x.pdf")


def test_filename_from_url_explicit_paper_id():
    assert _filename_from_url("https://example.com/x", "my-id") == "my-id.pdf"
    # Even when URL already has its own extension, paper_id wins.
    assert _filename_from_url("https://example.com/a.pdf", "my-id") == "my-id.pdf"


def test_filename_from_url_uses_path_basename_when_extension_known():
    assert _filename_from_url("https://example.com/papers/preprint.pdf", None) == "preprint.pdf"
    assert _filename_from_url("https://example.com/a/b/slides.pptx", None) == "slides.pptx"


def test_filename_from_url_arxiv_pdf_endpoint_adds_extension():
    # arxiv.org/pdf/<id> has no extension on disk; the helper must add .pdf
    # so derive_paper_id recognises the arxiv pattern on the stem.
    assert _filename_from_url("https://arxiv.org/pdf/2410.04059", None) == "2410.04059.pdf"
    assert _filename_from_url("https://arxiv.org/abs/2410.04059", None) == "2410.04059.pdf"


def test_filename_from_url_hash_fallback_for_extensionless_generic():
    """No paper_id, non-arxiv host, no recognized extension → url-hash fallback.

    Output must be deterministic for a given URL (sha1 of the URL string).
    """
    name = _filename_from_url("https://example.com/messy/?id=42", None)
    assert name.startswith("url-")
    assert name.endswith(".pdf")
    # Deterministic
    again = _filename_from_url("https://example.com/messy/?id=42", None)
    assert name == again


def test_fetch_and_ingest_rejects_bad_scheme(tmp_path):
    with pytest.raises(IngestError, match="invalid url"):
        fetch_and_ingest("ftp://example.com/x", tmp_path / "parsed",
                         runner=lambda *a, **kw: None)


def _ok_fetcher(body: bytes = b"%PDF-1.4 dummy"):
    """Factory: build a fetcher that writes `body` and records call kwargs."""
    from corpus_core.http_fetch import FetchResult
    captured: dict = {}

    def _fetch(url, dest_path, *, throttle, timeout_s):
        captured["url"] = url
        captured["dest_path"] = Path(dest_path)
        captured["throttle"] = throttle
        captured["timeout_s"] = timeout_s
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(body)
        return FetchResult(
            url=url, dest_path=Path(dest_path),
            ok=True, status=200, n_bytes=len(body), error=None,
        )

    return _fetch, captured


def test_fetch_and_ingest_happy_path(tmp_path, fake_mineru_runner):
    fetcher, captured = _ok_fetcher()
    paper = fetch_and_ingest(
        "https://example.com/papers/x.pdf",
        tmp_path / "parsed",
        runner=fake_mineru_runner,
        fetcher=fetcher,
    )
    # Wrote to <parse_dir>/inbox/x.pdf
    assert captured["dest_path"] == tmp_path / "parsed" / "inbox" / "x.pdf"
    assert captured["throttle"] is None  # non-arxiv host
    # ingest_one happened — markdown + meta exist
    md = tmp_path / "parsed" / "sources" / f"{paper.paper_id}.md"
    meta = tmp_path / "parsed" / "sources" / f"{paper.paper_id}.meta.json"
    assert md.exists() and meta.exists()


def test_fetch_and_ingest_arxiv_host_wires_singleton_throttle(
    tmp_path, fake_mineru_runner,
):
    from corpus_core.http_fetch import get_arxiv_throttle
    fetcher, captured = _ok_fetcher()
    paper = fetch_and_ingest(
        "https://arxiv.org/pdf/2410.04059",
        tmp_path / "parsed",
        runner=fake_mineru_runner,
        fetcher=fetcher,
    )
    assert captured["throttle"] is get_arxiv_throttle()
    # paper_id derived from arxiv-id pattern on filename
    assert paper.paper_id == "2410.04059"
    assert paper.paper_id_kind == "arxiv_id"


def test_fetch_and_ingest_explicit_paper_id_propagates(tmp_path, fake_mineru_runner):
    fetcher, captured = _ok_fetcher()
    paper = fetch_and_ingest(
        "https://example.com/messy/?id=42",
        tmp_path / "parsed",
        paper_id="custom-7",
        runner=fake_mineru_runner,
        fetcher=fetcher,
    )
    assert captured["dest_path"].name == "custom-7.pdf"
    assert paper.paper_id == "custom-7"
    assert paper.paper_id_kind == "user_supplied"


def test_fetch_and_ingest_propagates_fetch_failure(tmp_path):
    from corpus_core.http_fetch import FetchResult

    def _fail(url, dest_path, *, throttle, timeout_s):
        return FetchResult(
            url=url, dest_path=None, ok=False,
            status=503, n_bytes=0, error="http 503",
        )

    with pytest.raises(IngestError, match="fetch failed"):
        fetch_and_ingest(
            "https://example.com/x.pdf",
            tmp_path / "parsed",
            runner=lambda *a, **kw: None,
            fetcher=_fail,
        )


def test_fetch_and_ingest_propagates_mineru_failure(tmp_path):
    """Successful fetch + failing MinerU surfaces IngestError unchanged."""
    fetcher, _ = _ok_fetcher()

    def _boom(*_a, **_kw):
        raise IngestError("simulated parse failure")

    with pytest.raises(IngestError, match="simulated"):
        fetch_and_ingest(
            "https://example.com/x.pdf",
            tmp_path / "parsed",
            runner=_boom,
            fetcher=fetcher,
        )
