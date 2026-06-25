"""LabPaper schema + paper_id derivation tests."""
from __future__ import annotations

from pathlib import Path


from lab_corpus_mcp.corpus import (
    LabPaper,
    derive_paper_id,
    extract_title_from_markdown,
    load_lab_papers,
    read_meta,
    sha256_prefix,
    source_kind_for,
    write_meta,
)


def test_derive_paper_id_arxiv_style(tmp_path):
    p = tmp_path / "2503.12345v2.pdf"
    p.write_bytes(b"%PDF-1.4")
    paper_id, kind = derive_paper_id(p)
    assert paper_id == "2503.12345"
    assert kind == "arxiv_id"


def test_derive_paper_id_arxiv_style_no_version(tmp_path):
    p = tmp_path / "2503.99999.pdf"
    p.write_bytes(b"%PDF-1.4")
    paper_id, kind = derive_paper_id(p)
    assert paper_id == "2503.99999"
    assert kind == "arxiv_id"


def test_derive_paper_id_falls_back_to_sha256(tmp_path):
    p = tmp_path / "random-filename.pdf"
    p.write_bytes(b"deterministic-bytes-for-test")
    paper_id, kind = derive_paper_id(p)
    assert kind == "sha256"
    assert paper_id.startswith("sha256-")
    assert len(paper_id) == len("sha256-") + 16


def test_derive_paper_id_content_override(tmp_path):
    p = tmp_path / "any.pdf"
    p.write_bytes(b"on-disk-content")
    pid_disk, _ = derive_paper_id(p)
    pid_buf, _ = derive_paper_id(p, content=b"different-content")
    # Content override takes priority over reading from disk.
    assert pid_disk != pid_buf


def test_sha256_prefix_streams_large_files(tmp_path):
    p = tmp_path / "blob.bin"
    # Bigger than the 1 MiB read chunk so streaming logic actually executes.
    p.write_bytes(b"\x00" * (2 * 1024 * 1024 + 7))
    h = sha256_prefix(p, length=8)
    assert len(h) == 8
    # Same content → same hash; deterministic.
    assert sha256_prefix(p, length=8) == h


def test_source_kind_for_known_extensions(tmp_path):
    cases = {
        "x.pdf": "pdf", "x.PDF": "pdf",
        "x.docx": "docx", "x.pptx": "pptx",
        "x.png": "png", "x.JPG": "jpg",
        "x.txt": "unknown", "x": "unknown",
    }
    for name, expected in cases.items():
        assert source_kind_for(Path(name)) == expected


def test_extract_title_from_markdown_first_heading():
    md = "Some preamble line\n# Real Title\n\n## Section\nbody\n"
    assert extract_title_from_markdown(md) == "Real Title"


def test_extract_title_from_markdown_no_heading():
    assert extract_title_from_markdown("just paragraphs of text\nno hash\n") is None


def test_extract_title_from_markdown_skips_subheadings():
    """Only `# heading` counts (top level); `##` is a section."""
    assert extract_title_from_markdown("## Section first\n# Title\n") == "Title"


def test_extract_title_only_scans_first_20_lines():
    # 19 filler lines, then # Title — should be picked up.
    md = "\n".join(["filler"] * 19 + ["# Title late"])
    assert extract_title_from_markdown(md) == "Title late"
    # Too late to find — past the 20-line cutoff.
    md2 = "\n".join(["filler"] * 25 + ["# Too late"])
    assert extract_title_from_markdown(md2) is None


def test_meta_roundtrip(tmp_path):
    paper = LabPaper(
        paper_id="x", paper_id_kind="sha256", title="t", source_kind="pdf",
        source_path="/p.pdf", parsed_path="/p.md", n_chars=10,
        n_chunks=3, ingested_at="2026-05-09T10:00:00+00:00",
        figures_dir=None, extra={"k": "v"},
    )
    meta_path = tmp_path / "x.meta.json"
    write_meta(meta_path, paper)
    loaded = read_meta(meta_path)
    assert loaded == paper


def test_read_meta_corrupt_returns_none(tmp_path):
    bad = tmp_path / "bad.meta.json"
    bad.write_text("this is not json", encoding="utf-8")
    assert read_meta(bad) is None


def test_read_meta_missing_returns_none(tmp_path):
    assert read_meta(tmp_path / "missing.meta.json") is None


def test_load_lab_papers_empty_dir(tmp_path):
    """No sources/ subdir → empty mapping."""
    assert load_lab_papers(tmp_path) == {}


def test_load_lab_papers_round_trip(tmp_path):
    sources = tmp_path / "sources"
    sources.mkdir()
    md = sources / "x.md"
    md.write_text("# Body\n", encoding="utf-8")
    paper = LabPaper(
        paper_id="x", paper_id_kind="sha256", title="Body", source_kind="pdf",
        source_path="/orig.pdf", parsed_path=str(md), n_chars=8, n_chunks=0,
        ingested_at="2026-05-09T10:00:00+00:00", figures_dir=None,
    )
    write_meta(sources / "x.meta.json", paper)

    out = load_lab_papers(tmp_path)
    assert list(out.keys()) == ["x"]
    assert out["x"].title == "Body"


def test_load_lab_papers_skips_orphan_meta(tmp_path):
    """meta.json without the matching markdown source is dropped."""
    sources = tmp_path / "sources"
    sources.mkdir()
    paper = LabPaper(
        paper_id="orphan", paper_id_kind="sha256", title=None, source_kind="pdf",
        source_path="/o.pdf",
        parsed_path=str(sources / "orphan.md"),  # never created
        n_chars=0, n_chunks=0, ingested_at="", figures_dir=None,
    )
    write_meta(sources / "orphan.meta.json", paper)

    assert load_lab_papers(tmp_path) == {}


def test_load_lab_papers_skips_corrupt_meta(tmp_path):
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "broken.meta.json").write_text("not-json", encoding="utf-8")
    assert load_lab_papers(tmp_path) == {}
