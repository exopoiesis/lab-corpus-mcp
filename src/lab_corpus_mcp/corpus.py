"""Lab corpus paper schema + on-disk loader.

Where `arxiv-radar-mcp` keys papers by `arxiv_id` parsed from a JSON
shard, lab-corpus-mcp's papers come from heterogeneous sources (PDFs,
DOCX, sideloaded preprints). The canonical store is the on-disk tree
under `<parse.dir>/sources/<paper_id>.{md,meta.json}` written by the
ingest pipeline (`ingest.py`).

`paper_id` ∈ {DOI, sha256-of-pdf, arxiv_id, url-hash}, distinguished by
`paper_id_kind`. Phase 2B-1 derives ids from the file's sha256 prefix
when the user doesn't supply one explicitly; DOI / arxiv-id extraction
from PDF metadata is a Phase 2B+ enhancement.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

PAPER_ID_KINDS = ("doi", "sha256", "arxiv_id", "url_hash", "user_supplied")

# Conservative arxiv-id pattern (post-2007 numeric, e.g. 2503.12345 / 2503.12345v2).
_ARXIV_RE = re.compile(r"\b(\d{4}\.\d{4,5})(?:v\d+)?\b")


@dataclass
class LabPaper:
    """Metadata for one ingested document, mirrored to `<id>.meta.json`."""
    paper_id: str
    paper_id_kind: str
    title: str | None
    source_kind: str          # "pdf" / "docx" / "pptx" / "image" / "unknown"
    source_path: str          # absolute path to original file (or URL)
    parsed_path: str          # absolute path to <parse.dir>/sources/<paper_id>.md
    n_chars: int
    n_chunks: int = 0         # populated post-reindex
    ingested_at: str = ""     # ISO-8601 UTC
    figures_dir: str | None = None
    extra: dict = field(default_factory=dict)


def utcnow_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def sha256_prefix(path: Path, *, length: int = 16) -> str:
    """Stream-hash a file and return the first `length` hex chars."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:length]


def derive_paper_id(path: Path, *, content: bytes | None = None) -> tuple[str, str]:
    """Decide `(paper_id, paper_id_kind)` for a file we're about to ingest.

    Phase 2B-1: filename-based arxiv_id detection (cheap, deterministic),
    else sha256 prefix. PDF-content DOI extraction lands in Phase 2B+ when
    we wire pypdf or similar.
    """
    m = _ARXIV_RE.search(path.stem)
    if m:
        return m.group(1), "arxiv_id"

    # sha256 of file contents (16 hex chars ≈ 64-bit collision-resistance,
    # plenty for a personal corpus). `content` lets callers pass an
    # already-loaded buffer for tests / streaming pipelines.
    if content is None:
        sha = sha256_prefix(path)
    else:
        sha = hashlib.sha256(content).hexdigest()[:16]
    return f"sha256-{sha}", "sha256"


def source_kind_for(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    return suffix if suffix in {"pdf", "docx", "pptx", "png", "jpg", "jpeg"} else "unknown"


def write_meta(meta_path: Path, paper: LabPaper) -> None:
    """Persist a LabPaper as JSON next to its parsed markdown."""
    meta_path.write_text(
        json.dumps(asdict(paper), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def read_meta(meta_path: Path) -> LabPaper | None:
    """Inverse of `write_meta`. Returns None on missing/corrupt JSON."""
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    return LabPaper(**data)


def load_lab_papers(parse_dir: Path) -> dict[str, LabPaper]:
    """Walk `<parse.dir>/sources/*.meta.json` → {paper_id: LabPaper}.

    Skips entries with no matching markdown source on disk (orphaned
    metadata). Cheap — millisecond-class for thousands of papers.
    """
    sources_dir = parse_dir / "sources"
    out: dict[str, LabPaper] = {}
    if not sources_dir.exists():
        return out
    for meta_path in sources_dir.glob("*.meta.json"):
        paper = read_meta(meta_path)
        if paper is None:
            continue
        if not Path(paper.parsed_path).exists():
            continue
        out[paper.paper_id] = paper
    return out


def extract_title_from_markdown(text: str) -> str | None:
    """Pluck a candidate title from the first `# heading` of a markdown.

    Falls back to None when nothing looks like a heading in the first
    20 lines (MinerU sometimes emits images / table-of-contents
    placeholders before the title).
    """
    for line in text.splitlines()[:20]:
        s = line.strip()
        if s.startswith("# ") and len(s) > 2:
            return s[2:].strip()
    return None
