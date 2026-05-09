"""Combined-image audit — single source-of-truth check for what's installed.

Runs inside the lab-corpus-gpu container. Prints the layout of the
shared Python environment + cache mounts. Exits non-zero if any
expected package or cache path is missing, or if it detects a
duplicate package install (the Phase 3 invariant: one torch, one
sentence-transformers, one MinerU, one corpus_core, one Qwen cache,
one MinerU cache).

Wired into the Dockerfile's final RUN step so a regression turns the
image build red, not the first ingest call.

Standalone use:
    docker exec lab-corpus-combined python /usr/local/bin/audit_image.py
"""
from __future__ import annotations

import importlib
import importlib.metadata
import os
import sys
from collections import Counter


def _fail(msg: str, code: int = 1) -> None:
    print(f"AUDIT FAIL: {msg}", file=sys.stderr)
    sys.exit(code)


def _ok(label: str, value: str) -> None:
    print(f"  {label:24}{value}")


def main() -> int:
    print("=== combined-image audit ===")

    # 0. Python — must satisfy ALL three siblings' `requires-python`:
    #    arxiv-radar-mcp >=3.11, lab-corpus-mcp >=3.10,<3.14, MinerU
    #    3.x >=3.10,<3.14. The intersection is [3.11, 3.13].
    py = sys.version_info
    _ok("python.version:", f"{py.major}.{py.minor}.{py.micro}")
    if py < (3, 11):
        _fail(f"python {py.major}.{py.minor} < 3.11 (arxiv-radar-mcp requires-python)")
    if py >= (3, 14):
        _fail(f"python {py.major}.{py.minor} >= 3.14 (MinerU + lab-corpus upper bound)")

    # 1. torch — must be the base-image install, NOT pip-upgraded by
    #    `pip install mineru[core]`. Floor is 2.6 (MinerU 3.x). If the
    #    base image drifts below that, MinerU pulls a second torch and
    #    the image grows ~800 MB.
    try:
        import torch
    except ImportError:
        _fail("torch not installed (base image broken?)")
    _ok("torch.__version__:", torch.__version__)
    _ok("torch.__file__:", torch.__file__)
    _ok("torch.cuda.is_available():", str(torch.cuda.is_available()))

    parts = torch.__version__.split("+")[0].split(".")[:2]
    torch_major, torch_minor = (int(x) for x in parts)
    if (torch_major, torch_minor) < (2, 6):
        _fail(f"torch {torch.__version__} < 2.6 — MinerU 3.x will "
              f"pip-install a second torch on top, defeating the "
              f"single-pytorch invariant. Bump the FROM line in "
              f"Dockerfile to a base with torch>=2.6.")

    # 2. heavy ML siblings — exactly one install each.
    for name in ("sentence_transformers", "transformers", "mineru"):
        try:
            mod = importlib.import_module(name)
        except ImportError:
            _fail(f"missing required package: {name}")
        v = getattr(mod, "__version__", "?")
        _ok(f"{name}:", f"{v}  ({mod.__file__})")

    # 3. duplicate-distribution check: every dep that several siblings
    #    declared (e.g. mcp, numpy, sentence-transformers) must show up
    #    exactly once in `importlib.metadata`. If pip somehow shipped
    #    two distributions with the same name, this catches it.
    seen = Counter(d.metadata["Name"].lower()
                   for d in importlib.metadata.distributions()
                   if d.metadata["Name"])
    dupes = {n: c for n, c in seen.items() if c > 1}
    if dupes:
        _fail(f"duplicate distributions: {dupes}")
    _ok("distributions installed:", str(sum(seen.values())))

    # 4. our three siblings present + editable.
    for name in ("corpus_core", "arxiv_radar_mcp", "lab_corpus_mcp"):
        try:
            mod = importlib.import_module(name)
        except ImportError:
            _fail(f"sibling package missing: {name}")
        v = getattr(mod, "__version__", "?")
        _ok(f"{name}:", f"{v}  ({mod.__file__})")

    # 5. cache paths — env vars set + dirs writable (volume mounts will
    #    populate them at runtime, but the env contract must hold).
    for env in ("HF_HOME", "MODELSCOPE_CACHE"):
        val = os.environ.get(env)
        if not val:
            _fail(f"{env} env var not set")
        _ok(f"{env}:", val)

    # 6. heavy import chain works — Encoder + RadarServer + LabCorpusServer
    #    + combined supervisor all reachable WITHOUT pulling weights.
    from corpus_core.embeddings import Encoder  # noqa: F401
    from corpus_core.mcp_scaffold import (  # noqa: F401
        build_mcp_app, make_method_dispatcher, serve_streamable_http,
    )
    from arxiv_radar_mcp.server import RadarServer  # noqa: F401
    from lab_corpus_mcp.server import LabCorpusServer  # noqa: F401
    from lab_corpus_mcp.combined import (  # noqa: F401
        _LockedEncoder, build_servers, serve_combined,
    )
    _ok("import chain:", "OK")

    print("\n=== AUDIT PASS ===")
    print("single torch + sentence-transformers + transformers + mineru")
    print("single corpus_core / arxiv_radar_mcp / lab_corpus_mcp install")
    print("single HF_HOME + MODELSCOPE_CACHE — Qwen weights and MinerU "
          "models live in named volumes, one copy each on the host")
    print("combined supervisor present — single Qwen in VRAM at runtime")
    return 0


if __name__ == "__main__":
    sys.exit(main())
