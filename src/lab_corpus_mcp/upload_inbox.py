#!/usr/bin/env python3
"""Upload local PDFs (or other documents) to a running lab-corpus-mcp HTTP server.

Files are POSTed to the server's /upload endpoint and written to its inbox
directory (<parse.dir>/inbox/). No docker cp / scp required.

After upload, ingest_inbox() is triggered automatically unless --no-ingest
is passed. You can then track the MinerU job via job_status in Claude.

Usage:
    # Upload a directory of PDFs and start ingest:
    lab-corpus-upload ~/papers/ http://gomer:8766

    # Upload without auto-ingest (trigger manually in Claude later):
    lab-corpus-upload ~/papers/ http://gomer:8766 --no-ingest

    # Upload PPTX slides:
    lab-corpus-upload ~/slides/ http://gomer:8766 --glob "*.pptx"

    # Upload a single file:
    lab-corpus-upload paper.pdf http://gomer:8766

    # Recurse into subdirectories:
    lab-corpus-upload ~/library/ http://gomer:8766 --recursive

Prerequisites: pip install lab-corpus-mcp  (httpx pulled in transitively)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _require_httpx():
    try:
        import httpx  # noqa: PLC0415
        return httpx
    except ImportError:
        sys.exit(
            "httpx not found.\n"
            "Install it with:  pip install httpx\n"
            "Or:               pip install lab-corpus-mcp"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("source",
                   help="Local directory or single file to upload.")
    p.add_argument("server",
                   help="lab-corpus-mcp HTTP base URL, e.g. http://localhost:8766")
    p.add_argument("--glob", default="*.pdf",
                   help="Glob pattern when source is a directory (default: %(default)s).")
    p.add_argument("--recursive", action="store_true",
                   help="Recurse into subdirectories.")
    p.add_argument("--no-ingest", dest="ingest", action="store_false", default=True,
                   help="Upload only — do not trigger ingest_inbox automatically.")
    p.add_argument("--timeout", type=float, default=120.0,
                   help="HTTP request timeout per file in seconds (default: %(default)s).")
    return p.parse_args()


def collect_files(source: str, glob: str, recursive: bool) -> list[Path]:
    p = Path(source).expanduser()
    if p.is_file():
        return [p]
    if p.is_dir():
        fn = p.rglob if recursive else p.glob
        return sorted(f for f in fn(glob) if f.is_file())
    sys.exit(f"error: source not found or not a file/directory: {source}")


def upload_file(client, url: str, filepath: Path, timeout: float) -> dict:
    with filepath.open("rb") as fh:
        r = client.post(url, files={"file": (filepath.name, fh)}, timeout=timeout)
    r.raise_for_status()
    return r.json()


def main() -> None:
    args = parse_args()
    httpx = _require_httpx()

    files = collect_files(args.source, args.glob, args.recursive)
    if not files:
        sys.exit(f"no files matching {args.glob!r} in {args.source}")

    base = args.server.rstrip("/")
    upload_url = f"{base}/upload"

    print(f"Uploading {len(files)} file(s) to {upload_url} ...")

    saved: list[str] = []
    errors: list[dict] = []
    last_job_id: str | None = None

    with httpx.Client() as client:
        for i, filepath in enumerate(files, 1):
            size_kb = filepath.stat().st_size // 1024
            is_last = i == len(files)
            trigger = args.ingest and is_last
            url = f"{upload_url}?ingest=true" if trigger else upload_url

            print(f"  [{i}/{len(files)}] {filepath.name} ({size_kb} KB) ... ",
                  end="", flush=True)
            try:
                result = upload_file(client, url, filepath, args.timeout)
                if result.get("saved"):
                    saved.extend(result["saved"])
                    job_id = result.get("job_id")
                    suffix = f" + ingest job {job_id}" if job_id else ""
                    print(f"OK{suffix}")
                    if job_id:
                        last_job_id = job_id
                else:
                    errs = result.get("errors", [])
                    msg = errs[0]["error"] if errs else "unknown error"
                    print(f"FAIL ({msg})")
                    errors.append({"file": filepath.name, "error": msg})
            except httpx.HTTPStatusError as e:
                print(f"HTTP {e.response.status_code}: {e.response.text[:120]}")
                errors.append({"file": filepath.name,
                                "error": f"HTTP {e.response.status_code}"})
            except httpx.ConnectError:
                print(f"connection refused — is lab-corpus-mcp running at {base}?")
                errors.append({"file": filepath.name, "error": "connection refused"})
                break
            except Exception as e:  # noqa: BLE001
                print(f"ERROR: {e}")
                errors.append({"file": filepath.name, "error": str(e)})

    print()
    print(f"Done: {len(saved)} uploaded, {len(errors)} failed.")
    if errors:
        for err in errors:
            print(f"  FAIL  {err['file']}: {err['error']}")

    if not saved:
        sys.exit(1)

    if not args.ingest:
        print()
        print("Files are in the server inbox. To start MinerU parsing:")
        print("  → call ingest_inbox() in Claude, or")
        print(f"  → lab-corpus-upload <any-file> {base} --no-ingest  # then trigger manually")
    elif last_job_id:
        print()
        print(f"MinerU ingest running (job_id={last_job_id}).")
        print("Track progress: job_status() in Claude Desktop.")


if __name__ == "__main__":
    main()
