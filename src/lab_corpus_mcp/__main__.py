"""CLI entrypoint: `lab-corpus-mcp`.

Modes:

  Local stdio (default — Claude Desktop or local agent):
      lab-corpus-mcp [--config PATH]

  Remote HTTP backend (long-running on GPU host):
      lab-corpus-mcp --transport http [--bind HOST] [--port PORT]
                     [--config PATH]

  Local stdio→remote-HTTP proxy (for Claude Desktop pointing at a remote backend):
      lab-corpus-mcp --remote user@host [--remote-port 8765]

Phase 2A — the tool surface is the four-tool skeleton in `server.py`
(`corpus_stats`, `list_corpus`, `job_status`, `job_list`). Ingest /
search tools land in Phase 2B.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="lab-corpus-mcp",
        description="MCP server for a personal lab corpus (PDF/video/notes).",
    )
    parser.add_argument("--config", type=Path, default=None,
                        help="path to radar.toml (default: platform user-config)")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    transport_group = parser.add_argument_group("transport")
    transport_group.add_argument(
        "--transport", choices=["stdio", "http"], default="stdio",
        help="MCP transport (default: stdio for direct Claude Desktop use; "
             "use 'http' for a long-running backend on a GPU host)",
    )
    transport_group.add_argument(
        "--bind", default="127.0.0.1",
        help="host to bind for --transport=http (default: 127.0.0.1 — "
             "expose only via SSH tunnel, see README)",
    )
    transport_group.add_argument(
        "--port", type=int, default=8766,
        help="port for --transport=http (default: 8766; arxiv-radar-mcp uses 8765)",
    )

    remote_group = parser.add_argument_group("remote-proxy mode")
    remote_group.add_argument(
        "--remote", default=None, metavar="USER@HOST",
        help="run as stdio→HTTP proxy: open SSH tunnel to USER@HOST and "
             "forward MCP traffic to the backend. Mutually exclusive with --transport.",
    )
    remote_group.add_argument(
        "--remote-port", type=int, default=8766,
        help="remote backend port for SSH tunnel (default: 8766)",
    )
    remote_group.add_argument(
        "--ssh-binary", default="ssh",
        help="path to ssh binary (default: 'ssh' on PATH)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        stream=sys.stderr,
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.remote and args.transport != "stdio":
        parser.error("--remote and --transport=http are mutually exclusive — "
                     "the proxy itself runs on stdio")

    if args.remote:
        from corpus_core.proxy import run_proxy
        return run_proxy(
            target=args.remote,
            remote_port=args.remote_port,
            ssh_binary=args.ssh_binary,
        )

    if args.transport == "http":
        from lab_corpus_mcp.server import serve_http
        serve_http(host=args.bind, port=args.port, config_path=args.config)
        return 0

    from lab_corpus_mcp.server import serve
    serve(config_path=args.config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
