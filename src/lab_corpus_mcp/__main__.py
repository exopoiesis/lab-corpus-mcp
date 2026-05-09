"""CLI entrypoint: `lab-corpus-mcp`.

Modes:

  Local stdio (default — Claude Desktop or local agent):
      lab-corpus-mcp [--config PATH]

  Remote HTTP backend (long-running on GPU host):
      lab-corpus-mcp --transport http [--bind HOST] [--port PORT]
                     [--config PATH]

  Local stdio→remote-HTTP proxy (for Claude Desktop pointing at a remote backend):
      lab-corpus-mcp --remote user@host [--remote-port 8765]

  Combined arxiv-radar + lab-corpus on one container, sharing one Qwen
  copy in VRAM (12 GB GPU friendly):
      lab-corpus-mcp --mode combined
                     --arxiv-config /srv/arxiv-radar/radar.toml
                     --lab-config   /srv/lab-corpus/radar.toml
                     [--bind 0.0.0.0] [--arxiv-port 8765] [--lab-port 8766]
                     [--no-encoder-lock]
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
    parser.add_argument(
        "--mode", choices=["single", "combined"], default="single",
        help="`single` (default) — only this server runs. "
             "`combined` — arxiv-radar + lab-corpus in one process, "
             "sharing one Qwen instance in VRAM. Implies --transport=http.",
    )
    parser.add_argument("--config", type=Path, default=None,
                        help="path to radar.toml (single-mode only; default: platform user-config)")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    transport_group = parser.add_argument_group("transport (single mode)")
    transport_group.add_argument(
        "--transport", choices=["stdio", "http"], default="stdio",
        help="MCP transport (default: stdio for direct Claude Desktop use; "
             "use 'http' for a long-running backend on a GPU host)",
    )
    transport_group.add_argument(
        "--bind", default="127.0.0.1",
        help="host to bind for --transport=http or --mode=combined "
             "(default: 127.0.0.1; combined mode auto-overrides to 0.0.0.0 "
             "if you don't pass --bind)",
    )
    transport_group.add_argument(
        "--port", type=int, default=8766,
        help="port for --transport=http (default: 8766)",
    )

    remote_group = parser.add_argument_group("remote-proxy mode")
    remote_group.add_argument(
        "--remote", default=None, metavar="USER@HOST",
        help="run as stdio→HTTP proxy. Mutually exclusive with --transport / --mode=combined.",
    )
    remote_group.add_argument(
        "--remote-port", type=int, default=8766,
        help="remote backend port for SSH tunnel (default: 8766)",
    )
    remote_group.add_argument(
        "--ssh-binary", default="ssh",
        help="path to ssh binary (default: 'ssh' on PATH)",
    )

    combined_group = parser.add_argument_group("combined mode")
    combined_group.add_argument(
        "--arxiv-config", type=Path, default=None,
        help="path to arxiv-radar-mcp's radar.toml (default: platform user-config)",
    )
    combined_group.add_argument(
        "--lab-config", type=Path, default=None,
        help="path to lab-corpus-mcp's radar.toml (default: platform user-config)",
    )
    combined_group.add_argument(
        "--arxiv-port", type=int, default=8765,
        help="HTTP port for the arxiv-radar backend (default: 8765)",
    )
    combined_group.add_argument(
        "--lab-port", type=int, default=8766,
        help="HTTP port for the lab-corpus backend (default: 8766)",
    )
    combined_group.add_argument(
        "--no-encoder-lock", action="store_true",
        help="disable the threading.Lock around shared Encoder calls "
             "(use only when you have VRAM headroom and want concurrent "
             "encode throughput).",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        stream=sys.stderr,
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.mode == "combined":
        if args.remote:
            parser.error("--remote and --mode=combined are mutually exclusive")
        # Default 0.0.0.0 in combined mode unless user explicitly bound to localhost.
        host = args.bind if args.bind != "127.0.0.1" else "0.0.0.0"
        from lab_corpus_mcp.combined import serve_combined
        serve_combined(
            arxiv_config_path=args.arxiv_config,
            lab_config_path=args.lab_config,
            host=host,
            arxiv_port=args.arxiv_port,
            lab_port=args.lab_port,
            encoder_lock=not args.no_encoder_lock,
        )
        return 0

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
