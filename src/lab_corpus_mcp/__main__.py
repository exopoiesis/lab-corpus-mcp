"""CLI entry point. Delegates to arxiv_radar_mcp's MCP server for now;
lab-specific subcommands (upload, jobs, parse) hook in here as they
land.
"""
from __future__ import annotations

import sys


def main() -> int:
    # No lab-specific CLI surface yet — pass through to arxiv_radar_mcp.
    # When upload / jobs / corpus_stats tools are ready they'll be wired
    # into the MCP server's TOOL_SPECS via a subclass of RadarServer in
    # this package, not as separate CLI verbs.
    from arxiv_radar_mcp.__main__ import main as _radar_main
    return _radar_main()


if __name__ == "__main__":
    sys.exit(main())
