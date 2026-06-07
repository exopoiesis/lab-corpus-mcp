#!/usr/bin/env python3
"""Thin shim — delegates to lab_corpus_mcp.upload_inbox.

Repo users can run this directly.  Users who installed via pip get the
`lab-corpus-upload` console script instead (same code, proper entry point).
"""
from lab_corpus_mcp.upload_inbox import main

if __name__ == "__main__":
    main()
