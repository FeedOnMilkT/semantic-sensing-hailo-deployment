#!/usr/bin/env python3
"""Thin wrapper → pipelines.parse_native with --no-native."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipelines.parse_native import main

if __name__ == "__main__":
    raise SystemExit(main(["--profile", "demo3", "--no-native", *sys.argv[1:]]))
