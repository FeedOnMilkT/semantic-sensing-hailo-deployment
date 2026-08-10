#!/usr/bin/env python3
"""Fused mid-split parse only (no NATIVE)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipelines.parse_native import main

if __name__ == "__main__":
    raise SystemExit(
        main(
            [
                "--profile",
                "fused",
                "--no-native",
                "--section",
                "Fused mid-split · parse",
                "--results-name",
                "fused_parse",
                *sys.argv[1:],
            ]
        )
    )
