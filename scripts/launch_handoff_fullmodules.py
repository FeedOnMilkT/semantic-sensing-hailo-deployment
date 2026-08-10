#!/usr/bin/env python3
"""Thin wrapper → pipelines.fullmodule_chain."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipelines.fullmodule_chain import main

if __name__ == "__main__":
    raise SystemExit(main())
