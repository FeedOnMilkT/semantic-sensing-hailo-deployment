#!/usr/bin/env python3
"""Thin wrapper → hailo_deploy.logging."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hailo_deploy.logging import main, reexec_if_unlogged  # noqa: F401

if __name__ == "__main__":
    raise SystemExit(main())
