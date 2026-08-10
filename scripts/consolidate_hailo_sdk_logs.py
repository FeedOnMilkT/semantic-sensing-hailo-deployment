#!/usr/bin/env python3
"""Thin wrapper → hailo_deploy.artifacts.sdk_logs."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hailo_deploy.artifacts.sdk_logs import (  # noqa: F401
    DEFAULT_LOG_ROOT,
    consolidate,
    main,
    timestamp,
)

if __name__ == "__main__":
    raise SystemExit(main())
