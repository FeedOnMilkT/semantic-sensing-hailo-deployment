#!/usr/bin/env python3
"""Thin wrapper → pipelines.module_ptq fullmodule mode.

Prefer: ``python pipelines/module_ptq.py fullmodule ...``
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Re-export library helpers for scripts that still ``from run_fullmodule_real_ptq import ...``.
from hailo_deploy.io.calib import load_loader, map_hn_inputs  # noqa: F401
from hailo_deploy.io.hash import sha256  # noqa: F401
from hailo_deploy.io.handoff import (  # noqa: F401
    FORMAL_CALIBRATION_COUNT,
    FORMAL_GOLDEN_COUNT,
    first_existing,
    load_json,
    validate_handoff,
)
from hailo_deploy.io.metrics import metric, summarize  # noqa: F401
from hailo_deploy.gates.quantized_check import heldout_quantized_check  # noqa: F401


def main() -> int:
    from pipelines.module_ptq import main as pipeline_main

    return pipeline_main(["fullmodule", *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
