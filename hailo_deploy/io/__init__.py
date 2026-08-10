"""I/O helpers shared by Hailo deployment runners."""

from .hash import sha256
from .calib import load_calib_handoff, load_loader, map_hn_inputs
from .metrics import metric, summarize
from .handoff import (
    FORMAL_CALIBRATION_COUNT,
    FORMAL_GOLDEN_COUNT,
    first_existing,
    load_json,
    validate_handoff,
)

__all__ = [
    "sha256",
    "load_calib_handoff",
    "load_loader",
    "map_hn_inputs",
    "metric",
    "summarize",
    "FORMAL_CALIBRATION_COUNT",
    "FORMAL_GOLDEN_COUNT",
    "first_existing",
    "load_json",
    "validate_handoff",
]
