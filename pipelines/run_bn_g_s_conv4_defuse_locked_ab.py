#!/usr/bin/env python3
"""Locked compiler A/B: resize2 N2 + conv4 S4 + conv5 N4.

This is a deliberately narrow configuration of
``run_bn_g_s_conv4_defuse_ab``.  It schedules one compile-only arm and rejects
the compiled auto.alls unless all three allocation commands are realized
exactly.  Parse/optimize are never called.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import pipelines.run_bn_g_s_conv4_defuse_ab as campaign


LOCKED_ARM = "conv4_spatial4_locked"
LOCKED_SCRIPT = (
    "lock_r2_d0, lock_r2_d1, lock_r2_dc = "
    "defuse(bn_g_s/resize2, 2)\n"
    "lock_c4_sd0, lock_c4_sd1, lock_c4_sd2, lock_c4_sd3, lock_c4_sdc = "
    "defuse(bn_g_s/conv4, 4, defuse_type=SPATIAL)\n"
    "lock_c5_d0, lock_c5_d1, lock_c5_d2, lock_c5_d3, lock_c5_dc = "
    "defuse(bn_g_s/conv5, 4)\n"
)


def configure() -> None:
    campaign.SCHEDULED_ARM = LOCKED_ARM
    campaign.CANDIDATES = {
        LOCKED_ARM: campaign.Candidate(
            LOCKED_SCRIPT,
            "SCHEDULED_LOCKED_TOPOLOGY",
            "only resize2 NORMAL-2 + conv4 SPATIAL-4 + conv5 NORMAL-4 is allowed",
        )
    }
    campaign.EXPECTED_REALIZED_DEFUSES = {
        "resize2": {"count": 2, "type": "NORMAL"},
        "conv4": {"count": 4, "type": "SPATIAL"},
        "conv5": {"count": 4, "type": "NORMAL"},
    }
    # Commands persisted by the shared implementation must point back to this
    # locked entry point, not to the generic spatial-count campaign.
    campaign.__file__ = __file__


def main() -> int:
    configure()
    return campaign.main()


if __name__ == "__main__":
    sys.exit(main())
