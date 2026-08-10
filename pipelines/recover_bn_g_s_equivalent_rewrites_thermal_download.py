#!/usr/bin/env python3
"""Recover a completed HT2 result after a transient SSH download failure."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hailo_deploy.io.hash import sha256
from pipelines.run_bn_g_s_clip_feasibility import atomic_write, update_state
import pipelines.continue_bn_g_s_equivalent_rewrites_thermal as thermal
import pipelines.run_bn_g_s_branch_campaign as formal
import pipelines.run_bn_g_s_equivalent_rewrites_formal as campaign


HT2 = "spatial_height_split_2"
HIGH_SUPPORT = [7, 10, 15, 20, 23, 30, 37, 83, 102]


def high_support_iou(result: dict) -> float:
    values = result["semantic"]["per_class_iou"]
    return sum(float(values[index]) for index in HIGH_SUPPORT) / len(HIGH_SUPPORT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.out_root.resolve()
    state_path = root / "campaign_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    error = str(state.get("error", ""))
    if state.get("status") != "FAIL" or state.get("stage") != "failed":
        raise ValueError("campaign is not in the expected failed state")
    if "result.json" not in error or "exit status 255" not in error:
        raise ValueError("failure is not the reviewed transient SSH download error")
    if state.get("remote_stage_status") != "COMPLETE" or state.get("remote_stage_sample_count") != 976:
        raise ValueError("remote HT2 stage is not recorded COMPLETE at 976 samples")

    destination = root / "formal_results" / HT2 / "accuracy976.json"
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite recovered result: {destination}")
    remote = f"{state['remote_root']}/experiment/{HT2}_accuracy976/result.json"
    board_args = campaign.board_args(root, str(state["remote_root"]))
    result = formal.download_json(board_args, remote, destination)
    if result.get("sample_count") != 976:
        raise ValueError("recovered HT2 sample_count mismatch")
    if tuple(result.get("val_indices", ())) != campaign.FULL_INDICES:
        raise ValueError("recovered HT2 held-out indices mismatch")
    if result.get("checkpoint_sha256") != campaign.CHECKPOINT_SHA:
        raise ValueError("recovered HT2 checkpoint mismatch")

    oc4_path = root / "formal_results" / "output_channel_split_4" / "accuracy976.json"
    oc4 = json.loads(oc4_path.read_text(encoding="utf-8"))
    _, _, compile_reports, golden = thermal.validate_prior()
    results = {"output_channel_split_4": oc4, HT2: result}
    summary = campaign.build_summary(
        root, compile_reports, golden, results, float(state.get("elapsed_seconds", 0.0))
    )
    for name, data in results.items():
        summary["candidates"][name]["high_support_9_macro_iou"] = high_support_iou(data)
    summary.update({
        "selection_metric": "high_support_9_macro_iou",
        "high_support_rule": {
            "source_set": "fixed_19",
            "minimum_positive_images": 20,
            "class_indices": HIGH_SUPPORT,
        },
        "fixed_19_role": "retained_diagnostic",
        "continued_from": str(thermal.PRIOR_ROOT),
        "thermal_policy": thermal.THERMAL_POLICY,
        "accuracy_interval_seconds": thermal.ACCURACY_INTERVAL_SECONDS,
        "timing_comparison_allowed": False,
        "repair_scope": "download-only recovery; no inference, model, HEF, or topology change",
        "recovered_result_sha256": sha256(destination),
    })
    atomic_write(root / "summary.json", summary)
    atomic_write(root / "download_recovery.json", {
        "status": "PASS",
        "recovered_utc_unix_seconds": time.time(),
        "remote": remote,
        "local": str(destination),
        "sha256": sha256(destination),
        "sample_count": 976,
        "val_index_first": 1024,
        "val_index_last": 1999,
        "recovery_scope": "download only",
        "superseded_failure": error.splitlines()[0],
    })
    state.setdefault("completed", []).append(f"{HT2}_accuracy976")
    update_state(
        root,
        state,
        status="PASS_AWAITING_REVIEW",
        stage="complete",
        error=None,
        recovered_error=error.splitlines()[0],
        recovered_result_sha256=sha256(destination),
        finished_unix_seconds=time.time(),
    )
    print(json.dumps({
        "status": "PASS_AWAITING_REVIEW",
        "result": str(destination),
        "summary": str(root / "summary.json"),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
