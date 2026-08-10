#!/usr/bin/env python3
"""Restart the two formal rewrite arms with an ambient-safe thermal policy.

The previous continuation completed golden8 but its formal OC4 arm stopped at
160 diagnostic samples because the board could not cool from 70 C to the exact
60 C resume threshold within ten minutes.  This run reuses the immutable HEFs
and the passed golden8 gate, preserves the 70 C start guard, and changes only
the non-semantic cooling schedule for both candidates.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shlex
import sys
import time
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hailo_deploy.io.hash import sha256  # noqa: E402
from pipelines.run_bn_g_s_clip_feasibility import (  # noqa: E402
    BASE_MANIFEST,
    CHECKPOINT_SHA,
    atomic_write,
    ssh,
    update_state,
)
import pipelines.continue_bn_g_s_equivalent_rewrites_formal as first_continuation  # noqa: E402
import pipelines.run_bn_g_s_branch_campaign as formal  # noqa: E402
import pipelines.run_bn_g_s_equivalent_rewrites_formal as campaign  # noqa: E402


PRIOR_ROOT = Path(
    "/local/shared_with_docker/deplyment-demo/artifacts/fused/experiments/"
    "bn-gs-equivalent-rewrites-formal-continuation-20260806T093058Z"
)
PRIOR_REMOTE_ROOT = (
    ".local/share/semantic-sensing-board/hybrid-ort/"
    "bn-gs-equivalent-rewrites-formal-continuation-20260806T093058Z"
)
THERMAL_POLICY = {
    "max_start_temp_c": 70.0,
    "resume_temp_c": 62.0,
    "poll_seconds": 2.0,
    "max_wait_seconds": 1800.0,
}
ACCURACY_INTERVAL_SECONDS = 2.0
PRELAUNCH_TEMP_C = 62.0


def validate_prior() -> tuple[dict, dict[str, dict], dict, dict]:
    validated, arms, compile_reports = first_continuation.validate_compiled_source()
    state_path = PRIOR_ROOT / "campaign_state.json"
    golden_path = PRIOR_ROOT / "golden8_summary.json"
    if not state_path.is_file() or not golden_path.is_file():
        raise FileNotFoundError("prior continuation state/golden8 summary is missing")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("status") != "FAIL":
        raise ValueError("prior continuation is not the reviewed failed run")
    error = state.get("error", "")
    if "thermal guard timed out" not in error or "resume=60.00C" not in error:
        raise ValueError("prior continuation did not fail at the reviewed thermal guard")
    if state.get("remote_stage_sample_count") != 160:
        raise ValueError("prior thermal failure sample count changed")
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    if golden.get("status") != "PASS" or golden.get("sample_count") != 8:
        raise ValueError("prior golden8 gate is incomplete")
    if tuple(golden.get("val_indices", ())) != campaign.SAMPLE_INDICES:
        raise ValueError("prior golden8 held-out indices changed")
    return validated, arms, compile_reports, golden


def link_remote_hefs(remote_root: str, arms: dict[str, dict]) -> None:
    commands = [f"mkdir -p {shlex.quote(remote_root + '/hef_equivalent')}"]
    for name, spec in arms.items():
        source = f"{PRIOR_REMOTE_ROOT}/hef_equivalent/{name}.hef"
        destination = f"{remote_root}/hef_equivalent/{name}.hef"
        digest = spec["hef_sha256"]
        commands.extend([
            f"test $(sha256sum {shlex.quote(source)} | cut -d' ' -f1) = {shlex.quote(digest)}",
            f"test ! -e {shlex.quote(destination)}",
            f"ln {shlex.quote(source)} {shlex.quote(destination)}",
            f"test $(sha256sum {shlex.quote(destination)} | cut -d' ' -f1) = {shlex.quote(digest)}",
        ])
    ssh(" && ".join(commands))


def wait_until_cool(args, state: dict, label: str) -> None:
    deadline = time.monotonic() + THERMAL_POLICY["max_wait_seconds"]
    while True:
        temperature = formal.board_temperature(args)
        update_state(
            args.out_root,
            state,
            stage=f"{label}:cooldown",
            waiting_for_cool=True,
            board_temperature_c=temperature,
        )
        if temperature <= PRELAUNCH_TEMP_C:
            update_state(args.out_root, state, waiting_for_cool=False)
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"board did not cool to {PRELAUNCH_TEMP_C:.1f} C within "
                f"{THERMAL_POLICY['max_wait_seconds']:.0f} seconds"
            )
        time.sleep(10.0)


def run_formal_arms(
    root: Path,
    state: dict,
    remote_root: str,
    remote_manifests: dict[str, str],
) -> dict[str, dict]:
    args = campaign.board_args(root, remote_root)
    results_root = root / "formal_results"
    results_root.mkdir(exist_ok=False)
    results: dict[str, dict] = {}
    for name, manifest in remote_manifests.items():
        stages = (
            (1, "preflight1", 0.0),
            (976, "accuracy976", ACCURACY_INTERVAL_SECONDS),
        )
        for limit, suffix, interval in stages:
            label = f"{name}_{suffix}"
            wait_until_cool(args, state, label)
            result, progress, pid_file, log = formal.launch_stage(
                args,
                label=label,
                evaluator="evaluate_hybrid_ort_local_dataset.py",
                manifest=manifest,
                limit=limit,
                interval=interval,
            )
            formal.poll_stage(args, state, label, result, progress, pid_file, log)
            data = formal.download_json(args, result, results_root / name / f"{suffix}.json")
            if data.get("sample_count") != limit:
                raise ValueError(f"{label}: sample_count mismatch")
            if tuple(data.get("val_indices", ())) != campaign.FULL_INDICES[:limit]:
                raise ValueError(f"{label}: held-out index sequence mismatch")
            if limit == 976:
                results[name] = data
            state.setdefault("completed", []).append(label)
            update_state(root, state)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    validated, arms, compile_reports, golden = validate_prior()
    if args.validate_only:
        print(json.dumps({
            "status": "PASS",
            "checkpoint_sha256": CHECKPOINT_SHA,
            "prior_root": str(PRIOR_ROOT),
            "prior_golden8_sha256": sha256(PRIOR_ROOT / "golden8_summary.json"),
            "thermal_policy": THERMAL_POLICY,
            "accuracy_interval_seconds": ACCURACY_INTERVAL_SECONDS,
            "reused_hefs": {name: spec["hef_sha256"] for name, spec in arms.items()},
        }, indent=2, sort_keys=True))
        return 0

    root = args.out_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    unexpected = [path.name for path in root.iterdir() if path.name not in {"run.log", "run.pid"}]
    if unexpected:
        raise FileExistsError(f"refusing non-empty thermal continuation root: {root}: {unexpected}")
    started = time.monotonic()
    state = {
        "kind": "bn_g_s_equivalent_rewrites_thermal_continuation_state",
        "status": "RUNNING",
        "stage": "prior_gates_validated",
        "started_unix_seconds": time.time(),
        "continued_from": str(PRIOR_ROOT),
        "completed": ["compiled_hefs_reused", "golden8_reused"],
        "thermal_policy": THERMAL_POLICY,
        "accuracy_interval_seconds": ACCURACY_INTERVAL_SECONDS,
        "error": None,
    }
    update_state(root, state)
    try:
        base = copy.deepcopy(validated["base_manifest"])
        base["thermal_guard"] = dict(THERMAL_POLICY)
        formal.validate_base_manifest(base)
        atomic_write(root / "frozen_thermal_continuation_inputs.json", {
            "kind": "bn_g_s_equivalent_rewrites_thermal_continuation_inputs",
            "status": "PASS",
            "checkpoint_sha256": CHECKPOINT_SHA,
            "continued_from": str(PRIOR_ROOT),
            "prior_state_sha256": sha256(PRIOR_ROOT / "campaign_state.json"),
            "prior_golden8": {
                "path": str(PRIOR_ROOT / "golden8_summary.json"),
                "sha256": sha256(PRIOR_ROOT / "golden8_summary.json"),
                "status": golden["status"],
                "sample_count": golden["sample_count"],
            },
            "base_manifest": {"path": str(BASE_MANIFEST), "sha256": sha256(BASE_MANIFEST)},
            "thermal_policy": THERMAL_POLICY,
            "accuracy_interval_seconds": ACCURACY_INTERVAL_SECONDS,
            "semantic_effect": "none; cooling schedule only",
            "timing_comparison_allowed": False,
            "reused_arms": {
                name: {
                    "hef": {"path": str(spec["hef"]), "sha256": spec["hef_sha256"]},
                    "qhar": {"path": str(spec["har"]), "sha256": spec["har_sha256"]},
                }
                for name, spec in arms.items()
            },
        })

        remote_root = f".local/share/semantic-sensing-board/hybrid-ort/{root.name}"
        update_state(root, state, stage="remote_setup")
        campaign.ensure_remote_root(remote_root)
        link_remote_hefs(remote_root, arms)
        state["remote_root"] = remote_root
        state["completed"].append("remote_setup")
        update_state(root, state, stage="prepare_full976")
        manifests = campaign.upload_runtime_manifests(root, remote_root, base, arms)
        for path in (root / "runtime_manifests").glob("*.json"):
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if manifest.get("thermal_guard") != THERMAL_POLICY:
                raise ValueError(f"thermal policy was not frozen in {path.name}")

        update_state(root, state, stage="full976")
        results = run_formal_arms(root, state, remote_root, manifests)
        if set(results) != {"output_channel_split_4", "spatial_height_split_2"}:
            raise RuntimeError("thermal continuation did not complete both formal arms")
        summary = campaign.build_summary(
            root, compile_reports, golden, results, time.monotonic() - started
        )
        summary.update({
            "continued_from": str(PRIOR_ROOT),
            "thermal_policy": THERMAL_POLICY,
            "accuracy_interval_seconds": ACCURACY_INTERVAL_SECONDS,
            "timing_comparison_allowed": False,
            "repair_scope": "cooling schedule only; model, HEFs and runtime topology unchanged",
        })
        atomic_write(root / "summary.json", summary)
        update_state(
            root,
            state,
            status="PASS_AWAITING_REVIEW",
            stage="complete",
            finished_unix_seconds=time.time(),
            elapsed_seconds=summary["elapsed_seconds"],
        )
        print(json.dumps({
            "status": "PASS_AWAITING_REVIEW",
            "summary": str(root / "summary.json"),
        }, sort_keys=True), flush=True)
        return 0
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        atomic_write(root / "failure.json", {"status": "FAIL", "error": error})
        update_state(
            root,
            state,
            status="FAIL",
            stage="failed",
            error=error,
            finished_unix_seconds=time.time(),
            elapsed_seconds=time.monotonic() - started,
        )
        print(error, file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
