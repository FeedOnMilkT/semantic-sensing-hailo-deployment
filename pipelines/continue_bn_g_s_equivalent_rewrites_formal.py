#!/usr/bin/env python3
"""Continue the exact Conv4 rewrite campaign after an orchestration-only failure.

This continuation validates and reuses the already compiled HEFs and quantized
HARs from the original immutable run.  It starts at host/board golden8 and then
runs the approved OC4 and HT2 full976 formal evaluations in a new local and
remote result root.
"""
from __future__ import annotations

import argparse
import json
import os
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
    update_state,
)
import pipelines.run_bn_g_s_equivalent_rewrites_formal as campaign  # noqa: E402
import pipelines.run_bn_g_s_vstream_parity as parity  # noqa: E402


SOURCE_ROOT = Path(
    "/local/shared_with_docker/deplyment-demo/artifacts/fused/experiments/"
    "bn-gs-equivalent-rewrites-formal-20260806T090000Z"
)


def validate_compiled_source() -> tuple[dict, dict[str, dict], dict]:
    validated = campaign.validate_sources()
    state_path = SOURCE_ROOT / "campaign_state.json"
    failure_path = SOURCE_ROOT / "failure.json"
    compile_path = SOURCE_ROOT / "compile_results.json"
    host_sdk_path = SOURCE_ROOT / "host_sdk8.json"
    for path in (state_path, failure_path, compile_path, host_sdk_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    source_state = json.loads(state_path.read_text(encoding="utf-8"))
    if source_state.get("status") != "FAIL":
        raise ValueError("source campaign is not the frozen failed run")
    source_error = source_state.get("error", "")
    if "golden8/.result.progress.json" not in source_error or "FileNotFoundError" not in source_error:
        raise ValueError("source failure is not the reviewed result-directory bug")

    compile_reports = json.loads(compile_path.read_text(encoding="utf-8"))
    host_sdk = json.loads(host_sdk_path.read_text(encoding="utf-8"))
    if host_sdk.get("status") != "PASS" or host_sdk.get("sample_count") != 8:
        raise ValueError("source host SDK8 gate is incomplete")
    if tuple(host_sdk.get("val_indices", ())) != campaign.SAMPLE_INDICES:
        raise ValueError("source host SDK8 held-out indices changed")

    arms: dict[str, dict] = {}
    expected_contract = {((64, 64, 128), "UINT8"), ((32, 32, 256), "UINT16")}
    for name, source in campaign.ARMS.items():
        report = compile_reports.get(name)
        if not report or report.get("status") != "PASS":
            raise ValueError(f"{name}: source compile report is incomplete")
        hef = SOURCE_ROOT / "compile" / name / "bn_g_s.hef"
        expected_hef_sha = report["hef"]["sha256"]
        if not hef.is_file() or sha256(hef) != expected_hef_sha:
            raise ValueError(f"{name}: compiled HEF identity mismatch")
        qhar = source["qhar"]
        if not qhar.is_file() or sha256(qhar) != source["qhar_sha256"]:
            raise ValueError(f"{name}: quantized HAR identity mismatch")
        static = parity.hef_static(hef)
        contract = {
            (tuple(item["shape_hwc"]), item["format_type"])
            for item in static["outputs"]
        }
        if contract != expected_contract or len(static["network_groups"]) != 1:
            raise ValueError(f"{name}: compiled HEF contract changed")
        arms[name] = {
            "hef": hef,
            "hef_sha256": expected_hef_sha,
            "har": qhar,
            "har_sha256": source["qhar_sha256"],
            "role": source["role"],
        }
    return validated, arms, compile_reports


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--cuda-visible-devices", default="2")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    validated, arms, compile_reports = validate_compiled_source()
    if args.validate_only:
        print(json.dumps({
            "status": "PASS",
            "continued_from": str(SOURCE_ROOT),
            "checkpoint_sha256": CHECKPOINT_SHA,
            "reused_arms": {
                name: {"hef": str(spec["hef"]), "hef_sha256": spec["hef_sha256"]}
                for name, spec in arms.items()
            },
        }, indent=2, sort_keys=True))
        return 0

    root = args.out_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    unexpected = [path.name for path in root.iterdir() if path.name not in {"run.log", "run.pid"}]
    if unexpected:
        raise FileExistsError(f"refusing non-empty continuation root: {root}: {unexpected}")
    (root / "logs").mkdir(exist_ok=False)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    os.environ["HAILO_SDK_LOG_DIR"] = str(root / "logs")
    started = time.monotonic()
    state = {
        "kind": "bn_g_s_equivalent_rewrites_formal_continuation_state",
        "status": "RUNNING",
        "stage": "compiled_source_validated",
        "started_unix_seconds": time.time(),
        "continued_from": str(SOURCE_ROOT),
        "completed": ["compiled_source_validated"],
        "error": None,
    }
    update_state(root, state)
    try:
        atomic_write(root / "frozen_continuation_inputs.json", {
            "kind": "bn_g_s_equivalent_rewrites_continuation_inputs",
            "status": "PASS",
            "continued_from": str(SOURCE_ROOT),
            "source_campaign_state_sha256": sha256(SOURCE_ROOT / "campaign_state.json"),
            "source_compile_results_sha256": sha256(SOURCE_ROOT / "compile_results.json"),
            "source_host_sdk8_sha256": sha256(SOURCE_ROOT / "host_sdk8.json"),
            "base_manifest": {"path": str(BASE_MANIFEST), "sha256": sha256(BASE_MANIFEST)},
            "reused_arms": {
                name: {
                    "hef": {"path": str(spec["hef"]), "sha256": spec["hef_sha256"]},
                    "qhar": {"path": str(spec["har"]), "sha256": spec["har_sha256"]},
                    "role": spec["role"],
                }
                for name, spec in arms.items()
            },
            "repair": {
                "cause": "golden8 result parent directory was not created",
                "board_atomic_json_parent_guard": True,
                "host_prelaunch_parent_guard": True,
                "model_or_quantization_changed": False,
            },
        })

        remote_root = f".local/share/semantic-sensing-board/hybrid-ort/{root.name}"
        update_state(root, state, stage="remote_setup")
        campaign.ensure_remote_root(remote_root)
        state["remote_root"] = remote_root
        state["completed"].append("remote_setup")
        update_state(root, state, stage="golden8")

        _board, golden_summary = campaign.run_golden8(root, state, remote_root, arms)
        state["completed"].append("golden8")
        update_state(root, state, stage="prepare_full976")
        manifests = campaign.upload_runtime_manifests(
            root, remote_root, validated["base_manifest"], arms
        )
        update_state(root, state, stage="full976")
        results = campaign.run_full976(root, state, remote_root, manifests)
        if set(results) != {"output_channel_split_4", "spatial_height_split_2"}:
            raise RuntimeError("continuation did not complete both approved full976 arms")

        summary = campaign.build_summary(
            root, compile_reports, golden_summary, results, time.monotonic() - started
        )
        summary["continued_from"] = str(SOURCE_ROOT)
        summary["repair_scope"] = "orchestration path guards only; model and runtime topology unchanged"
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
