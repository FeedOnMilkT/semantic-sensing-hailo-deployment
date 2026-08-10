#!/usr/bin/env python3
"""Compile and formally evaluate two exact ``bn_g_s`` Conv4 rewrites.

The campaign freezes one contemporary unsplit control and two mathematically
equivalent graph rewrites.  It compiles every arm with the same DFC settings,
runs an eight-sample board-vs-SDK diagnostic, then evaluates only the two graph
rewrites on the immutable ADE20K validation indices 1024..1999.  The permanent
hybrid Hailo/ORT FP32 runtime boundary is never changed.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shlex
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from board_eval.deploy import stream_upload  # noqa: E402
from hailo_deploy.gates.compile_real import compile_quantized  # noqa: E402
from hailo_deploy.io.hash import sha256  # noqa: E402
from pipelines.run_bn_g_s_clip_feasibility import (  # noqa: E402
    BASE_MANIFEST,
    CHECKPOINT_SHA,
    IDENTITY,
    KNOWN_HOSTS,
    ONNX,
    ONNX_SHA,
    PROXY,
    REMOTE_DATASET,
    REMOTE_PYTHON,
    REMOTE_SEED,
    SSH_TARGET,
    atomic_write,
    launch_remote,
    poll_remote,
    ssh,
    ssh_options,
    ssh_readonly,
    update_state,
)
import pipelines.run_bn_g_s_branch_campaign as formal  # noqa: E402
import pipelines.run_bn_g_s_vstream_parity as parity  # noqa: E402


CPU_MODULES = ("bn_align_h", "bn_g_a", "bn_h_a", "bn_h_s")
HAILO_MODULES = (
    "sender_backbone",
    "bn_g_s",
    "receiver_stage4",
    "receiver_stage5",
    "receiver_dappm",
    "receiver_head",
)
SAMPLE_INDICES = tuple(range(1024, 1032))
FULL_INDICES = tuple(range(1024, 2000))
GOLDEN = parity.GOLDEN
GOLDEN_RESULT_SHA = parity.GOLDEN_RESULT_SHA
REMOTE_GOLDEN = parity.REMOTE_GOLDEN
FORMAL_RESULT = Path(
    "/local/shared_with_docker/deplyment-demo/artifacts/fused/"
    "board-hybrid-ort-formal-20260805T030725Z/gs_only_hailo_fp32/accuracy976.json"
)
FORMAL_FIXED19 = 0.2779885810957219
FORMAL_MEAN_BPP = 0.2098663830366291

CONTROL_ROOT = Path(
    "/local/shared_with_docker/deplyment-demo/artifacts/fused/experiments/"
    "bn-gs-current-formal-control-l2-20260806T072500Z"
)
OC4_ROOT = Path(
    "/local/shared_with_docker/deplyment-demo/artifacts/fused/experiments/"
    "bn-gs-conv4-graph-equivalence-20260806T070529Z"
)
HT2_ROOT = Path(
    "/local/shared_with_docker/deplyment-demo/artifacts/fused/experiments/"
    "bn-gs-conv4-height-tiling-equivalence-20260806T073625Z"
)

ARMS = {
    "control_unsplit": {
        "qhar": CONTROL_ROOT / "ptq/formal_l2/bn_g_s_l2_quantized.har",
        "qhar_sha256": "4d818f8add3b45194caf51718aac5ac6255c4f00d20dd4cda58978ea35500704",
        "onnx": ONNX,
        "onnx_sha256": ONNX_SHA,
        "role": "contemporary unsplit Conv4 compiler control",
        "full976": False,
    },
    "output_channel_split_4": {
        "qhar": OC4_ROOT / "ptq/output_channel_split_4_l2/bn_g_s_l2_quantized.har",
        "qhar_sha256": "a5fae52e28ab74d7b17888796dba81bd78dd357cd483f3ffc4cdd4e4611f16ad",
        "onnx": OC4_ROOT / "onnx/output_channel_split_4/bn_g_s.onnx",
        "onnx_sha256": None,
        "role": "exact Conv4 output-channel split into four 384-to-32 convolutions",
        "full976": True,
    },
    "spatial_height_split_2": {
        "qhar": HT2_ROOT / "ptq/spatial_height_split_2_l2/bn_g_s_l2_quantized.har",
        "qhar_sha256": "86f34058e35cd64bbfca96fbe05ff587e089c33b1a02bca434edeafd35240b71",
        "onnx": HT2_ROOT / "onnx/spatial_height_split_2/bn_g_s.onnx",
        "onnx_sha256": None,
        "role": "exact Conv4 two-way output-height tiling with one-row halo",
        "full976": True,
    },
}


def validate_sources() -> dict:
    if sha256(ONNX) != ONNX_SHA:
        raise ValueError("immutable bn_g_s ONNX SHA mismatch")
    if not (GOLDEN / "result.json").is_file():
        raise FileNotFoundError(GOLDEN / "result.json")
    if sha256(GOLDEN / "result.json") != GOLDEN_RESULT_SHA:
        raise ValueError("golden8 source result SHA mismatch")
    golden = json.loads((GOLDEN / "result.json").read_text(encoding="utf-8"))
    if golden.get("status") != "PASS" or tuple(golden.get("val_indices", ())[:8]) != SAMPLE_INDICES:
        raise ValueError("golden8 source does not cover immutable indices 1024..1031")

    base = json.loads(BASE_MANIFEST.read_text(encoding="utf-8"))
    formal.validate_base_manifest(base)
    if tuple(base.get("cpu_modules", ())) != CPU_MODULES:
        raise ValueError("runtime manifest CPU boundary changed")
    if tuple(base.get("hailo_modules", ())) != HAILO_MODULES:
        raise ValueError("runtime manifest Hailo boundary changed")

    formal_result = json.loads(FORMAL_RESULT.read_text(encoding="utf-8"))
    if formal_result.get("status") != "PASS" or formal_result.get("sample_count") != 976:
        raise ValueError("canonical formal full976 result is incomplete")
    if tuple(formal_result.get("val_indices", ())) != FULL_INDICES:
        raise ValueError("canonical formal result does not cover 1024..1999 exactly")
    fixed19 = formal_result["semantic"]["fixed_19_large_object_macro_iou"]
    mean_bpp = formal_result["bitstream"]["mean_bpp"]
    if fixed19 != FORMAL_FIXED19 or mean_bpp != FORMAL_MEAN_BPP:
        raise ValueError("canonical formal metric identity changed")

    frozen = {}
    for name, spec in ARMS.items():
        qhar = spec["qhar"]
        if not qhar.is_file() or sha256(qhar) != spec["qhar_sha256"]:
            raise ValueError(f"{name}: quantized HAR mismatch")
        onnx_path = spec["onnx"]
        if not onnx_path.is_file():
            raise FileNotFoundError(onnx_path)
        onnx_digest = sha256(onnx_path)
        if spec["onnx_sha256"] is not None and onnx_digest != spec["onnx_sha256"]:
            raise ValueError(f"{name}: ONNX mismatch")
        frozen[name] = {
            "qhar": {"path": str(qhar), "sha256": spec["qhar_sha256"], "bytes": qhar.stat().st_size},
            "onnx": {"path": str(onnx_path), "sha256": onnx_digest, "bytes": onnx_path.stat().st_size},
            "role": spec["role"],
            "full976": spec["full976"],
        }
    return {"base_manifest": base, "arms": frozen}


def compile_arms(root: Path, state: dict) -> tuple[dict[str, dict], dict[str, dict]]:
    from hailo_sdk_client import ClientRunner

    arm_specs: dict[str, dict] = {}
    reports: dict[str, dict] = {}
    expected_contract = {((64, 64, 128), "UINT8"), ((32, 32, 256), "UINT16")}
    for name, source in ARMS.items():
        update_state(root, state, stage=f"compile:{name}")
        destination = root / "compile" / name
        destination.mkdir(parents=True, exist_ok=False)
        runner = ClientRunner(har=str(source["qhar"]))
        result = compile_quantized(
            runner,
            hef_path=destination / "bn_g_s.hef",
            compiled_har=destination / "bn_g_s_compiled.har",
            auto_alls=destination / "bn_g_s.auto.alls",
        )
        static = parity.hef_static(destination / "bn_g_s.hef")
        contract = {(tuple(item["shape_hwc"]), item["format_type"]) for item in static["outputs"]}
        if contract != expected_contract:
            raise ValueError(f"{name}: compiled HEF output contract {contract} != {expected_contract}")
        if len(static["network_groups"]) != 1 or len(static["inputs"]) != 1 or len(static["outputs"]) != 2:
            raise ValueError(f"{name}: compiled HEF topology is not 1-input/2-output")
        result.update({
            "status": "PASS",
            "source_quantized_har": {
                "path": str(source["qhar"]),
                "sha256": source["qhar_sha256"],
            },
            "hef_static": static,
            "same_compile_policy": True,
        })
        reports[name] = result
        arm_specs[name] = {
            "hef": destination / "bn_g_s.hef",
            "hef_sha256": static["sha256"],
            "har": source["qhar"],
            "har_sha256": source["qhar_sha256"],
            "role": source["role"],
        }
        del runner
        atomic_write(root / "compile_results.partial.json", reports)
    return arm_specs, reports


def ensure_remote_root(remote_root: str) -> None:
    if ssh_readonly(
        f"test ! -e {shlex.quote(remote_root)} && echo ABSENT || echo PRESENT"
    ).strip() != "ABSENT":
        raise FileExistsError(f"remote experiment root exists: {remote_root}")
    ssh(
        f"test -d {shlex.quote(REMOTE_SEED)} || exit 4; "
        f"mkdir -p {shlex.quote(remote_root)}; "
        f"cp -al {shlex.quote(REMOTE_SEED + '/.')} {shlex.quote(remote_root + '/')}"
    )


def assert_board_idle() -> str:
    probe = ssh_readonly(
        "pgrep -af '[e]valuate_hybrid_ort|[e]valuate_bn_g_s|[b]oard_gate8' || true; "
        "cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null || true"
    )
    lines = [line for line in probe.splitlines() if line.strip()]
    if any("evaluate_" in line or "board_gate8" in line for line in lines[:-1]):
        raise RuntimeError(f"board has a residual evaluator:\n{probe}")
    return probe


def run_golden8(
    root: Path,
    state: dict,
    remote_root: str,
    arms: dict[str, dict],
) -> tuple[dict, dict]:
    parity.ARMS = arms
    update_state(root, state, stage="host_sdk8")
    references_path, host_report = parity.make_host_references(root)

    options = ssh_options()
    evaluator = ROOT / "board_eval/evaluate_bn_g_s_vstream_parity.py"
    dependencies = (
        ROOT / "board_eval/bn_g_s_branch_attribution_runtime.py",
        ROOT / "board_eval/hybrid_ort_runtime_runner.py",
        ROOT / "board_eval/fullmodule_runtime_runner.py",
    )
    for source in (evaluator, *dependencies):
        stream_upload(source, f"{remote_root}/{source.name}", options, SSH_TARGET)
    stream_upload(ONNX, f"{remote_root}/onnx/bn_g_s.onnx", options, SSH_TARGET)
    stream_upload(references_path, f"{remote_root}/{references_path.name}", options, SSH_TARGET)

    manifest_arms = []
    for name, spec in arms.items():
        destination = f"hef_equivalent/{name}.hef"
        stream_upload(spec["hef"], f"{remote_root}/{destination}", options, SSH_TARGET)
        manifest_arms.append({
            "name": name,
            "role": spec["role"],
            "hef_filename": destination,
            "hef_sha256": spec["hef_sha256"],
            "har_sha256": spec["har_sha256"],
        })
    manifest = {
        "kind": "bn_g_s_equivalent_rewrites_golden8_manifest",
        "checkpoint_sha256": CHECKPOINT_SHA,
        "sample_indices": list(SAMPLE_INDICES),
        "ort_threads": 4,
        "onnx": {"filename": "onnx/bn_g_s.onnx", "sha256": ONNX_SHA},
        "golden_bundle": {"result_sha256": GOLDEN_RESULT_SHA},
        "references": {
            "filename": references_path.name,
            "sha256": host_report["bundle"]["sha256"],
            "arrays": host_report["bundle"]["arrays"],
        },
        "arms": manifest_arms,
        "policy": {
            "diagnostic_only": True,
            "semantic_ranking": False,
            "permanent_cpu_modules": list(CPU_MODULES),
            "permanent_hailo_modules": list(HAILO_MODULES),
        },
    }
    manifest_path = root / "golden8_manifest.json"
    atomic_write(manifest_path, manifest)
    stream_upload(manifest_path, f"{remote_root}/golden8_manifest.json", options, SSH_TARGET)

    remote_golden_sha = ssh_readonly(
        f"sha256sum {shlex.quote(REMOTE_GOLDEN + '/result.json')} | cut -d' ' -f1"
    ).strip()
    if remote_golden_sha != GOLDEN_RESULT_SHA:
        raise ValueError("remote golden bundle identity mismatch")
    prelaunch = assert_board_idle()
    update_state(root, state, stage="board_golden8", board_prelaunch=prelaunch)
    result_remote = f"{remote_root}/golden8/result.json"
    raw_remote = f"{remote_root}/golden8/raw_outputs.npz"
    ssh(f"mkdir -p {shlex.quote(remote_root + '/golden8')}")
    job = launch_remote(
        remote_root,
        "board_golden8",
        [
            REMOTE_PYTHON,
            f"{remote_root}/{evaluator.name}",
            "--manifest",
            f"{remote_root}/golden8_manifest.json",
            "--bundle-root",
            REMOTE_GOLDEN,
            "--out",
            result_remote,
            "--raw-out",
            raw_remote,
        ],
        result_override=result_remote,
        progress_override=f"{remote_root}/golden8/result.progress.json",
    )
    update_state(root, state, stage="board_golden8", remote_job=job)
    poll_remote(root, state, job, timeout=1800)
    result_path = root / "golden8_board_result.json"
    raw_path = root / "golden8_raw_outputs.npz"
    parity.remote_download(result_remote, result_path)
    parity.remote_download(raw_remote, raw_path)
    board = json.loads(result_path.read_text(encoding="utf-8"))
    if board.get("status") != "PASS" or board.get("sample_count") != 8:
        raise RuntimeError("board golden8 result is incomplete")
    if tuple(board.get("sample_indices", board.get("val_indices", ()))) not in ((), SAMPLE_INDICES):
        raise ValueError("board golden8 returned unexpected sample indices")
    if sha256(raw_path) != board["raw_outputs"]["sha256"]:
        raise ValueError("downloaded golden8 raw bundle SHA mismatch")

    concise = {}
    for name, row in board["arms"].items():
        concise[name] = {
            branch: {
                metric: row["comparisons"]["board_float_vs_sdk"][branch][metric]
                for metric in ("aggregate_nmse", "cosine", "mae", "max_abs")
            }
            for branch in ("x_h_hat", "x_l_hat")
        }
    golden_summary = {
        "kind": "bn_g_s_equivalent_rewrites_golden8_summary",
        "status": "PASS",
        "sample_count": 8,
        "val_indices": list(SAMPLE_INDICES),
        "scope": "diagnostic correctness/localization only; never semantic ranking",
        "board_vs_own_sdk_quantized": concise,
        "board_ort_gate": board.get("board_ort_gate"),
        "artifacts": {
            "board_result": {"path": str(result_path), "sha256": sha256(result_path)},
            "raw_outputs": {"path": str(raw_path), "sha256": sha256(raw_path)},
            "host_sdk8": {"path": str(root / "host_sdk8.json"), "sha256": sha256(root / "host_sdk8.json")},
        },
    }
    atomic_write(root / "golden8_summary.json", golden_summary)
    return board, golden_summary


def board_args(root: Path, remote_root: str) -> SimpleNamespace:
    return SimpleNamespace(
        out_root=root,
        remote_root=remote_root,
        remote_dataset_root=REMOTE_DATASET,
        remote_ort_python=REMOTE_PYTHON,
        ssh_target=SSH_TARGET,
        identity_file=IDENTITY,
        proxy_command=PROXY,
        known_hosts=KNOWN_HOSTS,
    )


def upload_runtime_manifests(
    root: Path,
    remote_root: str,
    base: dict,
    arms: dict[str, dict],
) -> dict[str, str]:
    options = ssh_options()
    manifests_root = root / "runtime_manifests"
    manifests_root.mkdir(exist_ok=False)
    remote_manifests = {}
    for name, spec in arms.items():
        if not ARMS[name]["full976"]:
            continue
        candidate = copy.deepcopy(base)
        candidate["arm"] = f"bn_g_s_equivalent_rewrite_{name}"
        candidate["bn_g_s_candidate_description"] = spec["role"]
        gs = next(item for item in candidate["modules"] if item["name"] == "bn_g_s")
        gs.update({
            "hef_filename": f"hef_equivalent/{name}.hef",
            "hef_sha256": spec["hef_sha256"],
            "hef_bytes": spec["hef"].stat().st_size,
        })
        formal.validate_base_manifest(candidate)
        local_path = manifests_root / f"runtime_manifest_{name}.json"
        atomic_write(local_path, candidate)
        remote_name = f"runtime_manifest_{name}.json"
        stream_upload(local_path, f"{remote_root}/{remote_name}", options, SSH_TARGET)
        remote_manifests[name] = remote_name
    return remote_manifests


def run_full976(
    root: Path,
    state: dict,
    remote_root: str,
    remote_manifests: dict[str, str],
) -> dict[str, dict]:
    args = board_args(root, remote_root)
    results_root = root / "formal_results"
    results_root.mkdir(exist_ok=False)
    results = {}
    for name, remote_manifest in remote_manifests.items():
        for limit, suffix, interval in ((1, "preflight1", 0.0), (976, "accuracy976", 1.0)):
            label = f"{name}_{suffix}"
            if limit == 976:
                formal.wait_until_cool(args, state, label)
            result, progress, pid_file, log = formal.launch_stage(
                args,
                label=label,
                evaluator="evaluate_hybrid_ort_local_dataset.py",
                manifest=remote_manifest,
                limit=limit,
                interval=interval,
            )
            formal.poll_stage(args, state, label, result, progress, pid_file, log)
            data = formal.download_json(args, result, results_root / name / f"{suffix}.json")
            if data.get("sample_count") != limit:
                raise ValueError(f"{label}: sample_count {data.get('sample_count')} != {limit}")
            if tuple(data.get("val_indices", ())) != FULL_INDICES[:limit]:
                raise ValueError(f"{label}: held-out index sequence mismatch")
            if limit == 976:
                results[name] = data
            state.setdefault("completed", []).append(label)
            update_state(root, state)
    return results


def build_summary(
    root: Path,
    compile_reports: dict[str, dict],
    golden: dict,
    results: dict[str, dict],
    elapsed: float,
) -> dict:
    candidates = {}
    for name, result in results.items():
        fixed19 = result["semantic"]["fixed_19_large_object_macro_iou"]
        mean_bpp = result["bitstream"]["mean_bpp"]
        candidates[name] = {
            "sample_count": result["sample_count"],
            "val_index_first": result["val_indices"][0],
            "val_index_last": result["val_indices"][-1],
            "fixed_19_large_object_macro_iou": fixed19,
            "mean_bpp": mean_bpp,
            "delta_fixed19_vs_current_formal": fixed19 - FORMAL_FIXED19,
            "raw_result": str(root / "formal_results" / name / "accuracy976.json"),
        }
    return {
        "kind": "bn_g_s_equivalent_rewrites_formal_full976_summary",
        "status": "PASS_AWAITING_REVIEW",
        "elapsed_seconds": elapsed,
        "selection_metric": "fixed_19_large_object_macro_iou",
        "ignored_metric": "ADE20K 150-class mIoU",
        "numeric_acceptance_threshold": None,
        "permanent_runtime_boundary": {
            "hailo": list(HAILO_MODULES),
            "ort_cpu_fp32": list(CPU_MODULES),
            "cpu_codec": ["rounding", "CDF/index", "rANS", "byte container"],
        },
        "current_formal_reference": {
            "sample_count": 976,
            "fixed_19_large_object_macro_iou": FORMAL_FIXED19,
            "mean_bpp": FORMAL_MEAN_BPP,
            "raw_result": str(FORMAL_RESULT),
        },
        "candidates": candidates,
        "golden8_diagnostic": golden,
        "compile": compile_reports,
        "automatic_selection": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--cuda-visible-devices", default="2")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    validated = validate_sources()
    if args.validate_only:
        print(json.dumps({
            "status": "PASS",
            "checkpoint_sha256": CHECKPOINT_SHA,
            "sample_indices": list(SAMPLE_INDICES),
            "full_indices": [FULL_INDICES[0], FULL_INDICES[-1]],
            "arms": validated["arms"],
        }, indent=2, sort_keys=True))
        return 0

    root = args.out_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    unexpected = [path.name for path in root.iterdir() if path.name not in {"run.log", "run.pid"}]
    if unexpected:
        raise FileExistsError(f"refusing non-empty experiment root: {root}: {unexpected}")
    (root / "logs").mkdir(exist_ok=False)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    os.environ["HAILO_SDK_LOG_DIR"] = str(root / "logs")
    started = time.monotonic()
    state = {
        "kind": "bn_g_s_equivalent_rewrites_formal_state",
        "status": "RUNNING",
        "stage": "sources_validated",
        "started_unix_seconds": time.time(),
        "completed": ["sources_validated"],
        "error": None,
    }
    update_state(root, state)
    try:
        atomic_write(root / "frozen_inputs.json", {
            "kind": "bn_g_s_equivalent_rewrites_frozen_inputs",
            "status": "PASS",
            "checkpoint_sha256": CHECKPOINT_SHA,
            "base_manifest": {"path": str(BASE_MANIFEST), "sha256": sha256(BASE_MANIFEST)},
            "formal_result": {"path": str(FORMAL_RESULT), "sha256": sha256(FORMAL_RESULT)},
            "golden8_result": {"path": str(GOLDEN / "result.json"), "sha256": GOLDEN_RESULT_SHA},
            "arms": validated["arms"],
        })
        update_state(root, state, stage="compile")
        arms, compile_reports = compile_arms(root, state)
        atomic_write(root / "compile_results.json", compile_reports)
        state["completed"].append("compile_and_hef_contract")
        update_state(root, state, stage="remote_setup")

        remote_root = f".local/share/semantic-sensing-board/hybrid-ort/{root.name}"
        ensure_remote_root(remote_root)
        state["remote_root"] = remote_root
        state["completed"].append("remote_setup")
        update_state(root, state, stage="golden8")
        _board, golden_summary = run_golden8(root, state, remote_root, arms)
        state["completed"].append("golden8")
        update_state(root, state, stage="prepare_full976")

        manifests = upload_runtime_manifests(root, remote_root, validated["base_manifest"], arms)
        update_state(root, state, stage="full976")
        results = run_full976(root, state, remote_root, manifests)
        if set(results) != {"output_channel_split_4", "spatial_height_split_2"}:
            raise RuntimeError("formal full976 did not complete both approved rewrite arms")

        summary = build_summary(
            root, compile_reports, golden_summary, results, time.monotonic() - started
        )
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
