#!/usr/bin/env python3
"""Detached compiler-only A/B localization for the formal ``bn_g_s`` HAR.

This campaign deliberately reuses one immutable quantized HAR.  It changes only
DFC allocation policy, then compares the resulting HEFs on the same eight
golden tensors.  The results are diagnostic and never perform semantic model
selection.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from board_eval.deploy import stream_upload  # noqa: E402
from hailo_deploy.gates.compile_real import compile_quantized  # noqa: E402
from hailo_deploy.io.hash import sha256  # noqa: E402
from pipelines.run_bn_g_s_clip_feasibility import (  # noqa: E402
    CHECKPOINT_SHA,
    ONNX,
    ONNX_SHA,
    REMOTE_PYTHON,
    SSH_TARGET,
    atomic_write,
    launch_remote,
    poll_remote,
    ssh,
    ssh_options,
    ssh_readonly,
    update_state,
)
import pipelines.run_bn_g_s_vstream_parity as parity  # noqa: E402


CANONICAL = Path(
    "/local/shared_with_docker/deplyment-demo/artifacts/fused/current-ort-mainline-20260805"
)
FORMAL_DIR = CANONICAL / "bn_g_s_candidates/x_l_low_spatial_a16_l2"
FORMAL_HAR = FORMAL_DIR / "bn_g_s_l2_quantized.har"
FORMAL_HAR_SHA = "9900bc1fbffbbcf49ead61df9a15283e52a9e6bd3327269f5872795a82a0d31a"
FORMAL_HEF = FORMAL_DIR / "bn_g_s_l2.hef"
FORMAL_HEF_SHA = "cadbdd1114ac3d75c6006d95804cfa6d86025490fc110cb900e91e51181a7856"
FORMAL_AUTO_ALLS = FORMAL_DIR / "bn_g_s_l2.auto.alls"

GOLDEN = parity.GOLDEN
GOLDEN_RESULT_SHA = parity.GOLDEN_RESULT_SHA
REMOTE_GOLDEN = parity.REMOTE_GOLDEN
SAMPLE_INDICES = parity.SAMPLE_INDICES
CPU_MODULES = parity.CPU_MODULES

LOGICAL_ARM_SCRIPTS = {
    "formal_existing": None,
    "recompile_default": "",
    "conv5_disable_16x4": "compilation_param(bn_g_s/conv5, use_16x4_sc=disabled)\n",
    "conv2_disable_16x4": "compilation_param(bn_g_s/conv2, use_16x4_sc=disabled)\n",
}


def exact_auto_arm_scripts() -> dict[str, str | None]:
    original = FORMAL_AUTO_ALLS.read_text(encoding="utf-8")

    def disable(layer_prefix: str) -> str:
        lines = []
        changed = 0
        for line in original.splitlines():
            if line.startswith(f"compilation_param({layer_prefix}") and "use_16x4_sc=enabled" in line:
                line = line.replace("use_16x4_sc=enabled", "use_16x4_sc=disabled")
                changed += 1
            lines.append(line)
        expected = 4 if layer_prefix == "conv5_d" else 1
        if changed != expected:
            raise ValueError(
                f"expected {expected} physical {layer_prefix} 16x4 edits, got {changed}"
            )
        return "\n".join(lines) + "\n"

    return {
        "formal_existing": None,
        "replay_exact_auto": original,
        "conv5_physical_disable_16x4": disable("conv5_d"),
        "conv2_physical_disable_16x4": disable("conv2,"),
    }


def allocation_facts(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    enabled = [line for line in text.splitlines() if "use_16x4_sc=enabled" in line]
    return {
        "sha256": sha256(path),
        "conv2_16x4_enabled": any("compilation_param(conv2," in line for line in enabled),
        "conv5_16x4_enabled_count": sum(
            "compilation_param(conv5" in line for line in enabled
        ),
        "conv3_spatial_defuse": "defuse(conv3, 8, defuse_type=SPATIAL)" in text,
        "resize2_defuse": "defuse(resize2, 2)" in text,
        "conv4_spatial_defuse": "defuse(conv4, 8, defuse_type=SPATIAL)" in text,
        "conv5_defuse": "defuse(conv5, 4)" in text,
    }


def validate_sources() -> dict:
    expected = {
        FORMAL_HAR: FORMAL_HAR_SHA,
        FORMAL_HEF: FORMAL_HEF_SHA,
        ONNX: ONNX_SHA,
        GOLDEN / "result.json": GOLDEN_RESULT_SHA,
    }
    for path, digest in expected.items():
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"immutable source mismatch: {path}")
    formal = parity.validate_sources()
    return {
        "formal": formal,
        "formal_allocation": allocation_facts(FORMAL_AUTO_ALLS),
    }


def compile_arms(root: Path, arm_scripts: dict[str, str | None]) -> tuple[dict, dict]:
    from hailo_sdk_client import ClientRunner

    arm_specs: dict[str, dict] = {
        "formal_existing": {
            "hef": FORMAL_HEF,
            "hef_sha256": FORMAL_HEF_SHA,
            "har": FORMAL_HAR,
            "har_sha256": FORMAL_HAR_SHA,
            "role": "immutable formal runtime baseline",
        }
    }
    reports: dict[str, dict] = {
        "formal_existing": {
            "status": "REUSED",
            "hef": {"path": str(FORMAL_HEF), "sha256": FORMAL_HEF_SHA},
            "auto_alls": {
                "path": str(FORMAL_AUTO_ALLS),
                **allocation_facts(FORMAL_AUTO_ALLS),
            },
        }
    }
    for name, script in arm_scripts.items():
        if script is None:
            continue
        destination = root / "compile" / name
        destination.mkdir(parents=True, exist_ok=False)
        script_path = destination / "compiler_override.alls"
        script_path.write_text(script, encoding="utf-8")
        runner = ClientRunner(har=str(FORMAL_HAR))
        if script:
            runner.load_model_script(script)
        result = compile_quantized(
            runner,
            hef_path=destination / "bn_g_s.hef",
            compiled_har=destination / "bn_g_s_compiled.har",
            auto_alls=destination / "bn_g_s.auto.alls",
        )
        result.update({
            "status": "PASS",
            "source_quantized_har_sha256": FORMAL_HAR_SHA,
            "compiler_override": script,
            "compiler_override_sha256": sha256(script_path),
            "allocation": allocation_facts(destination / "bn_g_s.auto.alls"),
        })
        reports[name] = result
        arm_specs[name] = {
            "hef": destination / "bn_g_s.hef",
            "hef_sha256": result["hef"]["sha256"],
            "har": FORMAL_HAR,
            "har_sha256": FORMAL_HAR_SHA,
            "role": f"compiler-only A/B: {script or 'default recompile'}",
        }
        del runner
    return arm_specs, reports


def build_summary(board: dict, compile_reports: dict, elapsed: float, mode: str) -> dict:
    per_arm = {}
    for name, arm in board["arms"].items():
        per_arm[name] = {
            branch: arm["comparisons"]["board_float_vs_sdk"][branch]["aggregate_nmse"]
            for branch in ("x_h_hat", "x_l_hat")
        }
    baseline = per_arm["formal_existing"]
    ratios = {
        name: {
            branch: value[branch] / baseline[branch]
            for branch in ("x_h_hat", "x_l_hat")
        }
        for name, value in per_arm.items()
    }
    if mode == "physical_auto":
        conv5_name = "conv5_physical_disable_16x4"
        conv2_name = "conv2_physical_disable_16x4"
        replay_name = "replay_exact_auto"
    else:
        conv5_name = "conv5_disable_16x4"
        conv2_name = "conv2_disable_16x4"
        replay_name = "recompile_default"
    if ratios[conv5_name]["x_h_hat"] < 0.7:
        verdict = "CONV5_16X4_IS_PRIMARY_COMPILER_GAP"
    elif max(ratios[conv2_name].values()) < 0.7:
        verdict = "CONV2_16X4_IS_PRIMARY_COMPILER_GAP"
    elif ratios[replay_name]["x_h_hat"] > 1.5 or ratios[replay_name]["x_h_hat"] < 0.67:
        verdict = "EXACT_REPLAY_MAPPING_SENSITIVE" if mode == "physical_auto" else "DEFAULT_RECOMPILE_MAPPING_SENSITIVE"
    else:
        verdict = "16X4_NOT_PRIMARY_CONTINUE_WITH_INTERMEDIATE_PROBES"
    return {
        "kind": "bn_g_s_compiler_only_ab_summary",
        "status": "PASS",
        "verdict": verdict,
        "compiler_ab_mode": mode,
        "elapsed_seconds": elapsed,
        "board_vs_sdk_aggregate_nmse": per_arm,
        "ratio_to_formal_existing": ratios,
        "compile": compile_reports,
        "policy": {
            "diagnostic_only": True,
            "full976_not_run": True,
            "semantic_ranking": False,
            "source_quantized_har_immutable": True,
            "bn_g_s_only": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--cuda-visible-devices", default="2")
    parser.add_argument(
        "--physical-auto",
        action="store_true",
        help="replay the formal auto.alls and edit defused physical layers exactly",
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    validated = validate_sources()
    if args.validate_only:
        print(json.dumps({"status": "PASS", **validated["formal_allocation"]}, indent=2))
        return 0

    mode = "physical_auto" if args.physical_auto else "logical_override"
    arm_scripts = exact_auto_arm_scripts() if args.physical_auto else LOGICAL_ARM_SCRIPTS
    root = args.out_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    unexpected = [p.name for p in root.iterdir() if p.name not in {"run.log", "run.pid"}]
    if unexpected:
        raise FileExistsError(f"refusing non-empty out-root: {root}: {unexpected}")
    (root / "logs").mkdir(exist_ok=False)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    os.environ["HAILO_SDK_LOG_DIR"] = str(root / "logs")
    started = time.monotonic()
    state = {
        "kind": "bn_g_s_compiler_only_ab_state",
        "compiler_ab_mode": mode,
        "status": "RUNNING",
        "stage": "validated",
        "error": None,
        "started_unix_seconds": time.time(),
    }
    update_state(root, state)
    try:
        atomic_write(root / "source_validation.json", {
            "status": "PASS",
            "checkpoint_sha256": CHECKPOINT_SHA,
            "quantized_har": {"path": str(FORMAL_HAR), "sha256": FORMAL_HAR_SHA},
            "formal_hef": {"path": str(FORMAL_HEF), "sha256": FORMAL_HEF_SHA},
            "formal_allocation": validated["formal_allocation"],
            "sample_indices": list(SAMPLE_INDICES),
        })
        update_state(root, state, stage="compile")
        arms, compile_reports = compile_arms(root, arm_scripts)
        atomic_write(root / "compile_results.json", compile_reports)

        if args.physical_auto:
            conv5_alloc = compile_reports["conv5_physical_disable_16x4"]["allocation"]
            conv2_alloc = compile_reports["conv2_physical_disable_16x4"]["allocation"]
            if conv5_alloc["conv5_16x4_enabled_count"] != 0:
                raise ValueError("physical conv5 16x4 override was not realized")
            if conv2_alloc["conv2_16x4_enabled"]:
                raise ValueError("physical conv2 16x4 override was not realized")

        for name, spec in arms.items():
            static = parity.hef_static(spec["hef"])
            contract = {(tuple(x["shape_hwc"]), x["format_type"]) for x in static["outputs"]}
            expected = {((64, 64, 128), "UINT8"), ((32, 32, 256), "UINT16")}
            if contract != expected:
                raise ValueError(f"{name}: output contract changed: {contract}")

        parity.ARMS = arms
        update_state(root, state, stage="host_sdk8")
        references_path, host_report = parity.make_host_references(root)

        remote_root = f".local/share/semantic-sensing-board/hybrid-ort/{root.name}"
        if ssh_readonly(
            f"test ! -e {shlex.quote(remote_root)} && echo ABSENT || echo PRESENT"
        ).strip() != "ABSENT":
            raise FileExistsError(f"remote result root exists: {remote_root}")
        ssh(f"mkdir -p {shlex.quote(remote_root + '/hef')} {shlex.quote(remote_root + '/onnx')}")
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
            destination = f"hef/{name}.hef"
            stream_upload(spec["hef"], f"{remote_root}/{destination}", options, SSH_TARGET)
            manifest_arms.append({
                "name": name,
                "role": spec["role"],
                "hef_filename": destination,
                "hef_sha256": spec["hef_sha256"],
                "har_sha256": FORMAL_HAR_SHA,
            })
        manifest = {
            "kind": "bn_g_s_compiler_only_ab_manifest",
            "compiler_ab_mode": mode,
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
            "versions": {
                "dfc": parity.package_version("hailo-dataflow-compiler"),
                "hailort_host": parity.package_version("hailort"),
                "hailort_board_expected": "4.23.0",
                "hw_arch": "hailo8",
            },
            "policy": {
                "diagnostic_only": True,
                "full976_not_run": True,
                "cpu_modules": CPU_MODULES,
                "source_quantized_har_sha256": FORMAL_HAR_SHA,
            },
        }
        manifest_path = root / "board_manifest.json"
        atomic_write(manifest_path, manifest)
        stream_upload(manifest_path, f"{remote_root}/board_manifest.json", options, SSH_TARGET)
        if ssh_readonly(
            f"sha256sum {shlex.quote(REMOTE_GOLDEN + '/result.json')} | cut -d' ' -f1"
        ).strip() != GOLDEN_RESULT_SHA:
            raise ValueError("remote golden bundle identity mismatch")
        prelaunch = ssh_readonly(
            "pgrep -af '[e]valuate_bn_g_s_vstream_parity|[e]valuate_bn_g_s_clip_sweep' || true; "
            "vcgencmd measure_temp 2>/dev/null || true"
        )
        if "evaluate_bn_g_s" in prelaunch:
            raise RuntimeError(f"board has residual evaluator:\n{prelaunch}")
        update_state(root, state, stage="board8", board_prelaunch=prelaunch)
        result_remote = f"{remote_root}/result.json"
        raw_remote = f"{remote_root}/raw_outputs.npz"
        job = launch_remote(
            remote_root,
            "board_compiler_ab_attempt0",
            [
                REMOTE_PYTHON,
                f"{remote_root}/{evaluator.name}",
                "--manifest",
                f"{remote_root}/board_manifest.json",
                "--bundle-root",
                REMOTE_GOLDEN,
                "--out",
                result_remote,
                "--raw-out",
                raw_remote,
            ],
            result_override=result_remote,
            progress_override=f"{remote_root}/result.progress.json",
        )
        update_state(root, state, stage="board8", remote_job=job)
        poll_remote(root, state, job, timeout=1800)
        result_path = root / "board_result.json"
        raw_path = root / "raw_outputs.npz"
        parity.remote_download(result_remote, result_path)
        parity.remote_download(raw_remote, raw_path)
        board = json.loads(result_path.read_text(encoding="utf-8"))
        if board.get("status") != "PASS" or board.get("sample_count") != len(SAMPLE_INDICES):
            raise RuntimeError("board result incomplete")
        if sha256(raw_path) != board["raw_outputs"]["sha256"]:
            raise ValueError("downloaded raw bundle hash mismatch")
        summary = build_summary(board, compile_reports, time.monotonic() - started, mode)
        summary["artifacts"] = {
            "board_result": {"path": str(result_path), "sha256": sha256(result_path)},
            "raw_outputs": {"path": str(raw_path), "sha256": sha256(raw_path)},
            "host_sdk8": {"path": str(root / 'host_sdk8.json'), "sha256": sha256(root / 'host_sdk8.json')},
        }
        atomic_write(root / "summary.json", summary)
        update_state(
            root,
            state,
            status="PASS",
            stage="complete",
            elapsed_seconds=summary["elapsed_seconds"],
            verdict=summary["verdict"],
        )
        print(json.dumps({"status": "PASS", "verdict": summary["verdict"]}, sort_keys=True))
        return 0
    except Exception as exc:
        import traceback

        error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        atomic_write(root / "failure.json", {"status": "FAIL", "error": error})
        update_state(root, state, status="FAIL", stage="failed", error=error)
        print(error, file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
