#!/usr/bin/env python3
"""Detached bn_g_s Gate-8 parity audit: SDK/ORT vs raw and FLOAT32 VStreams."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from board_eval.deploy import stream_upload  # noqa: E402
from hailo_deploy.emulate import _build_native_feed, _match_outputs  # noqa: E402
from hailo_deploy.io.hash import sha256  # noqa: E402
from hailo_deploy.io.metrics import metric, summarize  # noqa: E402
from hailo_deploy.onnx_io import run_onnxruntime, signature  # noqa: E402
from pipelines.run_bn_g_s_clip_feasibility import (  # noqa: E402
    CHECKPOINT_SHA, ONNX, ONNX_SHA, REMOTE_PYTHON, SSH_TARGET,
    atomic_write, launch_remote, poll_remote, ssh, ssh_options, ssh_readonly,
    update_state,
)


CANONICAL = Path(
    "/local/shared_with_docker/deplyment-demo/artifacts/fused/current-ort-mainline-20260805"
)
CLIP_ROOT = Path(
    "/local/shared_with_docker/deplyment-demo/artifacts/fused/experiments/"
    "bn-gs-input-clip-range-sweep-retry-20260806T025328Z"
)
FORMAL_MANIFEST = Path(
    "/local/shared_with_docker/deplyment-demo/artifacts/fused/"
    "board-hybrid-ort-formal-20260805T030725Z/runtime_manifests/"
    "runtime_manifest_gs_only_hailo_fp32.json"
)
GOLDEN = CLIP_ROOT / "board/clip_yhat_bundle"
GOLDEN_RESULT_SHA = "f9c14c3e3430774055bf0df0bc43d62d0af37f9378fd4392a32523fd4a484c7f"
REMOTE_GOLDEN = (
    ".local/share/semantic-sensing-board/hybrid-ort/"
    "bn-gs-input-force-range-continuation-retry-20260805T161225Z/clip_yhat_bundle"
)
SAMPLE_INDICES = tuple(range(1024, 1032))
CPU_MODULES = ["bn_align_h", "bn_g_a", "bn_h_a", "bn_h_s"]
ARMS = {
    "formal_xl_a16": {
        "hef": CANONICAL / "bn_g_s_candidates/x_l_low_spatial_a16_l2/bn_g_s_l2.hef",
        "hef_sha256": "cadbdd1114ac3d75c6006d95804cfa6d86025490fc110cb900e91e51181a7856",
        "har": CANONICAL / "bn_g_s_candidates/x_l_low_spatial_a16_l2/bn_g_s_l2_quantized.har",
        "har_sha256": "9900bc1fbffbbcf49ead61df9a15283e52a9e6bd3327269f5872795a82a0d31a",
        "role": "formal runtime baseline; x_h UINT8, x_l UINT16",
    },
    "c3_hist_equivalent": {
        "hef": CLIP_ROOT / "ptq/C3_dfc_hist_equivalent/bn_g_s_l2.hef",
        "hef_sha256": "637d9dd6a50e619a2675ebeade50bbe92bece5b1ecee704d7c11312f26c70be0",
        "har": CLIP_ROOT / "ptq/C3_dfc_hist_equivalent/bn_g_s_l2_quantized.har",
        "har_sha256": "aa87d3de81e45b995513bd559f8046b1f49e09d32d9b19e9234b120fab9ad421",
        "role": "best clip-sweep board32 diagnostic; x_h UINT8, x_l UINT16",
    },
    "both_outputs_a16_sentinel": {
        "hef": CANONICAL / "bn_g_s_candidates/both_outputs_a16_l2/bn_g_s_l2.hef",
        "hef_sha256": "bd997f19aa20717620ac0f2e0d416231545052dae8a35e77b2de15a78076b5c3",
        "har": CANONICAL / "bn_g_s_candidates/both_outputs_a16_l2/bn_g_s_l2_quantized.har",
        "har_sha256": "09e6186d3a5be7eb66e0e68df6608d8313d790410a3e8a2c1ab3fe4cb6e8e578",
        "role": "existing x_h A16 precision sentinel; both outputs UINT16",
    },
}
EXPECTED_SOURCE_FILES = {
    "formal_manifest": (FORMAL_MANIFEST, None),
    "golden_result": (GOLDEN / "result.json", GOLDEN_RESULT_SHA),
    "onnx": (ONNX, ONNX_SHA),
}


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def array_sha(value: np.ndarray) -> str:
    import hashlib

    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def hef_static(path: Path) -> dict:
    from hailo_platform import HEF

    hef = HEF(str(path))

    def info(item: object) -> dict:
        quant = item.quant_info
        return {
            "name": item.name,
            "shape_hwc": [int(x) for x in item.shape],
            "format_type": str(item.format.type.name),
            "format_order": str(item.format.order.name),
            "quantization": {
                "qp_scale": float(quant.qp_scale), "qp_zp": float(quant.qp_zp),
                "limvals_min": float(quant.limvals_min),
                "limvals_max": float(quant.limvals_max),
            },
        }

    return {
        "sha256": sha256(path), "bytes": path.stat().st_size,
        "network_groups": list(hef.get_network_group_names()),
        "inputs": [info(x) for x in hef.get_input_vstream_infos()],
        "outputs": [info(x) for x in hef.get_output_vstream_infos()],
    }


def validate_sources() -> dict:
    for label, (path, expected) in EXPECTED_SOURCE_FILES.items():
        if not path.is_file():
            raise FileNotFoundError(f"{label}: {path}")
        if expected is not None and sha256(path) != expected:
            raise ValueError(f"{label}: immutable source SHA mismatch")
    if sha256(ONNX) != ONNX_SHA:
        raise ValueError("ONNX SHA mismatch")
    for name, arm in ARMS.items():
        for kind in ("hef", "har"):
            path = arm[kind]
            expected = arm[f"{kind}_sha256"]
            if not path.is_file() or sha256(path) != expected:
                raise ValueError(f"{name}/{kind}: immutable source mismatch")

    formal = json.loads(FORMAL_MANIFEST.read_text(encoding="utf-8"))
    if formal.get("cpu_modules") != CPU_MODULES:
        raise ValueError("formal manifest ORT boundary differs from permanent policy")
    formal_gs = next(item for item in formal["modules"] if item["name"] == "bn_g_s")
    if formal_gs.get("hef_sha256") != ARMS["formal_xl_a16"]["hef_sha256"]:
        raise ValueError("formal runtime does not select the declared baseline bn_g_s HEF")
    if formal_gs.get("engine") != "hailort_float32_vstream":
        raise ValueError("formal bn_g_s engine is not HailoRT FLOAT32 VStream")

    golden = json.loads((GOLDEN / "result.json").read_text(encoding="utf-8"))
    if golden.get("status") != "PASS" or golden.get("sample_count") != 32:
        raise ValueError("golden y_hat bundle is incomplete")
    if golden.get("checkpoint_sha256") != CHECKPOINT_SHA:
        raise ValueError("golden bundle checkpoint mismatch")
    if tuple(golden.get("val_indices", ())[:8]) != SAMPLE_INDICES:
        raise ValueError("golden bundle first-eight index contract changed")
    rows = {int(row["val_index"]): row for row in golden["rows"]}
    for index in SAMPLE_INDICES:
        path = GOLDEN / "latent_y_hat" / f"{index}.npy"
        value = np.load(path, allow_pickle=False)
        if array_sha(value) != rows[index]["sha256"]:
            raise ValueError(f"golden y_hat array SHA mismatch: {index}")

    static = {name: hef_static(arm["hef"]) for name, arm in ARMS.items()}
    for name, row in static.items():
        if len(row["network_groups"]) != 1 or len(row["inputs"]) != 1 or len(row["outputs"]) != 2:
            raise ValueError(f"{name}: unexpected HEF topology")
        if row["inputs"][0]["shape_hwc"] != [16, 16, 192] or row["inputs"][0]["format_type"] != "UINT8":
            raise ValueError(f"{name}: unexpected HEF input")
        shapes = {(tuple(x["shape_hwc"]), x["format_type"]) for x in row["outputs"]}
        expected = (
            {((64, 64, 128), "UINT16"), ((32, 32, 256), "UINT16")}
            if name == "both_outputs_a16_sentinel"
            else {((64, 64, 128), "UINT8"), ((32, 32, 256), "UINT16")}
        )
        if shapes != expected:
            raise ValueError(f"{name}: HEF output contract {shapes} != {expected}")
    return {"formal_manifest": formal_gs, "golden": golden, "hef_static": static}


def add_metric(accumulator: dict, name: str, reference: np.ndarray, actual: np.ndarray) -> None:
    row = metric(reference, actual)
    ref64 = np.asarray(reference, dtype=np.float64)
    delta = np.asarray(actual, dtype=np.float64) - ref64
    row.update({
        "sse": float(np.sum(delta * delta)),
        "reference_energy": float(np.sum(ref64 * ref64)),
    })
    accumulator.setdefault(name, []).append(row)


def summarize_metric_rows(rows: list[dict]) -> dict:
    result = summarize(rows)
    sse = float(sum(row["sse"] for row in rows))
    energy = float(sum(row["reference_energy"] for row in rows))
    result["aggregate_nmse"] = None if energy == 0.0 else float(sse / energy)
    return result


def make_host_references(root: Path) -> tuple[Path, dict]:
    destination = root / "host_references.npz"
    report_path = root / "host_sdk8.json"
    arrays: dict[str, np.ndarray] = {}
    reference_by_index: dict[int, dict[str, np.ndarray]] = {}
    sig = signature(ONNX)
    for index in SAMPLE_INDICES:
        y_hwc = np.ascontiguousarray(
            np.load(GOLDEN / "latent_y_hat" / f"{index}.npy", allow_pickle=False),
            dtype=np.float32,
        )
        sample = {"latent_y_hat": y_hwc.transpose(2, 0, 1)[None]}
        reference = run_onnxruntime(ONNX, sample)
        reference_by_index[index] = reference
        for branch, value in reference.items():
            arrays[f"ort__{index}__{branch}"] = np.ascontiguousarray(
                value[0].transpose(1, 2, 0), dtype=np.float32
            )

    from hailo_sdk_client import ClientRunner, InferenceContext

    arm_reports = {}
    for name, arm in ARMS.items():
        update_state(root, json.loads((root / "campaign_state.json").read_text(encoding="utf-8")),
                     stage=f"host_sdk8_{name}")
        runner = ClientRunner(har=str(arm["har"]))
        notes: list[str] = []
        rows = {branch: [] for branch in ("x_h_hat", "x_l_hat")}
        with runner.infer_context(InferenceContext.SDK_QUANTIZED) as context:
            for index in SAMPLE_INDICES:
                y_hwc = np.ascontiguousarray(
                    np.load(GOLDEN / "latent_y_hat" / f"{index}.npy", allow_pickle=False),
                    dtype=np.float32,
                )
                sample = {"latent_y_hat": y_hwc.transpose(2, 0, 1)[None]}
                raw = runner.infer(context, _build_native_feed(runner, sig, sample, notes))
                pairs = _match_outputs(runner, reference_by_index[index], raw, notes)
                if set(pairs) != {"x_h_hat", "x_l_hat"}:
                    raise ValueError(f"{name}: SDK outputs did not map to both semantic branches")
                for branch, (reference, got) in pairs.items():
                    add_metric(rows, branch, reference, got)
                    arrays[f"sdk__{name}__{index}__{branch}"] = np.ascontiguousarray(
                        got[0].transpose(1, 2, 0), dtype=np.float32
                    )
        arm_reports[name] = {
            "outputs_vs_host_ort": {
                branch: summarize_metric_rows(values) for branch, values in rows.items()
            },
            "notes": sorted(set(notes)),
        }

    temporary = destination.with_name(f".{destination.stem}.{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, destination)
    report = {
        "kind": "bn_g_s_host_sdk_quantized_reference8",
        "status": "PASS", "sample_count": len(SAMPLE_INDICES),
        "val_indices": list(SAMPLE_INDICES), "checkpoint_sha256": CHECKPOINT_SHA,
        "onnx_sha256": ONNX_SHA,
        "arms": arm_reports,
        "bundle": {
            "path": str(destination), "sha256": sha256(destination),
            "bytes": destination.stat().st_size,
            "arrays": {key: array_sha(value) for key, value in arrays.items()},
        },
        "policy": {"diagnostic_only": True, "full976_not_run": True},
    }
    atomic_write(report_path, report)
    return destination, report


def remote_download(remote: str, local: Path) -> None:
    temporary = local.with_name(f".{local.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        completed = subprocess.run(
            ["ssh", *ssh_options(), SSH_TARGET, "cat", remote],
            stdout=handle, stderr=subprocess.PIPE,
        )
    if completed.returncode:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(completed.stderr.decode("utf-8", errors="replace"))
    os.replace(temporary, local)


def build_interpretation(result: dict, host: dict) -> dict:
    outputs = ("x_h_hat", "x_l_hat")
    transform_fail = []
    input_fail = []
    for name, arm in result["arms"].items():
        if arm["gates"]["hailort_float_output_equals_official_raw_dequant"]["status"] != "PASS":
            transform_fail.append(name)
        if arm["gates"]["float_input_auto_quant_equals_official_raw_input"]["status"] != "PASS":
            input_fail.append(name)
    formal_xh = result["arms"]["formal_xl_a16"]["comparisons"][
        "board_float_vs_board_ort"
    ]["x_h_hat"]["aggregate_nmse"]
    both_xh = result["arms"]["both_outputs_a16_sentinel"]["comparisons"][
        "board_float_vs_board_ort"
    ]["x_h_hat"]["aggregate_nmse"]
    board_sdk = {
        name: {
            branch: arm["comparisons"]["board_float_vs_sdk"][branch]["aggregate_nmse"]
            for branch in outputs
        }
        for name, arm in result["arms"].items()
    }
    if result["board_ort_gate"]["status"] != "PASS":
        verdict = "HOST_VS_BOARD_ORT_REFERENCE_MISMATCH"
    elif transform_fail:
        verdict = "HAILORT_OUTPUT_VSTREAM_CONVERSION_MISMATCH"
    elif input_fail:
        verdict = "HAILORT_INPUT_VSTREAM_CONVERSION_MISMATCH"
    elif both_xh < 0.5 * formal_xh:
        verdict = "X_H_UINT8_PRECISION_SENTINEL_POSITIVE"
    else:
        verdict = "COMPILED_HEF_HARDWARE_VS_SDK_GAP_REMAINS"
    return {
        "verdict": verdict,
        "board_ort_gate": result["board_ort_gate"],
        "output_transform_failures": transform_fail,
        "input_transform_failures": input_fail,
        "formal_xh_board_vs_ort_nmse": formal_xh,
        "both_a16_xh_board_vs_ort_nmse": both_xh,
        "both_a16_over_formal_xh_nmse_ratio": float(both_xh / formal_xh),
        "board_float_vs_sdk_nmse": board_sdk,
        "host_sdk_vs_ort_nmse": {
            name: {
                branch: arm["outputs_vs_host_ort"][branch]["aggregate_nmse"]
                for branch in outputs
            }
            for name, arm in host["arms"].items()
        },
        "scope_note": "diagnostic tensor parity only; no semantic candidate selection",
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
            "status": "PASS", "checkpoint_sha256": CHECKPOINT_SHA,
            "sample_indices": SAMPLE_INDICES,
            "arms": {name: validated["hef_static"][name] for name in ARMS},
        }, indent=2, sort_keys=True))
        return 0

    root = args.out_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    unexpected = [path.name for path in root.iterdir() if path.name not in {"run.log", "run.pid"}]
    if unexpected:
        raise FileExistsError(f"refusing non-empty out-root: {root}: {unexpected}")
    started = time.monotonic()
    state = {
        "kind": "bn_g_s_vstream_parity_state", "status": "RUNNING",
        "stage": "validated", "error": None, "started_unix_seconds": time.time(),
    }
    update_state(root, state)
    try:
        atomic_write(root / "source_validation.json", {
            "status": "PASS", "checkpoint_sha256": CHECKPOINT_SHA,
            "formal_runtime_bn_g_s": validated["formal_manifest"],
            "sample_indices": list(SAMPLE_INDICES),
            "golden_result_sha256": GOLDEN_RESULT_SHA,
            "hef_static": validated["hef_static"],
        })
        update_state(root, state, stage="host_sdk8")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
        references_path, host_report = make_host_references(root)

        remote_root = f".local/share/semantic-sensing-board/hybrid-ort/{root.name}"
        if ssh_readonly(f"test ! -e {shlex.quote(remote_root)} && echo ABSENT || echo PRESENT").strip() != "ABSENT":
            raise FileExistsError(f"remote result root already exists: {remote_root}")
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
        for name, arm in ARMS.items():
            destination = f"hef/{name}.hef"
            stream_upload(arm["hef"], f"{remote_root}/{destination}", options, SSH_TARGET)
            manifest_arms.append({
                "name": name, "role": arm["role"], "hef_filename": destination,
                "hef_sha256": arm["hef_sha256"], "har_sha256": arm["har_sha256"],
            })
        manifest = {
            "kind": "bn_g_s_vstream_parity_manifest",
            "checkpoint_sha256": CHECKPOINT_SHA, "sample_indices": list(SAMPLE_INDICES),
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
                "dfc": package_version("hailo-dataflow-compiler"),
                "hailort_host": package_version("hailort"),
                "model_zoo": package_version("hailo-model-zoo"),
                "hailort_board_expected": "4.23.0", "hw_arch": "hailo8",
            },
            "policy": {
                "bn_g_s_only": True, "full976_not_run": True,
                "runtime_topology_unchanged": True,
                "cpu_modules": CPU_MODULES,
            },
        }
        manifest_path = root / "board_manifest.json"
        atomic_write(manifest_path, manifest)
        stream_upload(manifest_path, f"{remote_root}/board_manifest.json", options, SSH_TARGET)

        remote_golden_sha = ssh_readonly(
            f"sha256sum {shlex.quote(REMOTE_GOLDEN + '/result.json')} | cut -d' ' -f1"
        ).strip()
        if remote_golden_sha != GOLDEN_RESULT_SHA:
            raise ValueError("remote golden bundle identity mismatch")
        prelaunch = ssh_readonly(
            "pgrep -af '[e]valuate_bn_g_s_vstream_parity|[e]valuate_bn_g_s_clip_sweep' || true; "
            "vcgencmd measure_temp 2>/dev/null || true"
        )
        if "evaluate_bn_g_s" in prelaunch:
            raise RuntimeError(f"board has a residual bn_g_s evaluator:\n{prelaunch}")
        update_state(root, state, stage="board8", board_prelaunch=prelaunch)
        result_remote = f"{remote_root}/result.json"
        raw_remote = f"{remote_root}/raw_outputs.npz"
        job = launch_remote(
            remote_root, "board_vstream_parity_attempt0",
            [
                REMOTE_PYTHON, f"{remote_root}/{evaluator.name}",
                "--manifest", f"{remote_root}/board_manifest.json",
                "--bundle-root", REMOTE_GOLDEN,
                "--out", result_remote, "--raw-out", raw_remote,
            ],
            result_override=result_remote,
            progress_override=f"{remote_root}/result.progress.json",
        )
        update_state(root, state, stage="board8", remote_job=job)
        poll_remote(root, state, job, timeout=1800)
        result_path = root / "board_result.json"
        raw_path = root / "raw_outputs.npz"
        remote_download(result_remote, result_path)
        remote_download(raw_remote, raw_path)
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "PASS" or result.get("sample_count") != len(SAMPLE_INDICES):
            raise RuntimeError("board parity result is incomplete")
        if sha256(raw_path) != result["raw_outputs"]["sha256"]:
            raise ValueError("downloaded raw output bundle hash mismatch")
        interpretation = build_interpretation(result, host_report)
        summary = {
            "kind": "bn_g_s_vstream_parity_campaign_summary", "status": "PASS",
            "elapsed_seconds": time.monotonic() - started,
            "source_validation": str(root / "source_validation.json"),
            "host_sdk8": host_report, "board_result": result,
            "interpretation": interpretation,
            "artifacts": {
                "board_result": {"path": str(result_path), "sha256": sha256(result_path)},
                "raw_outputs": {"path": str(raw_path), "sha256": sha256(raw_path)},
            },
        }
        atomic_write(root / "summary.json", summary)
        update_state(root, state, status="PASS", stage="complete",
                     elapsed_seconds=summary["elapsed_seconds"],
                     verdict=interpretation["verdict"])
        print(json.dumps({"status": "PASS", "verdict": interpretation["verdict"],
                          "summary": str(root / "summary.json")}, sort_keys=True), flush=True)
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
