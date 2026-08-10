#!/usr/bin/env python3
"""Prepare the bn_g_s matched-precision causal arm and stop before compile.

This experiment re-runs only bn_g_s L2 PTQ from the current native HAR and
the formal 1024-sample calibration.  Relative to the formal x_l-A16 policy it
adds one causal treatment command in DFC's explicit input/weight/output form::

    quantization_param(bn_g_s/conv5, precision_mode=a16_w16_a8)

The three-part form is necessary because DFC 3.33 propagates the two-part
``a16_w16`` request to the output activation, and a downstream
``output_layer2 a8_w8`` command does not override that propagation.

The candidate is publishable for a later compile decision only when its HN
has the requested mixed precision and the frozen golden8 SDK_QUANTIZED codes
at resize2 and conv3/x_l are bit-exact with the formal quantized HAR.  This
entry point has no compile or board phase and rejects any HEF-like artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    from hailo_deploy.logging import reexec_if_unlogged

    reexec_if_unlogged(Path(__file__), sys.argv[1:])

from board_eval.compiler_probe_metrics import (  # noqa: E402
    code_comparison,
    sdk_codes_to_integer,
    summarize_code_comparisons,
)
from hailo_deploy.artifacts.layout import refuse_existing  # noqa: E402
from hailo_deploy.gates.compiler_probe import (  # noqa: E402
    hn_sha256,
    optimized_params_sha256,
)
from hailo_deploy.io.hash import sha256  # noqa: E402
from pipelines.run_bn_g_s_compiler_probe import _validate_golden  # noqa: E402
from pipelines.run_bn_g_s_vstream_parity import (  # noqa: E402
    GOLDEN,
    SAMPLE_INDICES,
    array_sha,
)


CHECKPOINT_SHA = "13dc6c4b3a3ea630af6d2d547808137be678bd909f9789f5594b7bdc76c36b90"
ONNX_SHA = "b292687ee569fe2f523ae3704cfca92bc84e4bc7e4041ea5c9fa41ac8c0e0c4f"
NATIVE_HAR_SHA = "9d6b2fc89deb9a37e296c2b4cbd579e981f3c4873663c597ef475451e2dabaeb"
FORMAL_QHAR_SHA = "9900bc1fbffbbcf49ead61df9a15283e52a9e6bd3327269f5872795a82a0d31a"
FORMAL_SCRIPT_SHA = "0224eb7746979b9a811adc01e913d74afedf6ddfd6f87806ec91d0c3dbbfc10f"
CALIB_META_SHA = "2be6115ea5105c63eed36a458ee738daaf86780e08de79f531726ce52ca58c02"

HANDOFF = Path(
    "/local/shared_with_docker/hailo_full_candidate/"
    "convbn-gfa-gs-3a3ea630af6d2d547808137be678bd909f9789f5594b7bdc76c36b90"
)
ONNX = HANDOFF / "onnx/bn_g_s.onnx"
CALIB = HANDOFF / "calibration/bn_g_s"
NATIVE_HAR = Path(
    "/local/shared_with_docker/deplyment-demo/artifacts/fused/experiments/"
    "bn-gs-input-clip-range-sweep-retry-20260806T025328Z/"
    "parse/har/bn_g_s_hailo_model.har"
)
FORMAL_DIR = Path(
    "/local/shared_with_docker/deplyment-demo/artifacts/fused/"
    "current-ort-mainline-20260805/bn_g_s_candidates/x_l_low_spatial_a16_l2"
)
FORMAL_QHAR = FORMAL_DIR / "bn_g_s_l2_quantized.har"
FORMAL_SCRIPT = FORMAL_DIR / "model_script.alls"

FORMAL_RECORDED_NATIVE_HAR = Path(
    "/local/shared_with_docker/deplyment-demo/artifacts/fused/candidates/"
    "convbn-gfa-gs-fullmodules-20260722T1630Z/har/bn_g_s_hailo_model.har"
)
FORMAL_RECORDED_NATIVE_HAR_SHA = (
    "2330ca78488ea5c93fd5c7492ddedbdad18cd94d1940854e97b06f7fa4562dfd"
)

EXTRA_SCRIPT = (
    "# Formal x_l low-spatial A16 policy.\n"
    "quantization_param(bn_g_s/conv3, precision_mode=a16_w16)\n"
    "quantization_param(bn_g_s/output_layer1, precision_mode=a16_w16)\n"
    "# Matched-precision causal arm: A16 conv5 compute, A8 x_h interface.\n"
    "quantization_param(bn_g_s/conv5, precision_mode=a16_w16_a8)\n"
)
EXPECTED_FULL_SCRIPT = (
    "model_optimization_flavor(optimization_level=2, compression_level=0, batch_size=1)\n"
    "model_optimization_config(calibration, batch_size=1, calibset_size=1024)\n\n"
    + EXTRA_SCRIPT
)

EXPECTED_PRECISION = {
    "bn_g_s/resize2": "a8_w8_a8",
    "bn_g_s/conv4": "a8_w8_a16",
    "bn_g_s/conv5": "a16_w16_a8",
    "bn_g_s/conv3": "a16_w16_a16",
    "bn_g_s/output_layer1": "a16_w16_a16",
    "bn_g_s/output_layer2": "a8_w8_a8",
}
EXPECTED_SHAPES = {
    "bn_g_s/resize2": (64, 64, 384),
    "bn_g_s/conv4": (64, 64, 128),
    "bn_g_s/conv5": (64, 64, 128),
    "bn_g_s/conv3": (32, 32, 256),
}
INTERNALS = {
    "resize2": ("bn_g_s/resize2", (64, 64, 384), "UINT8"),
    "x_l": ("bn_g_s/conv3", (32, 32, 256), "UINT16"),
}
FINAL_SHAPES = {"x_h": (64, 64, 128), "x_l": (32, 32, 256)}


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def require_hash(path: Path, expected: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"source hash mismatch: {path}: {actual} != {expected}")
    return {"path": str(path), "sha256": actual, "bytes": path.stat().st_size}


def gpu_gate(index: str) -> dict[str, Any]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    rows = []
    for line in output.strip().splitlines():
        fields = [item.strip() for item in line.split(",")]
        rows.append(
            {
                "index": fields[0],
                "name": fields[1],
                "memory_total_mib": int(fields[2]),
                "memory_used_mib": int(fields[3]),
                "utilization_percent": int(fields[4]),
            }
        )
    matches = [row for row in rows if row["index"] == index]
    if len(matches) != 1:
        raise ValueError(f"requested GPU {index} not present: {rows}")
    selected = matches[0]
    free_mib = selected["memory_total_mib"] - selected["memory_used_mib"]
    if free_mib < 20_000:
        raise RuntimeError(f"GPU {index} has only {free_mib} MiB free")
    return {"status": "PASS", "selected": selected, "free_mib": free_mib, "all": rows}


def source_gate(gpu: str) -> dict[str, Any]:
    sources = {
        "native_har": require_hash(NATIVE_HAR, NATIVE_HAR_SHA),
        "onnx": require_hash(ONNX, ONNX_SHA),
        "formal_quantized_har": require_hash(FORMAL_QHAR, FORMAL_QHAR_SHA),
        "formal_model_script": require_hash(FORMAL_SCRIPT, FORMAL_SCRIPT_SHA),
        "calibration_meta": require_hash(CALIB / "meta.json", CALIB_META_SHA),
    }
    meta = json.loads((CALIB / "meta.json").read_text(encoding="utf-8"))
    if meta.get("sample_count") != 1024:
        raise ValueError(f"formal L2 calibration count is not 1024: {meta.get('sample_count')}")
    if meta.get("checkpoint", {}).get("sha256") != CHECKPOINT_SHA:
        raise ValueError("formal L2 calibration checkpoint mismatch")
    if meta.get("onnx", {}).get("sha256") != ONNX_SHA:
        raise ValueError("formal L2 calibration ONNX mismatch")
    calib_files = sorted((CALIB / "latent_y_hat").glob("*.npy"))
    if len(calib_files) != 1024:
        raise ValueError(f"expected 1024 calibration tensors, got {len(calib_files)}")

    from hailo_sdk_client import ClientRunner

    native = ClientRunner(har=str(NATIVE_HAR))
    native_state = str(native.state)
    native_hn = native.get_hn_dict()
    if native_state != "hailo_model":
        raise ValueError(f"native source state is {native_state}, not hailo_model")
    if native_hn.get("net_params", {}).get("output_layers_order") != [
        "bn_g_s/conv5", "bn_g_s/conv3"
    ]:
        raise ValueError("native source output order changed")
    native_hn_digest = hn_sha256(native)
    del native

    versions = {
        "dfc": package_version("hailo-dataflow-compiler"),
        "hailort_host": package_version("hailort"),
        "model_zoo": package_version("hailo-model-zoo"),
        "hw_arch": "hailo8",
    }
    if versions["dfc"] != "3.33.0" or versions["hailort_host"] != "4.23.0":
        raise ValueError(f"unexpected Hailo tool versions: {versions}")

    golden = _validate_golden()
    return {
        "status": "PASS",
        "sources": sources,
        "native_har": {
            "state": native_state,
            "hn_sha256": native_hn_digest,
            "output_layers_order": native_hn["net_params"]["output_layers_order"],
            "lineage_note": (
                "The native HAR recorded by the 2026-07 formal result is absent. "
                "This immutable source is the currently retained parse of the same locked ONNX; "
                "its different SHA is disclosed and no deleted artifact was reconstructed."
            ),
            "formal_recorded_path": str(FORMAL_RECORDED_NATIVE_HAR),
            "formal_recorded_sha256": FORMAL_RECORDED_NATIVE_HAR_SHA,
            "formal_recorded_path_exists": FORMAL_RECORDED_NATIVE_HAR.is_file(),
        },
        "calibration": {
            "root": str(CALIB),
            "sample_count": len(calib_files),
            "indices": [0, 1023],
            "checkpoint_sha256": CHECKPOINT_SHA,
            "onnx_sha256": ONNX_SHA,
        },
        "versions": versions,
        "gpu": gpu_gate(gpu),
        "golden8": golden,
    }


def run_ptq(root: Path, gpu: str) -> dict[str, Any]:
    scripts = root / "model_scripts"
    scripts.mkdir(parents=True, exist_ok=False)
    extra_path = scripts / "matched_conv5_a16.extra.alls"
    extra_path.write_text(EXTRA_SCRIPT, encoding="utf-8")
    ptq_dir = root / "ptq" / "matched_conv5_a16_l2"
    command = [
        sys.executable,
        str(ROOT / "pipelines/module_ptq.py"),
        "calib-root",
        "--module", "bn_g_s",
        "--onnx", str(ONNX),
        "--parse-har", str(NATIVE_HAR),
        "--calib-root", str(CALIB),
        "--out-dir", str(ptq_dir),
        "--n-calib", "1024",
        "--opt-level", "2",
        "--eval-samples", "8",
        "--extra-alls", str(extra_path),
        "--skip-compile",
        "--expect-onnx-sha",
        "--expect-checkpoint-sha", CHECKPOINT_SHA,
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    log_path = root / "logs" / "ptq.log"
    log_path.parent.mkdir(parents=True, exist_ok=False)
    print(json.dumps({"command": shlex.join(command), "gpu": gpu}), flush=True)
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"module_ptq failed with exit code {return_code}; see {log_path}")

    result_path = ptq_dir / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "PASS":
        raise RuntimeError(f"PTQ result did not pass: {result.get('error')}")
    if result.get("compile") != "SKIP by --skip-compile":
        raise ValueError(f"compile policy violation: {result.get('compile')}")
    if result.get("n_calib") != 1024 or result.get("optimization_level") != 2:
        raise ValueError("PTQ did not preserve the formal L2 recipe")
    generated_script = Path(result["model_script"]["path"])
    if generated_script.read_text(encoding="utf-8") != EXPECTED_FULL_SCRIPT:
        raise ValueError("generated model script is not the locked formal recipe plus conv5 A16")
    forbidden = [
        path for path in root.rglob("*")
        if path.is_file() and (
            path.suffix.lower() == ".hef"
            or path.name.endswith("_compiled.har")
            or path.name.endswith(".auto.alls")
        )
    ]
    if forbidden:
        raise ValueError(f"compile artifacts unexpectedly exist: {forbidden}")
    return {
        "status": "PASS",
        "command": command,
        "cuda_visible_devices": gpu,
        "result": {"path": str(result_path), "sha256": sha256(result_path)},
        "log": {"path": str(log_path), "sha256": sha256(log_path)},
        "model_script": {
            "path": str(generated_script),
            "sha256": sha256(generated_script),
            "delta_from_formal": [
                "quantization_param(bn_g_s/conv5, precision_mode=a16_w16_a8)",
            ],
            "causal_treatment": "conv5 a16_w16 compute with explicit A8 output",
        },
        "quantized_har": result["quantized_har"],
        "compile": result["compile"],
        "forbidden_compile_artifacts": [],
        "ort_quantized_check": result.get("quantized_check"),
    }


def hn_gate(qhar: Path) -> dict[str, Any]:
    from hailo_sdk_client import ClientRunner

    runner = ClientRunner(har=str(qhar))
    if str(runner.state) != "quantized_model":
        raise ValueError(f"candidate qHAR state is not quantized_model: {runner.state}")
    hn = runner.get_hn_dict()
    order = hn.get("net_params", {}).get("output_layers_order")
    if order != ["bn_g_s/conv5", "bn_g_s/conv3"]:
        raise ValueError(f"candidate output order changed: {order}")
    layers = {}
    for name, expected_precision in EXPECTED_PRECISION.items():
        layer = hn.get("layers", {}).get(name)
        if layer is None:
            raise ValueError(f"candidate HN is missing {name}")
        actual = layer.get("quantization_params", {}).get("precision_mode")
        if actual != expected_precision:
            raise ValueError(f"{name} precision {actual} != {expected_precision}")
        shapes = [list(shape) for shape in layer.get("output_shapes", [])]
        if name in EXPECTED_SHAPES:
            if EXPECTED_SHAPES[name] not in {tuple(shape[1:]) for shape in shapes}:
                raise ValueError(f"{name} output shape mismatch: {shapes}")
        layers[name] = {
            "type": layer.get("type"),
            "precision_mode": actual,
            "output_shapes_nhwc": shapes,
            "inputs": list(layer.get("input") or []),
            "outputs": list(layer.get("output") or []),
        }
    result = {
        "status": "PASS",
        "state": str(runner.state),
        "hn_sha256": hn_sha256(runner),
        "optimized_params_sha256": optimized_params_sha256(runner),
        "output_layers_order": order,
        "layers": layers,
        "runtime_contract_without_hef": {
            "x_h_layer": "bn_g_s/conv5",
            "x_h_precision": layers["bn_g_s/conv5"]["precision_mode"],
            "x_h_expected_format": "UINT8",
            "x_l_layer": "bn_g_s/conv3",
            "x_l_precision": layers["bn_g_s/conv3"]["precision_mode"],
            "x_l_expected_format": "UINT16",
            "note": "HEF vstream format remains unverified until a separately authorized compile.",
        },
    }
    del runner
    return result


def _to_numpy(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _to_numpy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_numpy(item) for item in value]
    return np.asarray(value.numpy() if hasattr(value, "numpy") else value)


def _flatten_arrays(value: Any) -> list[np.ndarray]:
    if isinstance(value, dict):
        result: list[np.ndarray] = []
        for item in value.values():
            result.extend(_flatten_arrays(item))
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            result.extend(_flatten_arrays(item))
        return result
    return [np.asarray(value)]


def _map_shapes(value: Any, expected: dict[str, tuple[int, ...]]) -> dict[str, np.ndarray]:
    arrays = _flatten_arrays(value)
    result = {}
    for name, shape in expected.items():
        matches = [array for array in arrays if tuple(array.shape[1:]) == shape]
        if len(matches) != 1:
            raise ValueError(
                f"cannot map {name}/{shape}: {[tuple(array.shape) for array in arrays]}"
            )
        result[name] = np.ascontiguousarray(matches[0][0])
    return result


def sdk_capture(qhar: Path) -> dict[int, dict[str, np.ndarray]]:
    from hailo_sdk_client import ClientRunner, InferenceContext

    runner = ClientRunner(har=str(qhar))
    captured: dict[int, dict[str, np.ndarray]] = {}
    internal_shapes = {key: spec[1] for key, spec in INTERNALS.items()}
    internal_layers = [spec[0] for spec in INTERNALS.values()]
    with runner.infer_context(InferenceContext.SDK_QUANTIZED) as context:
        simulation = runner.get_keras_model(context)
        simulation.model.set_output_interal_layers(internal_layers)
        try:
            for index in SAMPLE_INDICES:
                y_hwc = np.ascontiguousarray(
                    np.load(GOLDEN / "latent_y_hat" / f"{index}.npy", allow_pickle=False),
                    dtype=np.float32,
                )
                raw = _to_numpy(simulation.model(y_hwc[None, ...], training=False))
                if not isinstance(raw, list) or len(raw) != 2:
                    raise ValueError(f"unexpected SDK internal output structure: {type(raw)}")
                final_values, internal_values = raw
                finals = _map_shapes(final_values, FINAL_SHAPES)
                internals = _map_shapes(internal_values, internal_shapes)
                captured[index] = {
                    "resize2_code": sdk_codes_to_integer(internals["resize2"], "UINT8"),
                    "x_l_code": sdk_codes_to_integer(internals["x_l"], "UINT16"),
                    "x_l_final": np.ascontiguousarray(finals["x_l"], dtype=np.float32),
                }
        finally:
            simulation.model.reset_output_interal_layers()
    del runner
    return captured


def sdk_bit_exact_gate(candidate_qhar: Path) -> dict[str, Any]:
    formal = sdk_capture(FORMAL_QHAR)
    candidate = sdk_capture(candidate_qhar)
    rows = {"resize2_code": [], "x_l_code": []}
    sample_rows = []
    final_x_l_exact = True
    for index in SAMPLE_INDICES:
        sample = {"val_index": index}
        for key in rows:
            metric = code_comparison(candidate[index][key], formal[index][key])
            rows[key].append(metric)
            sample[key] = {
                "formal_sha256": array_sha(formal[index][key]),
                "candidate_sha256": array_sha(candidate[index][key]),
                "mismatch_count": metric["mismatch_count"],
                "max_abs_code_diff": metric["max_abs_code_diff"],
            }
        exact_final = np.array_equal(candidate[index]["x_l_final"], formal[index]["x_l_final"])
        final_x_l_exact = final_x_l_exact and exact_final
        sample["x_l_final_float32"] = {
            "formal_sha256": array_sha(formal[index]["x_l_final"]),
            "candidate_sha256": array_sha(candidate[index]["x_l_final"]),
            "bit_exact": exact_final,
        }
        sample_rows.append(sample)
    summaries = {key: summarize_code_comparisons(value) for key, value in rows.items()}
    if any(value["mismatch_count"] != 0 for value in summaries.values()):
        raise ValueError(f"golden8 SDK internal-code bit-exact gate failed: {summaries}")
    if not final_x_l_exact:
        raise ValueError("golden8 SDK final x_l float32 bit-exact gate failed")
    return {
        "status": "PASS",
        "sample_count": len(SAMPLE_INDICES),
        "sample_indices": list(SAMPLE_INDICES),
        "api": "SDK_QUANTIZED set_output_interal_layers",
        "formal_quantized_har_sha256": FORMAL_QHAR_SHA,
        "candidate_quantized_har_sha256": sha256(candidate_qhar),
        "internal_layers": {
            "resize2_code": {
                "layer": INTERNALS["resize2"][0],
                "role": "exact conv4 input code",
                "shape_hwc": list(INTERNALS["resize2"][1]),
                "format": INTERNALS["resize2"][2],
            },
            "x_l_code": {
                "layer": INTERNALS["x_l"][0],
                "role": "exact conv3/x_l output code",
                "shape_hwc": list(INTERNALS["x_l"][1]),
                "format": INTERNALS["x_l"][2],
            },
        },
        "summary": summaries,
        "x_l_final_float32_bit_exact_all": final_x_l_exact,
        "samples": sample_rows,
    }


def run(root: Path, gpu: str) -> dict[str, Any]:
    refuse_existing(root)
    root.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    state = {
        "kind": "bn_g_s_matched_precision_precompile",
        "schema": "matched_precision_precompile.v1",
        "status": "RUNNING",
        "stage": "source_gate",
        "started_utc": started,
        "out_root": str(root),
        "policy": {
            "module": "bn_g_s only",
            "compile_allowed": False,
            "board_allowed": False,
            "full976_run": False,
            "semantic_ranking": False,
        },
    }
    atomic_json(root / "campaign_state.json", state)
    report = dict(state)
    try:
        report["source_gate"] = source_gate(gpu)
        state.update(stage="ptq")
        atomic_json(root / "campaign_state.json", state)
        report["ptq"] = run_ptq(root, gpu)
        qhar = Path(report["ptq"]["quantized_har"]["path"])
        state.update(stage="hn_precision_gate")
        atomic_json(root / "campaign_state.json", state)
        report["hn_precision_gate"] = hn_gate(qhar)
        state.update(stage="golden8_sdk_bit_exact_gate")
        atomic_json(root / "campaign_state.json", state)
        report["golden8_sdk_bit_exact_gate"] = sdk_bit_exact_gate(qhar)
        forbidden = [
            str(path) for path in root.rglob("*")
            if path.is_file() and (
                path.suffix.lower() == ".hef"
                or path.name.endswith("_compiled.har")
                or path.name.endswith(".auto.alls")
            )
        ]
        if forbidden:
            raise ValueError(f"precompile root contains compile artifacts: {forbidden}")
        report.update(
            status="WAITING_FOR_COMPILE_REVIEW",
            stage="precompile_complete",
            decision=(
                "Candidate satisfies the requested HN precision contract and both frozen "
                "golden8 causal bit-exact gates. No compile or board action was performed."
            ),
            compile_artifacts=[],
        )
        return report
    except Exception as exc:
        report.update(
            status="STOPPED_PRECOMPILE_GATE_FAILED",
            stage=state["stage"],
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        )
        return report
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(root / "precompile_report.json", report)
        state.update(
            status=report.get("status", "STOPPED_PRECOMPILE_GATE_FAILED"),
            stage=report.get("stage", state["stage"]),
            finished_utc=report["finished_utc"],
        )
        atomic_json(root / "campaign_state.json", state)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--gpu", default="2")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run(args.out_root.resolve(), args.gpu)
    print(
        json.dumps(
            {
                "status": report["status"],
                "stage": report["stage"],
                "out_root": str(args.out_root.resolve()),
                "precompile_report": str(args.out_root.resolve() / "precompile_report.json"),
                "error": report.get("error"),
            },
            indent=2,
        )
    )
    return 0 if report["status"] == "WAITING_FOR_COMPILE_REVIEW" else 1


if __name__ == "__main__":
    raise SystemExit(main())
