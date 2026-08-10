"""QUANTIZED emulator checks against golden tensors or ONNX Runtime."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from hailo_deploy.emulate import _build_native_feed, _match_outputs
from hailo_deploy.io.calib import map_hn_inputs
from hailo_deploy.io.metrics import (
    accumulate_abs_error,
    finalize_abs_error,
    metric,
    summarize,
)
from hailo_deploy.onnx_io import nchw_to_nhwc, nhwc_to_nchw, run_onnxruntime, signature


def heldout_quantized_check(
    runner,
    module: str,
    handoff: Path,
    meta: dict[str, Any],
    golden_indices: list[int],
) -> dict[str, Any]:
    """Compare SDK_QUANTIZED outputs to held-out FP32 golden module tensors."""
    from hailo_sdk_client import InferenceContext

    semantic_to_onnx = {
        semantic: item["onnx_name"] for semantic, item in meta["tensor_mapping"].items()
    }
    per_output: dict[str, list[dict[str, float | None]]] = {}
    samples: list[dict[str, Any]] = []
    notes: list[str] = []
    with runner.infer_context(InferenceContext.SDK_QUANTIZED) as context:
        for index in golden_indices:
            base = handoff / "evaluation" / "golden" / str(index) / module
            inputs_nchw: dict[str, np.ndarray] = {}
            for semantic, onnx_name in semantic_to_onnx.items():
                tensor = np.load(base / "inputs" / f"{semantic}.npy", allow_pickle=False)
                if tensor.dtype != np.float32 or tensor.ndim != 4 or not np.isfinite(tensor).all():
                    raise ValueError(f"invalid held-out input {module}/{semantic}/{index}")
                inputs_nchw[onnx_name] = tensor
            feed, _ = map_hn_inputs(
                runner, {name: nchw_to_nhwc(value) for name, value in inputs_nchw.items()}
            )
            raw = runner.infer(context, feed)
            expected = {
                name: np.load(base / "outputs" / f"{name}.npy", allow_pickle=False)
                for name in meta["onnx"]["output_names"]
            }
            pairs = _match_outputs(runner, expected, raw, notes)
            if set(pairs) != set(expected):
                raise ValueError(
                    f"could not map all quantized outputs for {module}/{index}: {notes[-3:]}"
                )
            sample = {"val_index": index, "outputs": {}}
            for name, (reference, actual) in pairs.items():
                if not np.isfinite(actual).all():
                    raise ValueError(f"non-finite SDK_QUANTIZED output {module}/{name}/{index}")
                values = metric(reference, actual)
                sample["outputs"][name] = values
                per_output.setdefault(name, []).append(values)
            samples.append(sample)
    return {
        "status": "PASS",
        "kind": "held_out_tensor_drift_not_semantic_acceptance",
        "indices": golden_indices,
        "per_output": {name: summarize(rows) for name, rows in per_output.items()},
        "samples": samples,
        "notes": notes,
    }


def ort_quantized_check(
    runner,
    onnx_path: Path,
    arrays: dict[str, np.ndarray],
    count: int,
) -> dict[str, Any]:
    """Compare SDK_QUANTIZED to ONNX Runtime on the tail of a calib batch."""
    from hailo_sdk_client import InferenceContext

    count = min(count, min(len(value) for value in arrays.values()))
    if count <= 0:
        return {"status": "SKIP", "reason": "eval_samples=0"}
    nchw = {name: nhwc_to_nchw(value[-count:]) for name, value in arrays.items()}
    sig = signature(onnx_path)
    notes: list[str] = []
    accumulators: dict[str, dict[str, Any]] = {}
    with runner.infer_context(InferenceContext.SDK_QUANTIZED) as ctx:
        for index in range(count):
            sample = {name: value[index : index + 1] for name, value in nchw.items()}
            reference = run_onnxruntime(onnx_path, sample)
            feed = _build_native_feed(runner, sig, sample, notes)
            actual = runner.infer(ctx, feed)
            for name, (ref, got) in _match_outputs(runner, reference, actual, notes).items():
                accumulate_abs_error(accumulators, name, ref, got)
    if not accumulators:
        return {"status": "FAIL", "error": "no output pairs", "notes": notes}
    return {
        "status": "PASS",
        "samples": count,
        "per_output": finalize_abs_error(accumulators),
        "notes": notes,
    }
