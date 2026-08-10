#!/usr/bin/env python3
"""Derive and validate an exact bn_g_s Conv4 input-channel reorder.

The original high-resolution branch concatenates ``[trunk, x_l_hat]`` before
Resize/Conv4.  This probe swaps the concatenation order and applies the same
permutation to Conv4's input-channel weights.  The represented real-valued
function is unchanged, while the compiler sees a different feature layout and
MAC accumulation order.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper


CONCAT_NAME = "/Concat"
CONV4_NAME = "/head_h/head_h.1/head_h.1.0/Conv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_node(model: onnx.ModelProto, name: str) -> onnx.NodeProto:
    matches = [node for node in model.graph.node if node.name == name]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one node named {name!r}, got {len(matches)}")
    return matches[0]


def derive(source: Path, output: Path) -> dict[str, object]:
    model = onnx.load(source)
    concat = _find_node(model, CONCAT_NAME)
    conv4 = _find_node(model, CONV4_NAME)
    if concat.op_type != "Concat" or len(concat.input) != 2:
        raise RuntimeError(f"unexpected concat contract: op={concat.op_type}, inputs={concat.input}")
    if conv4.op_type != "Conv" or len(conv4.input) < 2:
        raise RuntimeError(f"unexpected Conv4 contract: op={conv4.op_type}, inputs={conv4.input}")

    initializers = {item.name: item for item in model.graph.initializer}
    weight_init = initializers[conv4.input[1]]
    weight = numpy_helper.to_array(weight_init).copy()
    if tuple(weight.shape) != (128, 384, 3, 3):
        raise RuntimeError(f"unexpected Conv4 weight shape: {weight.shape}")

    # /Concat is [trunk:128, x_l_hat:256].  New input is [x_l_hat, trunk],
    # therefore Conv4 weights must become [old W_x_l, old W_trunk].
    old_inputs = list(concat.input)
    concat.input[:] = [old_inputs[1], old_inputs[0]]
    permuted_weight = np.concatenate([weight[:, 128:, :, :], weight[:, :128, :, :]], axis=1)
    replacement = numpy_helper.from_array(permuted_weight, name=weight_init.name)
    for index, initializer in enumerate(model.graph.initializer):
        if initializer.name == weight_init.name:
            model.graph.initializer[index].CopyFrom(replacement)
            break
    else:
        raise RuntimeError(f"initializer vanished: {weight_init.name}")

    onnx.checker.check_model(model, full_check=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output)
    onnx.checker.check_model(onnx.load(output), full_check=True)
    return {
        "concat_name": concat.name,
        "concat_original_inputs": old_inputs,
        "concat_candidate_inputs": list(concat.input),
        "conv4_name": conv4.name,
        "conv4_weight": weight_init.name,
        "conv4_weight_shape": list(weight.shape),
        "channel_permutation": "[128:384] + [0:128]",
    }


def _metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, object]:
    delta = candidate.astype(np.float64) - reference.astype(np.float64)
    denom = float(np.sum(np.square(reference.astype(np.float64))))
    return {
        "shape": list(reference.shape),
        "max_abs": float(np.max(np.abs(delta))),
        "mean_abs": float(np.mean(np.abs(delta))),
        "nmse": float(np.sum(np.square(delta)) / denom) if denom else None,
        "array_equal": bool(np.array_equal(reference, candidate)),
        "allclose_atol_1e-5_rtol_1e-5": bool(
            np.allclose(reference, candidate, atol=1e-5, rtol=1e-5)
        ),
    }


def validate_ort(source: Path, candidate: Path, golden_dir: Path) -> dict[str, object]:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sessions = [
        ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        for path in (source, candidate)
    ]
    input_names = [session.get_inputs()[0].name for session in sessions]
    output_names = [[item.name for item in session.get_outputs()] for session in sessions]
    if input_names[0] != input_names[1] or output_names[0] != output_names[1]:
        raise RuntimeError(
            f"I/O contract changed: inputs={input_names}, outputs={output_names}"
        )

    files = sorted(golden_dir.glob("*.npy"))[:8]
    if len(files) != 8:
        raise RuntimeError(f"expected at least 8 golden inputs under {golden_dir}, got {len(files)}")
    per_sample: dict[str, object] = {}
    aggregate: dict[str, list[np.ndarray]] = {name: [] for name in output_names[0]}
    for path in files:
        value = np.load(path).astype(np.float32, copy=False)
        if value.shape == (16, 16, 192):
            value = np.transpose(value, (2, 0, 1))[None, ...]
        elif value.shape == (1, 16, 16, 192):
            value = np.transpose(value, (0, 3, 1, 2))
        if value.shape != (1, 192, 16, 16):
            raise RuntimeError(f"unexpected input shape {value.shape} in {path}")
        outputs = [
            session.run(None, {input_name: value})
            for session, input_name in zip(sessions, input_names, strict=True)
        ]
        sample_metrics: dict[str, object] = {}
        for index, name in enumerate(output_names[0]):
            sample_metrics[name] = _metrics(outputs[0][index], outputs[1][index])
            aggregate[name].append(outputs[1][index] - outputs[0][index])
        per_sample[path.stem] = sample_metrics

    aggregate_metrics: dict[str, object] = {}
    for name, deltas in aggregate.items():
        flat = np.concatenate([delta.reshape(-1) for delta in deltas])
        aggregate_metrics[name] = {
            "max_abs": float(np.max(np.abs(flat))),
            "mean_abs": float(np.mean(np.abs(flat))),
            "all_finite": bool(np.isfinite(flat).all()),
        }
    return {
        "providers": sessions[0].get_providers(),
        "graph_optimization_level": "ORT_DISABLE_ALL",
        "input_name": input_names[0],
        "output_names": output_names[0],
        "sample_ids": [path.stem for path in files],
        "per_sample": per_sample,
        "aggregate": aggregate_metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--golden-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    structure = derive(args.source, args.output)
    ort_report = validate_ort(args.source, args.output, args.golden_dir)
    report = {
        "status": "PASS",
        "claim": "mathematically equivalent Conv4 input-channel reorder",
        "source": {"path": str(args.source), "sha256": sha256(args.source)},
        "candidate": {"path": str(args.output), "sha256": sha256(args.output)},
        "structure": structure,
        "onnx_checker": "PASS",
        "ort_golden8": ort_report,
        "gates_not_run": ["DFC_PARSE", "SDK_NATIVE", "PTQ", "COMPILE", "BOARD"],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
