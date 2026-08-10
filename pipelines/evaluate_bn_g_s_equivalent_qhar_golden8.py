#!/usr/bin/env python3
"""Compare an equivalent bn_g_s qHAR with the formal qHAR on golden8.

This is a precompile SDK_QUANTIZED diagnostic.  It never compiles a HEF and
never contacts the board.  The candidate ONNX must already have passed the
separate FP32 equivalence and DFC parse/NATIVE gates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hailo_deploy.emulate import _build_native_feed, _match_outputs
from hailo_deploy.io.hash import sha256
from hailo_deploy.io.metrics import metric, summarize
from hailo_deploy.onnx_io import signature


FORMAL_QHAR = Path(
    "/local/shared_with_docker/deplyment-demo/artifacts/fused/"
    "current-ort-mainline-20260805/bn_g_s_candidates/"
    "x_l_low_spatial_a16_l2/bn_g_s_l2_quantized.har"
)
HANDOFF = Path(
    "/local/shared_with_docker/hailo_full_candidate/"
    "convbn-gfa-gs-3a3ea630af6d2d547808137be678bd909f9789f5594b7bdc76c36b90"
)
FORMAL_ONNX = HANDOFF / "onnx/bn_g_s.onnx"
INDICES = tuple(range(1024, 1032))


def array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def _expected(index: int, output_names: list[str]) -> dict[str, np.ndarray]:
    base = HANDOFF / "evaluation/golden" / str(index) / "bn_g_s/outputs"
    return {
        name: np.load(base / f"{name}.npy", allow_pickle=False)
        for name in output_names
    }


def _input(index: int) -> dict[str, np.ndarray]:
    path = HANDOFF / "evaluation/golden" / str(index) / "bn_g_s/inputs/latent_y_hat.npy"
    value = np.load(path, allow_pickle=False)
    if value.shape != (1, 192, 16, 16) or value.dtype != np.float32:
        raise ValueError(f"unexpected golden input contract at {path}: {value.shape}/{value.dtype}")
    return {"latent_y_hat": value}


def infer_golden8(runner, onnx_path: Path, output_names: list[str]) -> tuple[dict[int, dict[str, np.ndarray]], list[str]]:
    from hailo_sdk_client import InferenceContext

    sig = signature(onnx_path)
    notes: list[str] = []
    results: dict[int, dict[str, np.ndarray]] = {}
    with runner.infer_context(InferenceContext.SDK_QUANTIZED) as context:
        for index in INDICES:
            inputs = _input(index)
            feed = _build_native_feed(runner, sig, inputs, notes)
            raw = runner.infer(context, feed)
            pairs = _match_outputs(runner, _expected(index, output_names), raw, notes)
            if set(pairs) != set(output_names):
                raise ValueError(f"could not map outputs for {index}: {notes[-4:]}")
            results[index] = {name: np.asarray(actual) for name, (_, actual) in pairs.items()}
    return results, notes


def _row(reference: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    values: dict[str, Any] = dict(metric(reference, actual))
    delta = actual.astype(np.float64) - reference.astype(np.float64)
    denom = float(np.sum(np.square(reference.astype(np.float64))))
    values.update(
        {
            "nmse": float(np.sum(np.square(delta)) / denom) if denom else None,
            "array_equal": bool(np.array_equal(reference, actual)),
            "reference_sha256": array_sha(reference),
            "actual_sha256": array_sha(actual),
        }
    )
    return values


def _scalar_summary(values: list[float | None]) -> dict[str, float | int | None]:
    array = np.asarray([value for value in values if value is not None], dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()) if array.size else None,
        "min": float(array.min()) if array.size else None,
        "p50": float(np.percentile(array, 50)) if array.size else None,
        "p95": float(np.percentile(array, 95)) if array.size else None,
        "max": float(array.max()) if array.size else None,
    }


def comparisons(
    formal: dict[int, dict[str, np.ndarray]],
    candidate: dict[int, dict[str, np.ndarray]],
    output_names: list[str],
) -> dict[str, Any]:
    arms = {
        "golden_vs_formal": [],
        "golden_vs_candidate": [],
        "formal_vs_candidate": [],
    }
    samples: list[dict[str, Any]] = []
    for index in INDICES:
        expected = _expected(index, output_names)
        sample: dict[str, Any] = {"val_index": index, "comparisons": {}}
        pairs = {
            "golden_vs_formal": (expected, formal[index]),
            "golden_vs_candidate": (expected, candidate[index]),
            "formal_vs_candidate": (formal[index], candidate[index]),
        }
        for label, (reference, actual) in pairs.items():
            per_output = {
                name: _row(reference[name], actual[name])
                for name in output_names
            }
            sample["comparisons"][label] = per_output
            arms[label].append(per_output)
        samples.append(sample)

    aggregate: dict[str, Any] = {}
    for label, rows in arms.items():
        aggregate[label] = {}
        for name in output_names:
            metric_rows = [row[name] for row in rows]
            aggregate[label][name] = summarize(metric_rows)
            aggregate[label][name]["all_array_equal"] = all(
                bool(row["array_equal"]) for row in metric_rows
            )
            aggregate[label][name]["nmse"] = _scalar_summary(
                [row["nmse"] for row in metric_rows]
            )
    return {"aggregate": aggregate, "samples": samples}


def graph_summary(runner) -> dict[str, Any]:
    hn = runner.get_hn_dict()
    rows: list[dict[str, Any]] = []
    for name, layer in hn.get("layers", {}).items():
        qparams = layer.get("quantization_params") or {}
        rows.append(
            {
                "name": name,
                "type": layer.get("type"),
                "input": list(layer.get("input") or []),
                "output": list(layer.get("output") or []),
                "original_names": list(layer.get("original_names") or []),
                "precision_mode": qparams.get("precision_mode"),
                "activation": (layer.get("params") or {}).get("activation"),
                "input_shapes": layer.get("input_shapes"),
                "output_shapes": layer.get("output_shapes"),
            }
        )
    return {
        "state": str(getattr(runner, "_state", None)),
        "layer_count": len(rows),
        "layers": rows,
        "output_layers": [row for row in rows if "output_layer" in str(row["type"])],
        "conv4_related": [
            row
            for row in rows
            if any(
                token in row["name"].lower()
                for token in ("conv4", "phase", "graph_equiv", "feature_splitter", "ew_add")
            )
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument("--candidate-qhar", type=Path, required=True)
    parser.add_argument("--candidate-onnx", type=Path, required=True)
    parser.add_argument("--formal-qhar", type=Path, default=FORMAL_QHAR)
    parser.add_argument("--formal-onnx", type=Path, default=FORMAL_ONNX)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    for path in (args.formal_qhar, args.formal_onnx, args.candidate_qhar, args.candidate_onnx):
        if not path.is_file():
            raise FileNotFoundError(path)
    from hailo_sdk_client import ClientRunner

    formal_runner = ClientRunner(har=str(args.formal_qhar))
    candidate_runner = ClientRunner(har=str(args.candidate_qhar))
    output_names = ["x_h_hat", "x_l_hat"]
    formal, formal_notes = infer_golden8(formal_runner, args.formal_onnx, output_names)
    candidate, candidate_notes = infer_golden8(candidate_runner, args.candidate_onnx, output_names)
    report = {
        "kind": "bn_g_s_equivalent_qhar_golden8_precompile_v1",
        "status": "PASS",
        "scope": "diagnostic golden8 only; not semantic ranking or acceptance",
        "candidate_label": args.candidate_label,
        "indices": list(INDICES),
        "formal": {
            "qhar": str(args.formal_qhar),
            "qhar_sha256": sha256(args.formal_qhar),
            "onnx": str(args.formal_onnx),
            "onnx_sha256": sha256(args.formal_onnx),
            "graph": graph_summary(formal_runner),
            "mapping_notes": formal_notes,
        },
        "candidate": {
            "qhar": str(args.candidate_qhar),
            "qhar_sha256": sha256(args.candidate_qhar),
            "onnx": str(args.candidate_onnx),
            "onnx_sha256": sha256(args.candidate_onnx),
            "graph": graph_summary(candidate_runner),
            "mapping_notes": candidate_notes,
        },
        "comparisons": comparisons(formal, candidate, output_names),
        "restrictions": {"compile": False, "board": False, "full976": False},
        "environment": {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")},
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "candidate": args.candidate_label,
                "report": str(args.report),
                "formal_vs_candidate": report["comparisons"]["aggregate"]["formal_vs_candidate"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
