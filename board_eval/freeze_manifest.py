#!/usr/bin/env python3
"""Freeze an auditable, calibration-disjoint receiver_stage45 board run.

The generated manifest is the only input accepted by the deploy and evaluator
tools.  It proves that the evaluation indices are a set difference against the
actual calibration metadata, not a convenient hand-written range.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hailo_deploy.io.hash import sha256

EXPECTED = {
    "x_h": {"shape": [64, 64, 128], "vstream": "receiver_stage45/input_layer1"},
    "x_l": {"shape": [32, 32, 256], "vstream": "receiver_stage45/input_layer2"},
    "high_feat": {"shape": [64, 64, 128], "vstream": "receiver_stage45/conv25"},
    "low_feat": {"shape": [16, 16, 512], "vstream": "receiver_stage45/conv20"},
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def check_indexed_dir(directory: Path, indices: list[int]) -> None:
    missing = [index for index in indices if not (directory / f"{index:05d}.npy").is_file()]
    if missing:
        raise ValueError(f"missing {len(missing)} indexed tensors in {directory}; first={missing[:5]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--calibration-meta", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True,
                        help="x_h/x_l/ground_truth root exported by the frozen candidate")
    parser.add_argument("--quantized-root", type=Path, required=True,
                        help="existing SDK_QUANTIZED high_feat/low_feat root")
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--quantized-har", type=Path, required=True)
    parser.add_argument("--hef", type=Path, required=True)
    parser.add_argument("--ptq-gate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite manifest: {args.out}")
    for path in (args.calibration_meta, args.onnx, args.quantized_har, args.hef, args.ptq_gate):
        if not path.is_file():
            raise FileNotFoundError(path)

    calibration = read_json(args.calibration_meta)
    inputs = read_json(args.input_root / "meta.json")
    quantized = read_json(args.quantized_root / "meta.json")
    gate = read_json(args.ptq_gate)
    calibration_indices = calibration.get("dataset", {}).get("val_indices")
    all_indices = inputs.get("val_indices")
    if not isinstance(calibration_indices, list) or not isinstance(all_indices, list):
        raise ValueError("calibration/input metadata lacks explicit val_indices; cannot prove disjointness")
    if len(set(calibration_indices)) != len(calibration_indices):
        raise ValueError("duplicate calibration indices")
    if len(set(all_indices)) != len(all_indices):
        raise ValueError("duplicate input indices")
    evaluation_indices = sorted(set(all_indices) - set(calibration_indices))
    if not evaluation_indices:
        raise ValueError("calibration consumes every available evaluation sample")
    overlap = sorted(set(evaluation_indices) & set(calibration_indices))
    if overlap:
        raise AssertionError(f"evaluation/calibration overlap: {overlap[:5]}")

    real = gate.get("real_level0") or {}
    artifact_har = Path(real.get("quantized_har", {}).get("path", ""))
    if gate.get("status") != "PASS" or real.get("status") != "PASS":
        raise ValueError("PTQ gate is not PASS")
    if artifact_har.resolve() != args.quantized_har.resolve():
        raise ValueError("quantized HAR is not the exact PTQ gate artifact")
    if quantized.get("quantized_har", {}).get("sha256") != sha256(args.quantized_har):
        raise ValueError("cached SDK_QUANTIZED outputs do not match the requested HAR")
    if inputs.get("onnx", {}).get("sha256") != sha256(args.onnx):
        raise ValueError("frozen candidate inputs do not match requested ONNX")
    calibration_onnx = calibration.get("onnx", {}).get("sha256")
    if calibration_onnx != sha256(args.onnx):
        raise ValueError("calibration does not match requested ONNX")

    for directory in (args.input_root / "x_h", args.input_root / "x_l",
                      args.input_root / "ground_truth", args.quantized_root / "high_feat",
                      args.quantized_root / "low_feat"):
        check_indexed_dir(directory, evaluation_indices)

    manifest = {
        "schema_version": 1,
        "run_tag": args.run_tag,
        "candidate": {
            "checkpoint": calibration.get("checkpoint"),
            "onnx": {"path": str(args.onnx.resolve()), "sha256": sha256(args.onnx)},
            "quantized_har": {"path": str(args.quantized_har.resolve()), "sha256": sha256(args.quantized_har)},
            "hef": {"path": str(args.hef.resolve()), "sha256": sha256(args.hef)},
            "ptq_gate": {"path": str(args.ptq_gate.resolve()), "sha256": sha256(args.ptq_gate)},
        },
        "calibration": {
            "meta": str(args.calibration_meta.resolve()),
            "meta_sha256": sha256(args.calibration_meta),
            "sample_count": len(calibration_indices),
            "val_indices": calibration_indices,
        },
        "evaluation": {
            "input_root": str(args.input_root.resolve()),
            "input_meta_sha256": sha256(args.input_root / "meta.json"),
            "quantized_root": str(args.quantized_root.resolve()),
            "quantized_meta_sha256": sha256(args.quantized_root / "meta.json"),
            "sample_count": len(evaluation_indices),
            "val_indices": evaluation_indices,
            "overlap_with_calibration": [],
        },
        "contract": {
            "dtype": "float32",
            "input_layout": "NHWC",
            "output_layout": "NHWC",
            "tensors": EXPECTED,
        },
        "board": {
            "runtime": "HailoRT Python InferVStreams over PCIe",
            "sample_transport": "persistent SSH framed NPZ in memory; no sample files on board",
            "hef_filename": args.hef.name,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "evaluation_samples": len(evaluation_indices),
                      "first": evaluation_indices[:3], "last": evaluation_indices[-3:]}, indent=2))


if __name__ == "__main__":
    main()
