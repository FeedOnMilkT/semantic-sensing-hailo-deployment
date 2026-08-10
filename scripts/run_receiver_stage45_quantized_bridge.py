#!/usr/bin/env python3
"""Emit full-val receiver_stage45 SDK_QUANTIZED outputs for the semantic gate.

Inputs are frozen codec reconstructions produced in the training container.  This
script has no RGB path: it accepts only the two real receiver activation tensors
and writes per-sample dequantized emulator outputs plus provenance.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if __name__ == "__main__":
    from hailo_deploy.logging import reexec_if_unlogged
    reexec_if_unlogged(Path(__file__), sys.argv[1:])

from hailo_deploy.io.hash import sha256
from hailo_deploy.io.handoff import load_json
from hailo_deploy.onnx_io import signature

CKPT_SHA = "1f9aef6272152396e53c827b834fe34d9b0bb6d0198f972ff26b537603fd0fb9"
ONNX_SHA = "0af3f4f9ae434c58d9d634f9735c1d2ff544acdff769e7b1ae0da4077cf688d1"


def names(count: int):
    return [f"{i:05d}.npy" for i in range(count)]


def validate_dir(path: Path, count: int, shape: tuple[int, ...]):
    if sorted(x.name for x in path.glob("*.npy")) != names(count):
        raise ValueError(f"non-contiguous inputs: {path}")
    for name in names(count):
        array = np.load(path / name, allow_pickle=False)
        if array.shape != shape or array.dtype != np.float32 or not np.isfinite(array).all():
            raise ValueError(f"invalid input {path / name}: {array.shape}/{array.dtype}")


def input_mapping(runner):
    result = {}
    for hn_name, layer in runner.get_hn_dict().get("layers", {}).items():
        if "input" not in str(layer.get("type", "")).lower():
            continue
        originals = layer.get("original_names") or []
        for name in ("x_h", "x_l"):
            if name in originals:
                if name in result:
                    raise ValueError(f"ambiguous HN input mapping for {name}")
                result[name] = hn_name
    if set(result) != {"x_h", "x_l"}:
        raise ValueError(f"unexpected HN input mapping: {result}")
    return result


def output_mapping(runner, raw, batch_count: int):
    expected = {"high_feat": (64, 64, 128), "low_feat": (16, 16, 512)}
    original_by_hn = {
        name: layer.get("original_names") or []
        for name, layer in runner.get_hn_dict().get("layers", {}).items()
        if "output" in str(layer.get("type", "")).lower()
    }
    result = {}
    values = raw.items() if isinstance(raw, dict) else []
    for hn_name, value in values:
        hit = [name for name in expected if name in original_by_hn.get(hn_name, [])]
        if len(hit) == 1:
            result[hit[0]] = np.asarray(value)
    if set(result) != set(expected):
        result = {}
        values = list(raw.values()) if isinstance(raw, dict) else list(raw if isinstance(raw, (list, tuple)) else [raw])
        for value in values:
            array = np.asarray(value)
            hit = [name for name, shape in expected.items()
                   if tuple(array.shape) == (batch_count, *shape)]
            if len(hit) == 1 and hit[0] not in result:
                result[hit[0]] = array
    if set(result) != set(expected):
        raise ValueError(f"could not map HN outputs: {list(result)}")
    normalized = {}
    for name, value in result.items():
        array = np.asarray(value)
        if (tuple(array.shape) != (batch_count, *expected[name])
                or not np.issubdtype(array.dtype, np.floating)
                or not np.isfinite(array).all()):
            raise ValueError(f"invalid output {name}: {array.shape}/{array.dtype}")
        normalized[name] = np.ascontiguousarray(array.astype(np.float32, copy=False))
    return normalized


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--quantized-har", type=Path, required=True)
    ap.add_argument("--ptq-gate-json", "--l0-gate-json", dest="ptq_gate_json", type=Path,
                    required=True, help="PASS gate JSON for the quantized HAR")
    ap.add_argument("--input-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=64,
                    help="SDK emulator batch size; does not alter per-sample outputs")
    args = ap.parse_args()
    if args.batch_size <= 0:
        ap.error("--batch-size must be positive")
    if args.out_dir.exists():
        ap.error(f"refusing to overwrite {args.out_dir}")
    if sha256(args.onnx) != ONNX_SHA:
        ap.error("candidate ONNX SHA mismatch")
    if dict(signature(args.onnx).inputs) != {"x_h": [1, 128, 64, 64], "x_l": [1, 256, 32, 32]}:
        ap.error("candidate ONNX input contract changed")
    gate = load_json(args.ptq_gate_json)
    real = gate.get("real_level0") or {}
    if gate.get("status") != "PASS" or real.get("status") != "PASS":
        ap.error("PTQ gate did not pass")
    if Path(real.get("quantized_har", {}).get("path", "")).resolve() != args.quantized_har.resolve():
        ap.error("quantized HAR is not the PTQ gate artifact")
    calibration = real.get("calibration") or {}
    if (calibration.get("checkpoint", {}).get("sha256") != CKPT_SHA
            or calibration.get("candidate_onnx_sha256") != ONNX_SHA):
        ap.error("PTQ calibration identity mismatch")
    input_meta_path = args.input_root / "meta.json"
    inputs = load_json(input_meta_path)
    count = int(inputs.get("sample_count", 0))
    if (count != 2000 or inputs.get("val_indices") != list(range(count))
            or inputs.get("checkpoint", {}).get("sha256") != CKPT_SHA
            or inputs.get("onnx", {}).get("sha256") != ONNX_SHA):
        ap.error("input bridge is not the exact 2000-sample frozen candidate contract")
    validate_dir(args.input_root / "x_h", count, (64, 64, 128))
    validate_dir(args.input_root / "x_l", count, (32, 32, 256))

    args.out_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out_dir.parent / f".{args.out_dir.name}.tmp-{os.getpid()}"
    if temporary.exists():
        ap.error(f"temporary directory exists: {temporary}")
    temporary.mkdir()
    try:
        from hailo_sdk_client import ClientRunner, InferenceContext
        runner = ClientRunner(har=str(args.quantized_har))
        mapping = input_mapping(runner)
        high_dir, low_dir = temporary / "high_feat", temporary / "low_feat"
        high_dir.mkdir()
        low_dir.mkdir()
        with runner.infer_context(InferenceContext.SDK_QUANTIZED) as context:
            for start in range(0, count, args.batch_size):
                stop = min(start + args.batch_size, count)
                feed = {
                    mapping["x_h"]: np.stack([
                        np.load(args.input_root / "x_h" / f"{index:05d}.npy", allow_pickle=False)
                        for index in range(start, stop)]),
                    mapping["x_l"]: np.stack([
                        np.load(args.input_root / "x_l" / f"{index:05d}.npy", allow_pickle=False)
                        for index in range(start, stop)]),
                }
                output = output_mapping(runner, runner.infer(
                    context, feed, batch_size=stop - start), stop - start)
                for local_index, index in enumerate(range(start, stop)):
                    np.save(high_dir / f"{index:05d}.npy", output["high_feat"][local_index])
                    np.save(low_dir / f"{index:05d}.npy", output["low_feat"][local_index])
                print(f"emitted {stop}/{count}", flush=True)
        meta = {
            "kind": "receiver_stage45_sdk_quantized_outputs", "emulator": "SDK_QUANTIZED",
            "quantized_har": {"path": str(args.quantized_har.resolve()), "sha256": sha256(args.quantized_har)},
            "ptq_gate_json": {"path": str(args.ptq_gate_json.resolve()), "sha256": sha256(args.ptq_gate_json)},
            "calibration": calibration, "input_root": str(args.input_root.resolve()),
            "input_metadata_sha256": sha256(input_meta_path), "sample_count": count,
            "val_indices": list(range(count)), "input_mapping": mapping,
            "sdk_emulator_batch_size": args.batch_size,
            "outputs": {
                "high_feat": {"layout": "NHWC", "dtype": "float32", "shape": [64, 64, 128]},
                "low_feat": {"layout": "NHWC", "dtype": "float32", "shape": [16, 16, 512]},
            },
        }
        (temporary / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        os.replace(temporary, args.out_dir)
        print(f"complete: {args.out_dir}")
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
