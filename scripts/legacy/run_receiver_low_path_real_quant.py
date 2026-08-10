#!/usr/bin/env python3
"""Quantize a receiver low-path candidate with real x_l activations only.

This is intentionally a quantization gate, not a compile driver.  It records
module-level emulator error so an unacceptably distorted candidate cannot be
advanced to HEF compilation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if __name__ == "__main__":
    from run_hailo_logged import reexec_if_unlogged

    reexec_if_unlogged(Path(__file__), sys.argv[1:])

from hailo_deploy.emulate import _build_native_feed, _match_outputs
from hailo_deploy.onnx_io import nhwc_to_nchw, run_onnxruntime, signature


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_x_l(root: Path, limit: int) -> tuple[np.ndarray, dict]:
    loader = root / "load_calib.py"
    metadata_path = root / "meta.json"
    if not loader.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"incomplete calibration handoff: {root}")
    spec = importlib.util.spec_from_file_location("receiver_stage45_calib", loader)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import calibration loader: {loader}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if limit > int(metadata["sample_count"]):
        raise ValueError(f"requested {limit} samples, only {metadata['sample_count']} available")
    arrays = module.load_calib(limit=limit)
    if "x_l" not in arrays:
        raise KeyError(f"expected source x_l, found {sorted(arrays)}")
    return arrays["x_l"], metadata


def _quantized_check(runner, onnx_path: Path, calibration_nhwc: np.ndarray,
                     count: int) -> dict:
    from hailo_sdk_client import InferenceContext

    count = min(count, len(calibration_nhwc))
    if count <= 0:
        return {"status": "SKIP", "reason": "eval_samples=0"}
    sig = signature(onnx_path)
    if sig.inputs != [("x_l", [1, 256, 32, 32])]:
        raise ValueError(f"candidate input contract changed unexpectedly: {sig.inputs}")
    nchw = nhwc_to_nchw(calibration_nhwc[-count:])
    notes: list[str] = []
    accumulators: dict[str, dict] = {}
    with runner.infer_context(InferenceContext.SDK_QUANTIZED) as ctx:
        for index in range(count):
            sample = {"x_l": nchw[index:index + 1]}
            ort_outputs = run_onnxruntime(onnx_path, sample)
            feed = _build_native_feed(runner, sig, sample, notes)
            quantized = runner.infer(ctx, feed)
            pairs = _match_outputs(runner, ort_outputs, quantized, notes)
            for name, (reference, actual) in pairs.items():
                delta = np.abs(reference.astype(np.float32) - actual.astype(np.float32))
                value = accumulators.setdefault(name, {
                    "max_abs_diff": 0.0, "absolute_error_sum": 0.0,
                    "reference_abs_sum": 0.0, "element_count": 0,
                })
                value["max_abs_diff"] = max(value["max_abs_diff"], float(delta.max()))
                value["absolute_error_sum"] += float(delta.astype(np.float64).sum())
                value["reference_abs_sum"] += float(np.abs(reference).astype(np.float64).sum())
                value["element_count"] += int(delta.size)
    if not accumulators:
        return {"status": "FAIL", "error": "no outputs matched", "notes": notes}
    outputs = {}
    for name, value in accumulators.items():
        mae = value["absolute_error_sum"] / value["element_count"]
        reference_mae = value["reference_abs_sum"] / value["element_count"]
        outputs[name] = {
            "max_abs_diff": value["max_abs_diff"],
            "mean_abs_diff": mae,
            "reference_abs_mean": reference_mae,
            "relative_mean_abs_error": mae / reference_mae if reference_mae else None,
        }
    return {"status": "PASS", "samples": count, "per_output": outputs, "notes": notes}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--har", type=Path, required=True)
    parser.add_argument("--calib-root", type=Path, default=Path(
        "/local/shared_with_docker/calib/receiver_stage45"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-calib", type=int, default=128)
    parser.add_argument("--opt-level", type=int, default=0)
    parser.add_argument("--eval-samples", type=int, default=4)
    args = parser.parse_args()
    if not args.onnx.is_file() or not args.har.is_file():
        parser.error("--onnx and --har must exist")
    if args.n_calib <= 0 or args.opt_level not in range(5) or args.eval_samples < 0:
        parser.error("invalid calibration, optimization, or evaluation count")
    if args.out_dir.exists():
        parser.error(f"refusing to overwrite candidate result directory: {args.out_dir}")

    result = {
        "status": "FAIL",
        "kind": "real_calibration_low_path_quantization",
        "onnx": {"path": str(args.onnx), "sha256": _sha256(args.onnx)},
        "har": {"path": str(args.har), "sha256": _sha256(args.har)},
        "calibration": {"source_root": str(args.calib_root), "count": args.n_calib},
        "optimization_level": args.opt_level,
        "quantized_har": None,
        "quantized_check": None,
        "error": None,
    }
    args.out_dir.mkdir(parents=True, exist_ok=False)
    try:
        from hailo_sdk_client import ClientRunner

        calibration, metadata = _load_x_l(args.calib_root, args.n_calib)
        runner = ClientRunner(har=str(args.har))
        runner.load_model_script(
            f"model_optimization_flavor(optimization_level={args.opt_level}, "
            "compression_level=0, batch_size=1)\n")
        runner.optimize(calibration)
        quantized_har = args.out_dir / "receiver_low_path_quantized.har"
        runner.save_har(str(quantized_har))
        result["quantized_har"] = {
            "path": str(quantized_har), "sha256": _sha256(quantized_har),
            "bytes": quantized_har.stat().st_size,
        }
        result["calibration"].update({
            "checkpoint": metadata["checkpoint"],
            "source_onnx": metadata["onnx"],
            "codec": metadata["codec"],
            "roundtrip_alignment": metadata["roundtrip_alignment"],
        })
        result["quantized_check"] = _quantized_check(
            runner, args.onnx, calibration, args.eval_samples)
        result["status"] = "PASS"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    result_path = args.out_dir / "result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"result: {result_path}")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
