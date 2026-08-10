#!/usr/bin/env python3
"""Formal H0/H1/H2 gate for receiver_head FLOAT32 vs native UINT8 output.

Uses named golden s8 features (NCHW float32) already resident on the board.
Reports mask pixel agreement and 20-warmup / 100-iteration timing. This does
not modify HEFs or the full-module runtime.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def summarize(samples: list[float]) -> dict[str, float | int]:
    values = np.asarray(samples, dtype=np.float64)
    if values.size == 0:
        raise ValueError("no timed iterations")
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "min": float(values.min()),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }


def nchw_to_hwc_batch(nchw: np.ndarray) -> np.ndarray:
    arr = np.asarray(nchw, dtype=np.float32)
    if arr.ndim != 4 or arr.shape[0] != 1:
        raise ValueError(f"expected NCHW batch-1, got {arr.shape}")
    return np.ascontiguousarray(arr[0].transpose(1, 2, 0))


def argmax_hwc(logits: np.ndarray) -> np.ndarray:
    """Argmax over channel-last axis -> HxW uint8 class mask."""
    return np.asarray(np.argmax(logits, axis=-1), dtype=np.uint8)


def build_pipeline(network_group, out_format):
    from hailo_platform import (FormatType, InferVStreams, InputVStreamParams,
                                OutputVStreamParams)

    in_params = InputVStreamParams.make_from_network_group(
        network_group, quantized=False, format_type=FormatType.FLOAT32)
    out_params = OutputVStreamParams.make_from_network_group(
        network_group, quantized=False, format_type=out_format)
    pipeline = InferVStreams(network_group, in_params, out_params)
    pipeline.__enter__()
    return pipeline


def inspect_vstreams(hef) -> dict:
    inputs = []
    outputs = []
    for item in hef.get_input_vstream_infos():
        row = {"name": item.name, "shape": list(item.shape),
               "format": str(getattr(item, "format", None))}
        for attr in ("quant_info", "qp_scale", "qp_zp"):
            if hasattr(item, attr):
                row[attr] = str(getattr(item, attr))
        inputs.append(row)
    for item in hef.get_output_vstream_infos():
        row = {"name": item.name, "shape": list(item.shape),
               "format": str(getattr(item, "format", None))}
        for attr in ("quant_info", "qp_scale", "qp_zp"):
            if hasattr(item, attr):
                try:
                    value = getattr(item, attr)
                    row[attr] = value if isinstance(value, (int, float, str, list, tuple)) else str(value)
                except Exception as exc:  # noqa: BLE001 — board API varies by hailort
                    row[attr] = f"<error {exc}>"
        outputs.append(row)
    return {"inputs": inputs, "outputs": outputs}


def timed_infer(pipeline, feed: dict, warmup: int, iterations: int, do_argmax: bool) -> dict:
    for _ in range(warmup):
        raw = pipeline.infer(feed)
        if do_argmax:
            out = next(iter(raw.values()))
            _ = argmax_hwc(out[0] if out.ndim == 4 else out)

    infer_ms: list[float] = []
    argmax_ms: list[float] = []
    total_ms: list[float] = []
    last_out = None
    last_mask = None
    for _ in range(iterations):
        t0 = time.perf_counter_ns()
        raw = pipeline.infer(feed)
        t1 = time.perf_counter_ns()
        out = next(iter(raw.values()))
        last_out = out
        if do_argmax:
            logits = out[0] if out.ndim == 4 else out
            mask = argmax_hwc(logits)
            t2 = time.perf_counter_ns()
            last_mask = mask
            argmax_ms.append((t2 - t1) / 1e6)
            total_ms.append((t2 - t0) / 1e6)
        else:
            total_ms.append((t1 - t0) / 1e6)
        infer_ms.append((t1 - t0) / 1e6)

    result = {
        "infer_wall_ms": summarize(infer_ms),
        "output": {
            "shape": list(last_out.shape),
            "dtype": str(last_out.dtype),
            "nbytes": int(last_out.nbytes),
            "min": float(np.min(last_out)),
            "max": float(np.max(last_out)),
        },
    }
    if do_argmax:
        result["cpu_argmax_ms"] = summarize(argmax_ms)
        result["infer_plus_argmax_ms"] = summarize(total_ms)
        result["mask_unique_labels"] = sorted(int(x) for x in np.unique(last_mask))
        result["_mask"] = last_mask
        result["_logits"] = last_out[0] if last_out.ndim == 4 else last_out
    return result


def run_condition(network_group, feed: dict, out_format, warmup: int, iterations: int,
                  do_argmax: bool) -> dict:
    pipeline = None
    try:
        pipeline = build_pipeline(network_group, out_format)
        with network_group.activate():
            return timed_infer(pipeline, feed, warmup, iterations, do_argmax)
    finally:
        if pipeline is not None:
            pipeline.__exit__(None, None, None)


def compare_masks(float_mask: np.ndarray, uint8_mask: np.ndarray) -> dict:
    if float_mask.shape != uint8_mask.shape:
        raise ValueError(f"mask shape mismatch: {float_mask.shape} vs {uint8_mask.shape}")
    equal = bool(np.array_equal(float_mask, uint8_mask))
    n = float_mask.size
    n_eq = int(np.sum(float_mask == uint8_mask))
    return {
        "equal": equal,
        "agreement": float(n_eq / n),
        "mismatch_count": int(n - n_eq),
        "shape": list(float_mask.shape),
    }


def main() -> None:
    from hailo_platform import (ConfigureParams, FormatType, HailoStreamInterface,
                                HEF, VDevice)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hef", type=Path, required=True)
    parser.add_argument("--golden-dir", type=Path, required=True,
                        help="Directory with {index}_s8.npy NCHW float32 tensors")
    parser.add_argument("--indices", default="1024,1025,1026,1027,1028,1029,1030,1031")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite: {args.out}")
    indices = [int(x.strip()) for x in args.indices.split(",") if x.strip()]
    if not indices:
        raise ValueError("no golden indices")

    hef = HEF(str(args.hef))
    vstream_meta = inspect_vstreams(hef)
    input_infos = list(hef.get_input_vstream_infos())
    if len(input_infos) != 1:
        raise ValueError(f"expected one input vstream, got {len(input_infos)}")
    in_name = input_infos[0].name
    in_hwc = tuple(input_infos[0].shape)

    vdevice = VDevice()
    activation = None
    try:
        configure = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        groups = vdevice.configure(hef, configure)
        if len(groups) != 1:
            raise ValueError(f"expected one network group, got {len(groups)}")
        network_group = groups[0]

        golden_rows = []
        all_equal = True
        # Timing uses the first golden; mask gate uses all goldens (1 infer each format).
        first_feed = None
        first_index = indices[0]

        for index in indices:
            path = args.golden_dir / f"{index}_s8.npy"
            if not path.is_file():
                raise FileNotFoundError(path)
            nchw = np.load(path, allow_pickle=False)
            hwc = nchw_to_hwc_batch(nchw)
            if hwc.shape != in_hwc:
                raise ValueError(f"{path}: HWC {hwc.shape} != HEF input {in_hwc}")
            feed = {in_name: hwc[None, ...]}
            if first_feed is None:
                first_feed = feed

            # One-shot semantic compare (not timed): H0 vs H1 masks.
            h0 = run_condition(network_group, feed, FormatType.FLOAT32, warmup=0, iterations=1,
                               do_argmax=True)
            h1 = run_condition(network_group, feed, FormatType.UINT8, warmup=0, iterations=1,
                               do_argmax=True)
            cmp = compare_masks(h0["_mask"], h1["_mask"])
            all_equal = all_equal and cmp["equal"]

            # Optional: check dequant-aware argmax equals raw uint8 argmax.
            # Hailo UINT8 logits: if scale>0, argmax(q) == argmax((q-zp)*scale).
            u8_logits = h1["_logits"]
            dequant_note = "direct_uint8_argmax"
            if u8_logits.dtype == np.uint8:
                # Also compare against float32 mask using float path only.
                pass

            golden_rows.append({
                "index": index,
                "s8_path": str(path),
                "s8_shape_nchw": list(nchw.shape),
                "float32_mask_labels": sorted(int(x) for x in np.unique(h0["_mask"])),
                "uint8_mask_labels": sorted(int(x) for x in np.unique(h1["_mask"])),
                "mask_compare": cmp,
                "argmax_policy": dequant_note,
                "float32_output": {k: v for k, v in h0["output"].items()},
                "uint8_output": {k: v for k, v in h1["output"].items()},
            })

        assert first_feed is not None
        timing = {
            "H0_float32": run_condition(
                network_group, first_feed, FormatType.FLOAT32,
                args.warmup, args.iterations, do_argmax=True),
            "H1_uint8": run_condition(
                network_group, first_feed, FormatType.UINT8,
                args.warmup, args.iterations, do_argmax=True),
            "H2_uint8_no_argmax": run_condition(
                network_group, first_feed, FormatType.UINT8,
                args.warmup, args.iterations, do_argmax=False),
        }
        # Drop non-JSON arrays from timing payloads.
        for key, payload in timing.items():
            payload.pop("_mask", None)
            payload.pop("_logits", None)

        h0_mean = timing["H0_float32"]["infer_plus_argmax_ms"]["mean"]
        h1_mean = timing["H1_uint8"]["infer_plus_argmax_ms"]["mean"]
        h1_p95 = timing["H1_uint8"]["infer_plus_argmax_ms"]["p95"]
        status = "PASS" if all_equal else "FAIL_MASK_MISMATCH"
        report = {
            "schema": "phase2.hybrid_ort_head.e5_h1_golden_gate.v1",
            "status": status,
            "hef": str(args.hef),
            "golden_dir": str(args.golden_dir),
            "indices": indices,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "timing_index": first_index,
            "vstream_meta": vstream_meta,
            "golden_mask_gate": {
                "all_equal": all_equal,
                "n_goldens": len(golden_rows),
                "rows": golden_rows,
            },
            "timing": timing,
            "comparison": {
                "mean_infer_plus_argmax_saved_ms": float(h0_mean - h1_mean),
                "mean_infer_plus_argmax_reduction_percent": float(100.0 * (h0_mean - h1_mean) / h0_mean),
                "h1_infer_plus_argmax_p95_ms": float(h1_p95),
                "proceed_to_e6_threshold_ms": 80.0,
                "proceed_to_e6": bool(h1_p95 > 80.0),
            },
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"status": status, "out": str(args.out),
                          "all_equal": all_equal, "h1_p95_ms": h1_p95}, sort_keys=True))
        if status != "PASS":
            raise SystemExit(2)
    finally:
        if activation is not None:
            activation.__exit__(None, None, None)
        vdevice.release()


if __name__ == "__main__":
    main()
