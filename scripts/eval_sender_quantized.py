#!/usr/bin/env python3
"""Evaluate a DFC-optimized sender model with the Hailo emulator (SDK_QUANTIZED).

Three-stage methodology (deployment guide §5.1):
  SDK_NATIVE      -> the parse is faithful (should match PyTorch/ORT)
  SDK_FP_OPTIMIZED-> the model rewrites (normalization etc.) are faithful
  SDK_QUANTIZED   -> the pure quantization loss (the number we report)

For the backbone the network input is the raw 0..255 NHWC image (the on-chip
`normalization` command does the /255 + ImageNet standardisation).  For the
bottleneck the network input is the pair (x_h, x_l) of already-normalised feature
maps, so the calibration/inference data is a dict keyed by the input layer names.

Usage (hailo-dfc env, from a scratch dir):
  python scripts/eval_sender_quantized.py --har ..._optimized.har --n 64
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np


def _cos(a, b):
    a = np.asarray(a, np.float64).reshape(-1)
    b = np.asarray(b, np.float64).reshape(-1)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _mse(a, b):
    return float(np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2))


def _rel_l2(a, b):
    a = np.asarray(a, np.float64).reshape(-1)
    b = np.asarray(b, np.float64).reshape(-1)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-12))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--har", required=True)
    ap.add_argument("--kind", choices=["backbone", "bottleneck"], default="backbone")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--calib", default="/root/deploy_artifacts/data/calib_u8.npy")
    ap.add_argument("--calib-xh", default="/root/deploy_artifacts/data/calib_xh.npy")
    ap.add_argument("--calib-xl", default="/root/deploy_artifacts/data/calib_xl.npy")
    ap.add_argument("--out", default="")
    ap.add_argument("--hw-arch", default="hailo8")
    a = ap.parse_args(argv)

    from hailo_sdk_client import ClientRunner
    from hailo_sdk_client.exposed_definitions import InferenceContext

    runner = ClientRunner(har=a.har, hw_arch=a.hw_arch)
    layers = runner.get_hn_dict()["layers"]
    items = layers.items() if isinstance(layers, dict) else \
        ((l.get("name"), l) for l in layers)
    input_names = [n for n, _ in items if "/input_layer" in str(n)]
    print(f"input layers: {input_names}", flush=True)

    if a.kind == "backbone":
        raw = np.load(a.calib, mmap_mode="r")[: a.n]
        # the on-chip normalization expects raw 0..255 NHWC
        feeds = {input_names[0]: np.ascontiguousarray(raw).astype(np.float32)}
        ref_feeds = feeds
    else:
        xh = np.ascontiguousarray(np.load(a.calib_xh, mmap_mode="r")[: a.n]).astype(np.float32)
        xl = np.ascontiguousarray(np.load(a.calib_xl, mmap_mode="r")[: a.n]).astype(np.float32)
        # Hailo network inputs are NHWC (channels last): the declared input shape of
        # input_layer1 is (64, 64, 128), not (128, 64, 64).  The .npy files are NCHW.
        xh = np.ascontiguousarray(np.transpose(xh, (0, 2, 3, 1)))
        xl = np.ascontiguousarray(np.transpose(xl, (0, 2, 3, 1)))
        feeds = {input_names[0]: xh, input_names[1]: xl}
        ref_feeds = feeds

    results = {}
    for ctx_name in ("SDK_NATIVE", "SDK_FP_OPTIMIZED", "SDK_QUANTIZED"):
        try:
            with runner.infer_context(getattr(InferenceContext, ctx_name)) as ctx:
                t0 = time.time()
                out = runner.infer(ctx, feeds, data_count=a.n, batch_size=1)
                dt = time.time() - t0
            arrs = [np.asarray(o) for o in out]
            results[ctx_name] = {"outputs": arrs, "sec": dt}
            print(f"[{ctx_name}] {len(arrs)} outputs in {dt:.1f}s "
                  f"shapes={[x.shape for x in arrs]}", flush=True)
        except Exception as e:
            print(f"[{ctx_name}] FAILED: {type(e).__name__}: {e}", flush=True)
            results[ctx_name] = None

    # compare each context against the native context
    native = results.get("SDK_NATIVE")
    summary = {"har": a.har, "kind": a.kind, "n": a.n, "contexts": {}}
    for ctx_name, r in results.items():
        if r is None:
            summary["contexts"][ctx_name] = {"ok": False}
            continue
        entry = {"ok": True, "sec": r["sec"], "outputs": []}
        if native is not None:
            for i, (o, ref) in enumerate(zip(r["outputs"], native["outputs"])):
                entry["outputs"].append({
                    "idx": i,
                    "shape": list(o.shape),
                    "cos": _cos(o, ref),
                    "mse": _mse(o, ref),
                    "rel_l2": _rel_l2(o, ref),
                })
        summary["contexts"][ctx_name] = entry

    print("\n===== QUANTIZATION SUMMARY =====")
    for ctx_name, e in summary["contexts"].items():
        if not e.get("ok"):
            print(f"{ctx_name:18s} FAILED")
            continue
        for o in e["outputs"]:
            print(f"{ctx_name:18s} out{o['idx']} shape={o['shape']} "
                  f"cos={o['cos']:.6f} mse={o['mse']:.4e} relL2={o['rel_l2']:.4e}")

    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(summary, f, indent=1)
        print(f"\nsaved {a.out}")


if __name__ == "__main__":
    raise SystemExit(main())
