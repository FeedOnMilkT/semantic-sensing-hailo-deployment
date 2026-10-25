#!/usr/bin/env python3
"""Evaluate a DFC-optimized sender model against the PyTorch FP32 reference.

WHY NOT "compare against SDK_NATIVE"
------------------------------------
The backbone's model script installs the on-chip `normalization(...)` command, so:

  * SDK_NATIVE        = the RAW parsed graph.  It does NOT apply the normalization, so
                        feeding it 0..255 images gives a meaningless result.
  * SDK_FP_OPTIMIZED  = normalization applied  -> this is the correct FP32 reference.
  * SDK_QUANTIZED     = normalization + quantization.

Comparing QUANTIZED against NATIVE therefore measures "normalization vs no
normalization", not the quantization loss.  The correct decomposition is:

    quant loss      = cos(QUANTIZED, FP_OPTIMIZED)
    end-to-end loss = cos(QUANTIZED, PyTorch FP32 reference)   <- what actually matters

This script reports both, against the .npy reference dumped by
`deploy/dump_sender_reference.py`.

Usage (hailo-dfc env, from a scratch dir):
  python scripts/eval_sender_vs_reference.py --har sender_backbone_optimized.har \
      --kind backbone --n 64 --out /root/deploy_artifacts/data/ref/backbone_vs_ref.json
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


def _pair_stats(o, r):
    o = np.asarray(o, np.float64)
    r = np.asarray(r, np.float64)
    n = min(len(o), len(r))
    o, r = o[:n], r[:n]
    cos_list = [_cos(o[i], r[i]) for i in range(n)]
    rel = float(np.linalg.norm((o - r).reshape(n, -1), axis=1).mean() /
                (np.linalg.norm(r.reshape(n, -1), axis=1).mean() + 1e-12))
    return {
        "min_cos": float(min(cos_list)),
        "mean_cos": float(np.mean(cos_list)),
        "mse": float(np.mean((o - r) ** 2)),
        "rel_l2": rel,
        "max_abs": float(np.abs(o - r).max()),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--har", required=True)
    ap.add_argument("--kind", choices=["backbone", "bottleneck"], default="backbone")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--ref-dir", default="/root/deploy_artifacts/data/ref")
    ap.add_argument("--out-dir", default="", help="also dump the QUANTIZED outputs here")
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
        imgs = np.load(os.path.join(a.ref_dir, "ref_images.npy"))[: a.n].astype(np.float32)
        feed = {input_names[0]: np.ascontiguousarray(imgs)}   # raw 0..255 NHWC
        ref_names = ["x_h", "x_l"]
    else:
        xh = np.load(os.path.join(a.ref_dir, "ref_x_h.npy"))[: a.n].astype(np.float32)
        xl = np.load(os.path.join(a.ref_dir, "ref_x_l.npy"))[: a.n].astype(np.float32)
        # Hailo network inputs are NHWC (channels last)
        feed = {input_names[0]: np.ascontiguousarray(np.transpose(xh, (0, 2, 3, 1))),
                input_names[1]: np.ascontiguousarray(np.transpose(xl, (0, 2, 3, 1)))}
        ref_names = ["y", "z_hat", "means", "scales"]

    ref = {}
    for k in ref_names:
        p = os.path.join(a.ref_dir, f"ref_{k}.npy")
        if os.path.isfile(p):
            ref[k] = np.load(p)[: a.n].astype(np.float64)

    outs = {}
    for ctx_name in ("SDK_FP_OPTIMIZED", "SDK_QUANTIZED"):
        try:
            with runner.infer_context(getattr(InferenceContext, ctx_name)) as ctx:
                t0 = time.time()
                raw = runner.infer(ctx, feed, data_count=a.n, batch_size=1)
                dt = time.time() - t0
            arrs = [np.asarray(x) for x in raw]
            # Hailo returns NHWC for 4-D conv outputs; the reference is NCHW
            arrs = [np.transpose(x, (0, 3, 1, 2)) if x.ndim == 4 else x for x in arrs]
            outs[ctx_name] = arrs
            print(f"[{ctx_name}] {dt:.1f}s shapes={[list(x.shape) for x in arrs]}", flush=True)
        except Exception as e:
            print(f"[{ctx_name}] FAILED: {type(e).__name__}: {e}", flush=True)

    summary = {"har": a.har, "kind": a.kind, "n": a.n, "quant_vs_fp_optimized": {},
               "quant_vs_pytorch_ref": {}}
    q = outs.get("SDK_QUANTIZED")
    f = outs.get("SDK_FP_OPTIMIZED")

    print("\n===== 1) QUANTIZATION LOSS  (QUANTIZED vs FP_OPTIMIZED, same DFC graph) =====")
    if q is not None and f is not None:
        for i, name in enumerate(ref_names):
            if i >= len(q):
                break
            s = _pair_stats(q[i], f[i])
            summary["quant_vs_fp_optimized"][name] = s
            print(f"  {name:8s} min_cos={s['min_cos']:.6f} mean_cos={s['mean_cos']:.6f} "
                  f"mse={s['mse']:.4e} rel_l2={s['rel_l2']:.4e}")

    print("\n===== 2) END-TO-END vs PyTorch FP32 reference =====")
    if q is not None:
        for i, name in enumerate(ref_names):
            if i >= len(q) or name not in ref:
                continue
            s = _pair_stats(q[i], ref[name])
            summary["quant_vs_pytorch_ref"][name] = s
            print(f"  {name:8s} min_cos={s['min_cos']:.6f} mean_cos={s['mean_cos']:.6f} "
                  f"mse={s['mse']:.4e} rel_l2={s['rel_l2']:.4e} max|diff|={s['max_abs']:.4e}")

    if a.out_dir and q is not None:
        os.makedirs(a.out_dir, exist_ok=True)
        for i, name in enumerate(ref_names):
            if i < len(q):
                np.save(os.path.join(a.out_dir, f"{name}.npy"), q[i].astype(np.float32))
        print(f"\ndumped QUANTIZED outputs -> {a.out_dir}")

    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "w") as fh:
            json.dump(summary, fh, indent=1)
        print(f"saved {a.out}")


if __name__ == "__main__":
    raise SystemExit(main())
