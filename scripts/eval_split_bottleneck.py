#!/usr/bin/env python3
"""Evaluate the ROUNDING-SPLIT sender bottleneck on Hailo (emulator), end to end.

The deployment is now TWO Hailo graphs with a host-side scalar rounding between them:

    HEF #1 "enc" : (x_h, x_l)      -> (y, z_pre)      [align_h + g_a + h_a]
    HOST         : z_hat = floor(z_pre + 0.5) + median
    HEF #2 "hs"  : z_hat            -> (means, scales) [h_s]

This script runs both HEFs in the DFC emulator, applies the host rounding, and compares
the resulting (y, z_hat, means, scales) against the PyTorch FP32 reference dumped by
`deploy/dump_sender_reference.py`.

It reports BOTH:
  * quantization loss vs the same graph in FP (SDK_FP_OPTIMIZED) -- isolates PTQ
  * end-to-end vs the PyTorch FP32 reference                  -- what actually matters

And, crucially, it checks the ROUNDING: `z_hat` must be integer-valued.  A non-integer
`z_hat` is the bug that inflated the bitstream 2.3x.

Usage (hailo-dfc env, from a scratch dir):
  python scripts/eval_split_bottleneck.py \
      --enc-har sender_bottleneck_enc_optimized.har \
      --hs-har  sender_bottleneck_hs_optimized.har \
      --n 64 --out-dir /root/deploy_artifacts/data/eval/hailo_split
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


def _stats(o, r):
    o = np.asarray(o, np.float64)
    r = np.asarray(r, np.float64)
    n = min(len(o), len(r))
    o, r = o[:n], r[:n]
    cos_list = [_cos(o[i], r[i]) for i in range(n)]
    return {
        "min_cos": float(min(cos_list)),
        "mean_cos": float(np.mean(cos_list)),
        "mse": float(np.mean((o - r) ** 2)),
        "rel_l2": float(np.linalg.norm((o - r).reshape(n, -1), axis=1).mean() /
                       (np.linalg.norm(r.reshape(n, -1), axis=1).mean() + 1e-12)),
        "max_abs": float(np.abs(o - r).max()),
    }


def _input_names(runner):
    layers = runner.get_hn_dict()["layers"]
    items = layers.items() if isinstance(layers, dict) else \
        ((l.get("name"), l) for l in layers)
    return [n for n, _ in items if "/input_layer" in str(n)]


def _infer(runner, ctx_name, feed, n):
    from hailo_sdk_client.exposed_definitions import InferenceContext
    with runner.infer_context(getattr(InferenceContext, ctx_name)) as ctx:
        t0 = time.time()
        raw = runner.infer(ctx, feed, data_count=n, batch_size=1)
        dt = time.time() - t0
    arrs = [np.asarray(x) for x in raw]
    # Hailo returns NHWC for 4-D conv outputs; the reference is NCHW
    arrs = [np.transpose(x, (0, 3, 1, 2)) if x.ndim == 4 else x for x in arrs]
    return arrs, dt


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--enc-har", required=True)
    ap.add_argument("--hs-har", required=True)
    ap.add_argument("--median-npy", default="/root/deploy_artifacts/onnx/sender_z_median.npy")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--ref-dir", default="/root/deploy_artifacts/data/ref")
    ap.add_argument("--feat-dir", default="",
                    help="use these x_h/x_l instead of the FP32 reference ones "
                         "(e.g. the QUANTIZED backbone output, for a true end-to-end arm)")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--hw-arch", default="hailo8")
    a = ap.parse_args(argv)

    from hailo_sdk_client import ClientRunner

    median = np.load(a.median_npy).reshape(1, -1, 1, 1).astype(np.float32)
    if a.feat_dir:
        # the QUANTIZED backbone's boundary features -> a TRUE end-to-end arm
        xh = np.load(os.path.join(a.feat_dir, "x_h.npy"))[: a.n].astype(np.float32)
        xl = np.load(os.path.join(a.feat_dir, "x_l.npy"))[: a.n].astype(np.float32)
        print(f"using backbone features from {a.feat_dir}")
    else:
        xh = np.load(os.path.join(a.ref_dir, "ref_x_h.npy"))[: a.n].astype(np.float32)
        xl = np.load(os.path.join(a.ref_dir, "ref_x_l.npy"))[: a.n].astype(np.float32)
    ref = {}
    for k in ("y", "z_hat", "means", "scales"):
        p = os.path.join(a.ref_dir, f"ref_{k}.npy")
        if os.path.isfile(p):
            ref[k] = np.load(p)[: a.n].astype(np.float64)

    r_enc = ClientRunner(har=a.enc_har, hw_arch=a.hw_arch)
    r_hs = ClientRunner(har=a.hs_har, hw_arch=a.hw_arch)
    enc_in = _input_names(r_enc)
    hs_in = _input_names(r_hs)
    print(f"enc inputs: {enc_in}   hs inputs: {hs_in}", flush=True)

    enc_feed = {enc_in[0]: np.ascontiguousarray(np.transpose(xh, (0, 2, 3, 1))),
                enc_in[1]: np.ascontiguousarray(np.transpose(xl, (0, 2, 3, 1)))}

    results = {}
    for ctx_name in ("SDK_FP_OPTIMIZED", "SDK_QUANTIZED"):
        enc_out, dt_e = _infer(r_enc, ctx_name, enc_feed, a.n)
        y, z_pre = enc_out[0], enc_out[1]
        # ---- THE HOST-SIDE ROUNDING (what replaces the broken in-graph Floor) ----
        z_hat = np.floor(z_pre + 0.5) + median
        hs_feed = {hs_in[0]: np.ascontiguousarray(np.transpose(z_hat, (0, 2, 3, 1)))}
        hs_out, dt_h = _infer(r_hs, ctx_name, hs_feed, a.n)
        means, scales = hs_out[0], hs_out[1]
        print(f"[{ctx_name}] enc {dt_e:.1f}s  hs {dt_h:.1f}s  "
              f"shapes y={list(y.shape)} z_hat={list(z_hat.shape)} "
              f"means={list(means.shape)}", flush=True)

        # ---- is the TRANSMITTED SYMBOL integer-valued?  (the bug this fixes) ----
        # CompressAI transmits `round(z - median)`; `z_hat` is that symbol PLUS the
        # median, so the symbols are (z_hat - median).  They must be integers --
        # non-integer symbols are what inflated the bitstream 2.3x.
        symbols = z_hat - median
        frac = np.abs(symbols - np.round(symbols))
        n_int = int((frac < 1e-4).sum())
        print(f"  symbols integer-valued: {n_int}/{symbols.size} "
              f"({100.0 * n_int / symbols.size:.2f}%)  "
              f"max|x-round(x)|={frac.max():.3e}", flush=True)

        results[ctx_name] = {"y": y, "z_hat": z_hat, "means": means, "scales": scales,
                             "int_frac": float(n_int / z_hat.size),
                             "max_frac": float(frac.max())}

    # ---- comparisons -------------------------------------------------------------
    print("\n===== 1) QUANTIZATION LOSS (QUANTIZED vs FP_OPTIMIZED) =====")
    q = results["SDK_QUANTIZED"]
    f = results["SDK_FP_OPTIMIZED"]
    quant_stats = {}
    for k in ("y", "z_hat", "means", "scales"):
        s = _stats(q[k], f[k])
        quant_stats[k] = s
        print(f"  {k:8s} min_cos={s['min_cos']:.6f} mean_cos={s['mean_cos']:.6f} "
              f"mse={s['mse']:.4e} rel_l2={s['rel_l2']:.4e}")

    print("\n===== 2) END-TO-END vs PyTorch FP32 reference =====")
    e2e_stats = {}
    for k in ("y", "z_hat", "means", "scales"):
        if k not in ref:
            continue
        s = _stats(q[k], ref[k])
        e2e_stats[k] = s
        print(f"  {k:8s} min_cos={s['min_cos']:.6f} mean_cos={s['mean_cos']:.6f} "
              f"mse={s['mse']:.4e} rel_l2={s['rel_l2']:.4e} max|diff|={s['max_abs']:.4e}")

    if a.out_dir:
        os.makedirs(a.out_dir, exist_ok=True)
        for k in ("y", "z_hat", "means", "scales"):
            np.save(os.path.join(a.out_dir, f"{k}.npy"), q[k].astype(np.float32))
        print(f"\ndumped QUANTIZED split outputs -> {a.out_dir}")

    summary = {"enc_har": a.enc_har, "hs_har": a.hs_har, "n": a.n,
               "z_hat_integer_fraction": q["int_frac"],
               "z_hat_max_frac_error": q["max_frac"],
               "quant_vs_fp_optimized": quant_stats,
               "quant_vs_pytorch_ref": e2e_stats}
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "w") as fh:
            json.dump(summary, fh, indent=1)
        print(f"saved {a.out}")


if __name__ == "__main__":
    raise SystemExit(main())
