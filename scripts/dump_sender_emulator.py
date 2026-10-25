#!/usr/bin/env python3
"""Dump the emulator outputs of a DFC-optimized sender model to .npy for comparison.

Runs the Hailo emulator (SDK_NATIVE / SDK_FP_OPTIMIZED / SDK_QUANTIZED) on the
reference images and writes each context's outputs as .npy files, so that
`compare_sender_routes.py` can compare them against the PyTorch FP32 reference.

Usage (hailo-dfc env, from a scratch dir):
  python scripts/dump_sender_emulator.py --har sender_backbone_optimized.har \
      --kind backbone --n 64 --out-dir /root/deploy_artifacts/data/eval/hailo
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--har", required=True)
    ap.add_argument("--kind", choices=["backbone", "bottleneck"], default="backbone")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--ref-dir", default="/root/deploy_artifacts/data/ref")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--contexts", default="SDK_NATIVE,SDK_FP_OPTIMIZED,SDK_QUANTIZED")
    ap.add_argument("--hw-arch", default="hailo8")
    a = ap.parse_args(argv)

    from hailo_sdk_client import ClientRunner
    from hailo_sdk_client.exposed_definitions import InferenceContext

    os.makedirs(a.out_dir, exist_ok=True)
    runner = ClientRunner(har=a.har, hw_arch=a.hw_arch)
    layers = runner.get_hn_dict()["layers"]
    items = layers.items() if isinstance(layers, dict) else \
        ((l.get("name"), l) for l in layers)
    input_names = [n for n, _ in items if "/input_layer" in str(n)]
    print(f"input layers: {input_names}", flush=True)

    ref = os.path.join(a.ref_dir, "ref_images.npy")
    if a.kind == "backbone":
        imgs = np.load(ref)[: a.n].astype(np.float32)          # raw 0..255 NHWC
        # the on-chip `normalization` command consumes raw 0..255 NHWC images
        feed = {input_names[0]: np.ascontiguousarray(imgs)}
    else:
        xh = np.load(os.path.join(a.ref_dir, "ref_x_h.npy"))[: a.n].astype(np.float32)
        xl = np.load(os.path.join(a.ref_dir, "ref_x_l.npy"))[: a.n].astype(np.float32)
        # Hailo network inputs are NHWC (channels last)
        feed = {input_names[0]: np.ascontiguousarray(np.transpose(xh, (0, 2, 3, 1))),
                input_names[1]: np.ascontiguousarray(np.transpose(xl, (0, 2, 3, 1)))}

    # reference tensors for the accuracy report
    if a.kind == "backbone":
        ref_names = ["x_h", "x_l"]
    else:
        ref_names = ["y", "z_hat", "means", "scales"]

    summary = {}
    for ctx_name in a.contexts.split(","):
        ctx_name = ctx_name.strip()
        try:
            with runner.infer_context(getattr(InferenceContext, ctx_name)) as ctx:
                t0 = time.time()
                out = runner.infer(ctx, feed, data_count=a.n, batch_size=1)
                dt = time.time() - t0
        except Exception as e:
            print(f"[{ctx_name}] FAILED: {type(e).__name__}: {e}", flush=True)
            summary[ctx_name] = {"ok": False, "error": str(e)[:300]}
            continue

        arrs = [np.asarray(o) for o in out]
        print(f"[{ctx_name}] {dt:.1f}s  shapes={[x.shape for x in arrs]}", flush=True)

        # Hailo returns NHWC for conv outputs; the reference is NCHW.
        conv_out = []
        for x in arrs:
            if x.ndim == 4:
                conv_out.append(np.transpose(x, (0, 3, 1, 2)))
            else:
                conv_out.append(x)
        for i, name in enumerate(ref_names):
            if i >= len(conv_out):
                break
            np.save(os.path.join(a.out_dir, f"{name}.npy"), conv_out[i].astype(np.float32))

        entry = {"ok": True, "sec": dt, "shapes": [list(x.shape) for x in arrs]}
        # accuracy vs the FP32 reference (when available for this kind)
        if a.kind == "bottleneck":
            for i, name in enumerate(ref_names):
                if i >= len(conv_out):
                    break
                p = os.path.join(a.ref_dir, f"ref_{name}.npy")
                if os.path.isfile(p):
                    r = np.load(p)[: a.n].astype(np.float64)
                    o = conv_out[i][: len(r)].astype(np.float64)
                    entry.setdefault("accuracy", {})[name] = {
                        "min_cos": float(min(_cos(o[j], r[j]) for j in range(len(o)))),
                        "mse": float(np.mean((o - r) ** 2)),
                    }
        summary[ctx_name] = entry

    print("\n===== SUMMARY =====")
    print(json.dumps(summary, indent=1))
    p = os.path.join(a.out_dir, "emulator_summary.json")
    with open(p, "w") as f:
        json.dump(summary, f, indent=1)
    print(f"saved {p}")


if __name__ == "__main__":
    raise SystemExit(main())
