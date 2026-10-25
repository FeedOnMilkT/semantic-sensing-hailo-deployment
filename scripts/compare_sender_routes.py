#!/usr/bin/env python3
"""End-to-end comparison of the two sender deployment routes on the same images.

Route A ("hailo"):   Hailo-8 backbone (quantized)  ->  Hailo-8 bottleneck (quantized)
Route B ("hybrid"):  Hailo-8 backbone (quantized)  ->  ORT fp16 bottleneck on the CPU
Reference:           PyTorch FP32 sender.

The comparison is done on the bottleneck's OUTPUT tensors (y, z_hat, means, scales),
because those are what actually cross the channel: the latent y is entropy-coded and
decoded at the receiver, so a change in y changes the bpp and the downstream mIoU.

The script consumes pre-computed .npy dumps:
  <dir>/<arm>_x_h.npy, <arm>_x_l.npy, <arm>_y.npy, <arm>_z_hat.npy,
            <arm>_means.npy, <arm>_scales.npy
and the FP32 reference in --ref-dir.

Usage:
  python scripts/compare_sender_routes.py --ref-dir /root/deploy_artifacts/data/ref \
      --arm hailo=/root/deploy_artifacts/data/eval/hailo \
      --arm hybrid=/root/deploy_artifacts/data/eval/hybrid
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def _cos(a, b):
    a = np.asarray(a, np.float64).reshape(-1)
    b = np.asarray(b, np.float64).reshape(-1)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--ref-dir", default="/root/deploy_artifacts/data/ref")
    ap.add_argument("--arm", action="append", default=[],
                    help="name=dir (repeatable)")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)

    names = ("x_h", "x_l", "y", "z_hat", "means", "scales")
    ref = {k: np.load(os.path.join(a.ref_dir, f"ref_{k}.npy"))[: a.n].astype(np.float64)
           for k in names}

    results = {}
    for spec in a.arm:
        arm, d = spec.split("=", 1)
        entry = {}
        for k in names:
            p = os.path.join(d, f"{k}.npy")
            if not os.path.isfile(p):
                continue
            arr = np.load(p)[: a.n].astype(np.float64)
            cos_list = [_cos(arr[i], ref[k][i]) for i in range(len(arr))]
            mse = float(np.mean((arr - ref[k][: len(arr)]) ** 2))
            rel = float(np.linalg.norm((arr - ref[k][: len(arr)]).reshape(len(arr), -1),
                                       axis=1).mean() /
                        (np.linalg.norm(ref[k][: len(arr)].reshape(len(arr), -1), axis=1).mean() + 1e-12))
            entry[k] = {
                "min_cos": float(min(cos_list)),
                "mean_cos": float(np.mean(cos_list)),
                "mse": mse,
                "rel_l2": rel,
                "max_abs": float(np.abs(arr - ref[k][: len(arr)]).max()),
            }
        results[arm] = entry

    print(f"{'arm':10s} {'tensor':8s} {'min_cos':>10s} {'mean_cos':>10s} "
          f"{'mse':>12s} {'rel_l2':>10s}")
    for arm, entry in results.items():
        for k, v in entry.items():
            print(f"{arm:10s} {k:8s} {v['min_cos']:10.6f} {v['mean_cos']:10.6f} "
                  f"{v['mse']:12.4e} {v['rel_l2']:10.4e}")

    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(results, f, indent=1)
        print(f"\nsaved {a.out}")


if __name__ == "__main__":
    raise SystemExit(main())
