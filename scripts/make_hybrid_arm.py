#!/usr/bin/env python3
"""Build the HYBRID arm's tensors: Hailo-8 backbone (quantized) -> ORT fp16 bottleneck.

The hybrid route runs the heavy conv backbone on the Hailo-8 and the bottleneck on the
CPU with ORT in fp16.  This script takes the Hailo backbone's QUANTIZED boundary
features (x_h, x_l) and pushes them through the ORT fp16 bottleneck, dumping the four
bottleneck outputs so the route comparison and the rate/distortion measurement can use
them exactly like the other arms.

It also verifies the ROUNDING: ORT implements `Floor` correctly, so the hybrid route
does NOT suffer the DFC emulator's rounding bug -- the transmitted symbols must be
integers.

Usage (ORT venv):
  python scripts/make_hybrid_arm.py \
      --backbone-dir /root/deploy_artifacts/data/eval/hailo_backbone \
      --onnx-fp16 /root/deploy_artifacts/onnx/sender_bottleneck_fp16.onnx \
      --median-npy /root/deploy_artifacts/onnx/sender_z_median.npy \
      --out-dir /root/deploy_artifacts/data/eval/hybrid
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import onnxruntime as ort


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--backbone-dir", default="/root/deploy_artifacts/data/eval/hailo_backbone")
    ap.add_argument("--onnx-fp16", default="/root/deploy_artifacts/onnx/sender_bottleneck_fp16.onnx")
    ap.add_argument("--median-npy", default="/root/deploy_artifacts/onnx/sender_z_median.npy")
    ap.add_argument("--out-dir", default="/root/deploy_artifacts/data/eval/hybrid")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args(argv)

    os.makedirs(a.out_dir, exist_ok=True)
    xh = np.load(os.path.join(a.backbone_dir, "x_h.npy"))[: a.n].astype(np.float32)
    xl = np.load(os.path.join(a.backbone_dir, "x_l.npy"))[: a.n].astype(np.float32)
    median = np.load(a.median_npy).reshape(1, -1, 1, 1).astype(np.float32)
    print(f"backbone features: x_h {xh.shape} x_l {xl.shape}")

    so = ort.SessionOptions()
    so.intra_op_num_threads = a.threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3
    sess = ort.InferenceSession(a.onnx_fp16, sess_options=so,
                                providers=["CPUExecutionProvider"])
    in_names = [i.name for i in sess.get_inputs()]
    out_names = [o.name for o in sess.get_outputs()]
    print(f"ort fp16 inputs={in_names} outputs={out_names}")

    for i in range(3):  # warm-up
        sess.run(None, {in_names[0]: xh[i % a.n][None], in_names[1]: xl[i % a.n][None]})

    acc = {n: [] for n in out_names}
    times = []
    for i in range(a.n):
        t0 = time.perf_counter_ns()
        outs = sess.run(None, {in_names[0]: xh[i][None], in_names[1]: xl[i][None]})
        times.append((time.perf_counter_ns() - t0) / 1e6)
        for name, o in zip(out_names, outs):
            acc[name].append(o[0])

    for name in out_names:
        arr = np.stack(acc[name]).astype(np.float32)
        np.save(os.path.join(a.out_dir, f"{name}.npy"), arr)
        print(f"  saved {name}.npy {arr.shape}")

    # copy the boundary features so the route comparison sees a complete arm
    np.save(os.path.join(a.out_dir, "x_h.npy"), xh)
    np.save(os.path.join(a.out_dir, "x_l.npy"), xl)

    # ---- rounding check: the transmitted symbols are (z_hat - median) ---------------
    z_hat = np.stack(acc["z_hat"])
    symbols = z_hat - median
    frac = np.abs(symbols - np.round(symbols))
    n_int = int((frac < 1e-4).sum())
    print(f"\nsymbols integer-valued: {n_int}/{symbols.size} "
          f"({100.0 * n_int / symbols.size:.2f}%)  max|x-round(x)|={frac.max():.3e}")

    t = np.array(times)
    print(f"ORT fp16 bottleneck latency: mean={t.mean():.2f} ms  median={np.median(t):.2f} ms  "
          f"p90={np.percentile(t, 90):.2f} ms  -> {1000.0 / t.mean():.1f} img/s "
          f"({a.threads} threads)")


if __name__ == "__main__":
    raise SystemExit(main())
