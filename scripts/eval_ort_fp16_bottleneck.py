#!/usr/bin/env python3
"""Evaluate the ORT-CPU bottleneck route: fp32 vs fp16, accuracy AND latency.

This is the fallback for the hybrid deployment (Hailo backbone + CPU bottleneck).  It
measures, on real validation images:

  * accuracy: cosine similarity / MSE / relative-L2 of the four bottleneck outputs
    (y, z_hat, means, scales) against the PyTorch FP32 reference dumped by
    `deploy/dump_sender_reference.py`.
  * latency: per-image wall-clock of the ORT session (threads configurable), plus the
    throughput.

The script has NO torch dependency (it runs in the plain ORT venv): the reference comes
from the .npy files.

Usage:
  python scripts/eval_ort_fp16_bottleneck.py \
      --onnx-fp32 /root/deploy_artifacts/onnx/sender_bottleneck_clean.onnx \
      --onnx-fp16 /root/deploy_artifacts/onnx/sender_bottleneck_fp16.onnx \
      --ref-dir /root/deploy_artifacts/data/ref --n 64 --threads 8
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import onnxruntime as ort


def _cos(a, b):
    a = np.asarray(a, np.float64).reshape(-1)
    b = np.asarray(b, np.float64).reshape(-1)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _stats(out, ref):
    return {
        "cos": _cos(out, ref),
        "mse": float(np.mean((np.asarray(out, np.float64) - np.asarray(ref, np.float64)) ** 2)),
        "rel_l2": float(np.linalg.norm((np.asarray(out, np.float64).reshape(-1) -
                                        np.asarray(ref, np.float64).reshape(-1))) /
                       (np.linalg.norm(np.asarray(ref, np.float64).reshape(-1)) + 1e-12)),
        "max_abs": float(np.abs(np.asarray(out, np.float64) -
                                np.asarray(ref, np.float64)).max()),
    }


def _make_session(path, threads):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3
    return ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--onnx-fp32", required=True)
    ap.add_argument("--onnx-fp16", default="")
    ap.add_argument("--ref-dir", default="/root/deploy_artifacts/data/ref")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)

    xh = np.load(os.path.join(a.ref_dir, "ref_x_h.npy"))[: a.n].astype(np.float32)
    xl = np.load(os.path.join(a.ref_dir, "ref_x_l.npy"))[: a.n].astype(np.float32)
    refs = {
        "y": np.load(os.path.join(a.ref_dir, "ref_y.npy"))[: a.n].astype(np.float32),
        "z_hat": np.load(os.path.join(a.ref_dir, "ref_z_hat.npy"))[: a.n].astype(np.float32),
        "means": np.load(os.path.join(a.ref_dir, "ref_means.npy"))[: a.n].astype(np.float32),
        "scales": np.load(os.path.join(a.ref_dir, "ref_scales.npy"))[: a.n].astype(np.float32),
    }
    print(f"ref images: x_h {xh.shape} x_l {xl.shape}")

    arms = {"ort_fp32": a.onnx_fp32}
    if a.onnx_fp16:
        arms["ort_fp16"] = a.onnx_fp16

    results = {}
    for arm, path in arms.items():
        sess = _make_session(path, a.threads)
        in_names = [i.name for i in sess.get_inputs()]
        out_names = [o.name for o in sess.get_outputs()]
        print(f"\n[{arm}] inputs={in_names} outputs={out_names}")

        # warm-up
        for i in range(a.warmup):
            sess.run(None, {in_names[0]: xh[i % a.n][None], in_names[1]: xl[i % a.n][None]})

        per_out = {n: {"cos": 1.0, "mse": 0.0, "rel_l2": 0.0, "max_abs": 0.0} for n in out_names}
        worst = {n: {"cos": 1.0, "mse": 0.0, "rel_l2": 0.0, "max_abs": 0.0} for n in out_names}
        times = []
        for i in range(a.n):
            # the exported graph has a STATIC batch dimension of 1
            feed = {in_names[0]: xh[i][None], in_names[1]: xl[i][None]}
            t0 = time.perf_counter_ns()
            outs = sess.run(None, feed)
            times.append((time.perf_counter_ns() - t0) / 1e6)
            for name, o in zip(out_names, outs):
                s = _stats(o, refs[name][i])
                per_out[name]["cos"] = min(per_out[name]["cos"], s["cos"])
                worst[name]["mse"] = max(worst[name]["mse"], s["mse"])
                worst[name]["rel_l2"] = max(worst[name]["rel_l2"], s["rel_l2"])
                worst[name]["max_abs"] = max(worst[name]["max_abs"], s["max_abs"])
                worst[name]["cos"] = min(worst[name]["cos"], s["cos"])

        arr = np.array(times)
        entry = {
            "onnx": path,
            "threads": a.threads,
            "latency_ms": {
                "mean": float(arr.mean()), "median": float(np.median(arr)),
                "p90": float(np.percentile(arr, 90)), "min": float(arr.min()),
                "max": float(arr.max()),
            },
            "fps": float(1000.0 / arr.mean()),
            "accuracy": {n: worst[n] for n in out_names},
        }
        results[arm] = entry
        print(f"  latency: mean={arr.mean():.1f} ms  median={np.median(arr):.1f} ms  "
              f"p90={np.percentile(arr, 90):.1f} ms  -> {1000.0 / arr.mean():.1f} img/s")
        for n in out_names:
            s = worst[n]
            print(f"  {n:8s} min cos={s['cos']:.6f}  max mse={s['mse']:.4e}  "
                  f"max relL2={s['rel_l2']:.4e}  max|diff|={s['max_abs']:.4e}")

    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(results, f, indent=1)
        print(f"\nsaved {a.out}")


if __name__ == "__main__":
    raise SystemExit(main())
