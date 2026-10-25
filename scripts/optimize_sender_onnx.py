#!/usr/bin/env python3
"""DFC-oriented ONNX cleanup for the SENDER graphs (multi-input aware).

`scripts/optimize_onnx_graph.py` (the teacher path) verifies the cleaned graph against
the original with ORT, but its ORT check feeds a SINGLE input, so it cannot verify the
two-input `sender_bottleneck.onnx` (x_h, x_l).  This script reuses the same passes
(static-shape folding, constant folding, identity drop, DCE) and adds a multi-input
ORT parity check.

Usage (run in the hailo-dfc env or the export venv):
  python scripts/optimize_sender_onnx.py --onnx sender_bottleneck.onnx \
      --out sender_bottleneck_clean.onnx
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import shape_inference

sys.path.insert(0, str(Path(__file__).resolve().parent))

from optimize_onnx_graph import (  # noqa: E402
    dce,
    drop_identity,
    fold_constants,
    fold_shapes,
    promote_attention_matmuls,
)


def _input_specs(model):
    """(name, shape) for every graph input, in declaration order."""
    specs = []
    for v in model.graph.input:
        dims = [d.dim_value for d in v.type.tensor_type.shape.dim]
        specs.append((v.name, dims))
    return specs


def _ort_run(model_bytes, feeds):
    import onnxruntime as ort
    s = ort.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])
    return s.run(None, feeds)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-ir", type=int, default=8, help="DFC 3.33 bundles onnx 1.16")
    ap.add_argument("--no-verify", action="store_true")
    a = ap.parse_args(argv)

    orig_bytes = onnx.load(a.onnx).SerializeToString()
    m = onnx.load(a.onnx)
    specs = _input_specs(m)
    print(f"in : nodes={len(m.graph.node)} "
          f"{dict(sorted(collections.Counter(n.op_type for n in m.graph.node).items()))}")
    print(f"     inputs={[(n, s) for n, s in specs]}")

    for fn in (fold_shapes, fold_constants, drop_identity, dce, promote_attention_matmuls):
        m, n = fn(m)
        print(f"{fn.__name__:28s} -> {n}")
    m, _ = fold_constants(m)
    m, _ = dce(m)
    m.ir_version = min(m.ir_version, a.max_ir)
    onnx.checker.check_model(m)
    ops = collections.Counter(n.op_type for n in m.graph.node)
    print(f"out: nodes={len(m.graph.node)} {dict(sorted(ops.items()))}")

    if not a.no_verify:
        shape_inference.infer_shapes(m)
        worst = 0.0
        for seed in (0, 1):
            rng = np.random.default_rng(seed)
            feeds = {n: rng.standard_normal(s).astype(np.float32) for n, s in specs}
            r = _ort_run(orig_bytes, feeds)
            o = _ort_run(m.SerializeToString(), feeds)
            for a_, b_ in zip(r, o):
                worst = max(worst, float(np.abs(a_ - b_).max()))
        print(f"ORT parity (orig vs optimized): max|diff|={worst:.3e}")
        if worst > 1e-3:
            print("ERROR: optimized graph diverges from original", file=sys.stderr)
            return 3

    onnx.save(m, a.out)
    print(f"saved {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
