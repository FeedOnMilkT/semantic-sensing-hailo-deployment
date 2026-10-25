#!/usr/bin/env python3
"""Convert an ONNX graph to fp16 for the ORT-CPU fallback route.

The hybrid route (Hailo backbone + CPU bottleneck) needs the bottleneck to run in fp16
on the CPU.  ONNXRuntime does not cast automatically, so the graph itself is converted
with `onnxruntime.transformers.float16.convert_float_to_float16`, keeping the
IO tensors in fp32 (keep_io_types=True) so the host can feed plain fp32 arrays.

Usage:
  python scripts/make_fp16_onnx.py --onnx sender_bottleneck.onnx \
      --out sender_bottleneck_fp16.onnx
"""
from __future__ import annotations

import argparse

import numpy as np
import onnx
from onnxruntime.transformers.float16 import convert_float_to_float16


def topological_sort(model):
    """Reorder graph nodes topologically.

    `convert_float_to_float16` inserts Cast nodes at the graph inputs but can leave them
    positioned after their consumers, which makes `onnx.checker` reject the graph
    ("Nodes in a graph must be topologically sorted").  ORT itself tolerates it, but the
    Hailo-side tooling and the checker do not, so the order is fixed here.
    """
    g = model.graph
    producer = {o: n for n in g.node for o in n.output}
    consumers = {}
    for n in g.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)

    state, order = {}, []

    def visit(node):
        st = state.get(id(node))
        if st == 2:
            return
        if st == 1:
            raise RuntimeError(f"cycle in graph at {node.name}")
        state[id(node)] = 1
        for inp in node.input:
            p = producer.get(inp)
            if p is not None and p is not node:
                visit(p)
        state[id(node)] = 2
        order.append(node)

    for n in g.node:
        visit(n)
    del g.node[:]
    g.node.extend(order)
    return model


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--opblock-list", default="",
                    help="comma-separated op types to keep in fp32")
    ap.add_argument("--min-positive-val", type=float, default=5.96e-08)
    ap.add_argument("--max-finite-val", type=float, default=65504.0)
    ap.add_argument("--no-verify", action="store_true")
    a = ap.parse_args(argv)

    model = onnx.load(a.onnx)
    block = [s.strip() for s in a.opblock_list.split(",") if s.strip()]
    m16 = convert_float_to_float16(
        model,
        keep_io_types=True,
        op_block_list=block or None,
        min_positive_val=a.min_positive_val,
        max_finite_val=a.max_finite_val,
    )
    m16 = topological_sort(m16)
    m16.ir_version = min(m16.ir_version, 8)
    onnx.checker.check_model(m16)
    onnx.save(m16, a.out)
    print(f"saved {a.out}")

    if not a.no_verify:
        import onnxruntime as ort
        s32 = ort.InferenceSession(a.onnx, providers=["CPUExecutionProvider"])
        s16 = ort.InferenceSession(a.out, providers=["CPUExecutionProvider"])
        names = [i.name for i in s32.get_inputs()]
        rng = np.random.default_rng(0)
        worst_cos, worst_rel = 1.0, 0.0
        for _ in range(3):
            feeds = {}
            for i in s32.get_inputs():
                shp = [d if isinstance(d, int) else 1 for d in i.shape]
                feeds[i.name] = rng.standard_normal(shp).astype(np.float32)
            r32 = s32.run(None, feeds)
            r16 = s16.run(None, feeds)
            for a_, b_ in zip(r32, r16):
                a64 = np.asarray(a_, np.float64).reshape(-1)
                b64 = np.asarray(b_, np.float64).reshape(-1)
                worst_cos = min(worst_cos, float(a64 @ b64 /
                                                 (np.linalg.norm(a64) * np.linalg.norm(b64) + 1e-12)))
                worst_rel = max(worst_rel, float(np.linalg.norm(a64 - b64) /
                                                 (np.linalg.norm(a64) + 1e-12)))
        print(f"fp16 vs fp32 (random data): min cos={worst_cos:.8f} max relL2={worst_rel:.3e}")


if __name__ == "__main__":
    raise SystemExit(main())
