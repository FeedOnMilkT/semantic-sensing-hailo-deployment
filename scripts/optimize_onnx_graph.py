#!/usr/bin/env python3
"""Hailo-DFC-oriented ONNX graph cleanup (static shape, GELU-preserving).

Why not onnxsim: its Mul-into-Conv fusion folds the trailing `Mul(0.5)` of the exported
GELU (Div(sqrt2) -> Erf -> Add(1) -> Mul(x, .) -> Mul(0.5)) into the next Conv's weights.
DFC 3.33 only recognises GELU as that full 5-node chain, so it then fails on a bare `Erf`.

Passes (in order):
  1. fold_shapes      Shape(x) -> int64 initializer, from static shape inference
  2. fold_constants   constant-only subgraphs (Gather/Unsqueeze/Cast/Concat/Slice/...) -> initializers
  3. drop_identity    remove Identity nodes
  4. dce              drop dead nodes / unused initializers
  4b. promote_attention_matmuls   2-D MatMul attention -> [1,1,N,C] form (DFC rejects 2-D)
  4c. rewrite_attn_double_norm   ReduceSum+Add(eps)+Div -> Log+grouped Softmax (opt-in)
  5. check_gelu       every Erf must still sit in the full GELU chain (exit code 2 otherwise)
Original `Constant` nodes that still feed live nodes are left untouched.

Usage (works in the hailo-dfc env and in .venv):
  python scripts/optimize_onnx_graph.py --onnx teacher_raw.onnx --out teacher.onnx
  python scripts/optimize_onnx_graph.py --onnx teacher_raw.onnx --out teacher.onnx \
      --rewrite-attn-double-norm
"""
from __future__ import annotations

import argparse
import collections
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper, shape_inference
from onnx.reference import ReferenceEvaluator

try:                                    # sibling import (script run from scripts/)
    from rewrite_attn_double_norm import rewrite_attn_double_norm
except ImportError:                     # package import (python -m scripts.optimize_onnx_graph)
    from .rewrite_attn_double_norm import rewrite_attn_double_norm

FOLDABLE = {"Shape", "Gather", "Unsqueeze", "Squeeze", "Cast", "Concat", "Identity", "Slice",
            "Add", "Sub", "Mul", "Div", "Reshape", "Transpose", "Range", "ConstantOfShape",
            "Floor", "Ceil", "Equal", "Where", "Not", "Less", "Greater", "Min", "Max"}


def _static_dims(vi):
    t = vi.type.tensor_type
    if not t.HasField("shape"):
        return None
    dims = [d.dim_value if d.HasField("dim_value") else 0 for d in t.shape.dim]
    return dims if all(d > 0 for d in dims) else None


def fold_shapes(m):
    m = shape_inference.infer_shapes(m, data_prop=True)
    g = m.graph
    known = {v.name: _static_dims(v) for v in [*g.input, *g.value_info, *g.output]}
    keep, n = [], 0
    for nd in g.node:
        dims = known.get(nd.input[0]) if nd.op_type == "Shape" and not nd.attribute else None
        if dims is not None:
            g.initializer.append(numpy_helper.from_array(np.array(dims, dtype=np.int64), nd.output[0]))
            n += 1
        else:
            keep.append(nd)
    del g.node[:]
    g.node.extend(keep)
    del g.value_info[:]
    return m, n


def _eval(m, nd, consts):
    ins = [i for i in nd.input if i]
    gin = [helper.make_tensor_value_info(i, helper.np_dtype_to_tensor_dtype(consts[i].dtype),
                                         consts[i].shape) for i in ins]
    gout = [helper.make_tensor_value_info(o, onnx.TensorProto.UNDEFINED, None) for o in nd.output]
    sub = helper.make_model(helper.make_graph([nd], "fold", gin, gout),
                            opset_imports=list(m.opset_import))
    sub.ir_version = m.ir_version
    return ReferenceEvaluator(sub).run(None, {i: consts[i] for i in ins})


def fold_constants(m, max_elems=1 << 20):
    g = m.graph
    consts = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    for nd in g.node:
        if nd.op_type == "Constant" and nd.attribute and nd.attribute[0].name == "value":
            consts[nd.output[0]] = numpy_helper.to_array(nd.attribute[0].t)
    keep, n = [], 0
    for nd in g.node:
        ins = [i for i in nd.input if i]
        if nd.op_type in FOLDABLE and ins and all(i in consts for i in ins):
            try:
                outs = [np.asarray(o) for o in _eval(m, nd, consts)]
            except Exception:
                outs = None
            if outs is not None and all(o.size <= max_elems for o in outs):
                for name, val in zip(nd.output, outs):
                    consts[name] = val
                    g.initializer.append(numpy_helper.from_array(val, name))
                n += 1
                continue
        keep.append(nd)
    del g.node[:]
    g.node.extend(keep)
    return m, n


def drop_identity(m):
    g = m.graph
    outs = {o.name for o in g.output}
    rename, keep = {}, []
    for nd in g.node:
        if nd.op_type == "Identity" and nd.output[0] not in outs:
            rename[nd.output[0]] = rename.get(nd.input[0], nd.input[0])
        else:
            keep.append(nd)
    for nd in keep:
        for k, i in enumerate(nd.input):
            if i in rename:
                nd.input[k] = rename[i]
    del g.node[:]
    g.node.extend(keep)
    return m, len(rename)


def dce(m):
    g = m.graph
    live = {o.name for o in g.output}
    keep = []
    for nd in reversed(g.node):
        if any(o in live for o in nd.output):
            keep.append(nd)
            live.update(i for i in nd.input if i)
    keep.reverse()
    removed = len(g.node) - len(keep)
    inits = [i for i in g.initializer if i.name in live]
    del g.node[:]
    g.node.extend(keep)
    del g.initializer[:]
    g.initializer.extend(inits)
    return m, removed


def _const_array(m, name):
    for i in m.graph.initializer:
        if i.name == name:
            return numpy_helper.to_array(i)
    for n in m.graph.node:
        if n.op_type == "Constant" and n.output[0] == name and n.attribute:
            return numpy_helper.to_array(n.attribute[0].t)
    return None


def promote_attention_matmuls(m):
    """2-D MatMul attention -> 4-D [1,1,rows,cols] MatMul.

    DFC 3.33 rejects `[N,C] @ [C,HW]` (it lowers it to a 4-D matmul with the wrong feature
    layout: "Unexpected input shapes at matmul layer ..."), but parses the same product as
    `[1,1,N,C] @ [1,1,C,HW]`.  Mathematically identical; only Reshape targets and the
    Softmax axis change:
        Reshape -> [d0,d1]      becomes  [1,1,d0,d1]
        Softmax axis a>=0       becomes  a+2   (negative axes unchanged)
    The 4-D-ness is propagated through scalar Mul / Softmax until the next MatMul or Reshape.
    """
    g = m.graph
    inf = shape_inference.infer_shapes(m)
    rank = {v.name: len(v.type.tensor_type.shape.dim)
            for v in [*inf.graph.value_info, *inf.graph.input, *inf.graph.output]
            if v.type.tensor_type.HasField("shape")}
    prod = {o: n for n in g.node for o in n.output}
    cons = collections.defaultdict(list)
    for n in g.node:
        for i in n.input:
            cons[i].append(n)

    promoted, done = set(), 0

    def promote_reshape(nd):
        shp = _const_array(m, nd.input[1])
        new = f"{nd.name}_shape4d"
        g.initializer.append(numpy_helper.from_array(np.concatenate([[1, 1], shp]).astype(np.int64), new))
        nd.input[1] = new
        promoted.add(nd.output[0])

    def reshape_ok(t, user):
        p = prod.get(t)
        return (p is not None and p.op_type == "Reshape" and len(cons[t]) == 1 and cons[t][0] is user
                and (c := _const_array(m, p.input[1])) is not None and c.size == 2 and rank.get(t) == 2)

    def propagate(t):
        for c in cons[t]:
            if c.op_type == "Mul" and all(rank.get(i, 0) in (0, 2) or i in promoted for i in c.input):
                promoted.add(c.output[0])
                propagate(c.output[0])
            elif c.op_type == "Softmax":
                for a in c.attribute:
                    if a.name == "axis" and a.i >= 0:
                        a.i += 2
                promoted.add(c.output[0])
                propagate(c.output[0])
            elif c.op_type in ("MatMul", "Reshape"):
                continue
            else:
                raise RuntimeError(f"promote_attention_matmuls: unsupported consumer {c.op_type} {c.name}")

    for mm in [n for n in g.node if n.op_type == "MatMul"]:
        if rank.get(mm.output[0]) != 2 or len(mm.input) != 2:
            continue
        if not all(i in promoted or reshape_ok(i, mm) for i in mm.input):
            continue
        for i in mm.input:
            if i not in promoted:
                promote_reshape(prod[i])
        promoted.add(mm.output[0])
        propagate(mm.output[0])
        done += 1
    del g.value_info[:]
    return m, done


def check_gelu(m):
    """Return Erf node names that are NOT inside Div -> Erf -> Add -> Mul -> Mul."""
    prod = {o: n for n in m.graph.node for o in n.output}
    cons = collections.defaultdict(list)
    for n in m.graph.node:
        for i in n.input:
            cons[i].append(n)

    def only(n, op):
        c = cons[n.output[0]] if n is not None else []
        return c[0] if len(c) == 1 and c[0].op_type == op else None

    bad = []
    for e in (n for n in m.graph.node if n.op_type == "Erf"):
        d = prod.get(e.input[0])
        ok = d is not None and d.op_type == "Div"
        ok = ok and only(only(only(e, "Add"), "Mul"), "Mul") is not None
        if not ok:
            bad.append(e.name)
    return bad


def _ort_run(model, x):
    import onnxruntime as ort
    s = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return s.run(None, {s.get_inputs()[0].name: x})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-ir", type=int, default=8, help="DFC 3.33 bundles onnx 1.16")
    ap.add_argument("--rewrite-attn-double-norm", action="store_true",
                    help="rewrite the RTFormer attn_l double-norm (ReduceSum+Add+Div) "
                         "into Log + grouped Softmax so the DFC can quantize it")
    ap.add_argument("--no-verify", action="store_true")
    a = ap.parse_args(argv)

    orig = onnx.load(a.onnx)
    m = onnx.load(a.onnx)
    print(f"in : nodes={len(m.graph.node)} {dict(sorted(collections.Counter(n.op_type for n in m.graph.node).items()))}")
    for fn in (fold_shapes, fold_constants, drop_identity, dce, promote_attention_matmuls):
        m, n = fn(m)
        print(f"{fn.__name__:15s} -> {n}")
    if a.rewrite_attn_double_norm:
        m, n = rewrite_attn_double_norm(m)
        print(f"{'attn_double_norm':15s} -> {n}")
    # one more round: folding can expose new Identity / dead producers
    m, _ = fold_constants(m)
    m, _ = dce(m)
    m.ir_version = min(m.ir_version, a.max_ir)
    onnx.checker.check_model(m)
    ops = collections.Counter(n.op_type for n in m.graph.node)
    print(f"out: nodes={len(m.graph.node)} {dict(sorted(ops.items()))}")

    bad = check_gelu(m)
    if bad:
        print(f"ERROR: Erf outside a full GELU chain (DFC cannot parse): {bad}", file=sys.stderr)
        return 2
    print(f"GELU chains intact: {ops.get('Erf', 0)}")

    if not a.no_verify:
        shape = [d.dim_value for d in m.graph.input[0].type.tensor_type.shape.dim]
        worst = 0.0
        for seed in (0, 1):
            x = np.random.default_rng(seed).standard_normal(shape).astype(np.float32)
            for r, o in zip(_ort_run(orig, x), _ort_run(m, x)):
                worst = max(worst, float(np.abs(r - o).max()))
        print(f"ORT parity (orig vs optimized): max|diff|={worst:.3e}")
        if worst > 1e-3:
            print("ERROR: optimized graph diverges from original", file=sys.stderr)
            return 3
    onnx.save(m, a.out)
    print(f"saved {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
