#!/usr/bin/env python3
"""Rewrite the teacher's wide ``attn_h`` blocks so DFC 3.33 / Hailo-8 can map them.

The original block (layer4/attn_h, layer5/attn_h) is, per the ONNX:

    K   = Reshape(BN(x))                [1,1,128,4096]
    Q   = Reshape(Reshape(Transpose(a))) [1,1,144,128]   a = Slice(kv)[1,128,12,12]
    V   = Reshape(Reshape(b))           [1,1,128,144]    b = Slice(kv)[1,128,12,12]
    S   = (Q @ K) * s                   [1,1,144,4096]
    P   = Softmax(S, axis=2)            softmax over the 144 axis
    O   = V @ P                         [1,1,128,4096]
    out = Reshape(O)                    [1,128,64,64]

Since the softmax axis is 144 and 4096 is a pure batch axis, every rewrite below
that only re-orders / chunks the 4096 axis is **mathematically exact**.

Modes
-----
``orig``  -- Q@K as above, optionally split into ``chunks`` column groups of 4096.
``T``     -- transposed form: S^T = K^T @ Q^T [4096,144], softmax on the last
             (features) axis, O^T = P^T @ V^T [4096,128].  Optional row chunking.

``softmax``: ``native`` (ONNX Softmax) or ``manual`` (ReduceMax/Sub/Exp/ReduceSum/Div,
only legal on the features axis => only for ``T``).

The functions locate attention blocks *structurally* (a Softmax whose input is
[1,1,144,4096]), never by hard-coded node index.
"""
from __future__ import annotations

import sys
from collections import defaultdict

import numpy as np
import onnx
from onnx import helper, numpy_helper


def _shapes(model):
    m = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    vs = {}
    for v in list(m.graph.value_info) + list(m.graph.input) + list(m.graph.output):
        vs[v.name] = [d.dim_value if d.dim_value > 0 else -1 for d in v.type.tensor_type.shape.dim]
    for i in m.graph.initializer:
        vs[i.name] = list(i.dims)
    return vs


def find_blocks(model, q=144, k=4096):
    """Return one dict per attention block found, by structure."""
    g = model.graph
    prod = {o: n for n in g.node for o in n.output}
    shapes = _shapes(model)
    blocks = []
    for sm in g.node:
        if sm.op_type != "Softmax":
            continue
        if shapes.get(sm.input[0]) != [1, 1, q, k]:
            continue
        mul = prod[sm.input[0]]
        assert mul.op_type == "Mul", mul.op_type
        mm1 = prod[mul.input[0]]
        assert mm1.op_type == "MatMul"
        cons = [n for n in g.node if sm.output[0] in n.input]
        assert len(cons) == 1 and cons[0].op_type == "MatMul"
        mm2 = cons[0]
        rs_out = [n for n in g.node if mm2.output[0] in n.input]
        assert len(rs_out) == 1 and rs_out[0].op_type == "Reshape"
        rs3 = rs_out[0]
        qt, kt = mm1.input[0], mm1.input[1]
        vt = mm2.input[0]
        # K: Reshape(x)
        kr = prod[kt]
        assert kr.op_type == "Reshape" and shapes[kr.input[0]] == [1, 128, 64, 64], shapes.get(kr.input[0])
        # Q: walk back through Reshape/Transpose to a Slice output
        def back_to_slice(t):
            while prod[t].op_type in ("Reshape", "Transpose"):
                t = prod[t].input[0]
            assert prod[t].op_type == "Slice", prod[t].op_type
            return t
        a, b = back_to_slice(qt), back_to_slice(vt)
        scale_nm = mul.input[1]
        blocks.append(dict(softmax=sm, mul=mul, mm1=mm1, mm2=mm2, rs3=rs3, kr=kr,
                           x=kr.input[0], a=a, b=b, scale=scale_nm, qt=qt, vt=vt,
                           out=rs3.output[0], tag="_".join(sm.name.strip("/").split("/")[-3:-1])))
    return blocks


class _B:
    def __init__(self, tag):
        self.tag, self.nodes, self.inits, self.i = tag, [], [], 0

    def uid(self, base):
        self.i += 1
        return f"{self.tag}__{base}_{self.i}"

    def const(self, arr, base="c"):
        nm = self.uid(base)
        self.inits.append(numpy_helper.from_array(np.asarray(arr), nm))
        return nm

    def op(self, t, ins, base=None, nout=1, **attrs):
        base = base or t.lower()
        outs = [self.uid(base + "_o") for _ in range(nout)]
        self.nodes.append(helper.make_node(t, list(ins), outs, name=self.uid(base), **attrs))
        return outs[0] if nout == 1 else outs

    def reshape(self, x, shape, base="rs"):
        return self.op("Reshape", [x, self.const(np.array(shape, np.int64), "shp")], base=base)

    def slice(self, x, s, e, ax):
        return self.op("Slice", [x, self.const(np.array([s], np.int64), "st"),
                                 self.const(np.array([e], np.int64), "en"),
                                 self.const(np.array([ax], np.int64), "ax")], base="sl")


def _softmax(b, s, axis, kind):
    if kind == "native":
        return b.op("Softmax", [s], axis=axis, base="sm")
    assert kind == "manual"
    mx = b.op("ReduceMax", [s], axes=[axis], keepdims=1, base="rmax")
    e = b.op("Exp", [b.op("Sub", [s, mx], base="sub")], base="exp")
    z = b.op("ReduceSum", [e, b.const(np.array([axis], np.int64), "rax")], keepdims=1, base="rsum")
    return b.op("Div", [e, z], base="div")


def build_replacement(bl, scale_val, mode, chunks, softmax, final_name):
    b = _B(bl["tag"] if isinstance(bl["tag"], str) else "attn")
    sc = b.const(np.array(scale_val, np.float32), "scale")
    if mode == "orig":
        q = bl["qt"]
        v = bl["vt"]
        kt = bl["kr"].output[0]
        w = 4096 // chunks
        outs = []
        for c in range(chunks):
            kc = kt if chunks == 1 else b.slice(kt, c * w, (c + 1) * w, 3)
            s = b.op("MatMul", [q, kc], base="qk")
            s = b.op("Mul", [s, sc], base="mul")
            p = _softmax(b, s, 2, softmax) if softmax == "native" else None
            outs.append(b.op("MatMul", [v, p], base="av"))
        o = outs[0] if chunks == 1 else b.op("Concat", outs, axis=3, base="cat")
        b.op("Reshape", [o, b.const(np.array([1, 128, 64, 64], np.int64), "shp")], base="rs_out")
        b.nodes[-1].output[0] = final_name
        return b
    assert mode == "T"
    x, a, bsl = bl["x"], bl["a"], bl["b"]
    xt = b.reshape(b.op("Transpose", [x], perm=[0, 2, 3, 1], base="tr_x"), [1, 1, 4096, 128], "xt")
    qT = b.reshape(a, [1, 1, 128, 144], "qT")
    vT = b.reshape(b.op("Transpose", [bsl], perm=[0, 2, 3, 1], base="tr_v"), [1, 1, 144, 128], "vT")
    w = 4096 // chunks
    outs = []
    for c in range(chunks):
        xc = xt if chunks == 1 else b.slice(xt, c * w, (c + 1) * w, 2)
        s = b.op("MatMul", [xc, qT], base="qkT")
        s = b.op("Mul", [s, sc], base="mul")
        p = _softmax(b, s, 3, softmax)
        outs.append(b.op("MatMul", [p, vT], base="avT"))
    o = outs[0] if chunks == 1 else b.op("Concat", outs, axis=2, base="cat")
    o = b.reshape(o, [1, 64, 64, 128], "o_rs")
    b.op("Transpose", [o], perm=[0, 3, 1, 2], base="tr_o")
    b.nodes[-1].output[0] = final_name
    return b


def rewrite_model(model, mode="T", chunks=1, softmax="native", only=None):
    """Rewrite every attention block found (or those whose tag is in ``only``)."""
    model = onnx.ModelProto.FromString(model.SerializeToString())
    blocks = find_blocks(model)
    assert blocks, "no attention block found"
    g = model.graph
    consts = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    for n in g.node:
        if n.op_type == "Constant":
            consts[n.output[0]] = numpy_helper.to_array(n.attribute[0].t)
    repl = {}
    drop = set()
    for bl in blocks:
        if only and bl["tag"] not in only:
            continue
        scale = float(np.asarray(consts[bl["scale"]]).reshape(-1)[0])
        built = build_replacement(bl, scale, mode, chunks, softmax, bl["out"])
        # the nodes are inserted where the old output Reshape was (all inputs precede it)
        repl[bl["rs3"].name] = built
        for n in (bl["softmax"], bl["mul"], bl["mm1"], bl["mm2"]):
            drop.add(n.name)
        if mode == "T":
            drop.add(bl["kr"].name)
        g.initializer.extend(built.inits)
    new_nodes = []
    for n in g.node:
        if n.name in repl:
            new_nodes.extend(repl[n.name].nodes)
        elif n.name in drop:
            continue
        else:
            new_nodes.append(n)
    del g.node[:]
    g.node.extend(new_nodes)
    _dce(model)
    return model


def _dce(model):
    """Remove nodes whose outputs are never used (iterate to fixpoint)."""
    g = model.graph
    while True:
        used = {t for n in g.node for t in n.input} | {o.name for o in g.output}
        keep = [n for n in g.node if any(o in used for o in n.output)]
        if len(keep) == len(g.node):
            break
        del g.node[:]
        g.node.extend(keep)
    used = {t for n in g.node for t in n.input}
    keep_i = [i for i in g.initializer if i.name in used or i.name in {x.name for x in g.input}]
    del g.initializer[:]
    g.initializer.extend(keep_i)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="T")
    ap.add_argument("--chunks", type=int, default=1)
    ap.add_argument("--softmax", default="native")
    a = ap.parse_args()
    m = onnx.load(a.onnx)
    for bl in find_blocks(m):
        print("found block", bl["tag"], bl["out"])
    onnx.save(rewrite_model(m, a.mode, a.chunks, a.softmax), a.out)
    print("saved", a.out)
