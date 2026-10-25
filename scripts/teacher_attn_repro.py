#!/usr/bin/env python3
"""Minimal reproduction graphs for the teacher ``attn_h`` block (DFC 3.33, Hailo-8).

Exact semantics of the original block (read from the ONNX, NOT assumed):

    x  [1,128,64,64] --Reshape--> Kt [1,1,128,4096]
    kv [1,256,12,12] --Slice--> a,b [1,128,12,12]
       a --Transpose(0,2,3,1)--Reshape--> Q [1,1,144,128]
       b --Reshape--> V [1,1,128,144]
    S   = Q @ Kt * 0.0833         [1,1,144,4096]
    P   = Softmax(S, axis=2)      softmax over the **144** axis (NOT over 4096)
    O   = V @ P                   [1,1,128,4096]
    out = Reshape(O) [1,128,64,64]

Important consequence: the 4096 axis is a pure *batch* axis for the softmax and
for the second MatMul.  Splitting 4096 into column chunks is therefore
**mathematically exact** (no running max / logsumexp merge needed).

Variants (``--variant``):
    pure    : inputs q,kt,v directly (only MatMul->Mul->Softmax->MatMul)
    ctx     : full context (reshapes/transposes/slices as in the teacher)
    ctx_nosm: ctx but Softmax replaced by Identity (reproduces the N_sm_id idea)
    ctx_1mm : ctx with only the first MatMul (+Mul+Softmax), output S-softmax
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import onnx
from onnx import TensorProto as TP, helper, numpy_helper

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SCALE = 1.0 / 12.0  # 0.0833 in the teacher (1/sqrt(144))


class G:
    """Tiny graph builder: unique names, nodes appended in topological order."""

    def __init__(self, name):
        self.name = name
        self.nodes, self.inits, self.inputs, self.outputs = [], [], [], []
        self._n = 0

    def uid(self, base):
        self._n += 1
        return f"{base}_{self._n}"

    def const(self, arr, base="c"):
        nm = self.uid(base)
        self.inits.append(numpy_helper.from_array(np.asarray(arr), nm))
        return nm

    def op(self, op_type, ins, nout=1, base=None, **attrs):
        base = base or op_type.lower()
        outs = [self.uid(base + "_o") for _ in range(nout)]
        self.nodes.append(helper.make_node(op_type, list(ins), outs,
                                           name=self.uid(base), **attrs))
        return outs[0] if nout == 1 else outs

    def reshape(self, x, shape, base="reshape"):
        return self.op("Reshape", [x, self.const(np.array(shape, np.int64), "shp")], base=base)

    def slice(self, x, start, end, axis):
        return self.op("Slice", [x, self.const(np.array([start], np.int64), "st"),
                                 self.const(np.array([end], np.int64), "en"),
                                 self.const(np.array([axis], np.int64), "ax")], base="slice")

    def reduce_sum(self, x, axis, keepdims=1):
        # opset >= 13: axes is an *input* for ReduceSum
        return self.op("ReduceSum", [x, self.const(np.array([axis], np.int64), "rax")],
                       keepdims=keepdims, base="reduce_sum")

    def reduce_max(self, x, axis, keepdims=1):
        # ReduceMax keeps axes as attribute until opset 18
        return self.op("ReduceMax", [x], axes=[axis], keepdims=keepdims, base="reduce_max")

    def add_input(self, name, shape):
        self.inputs.append(helper.make_tensor_value_info(name, TP.FLOAT, shape))

    def add_output(self, tensor, shape, name=None):
        if name and name != tensor:
            self.op("Identity", [tensor], base="out_id")
            self.nodes[-1].output[0] = name
            tensor = name
        self.outputs.append(helper.make_tensor_value_info(tensor, TP.FLOAT, shape))

    def model(self, opset=13):
        g = helper.make_graph(self.nodes, self.name, self.inputs, self.outputs,
                              initializer=self.inits)
        m = helper.make_model(g, opset_imports=[helper.make_opsetid("", opset)])
        m.ir_version = 7
        return m


# --------------------------------------------------------------------------- #
# attention core pieces (shared by repro + rewrites)
# --------------------------------------------------------------------------- #
def attn_original(g, q, kt, v, softmax="native"):
    """q [1,1,144,128], kt [1,1,128,4096], v [1,1,128,144] -> O [1,1,128,4096]."""
    s = g.op("MatMul", [q, kt], base="qk")
    s = g.op("Mul", [s, g.const(np.array(SCALE, np.float32), "scale")], base="scale")
    if softmax == "native":
        p = g.op("Softmax", [s], axis=2, base="softmax")
    elif softmax == "identity":
        p = g.op("Identity", [s], base="sm_id")
    elif softmax == "manual":
        p = manual_softmax(g, s, axis=2)
    else:
        raise ValueError(softmax)
    return g.op("MatMul", [v, p], base="av")


def manual_softmax(g, s, axis):
    m = g.reduce_max(s, axis)
    e = g.op("Exp", [g.op("Sub", [s, m], base="sm_sub")], base="sm_exp")
    z = g.reduce_sum(e, axis)
    return g.op("Div", [e, z], base="sm_div")


def attn_chunked(g, q, kt, v, n_chunks, softmax="native"):
    """Exact column-chunking along the 4096 axis (softmax is over the 144 axis)."""
    w = 4096 // n_chunks
    outs = []
    for c in range(n_chunks):
        kc = g.slice(kt, c * w, (c + 1) * w, 3)
        outs.append(attn_original(g, q, kc, v, softmax))
    return g.op("Concat", outs, axis=3, base="chunk_cat")


def attn_transposed(g, xt, qt, vt, chunks=1, softmax="native"):
    """Exact transposed form: softmax over the *channel* (144) axis.

    xt [1,1,4096,128] (K^T), qt [1,1,128,144] (Q^T), vt [1,1,144,128] (V^T)
    S' = xt @ qt            [1,1,4096,144]   (= S^T)
    P' = softmax_last(S'*s) [1,1,4096,144]   (= P^T)
    O' = P' @ vt            [1,1,4096,128]   (= O^T)
    Row-chunking of the 4096 axis is optional and exact.
    """
    def one(x):
        s = g.op("MatMul", [x, qt], base="qk_t")
        s = g.op("Mul", [s, g.const(np.array(SCALE, np.float32), "scale")], base="scale")
        if softmax == "native":
            p = g.op("Softmax", [s], axis=3, base="softmax_t")
        else:
            m = g.reduce_max(s, 3)
            e = g.op("Exp", [g.op("Sub", [s, m], base="sm_sub")], base="sm_exp")
            z = g.reduce_sum(e, 3)
            p = g.op("Div", [e, z], base="sm_div")
        return g.op("MatMul", [p, vt], base="av_t")
    if chunks == 1:
        return one(xt)
    w = 4096 // chunks
    return g.op("Concat", [one(g.slice(xt, c * w, (c + 1) * w, 2)) for c in range(chunks)],
                axis=2, base="chunk_cat")


def build(variant, chunks=1, softmax="native"):
    g = G(f"attn_{variant}")
    if variant == "ctx_T":
        g.add_input("x", [1, 128, 64, 64])
        g.add_input("kv", [1, 256, 12, 12])
        xt = g.reshape(g.op("Transpose", ["x"], perm=[0, 2, 3, 1], base="tr_x"), [1, 1, 4096, 128], "xt_rs")
        a = g.slice("kv", 0, 128, 1)
        b = g.slice("kv", 128, 256, 1)
        qt = g.reshape(a, [1, 1, 128, 144], "qt_rs")
        vt = g.reshape(g.op("Transpose", [b], perm=[0, 2, 3, 1], base="tr_v"), [1, 1, 144, 128], "vt_rs")
        o = attn_transposed(g, xt, qt, vt, chunks, softmax)
        o = g.reshape(o, [1, 64, 64, 128], "o_rs")
        o = g.op("Transpose", [o], perm=[0, 3, 1, 2], base="tr_o")
        g.add_output(o, [1, 128, 64, 64], "out")
        return g.model()
    if variant == "pure":
        g.add_input("q", [1, 1, 144, 128])
        g.add_input("kt", [1, 1, 128, 4096])
        g.add_input("v", [1, 1, 128, 144])
        o = (attn_chunked(g, "q", "kt", "v", chunks, softmax) if chunks > 1
             else attn_original(g, "q", "kt", "v", softmax))
        g.add_output(o, [1, 1, 128, 4096], "out")
        return g.model()
    # ctx variants
    g.add_input("x", [1, 128, 64, 64])
    g.add_input("kv", [1, 256, 12, 12])
    kt = g.reshape("x", [1, 1, 128, 4096], "kt_rs")
    a = g.slice("kv", 0, 128, 1)
    b = g.slice("kv", 128, 256, 1)
    qa = g.op("Transpose", [a], perm=[0, 2, 3, 1], base="tr")
    q = g.reshape(g.reshape(qa, [144, 128]), [1, 1, 144, 128], "q_rs")
    v = g.reshape(g.reshape(b, [128, 144]), [1, 1, 128, 144], "v_rs")
    if variant == "ctx_nosm":
        softmax = "identity"
    o = (attn_chunked(g, q, kt, v, chunks, softmax) if chunks > 1
         else attn_original(g, q, kt, v, softmax))
    out = g.reshape(o, [1, 128, 64, 64], "out_rs")
    g.add_output(out, [1, 128, 64, 64], "out")
    return g.model()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="ctx", choices=["pure", "ctx", "ctx_nosm", "ctx_T"])
    ap.add_argument("--chunks", type=int, default=1)
    ap.add_argument("--softmax", default="native", choices=["native", "manual", "identity"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    m = build(args.variant, args.chunks, args.softmax)
    onnx.save(m, args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
