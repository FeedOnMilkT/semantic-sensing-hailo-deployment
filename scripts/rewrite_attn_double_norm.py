#!/usr/bin/env python3
"""Rewrite the RTFormer `attn_l` double-normalisation into a DFC-friendly grouped softmax.

WHY
===
The teacher's external-attention block normalises twice:

    xk = Conv_k(BN(x))                     -> [1,512,16,16]
    z  = Reshape(xk, [1,8,64,256])         # head=8, chan/head=64, pos=256
    u  = Softmax(z, axis=-1)               # softmax over the 256 positions
    S  = ReduceSum(u, axis=2, keepdims=1)  # L1 over the 64 channels -> [1,8,1,256]
    y  = u / (S + 1e-6)

DFC 3.33 turns the `ReduceSum -> Add(eps) -> Div` into a *normalization* layer with
activation `inv_pos`, i.e. it computes the RECIPROCAL 1/(S+eps) on the APU.  Measured on
real calibration images (ORT):

    S      = 7.4e-3 .. 28.9  (layer4)      log2(S_max/S_min) = 11.9
    S      = 2.3e-4 .. 31.2  (layer5)      log2(S_max/S_min) = 17.0
    1/(S+eps) therefore needs 12..17 bits of APU exponent, but the exponent field is
    4 bit (APU_EXP_BITS=4 -> max 15) and a 16-bit activation only has 15 data bits
    -> `NegativeSlopeExponentNonFixable: Desired shift is 15.0, but op has only 15
       data bits`  (and "shift 13.0, but only 8 data bits" for the ew_mult behind it).

THE FIX (exact, no approximation)
=================================
    y = u / S   (the DFC already drops the 1e-6: teacher.hn has
                 `{"elementwise_add": false, "activation": "inv_pos"}`)
and
    log u[h,c,p] = z[h,c,p] - log(sum_q exp z[h,c,q])      (a per-(h,c) constant)
so softmax over the channel axis of log u is

    softmax_c(log u)[h,c,p] = exp(log u[h,c,p]) / sum_c' exp(log u[h,c',p])
                            = u[h,c,p] / sum_c' u[h,c',p]
                            = u / S                              <-- exactly the double-norm

=> Replace `ReduceSum(axis=2) -> Add(eps) -> Div(u, ·)` by
       `Log(u) -> Softmax(axis=2)`
   The channel axis (64 per head, 8 heads) becomes a *grouped* softmax, which the APU
   implements natively: no reciprocal, no negative slope, no exponent overflow.

   NOTE: this is NOT the same as softmax_c(z) (that was my first, wrong hypothesis):
   u = exp(z)/P with P depending on the channel, so softmax_c(z) != u/S.
   Measured: max|softmax_c(z) - u/S| = 0.87 (rel 1.09)  -> not usable.
   softmax_c(log u) is the exact identity (max|diff| 1.4e-4, cos 0.99999994; the
   residual is the float32 log/exp round trip).

Usage
-----
  python scripts/rewrite_attn_double_norm.py --onnx teacher.onnx --out teacher_rw.onnx
  python scripts/rewrite_attn_double_norm.py --onnx teacher.onnx --out teacher_rw.onnx \
      --check --calib /root/deploy_artifacts/data/calib_u8.npy --n-check 4
"""
from __future__ import annotations

import argparse
import collections
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper

MEAN255 = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD255 = np.array([58.395, 57.12, 57.375], dtype=np.float32)


# --------------------------------------------------------------------------- helpers
def _const_value(m, name):
    """Value of a constant/initializer tensor by name (None if not constant)."""
    for i in m.graph.initializer:
        if i.name == name:
            return numpy_helper.to_array(i)
    for n in m.graph.node:
        if n.op_type == "Constant" and n.output[0] == name and n.attribute:
            for a in n.attribute:
                if a.name == "value":
                    return numpy_helper.to_array(a.t)
    return None


def _axes_of(m, node):
    """Reduce axes of a ReduceSum node (as a list of ints)."""
    axes = None
    for a in node.attribute:
        if a.name == "axes":
            axes = list(a.ints)
    if axes is None and len(node.input) > 1:
        v = _const_value(m, node.input[1])
        if v is not None:
            axes = [int(x) for x in np.asarray(v).ravel()]
    return axes


def _softmax_axis(m, node):
    for a in node.attribute:
        if a.name == "axis":
            return int(a.i)
    return None


# --------------------------------------------------------------------------- the pass
def rewrite_attn_double_norm(m: onnx.ModelProto, verbose: bool = True):
    """`ReduceSum(axis=k) -> Add(eps) -> Div(u, ·)`  ==>  `Log(u) -> Softmax(axis=k)`.

    Returns (model, n_rewritten).  Only chains whose numerator is a Softmax output and
    whose denominator is `ReduceSum + const` are touched, so ordinary Div layers and
    the cross-attention blocks are left alone.
    """
    g = m.graph
    prod = {o: n for n in g.node for o in n.output}
    cons = collections.defaultdict(list)
    for n in g.node:
        for i in n.input:
            cons[i].append(n)

    new_nodes, removed, count = [], set(), 0
    for div in [n for n in g.node if n.op_type == "Div"]:
        num, den = div.input[0], div.input[1]
        add = prod.get(den)
        if add is None or add.op_type != "Add":
            continue
        rs = None
        for i in add.input:
            p = prod.get(i)
            if p is not None and p.op_type == "ReduceSum":
                rs = p
        if rs is None:
            continue
        # the numerator must be the ReduceSum's input (a Softmax output)
        u_name = rs.input[0]
        u = prod.get(u_name)
        if u is None or u.op_type != "Softmax":
            continue
        # the Add must be ReduceSum + a constant (the eps)
        other = [i for i in add.input if i != rs.output[0]]
        if len(other) != 1 or _const_value(m, other[0]) is None:
            continue
        # the Div must have a single consumer
        if len(cons[div.output[0]]) != 1:
            continue
        axes = _axes_of(m, rs)
        if not axes or len(axes) != 1:
            continue
        axis = axes[0]
        # the ReduceSum must keep the dims (so the softmax axis matches)
        keepdims = any(a.name == "keepdims" and a.i == 1 for a in rs.attribute)
        if not keepdims:
            continue

        log_name = f"{div.name}_log"
        sm_name = f"{div.name}_chan_softmax"
        new_nodes.append(helper.make_node("Log", [u_name], [log_name], name=log_name))
        new_nodes.append(helper.make_node("Softmax", [log_name], [sm_name],
                                          name=sm_name, axis=axis))
        # rewire the Div's consumer
        for c in cons[div.output[0]]:
            for k, i in enumerate(c.input):
                if i == div.output[0]:
                    c.input[k] = sm_name
        removed.update({div.name, add.name, rs.name})
        count += 1
        if verbose:
            print(f"  rewrote {u.name} -> {div.name}: "
                  f"ReduceSum(axis={axis})+Add(eps)+Div  ==>  Log+Softmax(axis={axis})")
        # the old Softmax stays (it is the numerator), nothing else to do

    if count:
        keep = [n for n in g.node if n.name not in removed]
        g.node.clear() if hasattr(g.node, "clear") else None
        del g.node[:]
        g.node.extend(keep)
        # insert the new nodes right after the numerator softmax
        for nn in new_nodes:
            g.node.append(nn)
        # keep topological order: sort stably by the order of the node they replace
        _topo_sort(g)
    return m, count


def _topo_sort(g):
    """Stable topological sort of the graph nodes (by name index)."""
    prod = {o: n for n in g.node for o in n.output}
    indeg = {n.name: 0 for n in g.node}
    adj = collections.defaultdict(list)
    for n in g.node:
        for i in n.input:
            p = prod.get(i)
            if p is not None and p.name != n.name:
                adj[p.name].append(n.name)
                indeg[n.name] += 1
    order, ready = [], [n.name for n in g.node if indeg[n.name] == 0]
    seen = set()
    while ready:
        nm = ready.pop(0)
        if nm in seen:
            continue
        seen.add(nm)
        order.append(nm)
        for s in adj[nm]:
            indeg[s] -= 1
            if indeg[s] == 0:
                ready.append(s)
    if len(order) != len(g.node):          # cycle -> keep the original order
        return
    by = {n.name: n for n in g.node}
    del g.node[:]
    g.node.extend(by[nm] for nm in order)


# --------------------------------------------------------------------------- check
def ort_logits(model_path, imgs_u8):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    s = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
    name = s.get_inputs()[0].name
    outs = []
    for u8 in imgs_u8:
        x = ((u8.astype(np.float32) - MEAN255) / STD255).transpose(2, 0, 1)[None]
        outs.append(s.run(None, {name: x})[0])
    return outs


def compare(a, b):
    """(max|diff|, cosine, argmax agreement) between two lists of logit arrays."""
    worst, cos_min, agree_min = 0.0, 1.0, 1.0
    for x, y in zip(a, b):
        d = float(np.abs(x - y).max())
        rel = d / (float(np.abs(x).max()) + 1e-12)
        c = float((x.ravel() @ y.ravel()) /
                  (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))
        ag = float((x.argmax(1) == y.argmax(1)).mean())
        worst = max(worst, rel)
        cos_min = min(cos_min, c)
        agree_min = min(agree_min, ag)
    return worst, cos_min, agree_min


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", action="store_true", help="ORT parity check against --onnx")
    ap.add_argument("--calib", default="/root/deploy_artifacts/data/calib_u8.npy")
    ap.add_argument("--n-check", type=int, default=4)
    a = ap.parse_args(argv)

    m = onnx.load(a.onnx)
    before = collections.Counter(n.op_type for n in m.graph.node)
    print(f"in : nodes={len(m.graph.node)}")
    m, n = rewrite_attn_double_norm(m)
    print(f"rewrote {n} attn double-norm chain(s)")
    after = collections.Counter(n.op_type for n in m.graph.node)
    print(f"out: nodes={len(m.graph.node)}  "
          f"Log {before['Log']}->{after['Log']}  Softmax {before['Softmax']}->{after['Softmax']}  "
          f"ReduceSum {before['ReduceSum']}->{after['ReduceSum']}  "
          f"Div {before['Div']}->{after['Div']}  Add {before['Add']}->{after['Add']}")
    if n == 0:
        print("ERROR: no chain rewritten", file=sys.stderr)
        return 2
    onnx.checker.check_model(m)
    onnx.save(m, a.out)
    print(f"saved {a.out}")

    if a.check:
        calib = np.load(a.calib, mmap_mode="r")[: a.n_check]
        imgs = [np.asarray(calib[i]) for i in range(len(calib))]
        ref = ort_logits(a.onnx, imgs)
        new = ort_logits(a.out, imgs)
        rel, cos, agree = compare(ref, new)
        print(f"ORT parity on {len(imgs)} real images: "
              f"max|rel diff|={rel:.3e}  min cosine={cos:.10f}  min argmax agreement={agree:.6f}")
        if rel > 1e-3 or cos < 0.99999:
            print("ERROR: rewritten graph diverges", file=sys.stderr)
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
