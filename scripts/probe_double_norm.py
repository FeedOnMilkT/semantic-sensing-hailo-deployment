#!/usr/bin/env python3
"""Probe: which ONNX form of the RTFormer `attn_l` double-norm does DFC 3.33 quantize?

Real chain (per attn_l block of the teacher):

    xk  = Conv_k(BN(x))                    -> [1,512,16,16]
    z   = Reshape(xk, [1,8,64,256])       # head=8, chan/head=64, pos=256
    u   = Softmax(z, axis=-1)             # softmax over pos
    S   = ReduceSum(u, axis=2, keepdims)  # L1 over the 64 channels -> [1,8,1,256]
    y   = u / (S + 1e-6)
    y2  = Reshape(y, [1,512,16,16])
    out = Conv_v(y2)

MEASURED problem (ORT, real calibration images, teacher_base.onnx):
    S spans 7.4e-3..28.9 (layer4) and 2.3e-4..31.2 (layer5)  ->  log2 ratio 11.9 / 17.0
    so 1/(S+eps) needs 11.9..17.0 bits of APU exponent.  The exponent field is 4 bit
    (APU_EXP_BITS=4 -> max shift 15) and the DFC additionally requires
    shift < output_bits, hence  NegativeSlopeExponentNonFixable.
    eps is NOT the driver (eps=1e-6 vs S_min=2.3e-4): the ratio comes from S_max/S_min.

This script builds a faithful single-block model (real Conv_v weights, real z values
captured from the teacher, so the numeric ranges are production ranges) for each
candidate rewrite and reports
  * ORT parity against the original chain (max|diff|, cosine)
  * whether DFC can PARSE and OPTIMIZE (quantize) the variant

Variants
  v0_baseline  u / (S+eps)                       current -> DFC fails
  v1_recip     u * Reciprocal(S+eps)             exact
  v2_eps       u / (S+0.05)                      approx: bounds the reciprocal
  v3_floor     u / Max(S, floor)                 approx: bounds the reciprocal
  v4_maxnorm   u/M / (sum_c(u/M) + eps/M)        EXACT: M=max_c u normalises the range
  v6_logsoftmax softmax_c(log u)                 EXACT (u/S == softmax_c(log u))

Usage (hailo-dfc env, from a scratch dir because the SDK writes logs into cwd):
  cd /tmp/probe && /root/miniconda3/envs/hailo-dfc/bin/python probe_double_norm.py
"""
from __future__ import annotations

import argparse
import os
import tempfile

import numpy as np
import onnx
from onnx import TensorProto as TP
from onnx import helper as h
from onnx import numpy_helper as nh

EPS = 1e-6


class B:
    def __init__(self):
        self.nodes, self.inits = [], []
        self.k = 0

    def n(self, op, ins, **attr):
        self.k += 1
        out = f"{op.lower()}_{self.k}"
        self.nodes.append(h.make_node(op, ins, [out], name=out, **attr))
        return out

    def c(self, arr, dtype=np.float32):
        self.k += 1
        name = f"c_{self.k}"
        self.inits.append(nh.from_array(np.asarray(arr, dtype=dtype), name))
        return name


def build(variant: str, vw: np.ndarray, eps: float = EPS, floor: float = 2e-3,
          out_hw=(16, 16)) -> onnx.ModelProto:
    b = B()
    xk = "xk"                                   # [1,512,16,16]  (Conv_k output)
    z = b.n("Reshape", [xk, b.c(np.array([1, 8, 64, 256], np.int64), np.int64)])
    u = b.n("Softmax", [z], axis=-1)            # over pos (256)
    S = b.n("ReduceSum", [u, b.c(np.array([2], np.int64), np.int64)], keepdims=1)
    ax2 = b.c(np.array([2], np.int64), np.int64)

    if variant == "v0_baseline":
        A = b.n("Add", [S, b.c(np.array(eps, np.float32))])
        y = b.n("Div", [u, A])

    elif variant == "v1_recip":
        A = b.n("Add", [S, b.c(np.array(eps, np.float32))])
        r = b.n("Reciprocal", [A])
        y = b.n("Mul", [u, r])

    elif variant == "v2_eps":
        A = b.n("Add", [S, b.c(np.array(eps, np.float32))])
        y = b.n("Div", [u, A])

    elif variant == "v3_floor":
        A = b.n("Max", [S, b.c(np.array(floor, np.float32))])
        y = b.n("Div", [u, A])

    elif variant == "v4_maxnorm":
        # EXACT rewrite:  M = max_c u ;  u' = u/M ;  S' = sum_c u' (= S/M in [1,64])
        #                  y = u' / (S' + eps/M)  ==  u / (S + eps)
        M = b.n("ReduceMax", [u], axes=[2], keepdims=1)          # [1,8,1,256]
        r = b.n("Reciprocal", [M])                              # 1/M in [1, 1700]
        up = b.n("Mul", [u, r])                                 # u' = u/M in [0,1]
        Sp = b.n("ReduceSum", [up, ax2], keepdims=1)            # S' in [1,64]
        ep = b.n("Mul", [r, b.c(np.array(eps, np.float32))])    # eps' = eps/M
        A = b.n("Add", [Sp, ep])
        y = b.n("Div", [up, A])

    elif variant == "v6_logsoftmax":
        lg = b.n("Log", [u])
        lt = b.n("Transpose", [lg], perm=[0, 1, 3, 2])      # [1,8,256,64]
        yt = b.n("Softmax", [lt], axis=-1)                  # over the 64 channels
        y = b.n("Transpose", [yt], perm=[0, 1, 3, 2])       # [1,8,64,256]

    elif variant == "v7_softmax_axis2":
        # EXACT: y = softmax over the 64-channel axis of log(u)  ==  u / sum_c u
        # (axis=2 is the channel axis in the DFC frame layout; groups=8 -> grouped softmax)
        lg = b.n("Log", [u])
        y = b.n("Softmax", [lg], axis=2)

    elif variant == "v8_manual_softmax":
        # the same, written in the pattern the DFC's SoftMaxMatch fuser recognises:
        #   reduce_max -> ew_sub -> exp -> reduce_sum(+inv_pos) -> ew_mult
        lg = b.n("Log", [u])
        Mx = b.n("ReduceMax", [lg], axes=[2], keepdims=1)
        df = b.n("Sub", [lg, Mx])
        ex = b.n("Exp", [df])
        ss = b.n("ReduceSum", [ex, ax2], keepdims=1)
        iv = b.n("Div", [ex, ss])
        y = b.n("Mul", [iv, b.c(np.array(1.0, np.float32))])

    elif variant == "v9_floor":
        A = b.n("Max", [S, b.c(np.array(floor, np.float32))])
        y = b.n("Div", [u, A])

    else:
        raise KeyError(variant)

    y = b.n("Reshape", [y, b.c(np.array([1, 512, out_hw[0], out_hw[1]], np.int64), np.int64)])
    out = b.n("Conv", [y, b.c(vw, np.float32)], kernel_shape=[1, 1])

    g = h.make_graph(
        b.nodes, variant,
        [h.make_tensor_value_info("xk", TP.FLOAT, [1, 512, out_hw[0], out_hw[1]])],
        [h.make_tensor_value_info(out, TP.FLOAT, None)], b.inits)
    m = h.make_model(g, opset_imports=[h.make_opsetid("", 13)])
    m.ir_version = 8
    return onnx.shape_inference.infer_shapes(m)


def reference(xk: np.ndarray, vw: np.ndarray, eps: float = EPS) -> np.ndarray:
    """FP32 reference of the original chain (numpy)."""
    z = xk.reshape(1, 8, 64, 256)
    e = np.exp(z - z.max(-1, keepdims=True))
    u = e / e.sum(-1, keepdims=True)
    S = u.sum(axis=2, keepdims=True)
    y = (u / (S + eps)).reshape(512, 256)
    return (vw.reshape(512, 512) @ y).reshape(1, 512, 16, 16)


def ort_run(m, xk):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    s = ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"])
    return s.run(None, {"xk": xk})


VARIANTS = ["v0_baseline", "v1_recip", "v2_eps", "v3_floor", "v4_maxnorm", "v6_logsoftmax"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/root/deploy_artifacts/hailo/scratch/probe_data.npz")
    ap.add_argument("--block", default="layer5", choices=["layer4", "layer5"])
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--no-dfc", action="store_true")
    ap.add_argument("--n-calib", type=int, default=8)
    ap.add_argument("--eps", type=float, default=EPS)
    ap.add_argument("--floor", type=float, default=2e-3)
    a = ap.parse_args()

    d = np.load(a.data)
    zs = d[f"z_{a.block}"].astype(np.float32)        # [N,1,8,64,256]
    vw = d[f"v_{a.block}"]
    xks = zs.reshape(-1, 1, 512, 16, 16)             # [N,1,512,16,16]
    xk0 = xks[0].astype(np.float32)
    ref = reference(xk0, vw)

    print(f"### block {a.block}: xk {xk0.shape}  v {vw.shape}", flush=True)
    models = {}
    for name in a.variants.split(","):
        m = build(name, vw, eps=a.eps, floor=a.floor)
        out = ort_run(m, xk0)[0]
        num = float(np.abs(out - ref).max())
        den = float(np.abs(ref).max()) + 1e-12
        cos = float((out.ravel() @ ref.ravel()) /
                    (np.linalg.norm(out) * np.linalg.norm(ref) + 1e-12))
        models[name] = m
        print(f"  {name:14s} ORT vs original: max|diff|={num:.4e} rel={num/den:.3e} cos={cos:.8f}", flush=True)

    if a.no_dfc:
        return

    from hailo_sdk_client import ClientRunner
    tmpd = tempfile.mkdtemp(prefix="dnprobe_")
    print(f"scratch: {tmpd}", flush=True)
    for name, m in models.items():
        p = os.path.join(tmpd, f"{name}.onnx")
        onnx.save(m, p)
        try:
            r = ClientRunner(hw_arch="hailo8")
            r.translate_onnx_model(p, name, start_node_names=["xk"],
                                   end_node_names=[m.graph.output[0].name],
                                   net_input_shapes={"xk": [1, 512, 16, 16]})
            print(f"[PARSE OK ] {name}", flush=True)
        except Exception as e:                                    # noqa: BLE001
            msg = str(e).strip().splitlines()
            print(f"[PARSE ERR] {name}: {type(e).__name__}: {(msg[0] if msg else '')[:300]}", flush=True)
            continue
        try:
            r.load_model_script(
                "model_optimization_flavor(optimization_level=0, compression_level=0, batch_size=1)\n"
                "model_optimization_config(calibration, batch_size=1, calibset_size=8)\n")
            # the DFC wants NHWC calibration data: xks is [N,1,512,16,16]
            calib = xks[: a.n_calib, 0].transpose(0, 2, 3, 1).astype(np.float32)
            r.optimize(calib)
            print(f"[OPT  OK ] {name}", flush=True)
            r.save_har(os.path.join(tmpd, f"{name}.har"))
        except Exception as e:                                    # noqa: BLE001
            import traceback
            traceback.print_exc()
            msg = str(e).strip().splitlines()
            print(f"[OPT  ERR] {name}: {type(e).__name__}: {(msg[-1] if msg else '')[:300]}", flush=True)


if __name__ == "__main__":
    main()
