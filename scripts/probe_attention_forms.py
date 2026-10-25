#!/usr/bin/env python3
"""Probe: which equivalent ONNX form of the cross-KV external attention does DFC 3.33 parse?

  S = softmax_n( K[N,c] @ X[c,HW] * N^-0.5 )      out = V[c,N] @ S[N,HW]

Run in the hailo-dfc env, from a scratch dir (the SDK drops logs into cwd):
  cd /tmp/probe && /root/miniconda3/envs/hailo-dfc/bin/python \
      /root/semantic-sensing-hailo-deployment/scripts/probe_attention_forms.py
"""
from __future__ import annotations

import sys

import numpy as np
import onnx
from onnx import TensorProto as TP
from onnx import helper as h
from onnx import numpy_helper as nh

C, N, SIDE, HW = 128, 144, 64, 64 * 64
rng = np.random.default_rng(0)


class B:
    def __init__(self):
        self.nodes, self.inits, self.k = [], [], 0

    def t(self, op, ins, **attr):
        self.k += 1
        out = f"{op.lower()}_{self.k}"
        self.nodes.append(h.make_node(op, ins, [out], name=out, **attr))
        return out

    def c(self, arr, dtype=np.int64):
        self.k += 1
        name = f"c_{self.k}"
        self.inits.append(nh.from_array(np.asarray(arr, dtype=dtype), name))
        return name

    def reshape(self, x, shape):
        return self.t("Reshape", [x, self.c(shape)])

    def conv1x1(self, x, cin, cout):
        w = self.c(rng.standard_normal((cout, cin, 1, 1)) * 0.1, np.float32)
        b = self.c(np.zeros(cout), np.float32)
        return self.t("Conv", [x, w, b], kernel_shape=[1, 1])


def build(variant: str) -> onnx.ModelProto:
    global rng
    rng = np.random.default_rng(0)   # identical weights for every variant -> comparable outputs
    b = B()
    x = b.conv1x1("x", C, C)          # [1,C,64,64]
    k = b.conv1x1("k", C, C)          # [1,C,12,12]   memory keys   (c on channels)
    v = b.conv1x1("v", C, C)          # [1,C,12,12]   memory values
    scale = b.c(np.array(N ** -0.5), np.float32)

    if variant == "v0_2d_current":
        kk = b.reshape(b.t("Transpose", [k], perm=[0, 2, 3, 1]), [N, C])
        xx = b.reshape(x, [C, HW])
        vv = b.reshape(v, [C, N])
        s = b.t("Softmax", [b.t("Mul", [b.t("MatMul", [kk, xx]), scale])], axis=0)
        o = b.reshape(b.t("MatMul", [vv, s]), [1, C, SIDE, SIDE])

    elif variant == "v1_4d_channel_first":
        kk = b.reshape(b.t("Transpose", [k], perm=[0, 2, 3, 1]), [1, 1, N, C])
        xx = b.reshape(x, [1, 1, C, HW])
        vv = b.reshape(v, [1, 1, C, N])
        s = b.t("Softmax", [b.t("Mul", [b.t("MatMul", [kk, xx]), scale])], axis=2)
        o = b.reshape(b.t("MatMul", [vv, s]), [1, C, SIDE, SIDE])

    elif variant == "v2_tokens_last_axis":
        # X^T[HW,C] @ K^T[C,N] -> [HW,N]; softmax over last axis; @ V^T[N,C] -> [HW,C]
        xt = b.t("Transpose", [b.reshape(x, [1, 1, C, HW])], perm=[0, 1, 3, 2])      # [1,1,HW,C]
        kt = b.reshape(k, [1, 1, C, N])                                              # [1,1,C,N]
        vt = b.t("Transpose", [b.reshape(v, [1, 1, C, N])], perm=[0, 1, 3, 2])       # [1,1,N,C]
        s = b.t("Softmax", [b.t("Mul", [b.t("MatMul", [xt, kt]), scale])], axis=3)   # [1,1,HW,N]
        ot = b.t("MatMul", [s, vt])                                                  # [1,1,HW,C]
        o = b.reshape(b.t("Transpose", [ot], perm=[0, 1, 3, 2]), [1, C, SIDE, SIDE])

    elif variant == "v3_tokens_3d":
        xt = b.t("Transpose", [b.reshape(x, [1, C, HW])], perm=[0, 2, 1])            # [1,HW,C]
        kt = b.reshape(k, [1, C, N])                                                 # [1,C,N]
        vt = b.t("Transpose", [b.reshape(v, [1, C, N])], perm=[0, 2, 1])             # [1,N,C]
        s = b.t("Softmax", [b.t("Mul", [b.t("MatMul", [xt, kt]), scale])], axis=2)
        ot = b.t("MatMul", [s, vt])                                                  # [1,HW,C]
        o = b.reshape(b.t("Transpose", [ot], perm=[0, 2, 1]), [1, C, SIDE, SIDE])
    else:
        raise KeyError(variant)

    g = h.make_graph(
        b.nodes, variant,
        [h.make_tensor_value_info("x", TP.FLOAT, [1, C, SIDE, SIDE]),
         h.make_tensor_value_info("k", TP.FLOAT, [1, C, 12, 12]),
         h.make_tensor_value_info("v", TP.FLOAT, [1, C, 12, 12])],
        [h.make_tensor_value_info(o, TP.FLOAT, None)], b.inits)
    m = h.make_model(g, opset_imports=[h.make_opsetid("", 13)])
    m.ir_version = 8
    return onnx.shape_inference.infer_shapes(m)


def ort_ref(m):
    import onnxruntime as ort
    s = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
    feeds = {i.name: np.random.default_rng(1).standard_normal(i.shape).astype(np.float32)
             for i in s.get_inputs()}
    return s.run(None, feeds)[0]


VARIANTS = ["v0_2d_current", "v1_4d_channel_first", "v2_tokens_last_axis", "v3_tokens_3d"]


def main():
    from hailo_sdk_client import ClientRunner
    ref = None
    for name in (sys.argv[1:] or VARIANTS):
        m = build(name)
        out = ort_ref(m)
        ref = out if ref is None else ref
        same = float(np.abs(out - ref).max())
        try:
            r = ClientRunner(hw_arch="hailo8")
            r.translate_onnx_model(_save(m, name), name,
                                   start_node_names=["x", "k", "v"],
                                   end_node_names=[m.graph.output[0].name],
                                   net_input_shapes={"x": [1, C, SIDE, SIDE], "k": [1, C, 12, 12], "v": [1, C, 12, 12]})
            print(f"[PARSE OK ] {name}  (vs v0 ORT max|diff|={same:.2e})")
        except Exception as e:                     # noqa: BLE001
            msg = str(e).strip().splitlines()
            print(f"[PARSE ERR] {name}: {type(e).__name__}: {(msg[0] if msg else '')[:260]}")


def _save(m, name):
    p = f"/tmp/probe_{name}.onnx"
    onnx.save(m, p)
    return p


if __name__ == "__main__":
    main()
