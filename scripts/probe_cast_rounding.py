#!/usr/bin/env python
"""Is `Cast` a working alternative to DFC's broken `Floor`?

Measured root cause: DFC's `Floor` is a NO-OP in the emulator (it returns the input
unchanged for every non-integer), while `Abs` is exact.  So `floor(d + 0.5)` yields
`d + 0.5` and the hyper-latent stops carrying integer symbols -> the factorized prior
sees outliers -> the z stream costs 52x more bits.

ONNX `Cast` to int32 truncates toward zero, so the rounding form

    round_half_up(d) = trunc(d + 1024.5) - 1024

is a true floor for every d in (-1024, 1024) (the shifted value is positive, where
trunc == floor), and the fp32 eps at that magnitude (~6e-5) is far below the 0.5 step.

This probe uses the [1,16,1,1] shape, which the DFC emulator accepts (the [1,128,8,8]
shape trips an "axes don't match array" error in the emulator's dataset builder).
"""
import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(OUT, exist_ok=True)

CH = 16
OFF = 1024.5


def build(name, nodes, inits=()):
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, CH, 1, 1])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, CH, 1, 1])
    g = helper.make_graph(nodes, name, [x], [y], list(inits))
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    p = os.path.join(OUT, f"{name}.onnx")
    onnx.save(m, p)
    return p


OFFC = helper.make_tensor("off", TensorProto.FLOAT, [], [OFF])

cases = {
    "cast_trunc": build("cast_trunc", [
        helper.make_node("Cast", ["x"], ["i"], to=TensorProto.INT32),
        helper.make_node("Cast", ["i"], ["y"], to=TensorProto.FLOAT),
    ]),
    "cast_floor": build("cast_floor", [
        helper.make_node("Add", ["x", "off"], ["t"]),
        helper.make_node("Cast", ["t"], ["i"], to=TensorProto.INT32),
        helper.make_node("Cast", ["i"], ["f"], to=TensorProto.FLOAT),
        helper.make_node("Sub", ["f", "off"], ["y"]),
    ], [OFFC]),
    "floor_ref": build("floor_ref", [helper.make_node("Floor", ["x"], ["y"])]),
}

from hailo_sdk_client import ClientRunner
from hailo_sdk_client.exposed_definitions import InferenceContext
import onnxruntime as ort

FLAT = np.array([-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, -0.25, 0.0,
                  0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.5], np.float32)
vals = FLAT.reshape(1, CH, 1, 1)

for name, path in sorted(cases.items()):
    print(f"\n########## {name}")
    try:
        runner = ClientRunner(hw_arch="hailo8")
        runner.translate_onnx_model(path, name)
        with runner.infer_context(InferenceContext.SDK_NATIVE) as ctx:
            out = runner.infer(ctx, {f"{name}/input_layer1":
                                     np.ascontiguousarray(vals.transpose(0, 2, 3, 1))},
                               data_count=1, batch_size=1)
        hailo = np.asarray(out[0]).reshape(-1)
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {str(e)[:200]}")
        continue

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    ref = sess.run(None, {"x": vals})[0].reshape(-1)
    d = np.abs(hailo - ref)
    print(f"  {'input':>7} {'ORT':>9} {'Hailo':>9} {'diff':>7}")
    for a, b, h in zip(FLAT, ref, hailo):
        flag = "  <-- MISMATCH" if abs(b - h) > 1e-5 else ""
        print(f"  {a:7.2f} {b:9.3f} {h:9.3f} {h - b:7.3f}{flag}")
    print(f"  mismatches: {int((d > 1e-5).sum())}/{len(ref)}  max|diff|={d.max():.6f}")
    print(f"  == ORT exactly? {bool(np.allclose(hailo, ref, atol=1e-5))}")
    print(f"  == trunc(x)?   {bool(np.allclose(hailo, np.trunc(FLAT), atol=1e-5))}")
    print(f"  == floor(x)?   {bool(np.allclose(hailo, np.floor(FLAT), atol=1e-5))}")
    print(f"  == identity?   {bool(np.allclose(hailo, FLAT, atol=1e-5))}")
