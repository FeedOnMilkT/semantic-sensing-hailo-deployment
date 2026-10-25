#!/usr/bin/env python
"""CONTROL probe: is the DFC emulator harness sound, and which elementwise ops work?

Each op gets its OWN single-output graph: a multi-output graph trips a DFC fuser
bug ("'shortcut1' is not in list"), which is unrelated to what we are testing.

  identity (Mul by 1)  -> must return x
  abs       (Abs)      -> known to work
  mul2      (Mul by 2) -> known to work
  floor     (Floor)    -> reported broken (returns x)
  cast      (Cast i32) -> reported broken (returns x)

If identity/abs/mul2 agree with ORT and floor/cast do not, the harness is sound and
the emulator genuinely lacks a working integer-rounding primitive.

The [1,16,1,1] shape is used because the emulator's dataset builder raises
"axes don't match array" for the [1,128,8,8] shape.
"""
import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(OUT, exist_ok=True)

CH = 16

FLAT = np.array([-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, -0.25, 0.0,
                  0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.5], np.float32)
vals = FLAT.reshape(1, CH, 1, 1)


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


CASES = {
    "id": (build("id", [helper.make_node("Mul", ["x", "one"], ["y"])],
                 [helper.make_tensor("one", TensorProto.FLOAT, [], [1.0])]), None),
    "ab": (build("ab", [helper.make_node("Abs", ["x"], ["y"])]), None),
    "m2": (build("m2", [helper.make_node("Mul", ["x", "two"], ["y"])],
                 [helper.make_tensor("two", TensorProto.FLOAT, [], [2.0])]), None),
    "fl": (build("fl", [helper.make_node("Floor", ["x"], ["y"])]), None),
    "ca": (build("ca", [
        helper.make_node("Cast", ["x"], ["i"], to=TensorProto.INT32),
        helper.make_node("Cast", ["i"], ["y"], to=TensorProto.FLOAT),
    ]), None),
}

from hailo_sdk_client import ClientRunner
from hailo_sdk_client.exposed_definitions import InferenceContext
import onnxruntime as ort

runner = ClientRunner(hw_arch="hailo8")
results = {}
for name, (path, _) in CASES.items():
    try:
        runner.translate_onnx_model(path, name)
        with runner.infer_context(InferenceContext.SDK_NATIVE) as ctx:
            res = runner.infer(ctx, {f"{name}/input_layer1":
                                     np.ascontiguousarray(vals.transpose(0, 2, 3, 1))},
                               data_count=1, batch_size=1)
        # the emulator returns {output_name: array}; the key is version-dependent,
        # so take the single output by position.
        arrays = list(res.values()) if isinstance(res, dict) else list(res)
        results[name] = np.asarray(arrays[0]).reshape(-1)
    except Exception as e:
        results[name] = None
        print(f"  {name}: FAILED {type(e).__name__}: {str(e)[:120]}")

print(f"\n{'input':>7} | " + " | ".join(f"{n:>7}" for n in CASES))
for i, v in enumerate(FLAT):
    row = []
    for n in CASES:
        h = results[n]
        row.append("   FAIL " if h is None else f"{h[i]:7.3f}")
    print(f"{v:7.2f} | " + " | ".join(row))

print("\nVERDICT (ORT is ground truth):")
for n in CASES:
    h = results[n]
    if h is None:
        print(f"  {n}: graph failed to parse/infer")
        continue
    sess = ort.InferenceSession(CASES[n][0], providers=["CPUExecutionProvider"])
    r = sess.run(None, {"x": vals})[0].reshape(-1)
    d = np.abs(h - r).max()
    if d < 1e-5:
        print(f"  {n}: CORRECT on Hailo (matches ORT)")
    elif np.allclose(h, FLAT, atol=1e-5):
        print(f"  {n}: NO-OP on Hailo (emulator returned the input unchanged)")
    else:
        print(f"  {n}: WRONG but not a no-op -> got {h[:6]}")
