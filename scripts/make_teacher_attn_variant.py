#!/usr/bin/env python3
"""Produce a rewritten teacher ONNX and run the mandatory self-check gate."""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import onnx
from teacher_attn_rewrite import rewrite_model, find_blocks
from teacher_attn_common import selfcheck_onnx

ap = argparse.ArgumentParser()
ap.add_argument("--onnx", default="/root/autodl-tmp/exp/teacher_static.onnx")
ap.add_argument("--out", required=True)
ap.add_argument("--mode", default="T")
ap.add_argument("--chunks", type=int, default=1)
ap.add_argument("--softmax", default="native")
ap.add_argument("--trials", type=int, default=2)
a = ap.parse_args()
m = onnx.load(a.onnx)
print("blocks:", [b["tag"] for b in find_blocks(m)])
m2 = rewrite_model(m, a.mode, a.chunks, a.softmax)
onnx.save(m2, a.out)
r = selfcheck_onnx(a.out, a.onnx, n_trials=a.trials, tol=1e-4, input_scale=1.0)
print(json.dumps(r))
