#!/usr/bin/env python3
"""Full-network parse -> optimize -> (compile) driver for the rewritten teacher.

Calibration = real RGB uint8-range float32 images (NHWC, 0..255) from
``calib_u8.npy`` (memory mapped); normalisation is part of the model script.

Examples (hailo-dfc python, run inside its own experiment dir):
  python run_teacher_full.py --onnx t_T4.onnx --name t_T4 --out-dir d --calib-n 32 --opt-level 0
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from teacher_attn_common import dfc_pipeline, MEAN255, STD255

ap = argparse.ArgumentParser()
ap.add_argument("--onnx", required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--out-dir", required=True)
ap.add_argument("--calib", default="/root/deploy_artifacts/data/calib_u8.npy")
ap.add_argument("--calib-n", type=int, default=32)
ap.add_argument("--opt-level", type=int, default=2)
ap.add_argument("--comp-level", type=int, default=0)
ap.add_argument("--extra", default="", help="';' separated model-script lines")
ap.add_argument("--no-compile", action="store_true")
a = ap.parse_args()

def calib_fn(layers):
    arr = np.load(a.calib, mmap_mode="r")[: a.calib_n]
    arr = np.ascontiguousarray(arr).astype(np.float32)
    assert len(layers) == 1
    return {layers[0][0]: arr}

lines = [f"normalization1 = normalization({MEAN255}, {STD255})"]
lines += [s.strip() for s in a.extra.split(";") if s.strip()]
res = dfc_pipeline(a.onnx, a.name, a.out_dir, calib_fn, script_lines=lines,
                   opt_level=a.opt_level, comp_level=a.comp_level,
                   calib_n=a.calib_n, do_compile=not a.no_compile)
