#!/usr/bin/env python3
"""Run a list of minimal-repro attention variants through parse->optimize->compile.

Usage (hailo-dfc python, each experiment in its own sub dir):
    python run_attn_repro.py --root /root/autodl-tmp/exp/repro \
        --variants pure:1:native ctx:1:native ctx:4:native
Variant spec = <variant>:<chunks>:<softmax>
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--opt-level", type=int, default=2)
    args = ap.parse_args()

    import onnx
    from teacher_attn_common import selfcheck_onnx, dfc_pipeline
    import teacher_attn_repro as R

    os.makedirs(args.root, exist_ok=True)
    summary = {}
    for spec in args.variants:
        variant, chunks, sm = spec.split(":")
        tag = f"{variant}_c{chunks}_{sm}"
        d = os.path.join(args.root, tag)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"{tag}.onnx")
        # reference = same variant, 1 chunk, native softmax
        ref_path = os.path.join(d, f"{tag}_ref.onnx")
        onnx.save(R.build("ctx" if variant == "ctx_T" else variant, 1, "native"), ref_path)
        onnx.save(R.build(variant, int(chunks), sm), path)
        chk = selfcheck_onnx(path, ref_path if sm != "identity" else None)
        summary[tag] = dict(selfcheck=chk)
        if not chk["ok"]:
            print(f"[{tag}] selfcheck FAILED, skipping DFC", flush=True)
            continue

        def calib_fn(layers, _n=8):
            rng = np.random.default_rng(1)
            return {h: rng.standard_normal([_n] + [1 if s < 0 else s for s in shp[1:]]).astype(np.float32)
                    for h, _o, shp in layers}

        res = dfc_pipeline(path, tag, d, calib_fn, calib_n=8,
                           opt_level=args.opt_level, do_compile=not args.no_compile)
        summary[tag].update({k: v for k, v in res.items() if k in
                             ("parse", "optimize", "compile", "hef_mib", "compile_s")})
        with open(os.path.join(args.root, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "selfcheck"}
                      for k, v in summary.items()}, indent=2))


if __name__ == "__main__":
    main()
