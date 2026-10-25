#!/usr/bin/env python3
"""Evaluate the split teacher chain in the DFC emulator against ORT FP32.

The teacher is split into 4 DFC graphs (see ``segs4/segments.json``).  Each
segment is a separate DFC graph, so the boundary tensors have to be handed over
on the host.  The emulator runs each segment in its own ``InferenceContext``
context, which reproduces the int8 behaviour of the real board.

Two contexts are measured:

* ``SDK_FP_OPTIMIZED`` -- no quantization.  This validates the *graph surgery*
  (the split, the staticised resize, the NHWC/NCHW conventions).  It must come
  out at cosine ~1.0; if it does not, the split itself is wrong and any
  remaining loss must not be attributed to quantization.
* ``SDK_QUANTIZED`` -- the actual int8 behaviour.  Scored by cosine similarity
  against ORT FP32.

Two details this script gets right that the naive version does not:

1. ``ClientRunner.infer`` keys its dataset by the **Hailo** layer name
   (``seg0/input_layer1``), not by the original ONNX tensor name (``image``),
   and it returns the outputs in the order of the segment's ``output_layer``
   entries.  The boundary hand-over therefore maps

       hailo output_layerN  ->  original ONNX tensor  ->  hailo input_layerM

2. The boundaries are **not a linear chain**: ``seg3`` consumes
   ``/neck/Resize_output_0`` (from ``seg0``), ``/neck/Resize_2_output_0`` (from
   ``seg1``) and ``/neck/Resize_1_output_0`` + ``/neck/Resize_3_output_0``
   (from ``seg2``).  Every segment's outputs are therefore accumulated in a
   global ``original tensor name -> array`` map, and the next segment picks
   the ones it declares as inputs.

The emulator returns NHWC for 4-D tensors, which is exactly the layout the next
segment's input layer expects, so no transpose is needed at the hand-over.  Only
the final logits need to be brought back to NCHW to match ORT.

Usage (hailo-dfc env, from a scratch dir):
    python eval_teacher_split_chain.py \
        --har-dir /root/autodl-tmp/scratch_teacher/work2 \
        --onnx /root/deploy_artifacts/hailo/teacher_fix/teacher_rw.onnx \
        --n 8 --context SDK_QUANTIZED
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

MEAN255 = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD255 = np.array([58.395, 57.12, 57.375], dtype=np.float32)


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_val_images(n, size=512, manifest="/root/deploy_artifacts/data/val_manifest.json"):
    """Real validation images as uint8 HWC RGB (bilinear resize to `size`)."""
    from PIL import Image
    with open(manifest) as f:
        man = json.load(f)
    items = man["subset"][:n]
    out = []
    for it in items:
        im = Image.open(it["image"]).convert("RGB").resize((size, size), Image.BILINEAR)
        out.append(np.asarray(im, np.uint8))
    return out, [it["name"] for it in items]


def ort_logits(onnx_path, imgs):
    """FP32 reference logits, one (1,150,128,128) array per image."""
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    outs = []
    for u8 in imgs:
        x = ((u8.astype(np.float32) - MEAN255) / STD255).transpose(2, 0, 1)[None]
        outs.append(np.asarray(sess.run(None, {name: x})[0]))
    return outs


# --------------------------------------------------------------------------- #
# emulator chain
# --------------------------------------------------------------------------- #
class SegmentChain:
    """Runs the split teacher segments back to back in the DFC emulator."""

    def __init__(self, har_dir, n_segs=4, hw_arch="hailo8"):
        from hailo_sdk_client import ClientRunner
        self.runners = []
        self.in_hailo = []    # hailo input-layer names per segment
        self.in_orig = []     # original ONNX tensor names per segment
        self.out_hailo = []   # hailo output-layer names per segment
        self.out_orig = []    # original ONNX tensor names per segment
        for k in range(n_segs):
            path = os.path.join(har_dir, f"s{k}", f"seg{k}_optimized.har")
            if not os.path.isfile(path):
                path = os.path.join(har_dir, f"seg{k}_optimized.har")
            r = ClientRunner(har=path, hw_arch=hw_arch)
            hn = r.get_hn()
            if isinstance(hn, str):
                hn = json.loads(hn)
            ins, ins_o, outs, outs_o = [], [], [], []
            for name, v in hn["layers"].items():
                t = v.get("type")
                orig = (v.get("original_names") or [name])[0]
                if t == "input_layer":
                    ins.append(name)
                    ins_o.append(orig)
                elif t == "output_layer":
                    outs.append(name)
                    outs_o.append(orig)
            self.runners.append(r)
            self.in_hailo.append(ins)
            self.in_orig.append(ins_o)
            self.out_hailo.append(outs)
            self.out_orig.append(outs_o)
        # global boundary map: original ONNX tensor name -> emulator array
        self._boundaries = {}

    def run(self, image_u8, context):
        """Runs the chain on one uint8 HWC image and returns the final logits."""
        env = {self.in_hailo[0][0]: image_u8.astype(np.float32)[None]}
        for k, r in enumerate(self.runners):
            with r.infer_context(context) as ctx:
                out = r.infer(ctx, env)
            if not isinstance(out, (list, tuple)):
                out = [out]
            out = [np.asarray(o) for o in out]
            for _hl, orig, arr in zip(self.out_hailo[k], self.out_orig[k], out):
                self._boundaries[orig] = arr
            if k + 1 < len(self.runners):
                env = {h: self._boundaries[o] for h, o in
                       zip(self.in_hailo[k + 1], self.in_orig[k + 1])}
            else:
                return out[0]
        raise RuntimeError("chain did not produce an output")


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def compare(ref, cand):
    """ref, cand: (1,150,128,128).  Returns (cosine, argmax agreement, max|diff|)."""
    a = ref.reshape(-1, ref.shape[1]).astype(np.float64)
    b = cand.reshape(-1, cand.shape[1]).astype(np.float64)
    cos = float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    agree = float((a.argmax(1) == b.argmax(1)).mean())
    mad = float(np.abs(a - b).max())
    return cos, agree, mad


def _to_nchw(arr, n_cls=150):
    """Emulator returns NHWC, ORT returns NCHW.  Bring both to NCHW."""
    arr = np.asarray(arr)
    if arr.ndim == 4 and arr.shape[-1] == n_cls and arr.shape[1] != n_cls:
        arr = arr.transpose(0, 3, 1, 2)
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--har-dir", default="/root/autodl-tmp/scratch_teacher/work2")
    ap.add_argument("--onnx", default="/root/deploy_artifacts/hailo/teacher_fix/teacher_rw.onnx")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--context", default="SDK_QUANTIZED",
                    choices=["SDK_QUANTIZED", "SDK_FP_OPTIMIZED"])
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from hailo_sdk_client import InferenceContext

    imgs, names = load_val_images(args.n)
    print(f"loaded {len(imgs)} validation images", flush=True)

    ref = ort_logits(args.onnx, imgs)
    print(f"ORT FP32 logits {ref[0].shape} "
          f"range=[{ref[0].min():.3f},{ref[0].max():.3f}]", flush=True)

    chain = SegmentChain(args.har_dir)
    for k in range(len(chain.runners)):
        print(f"seg{k}: in={chain.in_hailo[k]} -> {chain.in_orig[k]}\n"
              f"      out={chain.out_hailo[k]} -> {chain.out_orig[k]}", flush=True)

    ctx = getattr(InferenceContext, args.context)
    rows = []
    t0 = time.time()
    for i, (u8, nm) in enumerate(zip(imgs, names)):
        cand = _to_nchw(chain.run(u8, ctx))
        r = _to_nchw(ref[i])
        cos, agree, mad = compare(r, cand)
        rows.append(dict(name=nm, cos=cos, argmax_agree=agree, max_abs_diff=mad))
        print(f"  {nm}: cos={cos:.6f} argmax={agree*100:.2f}% mad={mad:.4f}", flush=True)
    dt = time.time() - t0

    cos = np.array([r["cos"] for r in rows])
    agree = np.array([r["argmax_agree"] for r in rows])
    summary = dict(
        context=args.context,
        n_images=len(rows),
        cos_mean=float(cos.mean()), cos_min=float(cos.min()),
        cos_max=float(cos.max()),
        argmax_agree_mean=float(agree.mean()),
        threshold=0.9,
        pass_=bool(cos.mean() >= 0.9),
        elapsed_s=round(dt, 1),
        per_image=rows,
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "per_image"}, indent=2))
    out = args.out or os.path.join(args.har_dir, f"eval_{args.context}.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
