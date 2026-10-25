#!/usr/bin/env python3
"""Quantized-sim sanity check: Hailo SDK_QUANTIZED vs FP32 ONNXRuntime logits.

Runs the optimized HAR through the DFC's *quantized* simulator (InferenceContext.
SDK_QUANTIZED) and compares the logits against ONNXRuntime on the same images.

Preprocessing contract (matches the model script's on-chip `normalization`):
    uint8 HWC RGB  ->  the HEF/normalization layer does (x-mean)/std with
                        mean=[123.675,116.28,103.53], std=[58.395,57.12,57.375]
    so the host feeds RAW 0..255 NHWC and the DFC applies the normalisation on chip.

Usage (hailo-dfc env, from a scratch dir):
  cd /tmp/eval && /root/miniconda3/envs/hailo-dfc/bin/python \
      /root/semantic-sensing-hailo-deployment/scripts/eval_hailo_quantized.py \
      --har /root/deploy_artifacts/hailo/teacher_fix/teacher_rw_optimized.har \
      --onnx /root/deploy_artifacts/hailo/teacher_fix/teacher_rw.onnx --n 8
"""
from __future__ import annotations

import argparse
import json

import numpy as np

MEAN255 = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD255 = np.array([58.395, 57.12, 57.375], dtype=np.float32)


def load_val_images(n: int, size: int = 512):
    """Real validation images as uint8 HWC RGB (resize to `size`x`size` bilinear)."""
    from PIL import Image
    with open("/root/deploy_artifacts/data/val_manifest.json") as f:
        man = json.load(f)
    items = man["subset"][:n]
    imgs = []
    for it in items:
        im = Image.open(it["image"]).convert("RGB").resize((size, size), Image.BILINEAR)
        imgs.append(np.asarray(im, np.uint8))
    return imgs, [it["name"] for it in items]


def ort_logits(onnx_path: str, imgs):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    outs = []
    for u8 in imgs:
        x = ((u8.astype(np.float32) - MEAN255) / STD255).transpose(2, 0, 1)[None]
        outs.append(sess.run(None, {name: x})[0])
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--har", required=True)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    from hailo_sdk_client import ClientRunner, InferenceContext

    imgs, names = load_val_images(a.n)
    print(f"loaded {len(imgs)} validation images", flush=True)

    ref = ort_logits(a.onnx, imgs)
    print("ORT (FP32) done", flush=True)

    runner = ClientRunner(har=a.har, hw_arch="hailo8")
    results = []
    with runner.infer_context(InferenceContext.SDK_QUANTIZED) as ctx:
        for u8, nm in zip(imgs, names):
            feed = u8[None].astype(np.float32)          # NHWC raw 0..255
            out = runner.infer(ctx, feed)
            arr = np.asarray(out if not isinstance(out, (list, tuple)) else out[0])
            if arr.ndim == 4:                            # NHWC -> NCHW
                arr = arr.transpose(0, 3, 1, 2)
            results.append(arr)
    print("SDK_QUANTIZED done", flush=True)

    rows = []
    for i, (r, q) in enumerate(zip(ref, results)):
        r = r.reshape(-1).astype(np.float64)
        q = q.reshape(-1).astype(np.float64)
        cos = float(r @ q / (np.linalg.norm(r) * np.linalg.norm(q) + 1e-12))
        rel = float(np.abs(r - q).max() / (np.abs(r).max() + 1e-12))
        # argmax agreement per pixel (class dim is 1 in NCHW [1,150,128,128])
        rp = ref[i].reshape(-1, ref[i].shape[1]) if ref[i].ndim == 4 else ref[i]
        qp = results[i].reshape(-1, results[i].shape[1]) if results[i].ndim == 4 else results[i]
        agree = float((rp.argmax(1) == qp.argmax(1)).mean())
        rows.append(dict(name=names[i], cosine=cos, max_rel_diff=rel, argmax_agreement=agree))
        print(f"  {names[i]}: cos={cos:.8f} max_rel={rel:.4e} argmax_agree={agree:.6f}", flush=True)

    summary = dict(
        n=len(rows),
        cosine_min=min(r["cosine"] for r in rows),
        cosine_mean=float(np.mean([r["cosine"] for r in rows])),
        max_rel_diff=max(r["max_rel_diff"] for r in rows),
        argmax_agreement_min=min(r["argmax_agreement"] for r in rows),
        argmax_agreement_mean=float(np.mean([r["argmax_agreement"] for r in rows])),
        per_image=rows,
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "per_image"}, indent=2))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(summary, f, indent=2)
        print("saved", a.out)


if __name__ == "__main__":
    main()
