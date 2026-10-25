#!/usr/bin/env python3
"""End-to-end mIoU evaluation of the TensorRT receiver on the sender bitstreams.

For every image in the sender package:

    *.rtff --ctypes libsscodec.so--> y_hat --TensorRT--> logits
           --bilinear 128->512--> argmax --> predicted label map
           --confusion matrix vs gt_labels.npy--> mIoU

The metric follows the training / evaluation code exactly
(``/root/Semantic-sensing/src/utils/metrics.py``, class ``SegMeter``):

    * the logits are bilinearly resized to the GT resolution with
      ``align_corners=False``,
    * ``pred = logits.argmax(1)``,
    * pixels whose GT label is ``ignore_index`` (255) are dropped,
    * the confusion matrix is accumulated over the whole set and
      ``mIoU`` is the mean IoU over the classes present in the GT
      (``union > 0``).

The script also reports per-image and per-stage latency (decode / inference /
post-processing) and, optionally, compares against the FP32 PyTorch reference
mIoU that was measured by the training repo's own dump script.

Usage
-----
    /root/autodl-tmp/venvs/trt/bin/python receiver_trt/eval_miou.py \
        --engine /root/autodl-tmp/deploy_artifacts/package/receiver/receiver_fp32.engine \
        --sender-dir /root/autodl-tmp/deploy_artifacts/package/sender_out \
        --out-json /root/autodl-tmp/deploy_artifacts/package/receiver/miou_fp32.json

    # only the first 200 images
    ... --limit 200
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from receiver_trt import (DEFAULT_LIB, DEFAULT_PACKAGE, DEFAULT_RECEIVER,  # noqa: E402
                          DEFAULT_SENDER, DEFAULT_TABLES, GT_SIZE, IGNORE_INDEX,
                          NUM_CLASSES, ReceiverPipeline)

# The FP32 PyTorch references measured by the training repo.
#
# (a) "sender arm" -- Semantic-sensing/deploy/dump_reference_500.py, full
#     2000-image val set.  It reconstructs the receiver input in PyTorch as
#     ``round(y - means) + means`` instead of decoding the rANS payload.
# (b) "receiver arm" -- receiver_trt/pytorch_reference_eval.py, the PyTorch
#     ``ReceiverNet`` on the *same* libsscodec.so-decoded bitstreams the
#     TensorRT receiver consumes.  This is the directly comparable one.
FP32_PYTORCH_REFERENCE_SENDER_ARM = {
    "source": "/dev/shm/hailo2000/ref/fp32_summary.json",
    "n": 2000,
    "global_miou": 0.3503605396561383,
    "pixel_acc": 0.7627143723830784,
    "per_image_miou_mean": 0.34906721806952673,
}

FP32_PYTORCH_REFERENCE_RECEIVER_ARM = {
    "source": "/root/autodl-tmp/deploy_artifacts/package/receiver/miou_pytorch_fp32.json",
    "n": 2000,
    "global_miou": 0.35020865384718824,
    "pixel_acc": 0.76223629573636,
    "per_image_miou_mean": 0.35154000944131236,
}


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def confusion(pred: np.ndarray, gt: np.ndarray, num_classes=NUM_CLASSES,
              ignore=IGNORE_INDEX) -> np.ndarray:
    """Per-image confusion matrix, exactly as ``SegMeter.update`` does it."""
    pred = np.asarray(pred).reshape(-1)
    gt = np.asarray(gt).reshape(-1)
    valid = gt != ignore
    p = pred[valid].astype(np.int64)
    g = gt[valid].astype(np.int64)
    idx = g * num_classes + p
    return np.bincount(idx, minlength=num_classes ** 2).reshape(num_classes, num_classes)


def miou_from_conf(conf: np.ndarray):
    """Global mIoU over the classes present in the GT (``union > 0``)."""
    conf = np.asarray(conf, np.float64)
    tp = np.diag(conf)
    union = conf.sum(0) + conf.sum(1) - tp
    present = union > 0
    if not present.any():
        return float("nan"), float("nan"), present
    return (float((tp[present] / union[present]).mean()),
            float(tp.sum() / max(conf.sum(), 1.0)),
            present)


def per_image_miou(conf: np.ndarray):
    """Per-image mIoU over the classes present in that image."""
    tp = np.diag(np.asarray(conf, np.float64))
    union = conf.sum(0) + conf.sum(1) - tp
    present = union > 0
    if not present.any():
        return float("nan")
    return float((tp[present] / union[present]).mean())


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
def load_manifest(sender_dir):
    with open(os.path.join(sender_dir, "manifest.json")) as f:
        man = json.load(f)
    return man


def evaluate(engine_path, sender_dir=DEFAULT_SENDER, limit=0, warmup=3,
             gt_size=GT_SIZE, lib_path=DEFAULT_LIB, tables_path=DEFAULT_TABLES,
             verbose_every=100, label_dump_dir=None):
    man = load_manifest(sender_dir)
    images = man["images"]
    if limit:
        images = images[:limit]

    gt = np.load(os.path.join(sender_dir, "gt_labels.npy"), mmap_mode="r")
    bit_dir = os.path.join(sender_dir, "bitstreams")

    pipe = ReceiverPipeline(engine_path, lib_path=lib_path, tables_path=tables_path,
                            gt_size=gt_size, warmup=warmup)

    conf_total = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)
    per_img_miou = []
    dec_ms, inf_ms, post_ms = [], [], []
    failures = []
    t_start = time.perf_counter()

    for i, it in enumerate(images):
        path = os.path.join(bit_dir, it["file"])
        try:
            res = pipe.infer_bitstream(path)
        except Exception as e:                                 # noqa: BLE001
            failures.append({"name": it["name"], "error": repr(e)})
            continue
        c = confusion(np.asarray(res.label).reshape(gt_size, gt_size),
                      np.asarray(gt[i]))
        conf_total += c
        per_img_miou.append(per_image_miou(c))
        dec_ms.append(res.decode_seconds * 1e3)
        inf_ms.append(res.infer_seconds * 1e3)
        post_ms.append(res.postprocess_seconds * 1e3)

        if label_dump_dir and i < 8:
            os.makedirs(label_dump_dir, exist_ok=True)
            np.save(os.path.join(label_dump_dir, f"{it['name']}.npy"), res.label[0])

        if verbose_every and (i + 1) % verbose_every == 0:
            el = time.perf_counter() - t_start
            print(f"  {i + 1}/{len(images)}  {el:.1f}s  "
                  f"({el / (i + 1) * 1000:.1f} ms/img)", flush=True)

    wall = time.perf_counter() - t_start
    pipe.close()

    miou, pacc, present = miou_from_conf(conf_total)
    pi = np.array(per_img_miou, np.float64)

    def stats(v):
        if len(v) == 0:
            return {}
        a = np.asarray(v, np.float64)
        return {"mean": float(a.mean()), "median": float(np.median(a)),
                "p95": float(np.percentile(a, 95)), "min": float(a.min()),
                "max": float(a.max()), "std": float(a.std())}

    result = {
        "engine": os.path.abspath(engine_path),
        "codec_version": pipe.codec_version,
        "sender_dir": os.path.abspath(sender_dir),
        "n_images": len(images),
        "n_evaluated": len(per_img_miou),
        "n_failed": len(failures),
        "failures": failures[:20],
        "miou": miou,
        "pixel_acc": pacc,
        "classes_present": int(present.sum()),
        "per_image_miou_mean": float(np.nanmean(pi)) if len(pi) else float("nan"),
        "per_image_miou_std": float(np.nanstd(pi)) if len(pi) else float("nan"),
        "latency_ms": {
            "decode": stats(dec_ms),
            "inference": stats(inf_ms),
            "postprocess": stats(post_ms),
            "decode_plus_inference": stats(np.asarray(dec_ms) + np.asarray(inf_ms)),
        },
        "wall_seconds": wall,
        "images_per_second": (len(per_img_miou) / wall) if wall > 0 else 0.0,
        "gt_size": gt_size,
        "num_classes": NUM_CLASSES,
        "ignore_index": IGNORE_INDEX,
        "metric_convention": "SegMeter (src/utils/metrics.py): bilinear "
                             "align_corners=False to GT size, argmax, ignore "
                             "pixels with GT==255, mIoU over classes with union>0",
    }
    return result, conf_total


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", default=os.path.join(DEFAULT_RECEIVER, "receiver_fp32.engine"))
    ap.add_argument("--sender-dir", default=DEFAULT_SENDER)
    ap.add_argument("--lib", default=DEFAULT_LIB)
    ap.add_argument("--tables", default=DEFAULT_TABLES)
    ap.add_argument("--limit", type=int, default=0, help="0 = all images")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--gt-size", type=int, default=GT_SIZE)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--save-conf", default=None, help="save the 150x150 confusion matrix as .npy")
    ap.add_argument("--label-dump-dir", default=None)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    print(f"[eval] engine: {a.engine}", flush=True)
    print(f"[eval] sender: {a.sender_dir}", flush=True)
    result, conf = evaluate(a.engine, sender_dir=a.sender_dir, limit=a.limit,
                             warmup=a.warmup, gt_size=a.gt_size, lib_path=a.lib,
                             tables_path=a.tables,
                             verbose_every=0 if a.quiet else 100,
                             label_dump_dir=a.label_dump_dir)

    result["fp32_pytorch_reference"] = {
        "sender_arm": FP32_PYTORCH_REFERENCE_SENDER_ARM,
        "receiver_arm": FP32_PYTORCH_REFERENCE_RECEIVER_ARM,
    }
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("failures", "latency_ms", "fp32_pytorch_reference")},
                     indent=1))
    print("[eval] latency (ms):")
    for k, v in result["latency_ms"].items():
        if v:
            print(f"  {k:22s} mean={v['mean']:.3f} median={v['median']:.3f} "
                  f"p95={v['p95']:.3f} min={v['min']:.3f} max={v['max']:.3f}")

    if a.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(a.out_json)), exist_ok=True)
        with open(a.out_json, "w") as f:
            json.dump(result, f, indent=1)
        print(f"[eval] summary -> {a.out_json}")
    if a.save_conf:
        np.save(a.save_conf, conf)
        print(f"[eval] confusion -> {a.save_conf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
