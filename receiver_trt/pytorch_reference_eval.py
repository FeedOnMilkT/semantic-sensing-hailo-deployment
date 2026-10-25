#!/usr/bin/env python3
"""FP32 PyTorch reference for the SAME bitstreams the TensorRT receiver decodes.

The TRT evaluation decodes the *Hailo-sender* bitstreams, so the reference
that makes the comparison meaningful is the PyTorch FP32 receiver running on
the *same* decoded ``y_hat`` tensors -- not the PyTorch FP32 sender arm.

This script therefore:

    *.rtff --ctypes libsscodec.so--> y_hat
           --PyTorch ReceiverNet (the same net the ONNX was traced from)-->
           logits [1,150,128,128]
           --bilinear 128->512 + argmax--> label map
           --confusion vs gt_labels.npy--> mIoU

It is the ground truth for "what would the deployment give if TensorRT were
replaced by the PyTorch FP32 model?", and it validates the ONNX/TRT export
end to end on the deployed bitstreams.

Run with the TORCH venv (NOT the TRT venv):

    /root/Semantic-sensing/venv-hailo/bin/python \
        receiver_trt/pytorch_reference_eval.py --n 2000 \
        --out-json /root/autodl-tmp/deploy_artifacts/package/receiver/miou_pytorch_fp32.json
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_ROOT = os.environ.get("SEMANTIC_SENSING_ROOT", "/root/Semantic-sensing")
for p in (HERE, TRAIN_ROOT, os.path.join(TRAIN_ROOT, "deploy")):
    if p not in sys.path:
        sys.path.insert(0, p)

DEFAULT_PACKAGE = "/root/autodl-tmp/deploy_artifacts/package"
DEFAULT_SENDER = os.path.join(DEFAULT_PACKAGE, "sender_out")
DEFAULT_RECEIVER = os.path.join(DEFAULT_PACKAGE, "receiver")
NUM_CLASSES = 150
IGNORE_INDEX = 255


# --------------------------------------------------------------------------- #
# the codec (same ctypes binding as receiver_trt.py, duplicated so this file
# runs under the torch venv without pulling in tensorrt)
# --------------------------------------------------------------------------- #
class SsCodec:
    def __init__(self, lib_path, tables_path):
        self._lib = ctypes.CDLL(lib_path)
        self._lib.ss_open.argtypes = [ctypes.c_char_p]
        self._lib.ss_open.restype = ctypes.c_void_p
        self._lib.ss_close.argtypes = [ctypes.c_void_p]
        self._lib.ss_decode.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                        ctypes.c_size_t, ctypes.c_void_p,
                                        ctypes.c_size_t, ctypes.c_void_p]
        self._lib.ss_decode.restype = ctypes.c_int
        self._handle = self._lib.ss_open(tables_path.encode())
        if not self._handle:
            raise RuntimeError(f"ss_open({tables_path}) failed")

    def decode(self, payload):
        y_hat = np.empty(192 * 16 * 16, dtype=np.float32)
        dims = (ctypes.c_int * 8)()
        src = ctypes.create_string_buffer(payload, len(payload))
        rc = self._lib.ss_decode(self._handle,
                                 ctypes.cast(src, ctypes.c_void_p), len(payload),
                                 y_hat.ctypes.data_as(ctypes.c_void_p),
                                 y_hat.size, dims)
        if rc != 0:
            raise RuntimeError(f"ss_decode failed rc={rc}")
        zh, zw, yh, yw = int(dims[0]), int(dims[1]), int(dims[2]), int(dims[3])
        return y_hat[:192 * yh * yw].reshape(1, 192, yh, yw)

    def close(self):
        if self._handle:
            self._lib.ss_close(self._handle)
            self._handle = None


# --------------------------------------------------------------------------- #
# metric (same as src/utils/metrics.py SegMeter)
# --------------------------------------------------------------------------- #
def confusion(pred, gt, num_classes=NUM_CLASSES, ignore=IGNORE_INDEX):
    valid = gt != ignore
    p = pred[valid].reshape(-1).astype(np.int64)
    g = gt[valid].reshape(-1).astype(np.int64)
    idx = g * num_classes + p
    return np.bincount(idx, minlength=num_classes ** 2).reshape(num_classes, num_classes)


def miou_from_conf(conf):
    conf = np.asarray(conf, np.float64)
    tp = np.diag(conf)
    union = conf.sum(0) + conf.sum(1) - tp
    present = union > 0
    if not present.any():
        return float("nan"), float("nan"), present
    return (float((tp[present] / union[present]).mean()),
            float(tp.sum() / max(conf.sum(), 1.0)), present)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sender-dir", default=DEFAULT_SENDER)
    ap.add_argument("--lib", default=os.path.join(DEFAULT_SENDER, "libsscodec.so"))
    ap.add_argument("--tables", default=os.path.join(DEFAULT_SENDER, "codec_tables.bin"))
    ap.add_argument("--ckpt", default=os.path.join(TRAIN_ROOT, "ckpt", "student_mid_b0.03_s2.pth"))
    ap.add_argument("--n", type=int, default=0, help="0 = all")
    ap.add_argument("--out-size", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--save-conf", default=None)
    a = ap.parse_args(argv)

    from export_receiver_onnx import ReceiverNet      # the traced net
    from export_sender import load_student

    device = torch.device(a.device)
    student, _ = load_student(a.ckpt)
    student = student.to(device).eval()
    net = ReceiverNet(student, out_size=a.out_size, batch=1).to(device).eval()

    man = json.load(open(os.path.join(a.sender_dir, "manifest.json")))
    images = man["images"]
    if a.n:
        images = images[:a.n]
    gt = np.load(os.path.join(a.sender_dir, "gt_labels.npy"), mmap_mode="r")
    bit_dir = os.path.join(a.sender_dir, "bitstreams")

    codec = SsCodec(a.lib, a.tables)
    conf_total = np.zeros((NUM_CLASSES, NUM_CLASSES), np.float64)
    per_img = []
    dec_ms, inf_ms = [], []
    t0 = time.perf_counter()
    with torch.no_grad():
        for i, it in enumerate(images):
            with open(os.path.join(bit_dir, it["file"]), "rb") as f:
                payload = f.read()
            t1 = time.perf_counter()
            y_hat = codec.decode(payload)
            t2 = time.perf_counter()
            logits = net(torch.from_numpy(y_hat).to(device)).float()
            t3 = time.perf_counter()
            if logits.shape[-2:] != gt.shape[-2:]:
                logits = F.interpolate(logits, size=gt.shape[-2:], mode="bilinear",
                                       align_corners=False)
            pred = logits.argmax(1)[0].cpu().numpy()
            c = confusion(pred, np.asarray(gt[i]))
            conf_total += c
            tp = np.diag(c.astype(np.float64))
            union = c.sum(0) + c.sum(1) - tp
            present = union > 0
            per_img.append(float((tp[present] / union[present]).mean())
                           if present.any() else float("nan"))
            dec_ms.append((t2 - t1) * 1e3)
            inf_ms.append((t3 - t2) * 1e3)
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(images)}  {time.perf_counter() - t0:.0f}s", flush=True)
    wall = time.perf_counter() - t0
    codec.close()

    miou, pacc, present = miou_from_conf(conf_total)
    pi = np.array(per_img, np.float64)
    result = {
        "note": "PyTorch FP32 receiver on the SAME (Hailo-sender) bitstreams "
                "the TensorRT receiver decodes",
        "ckpt": a.ckpt,
        "sender_dir": os.path.abspath(a.sender_dir),
        "n_images": len(images),
        "miou": miou,
        "pixel_acc": pacc,
        "classes_present": int(present.sum()),
        "per_image_miou_mean": float(np.nanmean(pi)),
        "per_image_miou_std": float(np.nanstd(pi)),
        "latency_ms": {
            "decode": {"mean": float(np.mean(dec_ms)), "median": float(np.median(dec_ms))},
            "inference": {"mean": float(np.mean(inf_ms)), "median": float(np.median(inf_ms))},
        },
        "wall_seconds": wall,
        "torch": torch.__version__,
        "device": str(device),
    }
    print(json.dumps(result, indent=1))
    if a.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(a.out_json)), exist_ok=True)
        with open(a.out_json, "w") as f:
            json.dump(result, f, indent=1)
        print(f"summary -> {a.out_json}")
    if a.save_conf:
        np.save(a.save_conf, conf_total)
        print(f"confusion -> {a.save_conf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
