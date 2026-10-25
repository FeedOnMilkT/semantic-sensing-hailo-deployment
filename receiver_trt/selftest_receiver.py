#!/usr/bin/env python3
"""Self-test for the TensorRT receiver.

Checks
------

1. **Codec vs the sender's own reconstruction** (the check that matters).
   The shipped bitstreams were encoded by the C++ sender with the *C++ h_s*
   means.  The receiver must therefore reconstruct exactly

       y_hat = round(y - means_cpp) + means_cpp

   where ``means_cpp`` is the deterministic h_s the codec computes from the
   bitstream's own z-stream.  This comparison must be EXACT (0.0 difference).

2. **Decoded z_hat vs the Hailo sender dump** -- proves the bitstream carries
   the Hailo sender's latent (the z-stream must be bit-identical).

3. **C++ h_s vs the PyTorch FP32 h_s** on the SAME z_hat -- this quantifies the
   known hyper-synthesis gap (the C++ h_s and the FP32 PyTorch h_s disagree on
   ~17% of the CDF "档位", which is exactly why the receiver must NOT
   recompute h_s in the network).

4. **C++ h_s vs the Hailo int8 h_s** -- the same gap for the Hailo arm.

5. **TRT vs ONNX Runtime** -- the TensorRT logits are compared against the ORT
   logits of ``receiver.onnx`` for the SAME ``y_hat`` (cosine + argmax).

6. **FP32 vs FP16 engine** -- the two engines on the same inputs.

7. **Label-map equivalence** -- the fused CUDA post-processing kernel must
   produce exactly the same labels as the numpy reference.

Usage
-----
    /root/autodl-tmp/venvs/trt/bin/python receiver_trt/selftest_receiver.py \
        --n 8 --out-json /root/autodl-tmp/deploy_artifacts/package/receiver/selftest.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from postprocess import make_postprocess, numpy_label_map   # noqa: E402
from receiver_trt import (DEFAULT_LIB, DEFAULT_RECEIVER, DEFAULT_SENDER,  # noqa: E402
                          DEFAULT_TABLES, SsCodec)

DEFAULT_FP32_ENGINE = os.path.join(DEFAULT_RECEIVER, "receiver_fp32.engine")
DEFAULT_FP16_ENGINE = os.path.join(DEFAULT_RECEIVER, "receiver_fp16.engine")
DEFAULT_ONNX = os.path.join(DEFAULT_RECEIVER, "receiver.onnx")
HAILO_CHUNKS = "/dev/shm/hailo2000/hailo"
REF_DIR = "/dev/shm/hailo2000/ref"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def cosine(a, b) -> float:
    a = np.asarray(a, np.float64).reshape(-1)
    b = np.asarray(b, np.float64).reshape(-1)
    den = np.linalg.norm(a) * np.linalg.norm(b)
    if den < 1e-12:
        return 1.0 if np.allclose(a, b) else 0.0
    return float((a @ b) / den)


def compare(ref, got):
    ref = np.asarray(ref, np.float32)
    got = np.asarray(got, np.float32)
    diff = np.abs(ref - got)
    return {
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "cosine": cosine(ref, got),
        "argmax_agree": float((ref.argmax(-1) == got.argmax(-1)).mean()),
    }


def summarize(rows, keys=("cosine", "argmax_agree", "max_abs_diff", "mean_abs_diff")):
    out = {}
    for k in keys:
        vals = [r[k] for r in rows]
        out[k] = {"mean": float(np.mean(vals)), "min": float(np.min(vals)),
                  "max": float(np.max(vals))}
    return out


def load_hailo_chunks(n):
    """The Hailo emulator dump: y / z_hat / means / scales per image."""
    files = sorted(glob.glob(os.path.join(HAILO_CHUNKS, "chunk_*.npz")))
    if not files:
        return None
    ys, zs, ms = [], [], []
    for f in files:
        d = np.load(f)
        ys.append(d["y"])
        zs.append(d["z_hat"])
        ms.append(d["means"])
        if sum(a.shape[0] for a in ys) >= n:
            break
    return (np.concatenate(ys, 0)[:n], np.concatenate(zs, 0)[:n],
            np.concatenate(ms, 0)[:n])


def load_ref(n):
    """The PyTorch FP32 sender reference: y / z_hat / means."""
    ry = os.path.join(REF_DIR, "ref_y.npy")
    rm = os.path.join(REF_DIR, "ref_means.npy")
    rz = os.path.join(REF_DIR, "ref_z_hat.npy")
    if not (os.path.isfile(ry) and os.path.isfile(rm) and os.path.isfile(rz)):
        return None
    return (np.load(ry, mmap_mode="r")[:n], np.load(rz, mmap_mode="r")[:n],
            np.load(rm, mmap_mode="r")[:n])


# --------------------------------------------------------------------------- #
# check 1-4: the codec
# --------------------------------------------------------------------------- #
def check_codec(codec, sender_dir, n):
    man = json.load(open(os.path.join(sender_dir, "manifest.json")))
    bit_dir = os.path.join(sender_dir, "bitstreams")
    hailo = load_hailo_chunks(n)
    ref = load_ref(n)

    out = {"n_images": min(n, len(man["images"]))}

    # ---- 1) the decoded y_hat vs the sender's own reconstruction ------------
    rows = []
    y_hats = []
    for i in range(out["n_images"]):
        path = os.path.join(bit_dir, man["images"][i]["file"])
        y, _ = codec.decode_file(path)
        y_hats.append(y.astype(np.float32))
        z, means_cpp, scales_cpp, indexes = codec.decode_z_and_hs(open(path, "rb").read())
        # the sender's reconstruction, using the SAME (C++) h_s means
        if hailo is not None:
            y_hailo = hailo[0][i]
            y_hat_ref = np.round(y_hailo - means_cpp[0]) + means_cpp[0]
            c = compare(y_hat_ref, y[0])
            sym_dec = np.round(y[0] - means_cpp[0])
            sym_ref = np.round(y_hailo - means_cpp[0])
            c["symbol_mismatches"] = int((sym_dec != sym_ref).sum())
            c["n_symbols"] = int(sym_ref.size)
            c["name"] = man["images"][i]["name"]
            rows.append(c)
    if rows:
        out["yhat_vs_sender_reconstruction"] = {
            "description": "decoded y_hat vs round(y_hailo - means_cpp) + means_cpp "
                           "(the sender's own reconstruction, exact match expected)",
            "per_image": rows,
            "summary": summarize(rows, ("cosine", "argmax_agree", "max_abs_diff",
                                         "mean_abs_diff")),
            "total_symbol_mismatches": int(sum(r["symbol_mismatches"] for r in rows)),
        }

    # ---- 2) the decoded z_hat vs the Hailo sender dump ---------------------
    if hailo is not None:
        zrows = []
        for i in range(out["n_images"]):
            path = os.path.join(bit_dir, man["images"][i]["file"])
            z, means_cpp, scales_cpp, indexes = codec.decode_z_and_hs(open(path, "rb").read())
            c = compare(hailo[1][i], z[0])
            c["name"] = man["images"][i]["name"]
            zrows.append(c)
        out["zhat_vs_hailo_sender"] = {
            "description": "z-stream decoded from the bitstream vs the Hailo "
                           "sender dump's z_hat (must be bit-identical)",
            "per_image": zrows,
            "summary": summarize(zrows),
        }

    # ---- 3) the C++ h_s vs the PyTorch FP32 h_s (same z_hat) --------------
    if ref is not None:
        prows = []
        for i in range(min(n, len(ref[1]))):
            z = np.ascontiguousarray(np.asarray(ref[1][i], np.float32))[None]
            means_cpp, scales_cpp, indexes = codec.hyper_synthesis(z)
            c = compare(ref[2][i], means_cpp[0])
            c["name"] = f"ref_{i:04d}"
            prows.append(c)
        out["cpp_hs_vs_pytorch_hs"] = {
            "description": "C++ h_s means vs the PyTorch FP32 h_s means on the "
                           "SAME z_hat (the known h_s gap -- the reason the "
                           "receiver must not recompute h_s in the graph)",
            "per_image": prows,
            "summary": summarize(prows),
        }

    # ---- 4) the C++ h_s vs the Hailo int8 h_s ------------------------------
    if hailo is not None:
        hrows = []
        for i in range(out["n_images"]):
            path = os.path.join(bit_dir, man["images"][i]["file"])
            z, means_cpp, scales_cpp, indexes = codec.decode_z_and_hs(open(path, "rb").read())
            c = compare(hailo[2][i], means_cpp[0])
            c["name"] = man["images"][i]["name"]
            hrows.append(c)
        out["cpp_hs_vs_hailo_int8_hs"] = {
            "description": "C++ h_s means vs the Hailo int8 h_s means (the "
                           "~17% CDF-index disagreement documented in cpp/README.md)",
            "per_image": hrows,
            "summary": summarize(hrows),
        }

    return out, y_hats


# --------------------------------------------------------------------------- #
# check 5-6: the engines
# --------------------------------------------------------------------------- #
def check_engines(onnx_path, fp32_engine, fp16_engine, y_hats):
    import onnxruntime as ort
    from receiver_trt import TrtEngine

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])

    e32 = TrtEngine(fp32_engine) if os.path.isfile(fp32_engine) else None
    e16 = TrtEngine(fp16_engine) if os.path.isfile(fp16_engine) else None

    out = {"ort_vs_trt_fp32": [], "ort_vs_trt_fp16": [], "trt_fp32_vs_trt_fp16": []}
    try:
        for y in y_hats:
            ort_out = sess.run(["logits"], {"y_hat": y})[0]
            t32 = e32.infer(y) if e32 is not None else None
            t16 = e16.infer(y) if e16 is not None else None
            if t32 is not None:
                out["ort_vs_trt_fp32"].append(compare(ort_out, t32))
            if t16 is not None:
                out["ort_vs_trt_fp16"].append(compare(ort_out, t16))
            if t32 is not None and t16 is not None:
                out["trt_fp32_vs_trt_fp16"].append(compare(t32, t16))
    finally:
        if e32 is not None:
            e32.close()
        if e16 is not None:
            e16.close()

    for k in list(out):
        if out[k]:
            out[k + "_summary"] = summarize(out[k])
    return out


# --------------------------------------------------------------------------- #
# check 7: the post-processing kernel
# --------------------------------------------------------------------------- #
def check_postprocess(logits):
    pp = make_postprocess("auto")
    try:
        cuda_labels = pp.label_map(logits, 512)
    finally:
        pp.close()
    numpy_labels = numpy_label_map(logits, 512)
    return {
        "implementation": type(pp).__name__,
        "identical": bool(np.array_equal(cuda_labels, numpy_labels)),
        "n_different_pixels": int((cuda_labels != numpy_labels).sum()),
    }


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sender-dir", default=DEFAULT_SENDER)
    ap.add_argument("--lib", default=DEFAULT_LIB)
    ap.add_argument("--tables", default=DEFAULT_TABLES)
    ap.add_argument("--onnx", default=DEFAULT_ONNX)
    ap.add_argument("--fp32-engine", default=DEFAULT_FP32_ENGINE)
    ap.add_argument("--fp16-engine", default=DEFAULT_FP16_ENGINE)
    ap.add_argument("--n", type=int, default=8, help="number of images to check")
    ap.add_argument("--out-json", default=None)
    a = ap.parse_args(argv)

    report = {"n_images": a.n, "sender_dir": os.path.abspath(a.sender_dir)}

    codec = SsCodec(a.lib, a.tables)
    try:
        report["codec_version"] = codec.version
        codec_report, y_hats = check_codec(codec, a.sender_dir, a.n)
        report.update(codec_report)

        print("[selftest] codec:")
        if "yhat_vs_sender_reconstruction" in codec_report:
            v = codec_report["yhat_vs_sender_reconstruction"]
            print(f"  y_hat vs sender reconstruction : cosine="
                  f"{v['summary']['cosine']['mean']:.9f} max|dy|="
                  f"{v['summary']['max_abs_diff']['max']:.3e} "
                  f"symbol mismatches={v['total_symbol_mismatches']}")
        if "zhat_vs_hailo_sender" in codec_report:
            v = codec_report["zhat_vs_hailo_sender"]
            print(f"  z_hat vs Hailo sender         : max|dz|="
                  f"{v['summary']['max_abs_diff']['max']:.3e}")
        if "cpp_hs_vs_pytorch_hs" in codec_report:
            v = codec_report["cpp_hs_vs_pytorch_hs"]
            print(f"  C++ h_s vs PyTorch FP32 h_s    : max|d|="
                  f"{v['summary']['max_abs_diff']['max']:.4e} "
                  f"mean|d|={v['summary']['mean_abs_diff']['mean']:.4e}")
        if "cpp_hs_vs_hailo_int8_hs" in codec_report:
            v = codec_report["cpp_hs_vs_hailo_int8_hs"]
            print(f"  C++ h_s vs Hailo int8 h_s      : max|d|="
                  f"{v['summary']['max_abs_diff']['max']:.4e}")
    finally:
        codec.close()

    # ---- engines -----------------------------------------------------------
    if os.path.isfile(a.onnx):
        report["engine_parity"] = check_engines(a.onnx, a.fp32_engine,
                                                a.fp16_engine, y_hats)
        print("[selftest] engines (TRT vs ONNX Runtime):")
        for k in ("ort_vs_trt_fp32_summary", "ort_vs_trt_fp16_summary",
                  "trt_fp32_vs_trt_fp16_summary"):
            if k in report["engine_parity"]:
                v = report["engine_parity"][k]
                print(f"  {k[:-7]:24s} cosine={v['cosine']['mean']:.9f} "
                      f"argmax={v['argmax_agree']['mean']:.6f} "
                      f"max|d|={v['max_abs_diff']['max']:.6g}")

    # ---- post-processing ---------------------------------------------------
    from receiver_trt import TrtEngine
    if os.path.isfile(a.fp32_engine) and y_hats:
        e = TrtEngine(a.fp32_engine)
        try:
            logits = e.infer(y_hats[0])
        finally:
            e.close()
        report["postprocess_check"] = check_postprocess(logits)
        print(f"[selftest] postprocess: {report['postprocess_check']}")

    if a.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(a.out_json)), exist_ok=True)
        with open(a.out_json, "w") as f:
            json.dump(report, f, indent=1)
        print(f"[selftest] report -> {a.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
