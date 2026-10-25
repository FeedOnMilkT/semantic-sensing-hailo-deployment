#!/usr/bin/env python3
"""Cross-validate libsscodec against CompressAI 1.2.8.

CHECK 1 (z stream)
------------------
The factorized-prior (z) stream does NOT depend on h_s: it only needs the
EntropyBottleneck CDFs and the medians.  So the C++ encoder must produce
BYTE-IDENTICAL output to `EntropyBottleneck.compress(z)`.

CHECK 2 (y stream)
------------------
The conditional (y) stream depends on the CDF index of every element, which
comes from h_s.  To compare against CompressAI independently of h_s, we take
the indexes (and means) that PYTHON computed -- via the real h_s -- and feed
them to BOTH encoders.  The bytes must then be identical.

CHECK 3 (round trip)
--------------------
The C++ decoder must reproduce the symbols exactly.

Usage (venv-hailo):
    /root/Semantic-sensing/venv-hailo/bin/python cpp/tests/crosscheck_compressai.py \
        --tables cpp/codec_tables.bin --vectors /root/autodl-tmp/work/vectors.bin \
        --n 20
"""
from __future__ import annotations

import argparse
import ctypes
import os
import sys

import numpy as np
import torch

REPO = "/root/Semantic-sensing"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "deploy"))


def load_lib(path):
    lib = ctypes.CDLL(path)
    lib.ss_open.argtypes = [ctypes.c_char_p]
    lib.ss_open.restype = ctypes.c_void_p
    lib.ss_close.argtypes = [ctypes.c_void_p]
    lib.ss_close.restype = None
    lib.ss_version.restype = ctypes.c_char_p
    lib.ss_hyper_synthesis.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int32),
    ]
    lib.ss_hyper_synthesis.restype = ctypes.c_int
    lib.ss_get_medians.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
                                   ctypes.c_size_t]
    lib.ss_get_medians.restype = ctypes.c_int
    lib.ss_encode_z_stream.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.ss_encode_z_stream.restype = ctypes.c_int
    lib.ss_encode_y_stream.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.ss_encode_y_stream.restype = ctypes.c_int
    lib.ss_decode_y_stream.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_int32), ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t,
    ]
    lib.ss_decode_y_stream.restype = ctypes.c_int
    lib.ss_decode_z_stream.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
        ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
    ]
    lib.ss_decode_z_stream.restype = ctypes.c_int
    return lib


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--tables",
                    default="/root/semantic-sensing-hailo-deployment/cpp/codec_tables.bin")
    ap.add_argument("--lib", default="/root/autodl-tmp/build/O2/libsscodec.so")
    ap.add_argument("--vectors", default="/root/autodl-tmp/work/vectors.bin")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--ckpt",
                    default="/root/Semantic-sensing/ckpt/student_mid_b0.03_s2.pth")
    args = ap.parse_args()

    lib = load_lib(args.lib)
    print("lib version:", lib.ss_version().decode())
    handle = lib.ss_open(args.tables.encode())
    if not handle:
        raise SystemExit("ss_open failed")

    # ---- the Python reference model -------------------------------------
    from export_sender import load_student
    student, cfg = load_student(args.ckpt)
    bn = student.bottleneck
    with torch.no_grad():
        bn.gaussian_conditional.update_scale_table(
            torch.exp(torch.linspace(np.log(0.11), np.log(256.0), 64)), force=True)
        bn.entropy_bottleneck.update(force=True)
    eb = bn.entropy_bottleneck
    gc = bn.gaussian_conditional
    medians_t = eb._get_medians().detach().reshape(-1).float()

    # ---- load the test vectors ------------------------------------------
    y_n = 192 * 16 * 16
    z_n = 128 * 8 * 8
    # NOTE: the vectors file is a RAW float32 stream, so the per-image stride
    # is the ELEMENT count (y_n + z_n), not the byte count.
    per = y_n + z_n
    raw = np.fromfile(args.vectors, dtype=np.float32)
    n = min(args.n, raw.size // per)

    # ---- check the medians agree -----------------------------------------
    med_c = np.zeros(128, np.float32)
    lib.ss_get_medians(handle, med_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 128)
    print(f"medians max|C++ - PyTorch| = "
          f"{np.abs(med_c - medians_t.numpy()).max():.3e}")

    z_ok_bytes = y_ok_bytes = rt_ok = 0
    max_hs_diff = 0.0
    max_index_diff = 0

    for i in range(n):
        base = i * per
        y = raw[base:base + y_n].reshape(1, 192, 16, 16)
        z_hat = raw[base + y_n:base + y_n + z_n].reshape(1, 128, 8, 8)

        # ---- the reference means/scales/indexes from PyTorch -------------
        with torch.no_grad():
            zt = torch.from_numpy(z_hat)
            hs = bn.h_s(zt)
            means_t, scales_t = hs.chunk(2, dim=1)
            if means_t.shape[-2:] != (16, 16):
                import torch.nn.functional as F
                means_t = F.interpolate(means_t, size=(16, 16), mode="bilinear",
                                        align_corners=False)
                scales_t = F.interpolate(scales_t, size=(16, 16), mode="bilinear",
                                         align_corners=False)
            indexes_t = gc.build_indexes(scales_t).int().reshape(-1)

        # ---- the C++ means/scales/indexes --------------------------------
        means_c = np.zeros(y_n, np.float32)
        scales_c = np.zeros(y_n, np.float32)
        indexes_c = np.zeros(y_n, np.int32)
        rc = lib.ss_hyper_synthesis(
            handle,
            np.ascontiguousarray(z_hat.reshape(-1)).ctypes.data_as(
                ctypes.POINTER(ctypes.c_float)), 8, 8,
            means_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            scales_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            indexes_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        )
        assert rc == 0, rc
        mdiff = np.abs(means_c - means_t.reshape(-1).numpy()).max()
        sdiff = np.abs(scales_c - scales_t.reshape(-1).numpy()).max()
        max_hs_diff = max(max_hs_diff, float(mdiff), float(sdiff))
        n_idx_diff = int((indexes_c != indexes_t.numpy()).sum())
        max_index_diff = max(max_index_diff, n_idx_diff)

        # ---- CHECK 1: z stream --------------------------------------------
        # CompressAI's EntropyBottleneck.compress(x) codes round(x - median).
        # The deployed sender's z_hat is ALREADY  floor(z_pre + 0.5) + median,
        # so the transmitted symbol is round(z_hat - median) == floor(z_pre+0.5).
        # BOTH encoders therefore get the SAME input (z_hat) and the SAME
        # medians -> the byte streams must be identical.
        # NOTE: compress() returns a LIST of strings (one per batch element).
        z_ref = eb.compress(torch.from_numpy(z_hat).contiguous())[0]
        out = np.zeros(1 << 20, np.uint8)
        n_out = ctypes.c_size_t(0)
        rc = lib.ss_encode_z_stream(
            handle,
            np.ascontiguousarray(z_hat.reshape(-1)).ctypes.data_as(
                ctypes.POINTER(ctypes.c_float)), 8, 8,
            medians_t.numpy().astype(np.float32).ctypes.data_as(
                ctypes.POINTER(ctypes.c_float)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), out.size,
            ctypes.byref(n_out))
        assert rc == 0, rc
        z_bytes = bytes(out[:n_out.value])
        if z_bytes == z_ref:
            z_ok_bytes += 1

        # ---- CHECK 2: y stream with PYTHON indexes ------------------------
        symbols_t = gc.quantize(torch.from_numpy(y).contiguous(), "symbols",
                                means_t).int().reshape(-1).numpy()
        # CompressAI's compress() wants indexes with the SAME SHAPE as the
        # input tensor.
        y_ref = gc.compress(torch.from_numpy(y).contiguous(),
                             indexes_t.reshape(1, 192, 16, 16),
                             means=means_t)[0]
        n_out2 = ctypes.c_size_t(0)
        rc = lib.ss_encode_y_stream(
            handle,
            np.ascontiguousarray(symbols_t, dtype=np.int32).ctypes.data_as(
                ctypes.POINTER(ctypes.c_int32)),
            np.ascontiguousarray(indexes_t.numpy(), dtype=np.int32).ctypes.data_as(
                ctypes.POINTER(ctypes.c_int32)),
            16, 16, out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), out.size,
            ctypes.byref(n_out2))
        assert rc == 0, rc
        y_bytes = bytes(out[:n_out2.value])
        if y_bytes == y_ref:
            y_ok_bytes += 1

        # ---- CHECK 3: decode the C++ y stream ----------------------------
        dec = np.zeros(y_n, np.int32)
        y_bytes_arr = np.frombuffer(y_bytes, dtype=np.uint8)
        rc = lib.ss_decode_y_stream(
            handle, y_bytes_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            len(y_bytes),
            np.ascontiguousarray(indexes_t.numpy(), dtype=np.int32).ctypes.data_as(
                ctypes.POINTER(ctypes.c_int32)),
            16, 16, dec.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), y_n)
        assert rc == 0, rc
        if np.array_equal(dec, symbols_t):
            rt_ok += 1

    lib.ss_close(handle)

    print()
    print(f"images                       : {n}")
    print(f"h_s max|C++ - PyTorch|       : {max_hs_diff:.3e} (means & scales)")
    print(f"h_s max CDF index difference : {max_index_diff} elements "
          f"(of {y_n} per image)")
    print(f"z stream byte-identical     : {z_ok_bytes}/{n}")
    print(f"y stream byte-identical     : {y_ok_bytes}/{n}")
    print(f"y round-trip exact           : {rt_ok}/{n}")

    ok = (z_ok_bytes == n) and (y_ok_bytes == n) and (rt_ok == n)
    print()
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())