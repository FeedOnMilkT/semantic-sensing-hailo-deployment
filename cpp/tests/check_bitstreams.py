#!/usr/bin/env python3
"""Cross-check the C++ sender's RTFF bitstreams against CompressAI 1.2.8.

For every image it takes the REAL sender tensors (y, z_hat from the Hailo
emulator dump) and compares:

  1. the z stream: the C++ bytes vs `EntropyBottleneck.compress(z_hat)`.
     The z stream does NOT depend on h_s, so it must be byte-identical.
  2. the y stream: the C++ bytes vs `GaussianConditional.compress(y, indexes,
     means=means)` where `indexes`/`means` are the ones the C++ actually used
     (queried through ss_hyper_synthesis).  This tests the rANS coder itself,
     independently of the h_s implementation.
  3. the whole container: the C++ .rtff bytes vs a container assembled by
     Python from the same two streams (byte-for-byte).
  4. the round trip: the C++ decoder reproduces round(y - means) exactly.

It ALSO reports how many CDF indexes the C++ h_s and the PyTorch h_s disagree
on.  A handful of disagreements is EXPECTED and harmless: the deployed
bitstream is defined by the C++ h_s, and the RECEIVER also uses the C++ h_s
(ss_hyper_synthesis), so both ends agree bit-exactly.  See cpp/README.md.

Usage (venv-hailo):
    /root/Semantic-sensing/venv-hailo/bin/python \
        cpp/tests/check_bitstreams.py \
        --rtff-dir /root/autodl-tmp/deploy_artifacts/package/sender_out/bitstreams \
        --vectors /root/autodl-tmp/work/vectors250.bin --n 25
"""
from __future__ import annotations

import argparse
import ctypes
import os
import struct
import sys

import numpy as np
import torch

REPO = "/root/Semantic-sensing"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "deploy"))

Y_C, Y_H, Y_W = 192, 16, 16
Z_C, Z_H, Z_W = 128, 8, 8


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
        ctypes.POINTER(ctypes.c_int32)]
    lib.ss_hyper_synthesis.restype = ctypes.c_int
    lib.ss_decode.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.POINTER(ctypes.c_int)]
    lib.ss_decode.restype = ctypes.c_int
    lib.ss_decode_z.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.POINTER(ctypes.c_int)]
    lib.ss_decode_z.restype = ctypes.c_int
    return lib


def parse_rtff(data):
    assert data[:4] == b"RTFF", "bad magic"
    z_h, z_w, y_h, y_w, fh_h, fh_w, fl_h, fl_w, B = struct.unpack_from(">HHHHHHHHB", data, 4)
    off = 21
    streams = []
    for _ in range(2):
        (ln,) = struct.unpack_from(">I", data, off)
        off += 4
        streams.append(data[off:off + ln])
        off += ln
    return (z_h, z_w, y_h, y_w, fh_h, fh_w, fl_h, fl_w, B), streams


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--rtff-dir",
                    default="/root/autodl-tmp/deploy_artifacts/package/sender_out/bitstreams")
    ap.add_argument("--tables",
                    default="/root/semantic-sensing-hailo-deployment/cpp/codec_tables.bin")
    ap.add_argument("--lib", default="/root/autodl-tmp/build/O2/libsscodec.so")
    ap.add_argument("--vectors", default="/root/autodl-tmp/work/vectors250.bin")
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--ckpt",
                    default="/root/Semantic-sensing/ckpt/student_mid_b0.03_s2.pth")
    args = ap.parse_args()

    lib = load_lib(args.lib)
    handle = lib.ss_open(args.tables.encode())
    if not handle:
        raise SystemExit("ss_open failed")

    from export_sender import load_student
    student, cfg = load_student(args.ckpt)
    bn = student.bottleneck
    with torch.no_grad():
        bn.gaussian_conditional.update_scale_table(
            torch.exp(torch.linspace(np.log(0.11), np.log(256.0), 64)), force=True)
        bn.entropy_bottleneck.update(force=True)
    eb, gc = bn.entropy_bottleneck, bn.gaussian_conditional

    names = sorted(f for f in os.listdir(args.rtff_dir) if f.endswith(".rtff"))

    y_n = Y_C * Y_H * Y_W
    z_n = Z_C * Z_H * Z_W
    # NOTE: the vectors file is a RAW float32 stream, so the per-image stride
    # is the ELEMENT count (y_n + z_n), not the byte count.
    per = y_n + z_n
    raw = np.fromfile(args.vectors, dtype=np.float32)
    n = min(args.n, raw.size // per, len(names))

    z_ok = y_ok = cont_ok = rt_ok = 0
    max_bpp_diff = 0.0
    py_bpp_sum = 0.0
    cpp_bpp_sum = 0.0
    idx_diff_total = 0
    hs_diff_max = 0.0

    for i in range(n):
        base = i * per
        y = raw[base:base + y_n].reshape(1, Y_C, Y_H, Y_W)
        z_hat = raw[base + y_n:base + y_n + z_n].reshape(1, Z_C, Z_H, Z_W)

        path = os.path.join(args.rtff_dir, names[i])
        with open(path, "rb") as f:
            data = f.read()
        hdr, streams = parse_rtff(data)
        z_bytes, y_bytes = streams

        # ---- the h_s the C++ actually used --------------------------------
        means_c = np.zeros(y_n, np.float32)
        scales_c = np.zeros(y_n, np.float32)
        idx_c = np.zeros(y_n, np.int32)
        lib.ss_hyper_synthesis(
            handle,
            np.ascontiguousarray(z_hat.reshape(-1)).ctypes.data_as(
                ctypes.POINTER(ctypes.c_float)), Z_H, Z_W,
            means_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            scales_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            idx_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
        idx_c_t = torch.from_numpy(idx_c.reshape(1, Y_C, Y_H, Y_W))
        means_c_t = torch.from_numpy(means_c.reshape(1, Y_C, Y_H, Y_W))

        # ---- the python h_s (for reference / diagnostics) -----------------
        with torch.no_grad():
            z_ref = eb.compress(torch.from_numpy(z_hat).contiguous())[0]
            means_t, scales_t = bn.h_s(torch.from_numpy(z_hat)).chunk(2, dim=1)
            if means_t.shape[-2:] != (Y_H, Y_W):
                import torch.nn.functional as F
                means_t = F.interpolate(means_t, size=(Y_H, Y_W), mode="bilinear",
                                        align_corners=False)
                scales_t = F.interpolate(scales_t, size=(Y_H, Y_W), mode="bilinear",
                                         align_corners=False)
            idx_py = gc.build_indexes(scales_t).int()
            hs_diff_max = max(hs_diff_max,
                              float(np.abs(means_t.numpy() - means_c.reshape(1, Y_C, Y_H, Y_W)).max()),
                              float(np.abs(scales_t.numpy() - scales_c.reshape(1, Y_C, Y_H, Y_W)).max()))
        idx_diff_total += int((idx_c_t != idx_py).sum())

        # ---- CHECK 1: the z stream (h_s independent) ----------------------
        if z_bytes == z_ref:
            z_ok += 1

        # ---- CHECK 2: the y stream, SAME indexes --------------------------
        y_ref = gc.compress(torch.from_numpy(y).contiguous(), idx_c_t,
                             means=means_c_t)[0]
        if y_bytes == y_ref:
            y_ok += 1

        # ---- CHECK 3: the whole container ----------------------------------
        buf = bytearray(b"RTFF")
        buf += struct.pack(">HHHHHHHHB", Z_H, Z_W, Y_H, Y_W, 64, 64, 32, 32, 1)
        for s in (z_ref, y_ref):
            buf += struct.pack(">I", len(s)) + s
        if bytes(buf) == data:
            cont_ok += 1

        # ---- CHECK 4: the C++ round trip -----------------------------------
        arr = np.frombuffer(data, dtype=np.uint8)
        y_hat = np.zeros(y_n, np.float32)
        dims = np.zeros(8, np.int32)
        rc = lib.ss_decode(handle,
                           arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                           len(data),
                           y_hat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                           y_hat.size,
                           dims.ctypes.data_as(ctypes.POINTER(ctypes.c_int)))
        assert rc == 0, rc
        z_dec = np.zeros(z_n, np.float32)
        rc = lib.ss_decode_z(handle,
                             arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                             len(data),
                             z_dec.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                             z_dec.size,
                             dims.ctypes.data_as(ctypes.POINTER(ctypes.c_int)))
        assert rc == 0, rc

        sym = np.round(y.reshape(-1) - means_c)
        err = np.abs(y_hat - (sym + means_c)).max()
        zerr = np.abs(z_dec - z_hat.reshape(-1)).max()
        if err < 1e-4 and zerr < 1e-4:
            rt_ok += 1

        cpp_bpp = len(data) * 8 / (512 * 512)
        py_bpp = (len(z_ref) + len(y_ref)) * 8 / (512 * 512)
        py_bpp_sum += py_bpp
        cpp_bpp_sum += cpp_bpp
        max_bpp_diff = max(max_bpp_diff, abs(cpp_bpp - py_bpp))

    lib.ss_close(handle)

    print()
    print(f"images checked               : {n}")
    print(f"z stream byte-identical      : {z_ok}/{n}")
    print(f"y stream byte-identical      : {y_ok}/{n}  (same indexes)")
    print(f"container byte-identical     : {cont_ok}/{n}")
    print(f"round trip (z+y) exact       : {rt_ok}/{n}")
    print(f"h_s max|C++ - PyTorch|       : {hs_diff_max:.3e}")
    print(f"CDF indexes differing vs Py  : {idx_diff_total} of {n * y_n} "
          f"({100.0 * idx_diff_total / (n * y_n):.4f}%)")
    print(f"avg bpp  C++ : {cpp_bpp_sum / n:.6f}")
    print(f"avg bpp  Py  : {py_bpp_sum / n:.6f}")
    print(f"max per-image bpp difference : {max_bpp_diff:.3e}")

    ok = (z_ok == n and y_ok == n and cont_ok == n and rt_ok == n)
    print()
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
