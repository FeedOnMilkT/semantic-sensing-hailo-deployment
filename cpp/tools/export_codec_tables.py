#!/usr/bin/env python3
"""Export the codec tables (codec_tables.bin) consumed by libsscodec.

WHAT IS EXPORTED
----------------
Everything the shared codec needs that is NOT computed at runtime:

  1. h_s (the hyper-synthesis) weights and structure, taken from the
     student's checkpoint.  For this model h_s is

         Conv(128->128, k3, p1, no bias)
         BatchNorm2d(128)          (eps = 1e-5)
         ReLU
         Upsample(bilinear, x2)
         Conv(128->128, k3, p1, no bias)
         BatchNorm2d(128)
         ReLU
         Conv(128->384, k3, p1, bias)

     The BatchNorm is stored as (gamma, beta, mean, invstd) with
     invstd = 1/sqrt(var + eps), so the C++ runtime performs NO square root
     and NO division.  The layer sequence is kept EXACTLY as PyTorch has it
     (BN is not folded into the preceding convolution) so the C++ evaluation
     follows the same operation order as the reference.

  2. The GaussianConditional scale table (64 log-spaced entries) and its
     quantised CDF / CDF lengths / offsets.

  3. The EntropyBottleneck quantiles (min/median/max per channel), its
     quantised CDF / CDF lengths / offsets, and the per-channel medians.

  4. The GaussianConditional scale bound (0.11).

  5. The sender's per-channel median vector (the same numbers the host
     rounding uses) -- the EntropyBottleneck medians.

The CDFs are built by calling `update(force=True)` on the real CompressAI
modules, so they are bit-identical to what the Python reference uses.

OUTPUT FORMAT
-------------
See cpp/src/tables_format.h for the exact byte layout (all little-endian).

Usage (in the venv-hailo python):
    /root/Semantic-sensing/venv-hailo/bin/python \
        cpp/tools/export_codec_tables.py \
        --ckpt /root/Semantic-sensing/ckpt/student_mid_b0.03_s2.pth \
        --out  cpp/codec_tables.bin
"""
from __future__ import annotations

import argparse
import hashlib
import os
import struct
import sys

import numpy as np
import torch

REPO = "/root/Semantic-sensing"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "deploy"))

# ---- constants mirroring cpp/src/tables_format.h ------------------------- #
MAGIC = b"SSTB"
FORMAT_VERSION = 1
HEADER_SIZE = 28
LAYER_DESC_SIZE = 56

K_CONV = 1
K_BN = 2
K_UPSAMPLE = 3

F_RELU_AFTER = 1 << 0
F_HAS_BIAS = 1 << 1

T_SCALE_TABLE = 1
T_GAUSS_CDF = 2
T_GAUSS_CDF_LENGTH = 3
T_GAUSS_OFFSET = 4
T_EB_CDF = 5
T_EB_CDF_LENGTH = 6
T_EB_OFFSET = 7
T_MEDIANS = 8
T_QUANTILES = 9
T_SCALE_BOUND = 10

DT_F32 = 0
DT_I32 = 1

# the number of table descriptors written by main()
N_TABLES = 10


# --------------------------------------------------------------------------- #
# h_s extraction
# --------------------------------------------------------------------------- #
def extract_hs_layers(hs_module):
    """Walk a `_Synthesis` module and emit the flat layer descriptor list.

    We rely on the KNOWN structure of `_Synthesis` (hyperprior.py):
        head  = _block(c_in, hidden, 1, act="convbn", inverse=True)
               = Conv2d(c_in, hidden, 3, 1, 1, bias=False) + BatchNorm2d + ReLU
        ups   = ModuleList of (Upsample(bilinear, x2) + _block(hidden, hidden, 1))
        proj  = Conv2d(hidden, c_out, 3, 1, 1)

    `_block(..., act="convbn")` is Conv(bias=False) -> BatchNorm2d -> ReLU.
    The `act` used for h_s is "convbn" (the hyper transforms stay conv+ReLU),
    so there is no GDN/IGDN anywhere in h_s.

    We VERIFY that assumption here (and fail loudly if it does not hold),
    because a GDN would introduce a square root into the runtime path.
    """
    layers = []

    conv, bn, relu = _expect_conv_bn_relu(hs_module.head, "head")
    layers.append(_conv_layer(conv))
    layers.append(_bn_layer(bn, relu))

    for i, up in enumerate(hs_module.ups):
        # up = Sequential(Upsample, Sequential(Conv, BN, ReLU))
        if len(up) != 2:
            raise RuntimeError(f"unexpected h_s ups[{i}] structure: {up}")
        us = up[0]
        if us.__class__.__name__ != "Upsample":
            raise RuntimeError(f"expected Upsample, got {us.__class__.__name__}")
        if us.mode != "bilinear":
            raise RuntimeError(f"expected bilinear upsample, got {us.mode}")
        if us.align_corners is not False:
            raise RuntimeError("expected align_corners=False")
        factor = float(us.scale_factor)
        if factor != int(factor):
            raise RuntimeError("non-integer upsample factor")
        layers.append({
            "kind": K_UPSAMPLE, "flags": 0,
            "in_channels": 0, "out_channels": 0,
            "kernel_h": 0, "kernel_w": 0,
            "stride_h": 0, "stride_w": 0,
            "pad_h": 0, "pad_w": 0,
            "scalar0": factor, "params": np.zeros(0, np.float32),
        })
        conv, bn, relu = _expect_conv_bn_relu(up[1], f"ups[{i}]")
        layers.append(_conv_layer(conv))
        layers.append(_bn_layer(bn, relu))

    proj = hs_module.proj
    if proj.__class__.__name__ != "Conv2d":
        raise RuntimeError("h_s proj is not a Conv2d")
    layers.append(_conv_layer(proj, has_bias=proj.bias is not None))
    return layers


def _expect_conv_bn_relu(seq, where):
    """`_block(..., act='convbn')` == Sequential(Conv2d, BatchNorm2d, ReLU)."""
    if len(seq) != 3:
        raise RuntimeError(f"{where}: expected 3 modules, got {len(seq)}")
    conv, bn, act = seq[0], seq[1], seq[2]
    if conv.__class__.__name__ != "Conv2d":
        raise RuntimeError(f"{where}.0 is {conv.__class__.__name__}, expected Conv2d")
    if bn.__class__.__name__ != "BatchNorm2d":
        raise RuntimeError(f"{where}.1 is {bn.__class__.__name__}, expected BatchNorm2d")
    if act.__class__.__name__ != "ReLU":
        raise RuntimeError(f"{where}.2 is {act.__class__.__name__}, expected ReLU")
    return conv, bn, True


def _conv_layer(conv, has_bias=None):
    if has_bias is None:
        has_bias = conv.bias is not None
    w = conv.weight.detach().cpu().numpy().astype(np.float32)  # OIHW
    params = [w.reshape(-1)]
    if has_bias:
        params.append(conv.bias.detach().cpu().numpy().astype(np.float32).reshape(-1))
    return {
        "kind": K_CONV,
        "flags": F_HAS_BIAS if has_bias else 0,
        "in_channels": int(conv.in_channels),
        "out_channels": int(conv.out_channels),
        "kernel_h": int(conv.kernel_size[0]),
        "kernel_w": int(conv.kernel_size[1]),
        "stride_h": int(conv.stride[0]),
        "stride_w": int(conv.stride[1]),
        "pad_h": int(conv.padding[0]),
        "pad_w": int(conv.padding[1]),
        "scalar0": 0.0,
        "params": np.concatenate(params).astype(np.float32),
    }


def _bn_layer(bn, relu_after):
    """BN -> (gamma, beta, mean, invstd); invstd precomputed, no runtime sqrt."""
    gamma = bn.weight.detach().cpu().numpy().astype(np.float32).reshape(-1)
    beta = bn.bias.detach().cpu().numpy().astype(np.float32).reshape(-1)
    mean = bn.running_mean.detach().cpu().numpy().astype(np.float32).reshape(-1)
    var = bn.running_var.detach().cpu().numpy().astype(np.float32).reshape(-1)
    eps = float(bn.eps)
    invstd = (1.0 / np.sqrt(var.astype(np.float64) + eps)).astype(np.float32)
    return {
        "kind": K_BN,
        "flags": F_RELU_AFTER if relu_after else 0,
        "in_channels": int(bn.num_features),
        "out_channels": int(bn.num_features),
        "kernel_h": 0, "kernel_w": 0,
        "stride_h": 0, "stride_w": 0,
        "pad_h": 0, "pad_w": 0,
        "scalar0": eps,
        "params": np.concatenate([gamma, beta, mean, invstd]).astype(np.float32),
    }


# --------------------------------------------------------------------------- #
# writing
# --------------------------------------------------------------------------- #
def write_layer(buf, L):
    params = np.ascontiguousarray(L["params"], dtype=np.float32)
    buf += struct.pack("<IIIIIIIIIII", L["kind"], L["flags"],
                        L["in_channels"], L["out_channels"],
                        L["kernel_h"], L["kernel_w"],
                        L["stride_h"], L["stride_w"],
                        L["pad_h"], L["pad_w"], 0)
    buf += struct.pack("<Q", params.size)
    buf += struct.pack("<f", float(L["scalar0"]))
    assert len(buf) % 1 == 0
    buf += params.tobytes()


def write_table(buf, tag, dtype, ndim, arr):
    a = np.ascontiguousarray(arr)
    if dtype == DT_F32:
        a = a.astype(np.float32)
    else:
        a = a.astype(np.int32)
    buf += struct.pack("<IIII", tag, dtype, ndim, 0)
    buf += struct.pack("<Q", a.size)
    buf += a.tobytes()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--ckpt", default="/root/Semantic-sensing/ckpt/student_mid_b0.03_s2.pth")
    ap.add_argument("--out", default="/root/semantic-sensing-hailo-deployment/cpp/codec_tables.bin")
    ap.add_argument("--median-npy",
                    default="/root/deploy_artifacts/onnx/sender_z_median.npy",
                    help="optional: cross-check the exported medians against this file")
    args = ap.parse_args()

    from export_sender import load_student

    student, cfg = load_student(args.ckpt)
    bn = student.bottleneck
    print(f"bottleneck : {type(bn).__name__}")
    print(f"cfg.bottleneck: {cfg.bottleneck}")

    # ---- build the CDF tables (force=True) --------------------------------- #
    # The order matters: the conditional coder needs its scale table first.
    with torch.no_grad():
        updated = bn.gaussian_conditional.update_scale_table(
            bn.gaussian_conditional.scale_table if bn.gaussian_conditional.scale_table.numel()
            else _scale_table(),
            force=True)
        updated |= bn.entropy_bottleneck.update(force=True)
    print(f"update(force=True) -> {updated}")

    gc = bn.gaussian_conditional
    eb = bn.entropy_bottleneck

    scale_table = gc.scale_table.detach().cpu().numpy().astype(np.float32).reshape(-1)
    gauss_cdf = gc._quantized_cdf.detach().cpu().numpy().astype(np.int32)
    gauss_cdf_len = gc._cdf_length.detach().cpu().numpy().astype(np.int32).reshape(-1)
    gauss_off = gc._offset.detach().cpu().numpy().astype(np.int32).reshape(-1)

    eb_cdf = eb._quantized_cdf.detach().cpu().numpy().astype(np.int32)
    eb_cdf_len = eb._cdf_length.detach().cpu().numpy().astype(np.int32).reshape(-1)
    eb_off = eb._offset.detach().cpu().numpy().astype(np.int32).reshape(-1)
    quantiles = eb.quantiles.detach().cpu().numpy().astype(np.float32).reshape(-1)
    medians = eb._get_medians().detach().cpu().numpy().astype(np.float32).reshape(-1)
    scale_bound = gc.scale_bound.detach().cpu().numpy().astype(np.float32).reshape(-1)

    # The GaussianConditional CDF is a 2-D tensor; we store it FLATTENED
    # (row-major) because the C++ side addresses it as a flat array.
    if gauss_cdf.ndim != 2:
        raise RuntimeError(f"unexpected gauss cdf shape {gauss_cdf.shape}")
    gauss_cdf_flat = gauss_cdf.reshape(-1)
    if eb_cdf.ndim != 2:
        raise RuntimeError(f"unexpected eb cdf shape {eb_cdf.shape}")
    eb_cdf_flat = eb_cdf.reshape(-1)

    # ---- h_s layers --------------------------------------------------------- #
    layers = extract_hs_layers(bn.h_s)
    print(f"h_s layers: {len(layers)}")
    for L in layers:
        print(f"   kind={L['kind']} flags={L['flags']} "
              f"in={L['in_channels']} out={L['out_channels']} "
              f"k={L['kernel_h']}x{L['kernel_w']} p={L['pad_h']} "
              f"scalar0={L['scalar0']} params={L['params'].size}")

    # ---- assemble ----------------------------------------------------------- #
    buf = bytearray()
    buf += MAGIC
    buf += struct.pack("<IIII", FORMAT_VERSION, 0, len(layers), N_TABLES)
    buf += struct.pack("<II", 0, 0)
    assert len(buf) == HEADER_SIZE, len(buf)

    for L in layers:
        write_layer(buf, L)

    write_table(buf, T_SCALE_TABLE, DT_F32, 1, scale_table)
    write_table(buf, T_GAUSS_CDF, DT_I32, 2, gauss_cdf_flat)
    write_table(buf, T_GAUSS_CDF_LENGTH, DT_I32, 1, gauss_cdf_len)
    write_table(buf, T_GAUSS_OFFSET, DT_I32, 1, gauss_off)
    write_table(buf, T_EB_CDF, DT_I32, 2, eb_cdf_flat)
    write_table(buf, T_EB_CDF_LENGTH, DT_I32, 1, eb_cdf_len)
    write_table(buf, T_EB_OFFSET, DT_I32, 1, eb_off)
    write_table(buf, T_MEDIANS, DT_F32, 1, medians)
    write_table(buf, T_QUANTILES, DT_F32, 2, quantiles.reshape(eb.channels, 3))
    write_table(buf, T_SCALE_BOUND, DT_F32, 1, scale_bound)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "wb") as f:
        f.write(buf)

    sha = hashlib.sha256(buf).hexdigest()
    print(f"\nwrote {args.out}")
    print(f"  bytes   : {len(buf)}")
    print(f"  sha256  : {sha}")
    print(f"  scale_table: {len(scale_table)} entries "
          f"[{scale_table[0]:.6f} .. {scale_table[-1]:.6f}]")
    print(f"  gauss cdf : {gauss_cdf.shape} -> flat {gauss_cdf_flat.size}")
    print(f"  eb cdf    : {eb_cdf.shape} -> flat {eb_cdf_flat.size}")
    print(f"  medians   : {len(medians)}")
    print(f"  scale_bound: {scale_bound}")

    # ---- cross-check the medians against the sender's median file ----------- #
    if args.median_npy and os.path.isfile(args.median_npy):
        ref = np.load(args.median_npy).astype(np.float32).reshape(-1)
        d = float(np.abs(ref - medians).max())
        print(f"  median vs {args.median_npy}: max|diff| = {d:.3e}")
        if d > 1e-6:
            print("  WARNING: medians differ from the sender median file!")


def _scale_table(smin=0.11, smax=256.0, levels=64):
    """Same as hyperprior._scale_table -- log-spaced Gaussian scale table."""
    import math
    return torch.exp(torch.linspace(math.log(smin), math.log(smax), levels))


if __name__ == "__main__":
    main()
