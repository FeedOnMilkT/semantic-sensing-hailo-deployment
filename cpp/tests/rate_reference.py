#!/usr/bin/env python3
"""Rate reference: recompute the real rANS bpp for the whole val set, in Python.

Three arms are computed so the numbers can be attributed:

  A. "cpp_h_s"      the Hailo y / Hailo z_hat, but with the FP32 h_s means and
                    scales -- this is what the C++ sender does now (the Hailo
                    `hs` HEF is not used any more).
  B. "hailo_h_s"    the Hailo y / Hailo z_hat with the HAILO int8 hs means and
                    scales (straight from the emulator dump) -- this is the
                    OLD deployed configuration and the number quoted in the
                    deployment guide (0.17583).
  C. "fp32"         the PyTorch FP32 reference y / z_hat with the FP32 h_s --
                    the guide's 0.17467.

Everything is coded with CompressAI's own entropy coder, so the numbers are
directly comparable with the guide.

Usage (venv-hailo):
    /root/Semantic-sensing/venv-hailo/bin/python cpp/tests/rate_reference.py \
        --chunks /root/autodl-tmp/data/hailo_chunks --ref-dir /root/autodl-tmp/data/ref \
        --n 2000 --out /root/autodl-tmp/work/rate_reference.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

REPO = "/root/Semantic-sensing"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "deploy"))

Y_C, Y_H, Y_W = 192, 16, 16
Z_C, Z_H, Z_W = 128, 8, 8


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--chunks", default="/root/autodl-tmp/data/hailo_chunks")
    ap.add_argument("--ref-dir", default="/root/autodl-tmp/data/ref")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out", default="/root/autodl-tmp/work/rate_reference.json")
    ap.add_argument("--ckpt",
                    default="/root/Semantic-sensing/ckpt/student_mid_b0.03_s2.pth")
    args = ap.parse_args()

    import glob
    from export_sender import load_student

    student, cfg = load_student(args.ckpt)
    bn = student.bottleneck
    with torch.no_grad():
        bn.gaussian_conditional.update_scale_table(
            torch.exp(torch.linspace(np.log(0.11), np.log(256.0), 64)), force=True)
        bn.entropy_bottleneck.update(force=True)
    eb, gc = bn.entropy_bottleneck, bn.gaussian_conditional

    files = sorted(glob.glob(os.path.join(args.chunks, "chunk_*.npz")))
    ref_y = np.load(os.path.join(args.ref_dir, "ref_y.npy"), mmap_mode="r")
    ref_z = np.load(os.path.join(args.ref_dir, "ref_z_hat.npy"), mmap_mode="r")

    bpp = {"cpp_h_s": [], "hailo_h_s": [], "fp32": []}
    zbits = {"cpp_h_s": [], "hailo_h_s": [], "fp32": []}
    ybits = {"cpp_h_s": [], "hailo_h_s": [], "fp32": []}

    idx = 0
    for fp in files:
        if idx >= args.n:
            break
        d = np.load(fp)
        y_h = np.ascontiguousarray(d["y"].astype(np.float32))
        z_h = np.ascontiguousarray(d["z_hat"].astype(np.float32))
        means_h = np.ascontiguousarray(d["means"].astype(np.float32))
        scales_h = np.ascontiguousarray(d["scales"].astype(np.float32))
        for i in range(y_h.shape[0]):
            if idx >= args.n:
                break
            y = torch.from_numpy(y_h[i:i + 1])
            z_hat = torch.from_numpy(z_h[i:i + 1])

            with torch.no_grad():
                # ---- arm B: the HAILO int8 hs (from the dump) -------------
                zi = gc.build_indexes(torch.from_numpy(scales_h[i:i + 1]))
                zb = eb.compress(z_hat)[0]
                yb = gc.compress(y, zi, means=torch.from_numpy(means_h[i:i + 1]))[0]
                bpp["hailo_h_s"].append((len(zb) + len(yb)) * 8 / (512 * 512))
                zbits["hailo_h_s"].append(len(zb) * 8)
                ybits["hailo_h_s"].append(len(yb) * 8)

                # ---- arm A: the FP32 h_s on the Hailo y / z_hat -------------
                m_t, s_t = bn.h_s(z_hat).chunk(2, dim=1)
                if m_t.shape[-2:] != (Y_H, Y_W):
                    import torch.nn.functional as F
                    m_t = F.interpolate(m_t, size=(Y_H, Y_W), mode="bilinear",
                                        align_corners=False)
                    s_t = F.interpolate(s_t, size=(Y_H, Y_W), mode="bilinear",
                                         align_corners=False)
                zi = gc.build_indexes(s_t)
                zb = eb.compress(z_hat)[0]
                yb = gc.compress(y, zi, means=m_t)[0]
                bpp["cpp_h_s"].append((len(zb) + len(yb)) * 8 / (512 * 512))
                zbits["cpp_h_s"].append(len(zb) * 8)
                ybits["cpp_h_s"].append(len(yb) * 8)

                # ---- arm C: the full FP32 reference --------------------------
                y_f = torch.from_numpy(np.asarray(ref_y[idx:idx + 1]).astype(np.float32))
                z_f = torch.from_numpy(np.asarray(ref_z[idx:idx + 1]).astype(np.float32))
                m_f, s_f = bn.h_s(z_f).chunk(2, dim=1)
                if m_f.shape[-2:] != (Y_H, Y_W):
                    import torch.nn.functional as F
                    m_f = F.interpolate(m_f, size=(Y_H, Y_W), mode="bilinear",
                                        align_corners=False)
                    s_f = F.interpolate(s_f, size=(Y_H, Y_W), mode="bilinear",
                                         align_corners=False)
                zi = gc.build_indexes(s_f)
                zb = eb.compress(z_f)[0]
                yb = gc.compress(y_f, zi, means=m_f)[0]
                bpp["fp32"].append((len(zb) + len(yb)) * 8 / (512 * 512))
                zbits["fp32"].append(len(zb) * 8)
                ybits["fp32"].append(len(yb) * 8)

            idx += 1
        print(f"  {idx} images", flush=True)

    summary = {"n": idx}
    for k in bpp:
        summary[k] = {
            "bpp": float(np.mean(bpp[k])),
            "bpp_std": float(np.std(bpp[k])),
            "z_bits": float(np.mean(zbits[k])),
            "y_bits": float(np.mean(ybits[k])),
        }
        print(f"{k:12s} bpp={summary[k]['bpp']:.6f} +- {summary[k]['bpp_std']:.6f}  "
              f"z_bits={summary[k]['z_bits']:.1f}  y_bits={summary[k]['y_bits']:.1f}")

    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {args.out}")
    print("guide reference: Hailo int8 (old, with the hs HEF) = 0.17583, "
          "PyTorch FP32 = 0.17467")


if __name__ == "__main__":
    main()
