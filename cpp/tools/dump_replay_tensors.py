#!/usr/bin/env python3
"""Dump the sender tensors (y, z_pre, z_hat) into the replay-backend format.

The C++ replay backend expects ONE raw binary file per image:

    float32 y     [192*16*16]   NCHW
    float32 z_pre [128*8*8]     NCHW     (pre-rounding)
    float32 z_hat [128*8*8]     NCHW     (optional, already rounded)

The Hailo DFC emulator dumps (chunk_XXXXX.npz) contain y and the ALREADY
ROUNDED z_hat (z_hat = floor(z_pre + 0.5) + median).  The pre-rounding z_pre
is NOT recoverable from them, so the zhat-mode of the replay backend is used
for those images (it skips the host rounding).

For the PyTorch FP32 reference (ref_z_hat.npy) the same applies.

Usage (venv-hailo):
    /root/Semantic-sensing/venv-hailo/bin/python \
        cpp/tools/dump_replay_tensors.py \
        --chunks /dev/shm/hailo2000/hailo \
        --out-dir /root/autodl-tmp/work/replay_hailo
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

Y_C, Y_H, Y_W = 192, 16, 16
Z_C, Z_H, Z_W = 128, 8, 8


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--chunks", default="/dev/shm/hailo2000/hailo",
                    help="directory with chunk_XXXXX.npz (Hailo emulator dump)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--names", default=None,
                    help="optional names.json (for the name <-> index mapping)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.chunks, "chunk_*.npz")))
    if not files:
        raise SystemExit(f"no chunk files in {args.chunks}")

    names = None
    if args.names and os.path.isfile(args.names):
        with open(args.names) as f:
            import json
            names = json.load(f)
        if isinstance(names, dict):
            names = names.get("names") or names.get("full")
        names = list(names)

    os.makedirs(args.out_dir, exist_ok=True)
    idx = 0
    for fp in files:
        d = np.load(fp)
        y = np.ascontiguousarray(d["y"].astype(np.float32))
        z_hat = np.ascontiguousarray(d["z_hat"].astype(np.float32))
        for i in range(y.shape[0]):
            if args.limit and idx >= args.limit:
                break
            if names is not None and idx < len(names):
                name = names[idx]
            else:
                name = f"img_{idx:06d}"
            yi = y[i].reshape(-1)
            zi = z_hat[i].reshape(-1)
            # z_pre is not recoverable from the emulator dump -> we write the
            # z_hat block twice; the C++ zhat-mode reads the THIRD block.
            out = np.concatenate([yi, zi, zi]).astype(np.float32)
            out.tofile(os.path.join(args.out_dir, name + ".bin"))
            idx += 1
        if args.limit and idx >= args.limit:
            break

    print(f"wrote {idx} tensor file(s) to {args.out_dir}")
    if names is not None:
        print(f"names taken from {args.names} (first: {names[0] if names else '-'})")


if __name__ == "__main__":
    main()
