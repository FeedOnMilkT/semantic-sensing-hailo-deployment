#!/usr/bin/env python3
"""Build the raw test-vector files used by the C++ tests.

The vectors come from the REAL all-Hailo sender output (the DFC emulator's
int8 tensors in /dev/shm/hailo2000/hailo/chunk_XXXXX.npz), i.e. exactly the
y and z_hat the deployed sender produces.

Output (raw, little-endian float32):
    vectors.bin : n_images * ( y[192*16*16] , z_hat[128*8*8] )   NCHW

Usage (venv-hailo):
    /root/Semantic-sensing/venv-hailo/bin/python \
        cpp/tests/make_test_vectors.py \
        --chunks /dev/shm/hailo2000/hailo --n 250 --out /root/autodl-tmp/work/vectors.bin
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--chunks", default="/dev/shm/hailo2000/hailo")
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--out", default="/root/autodl-tmp/work/vectors.bin")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.chunks, "chunk_*.npz")))
    if not files:
        raise SystemExit(f"no chunks found in {args.chunks}")
    print(f"{len(files)} chunk file(s)")

    out = []
    total = 0
    for fp in files:
        d = np.load(fp)
        y = np.ascontiguousarray(d["y"].astype(np.float32))
        z = np.ascontiguousarray(d["z_hat"].astype(np.float32))
        for i in range(y.shape[0]):
            if total >= args.n:
                break
            out.append(y[i].reshape(-1))
            out.append(z[i].reshape(-1))
            total += 1
        if total >= args.n:
            break

    arr = np.concatenate(out).astype(np.float32)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    arr.tofile(args.out)
    print(f"wrote {args.out}: {total} image(s), {arr.nbytes} bytes")

    # sanity: z_hat - median must be integer valued (the transmitted symbol)
    n_img = total
    z_all = arr.reshape(n_img, -1)[:, 192 * 16 * 16:]
    print(f"z_hat max|x-round(x)| = {np.abs(z_all - np.round(z_all)).max():.3e}")
    med = np.load("/root/deploy_artifacts/onnx/sender_z_median.npy").astype(np.float32)
    sym = z_all - np.repeat(med.reshape(1, -1), 8 * 8, axis=1)
    print(f"z symbols max|x-round(x)| = {np.abs(sym - np.round(sym)).max():.3e}")


if __name__ == "__main__":
    main()
