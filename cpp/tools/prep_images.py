#!/usr/bin/env python3
"""Prepare 512x512x3 uint8 image files for the C++ sender pipeline.

The sender pipeline consumes RAW uint8 pixels (0..255, RGB, NHWC) -- the Hailo
graph does the ImageNet normalisation on-chip, so the host must NOT
normalise.  This script produces exactly the same pixels the Python reference
and the DFC emulator saw, using the SAME loader path:

    PIL.Image.open(path).convert("RGB").resize((512, 512), Image.BILINEAR)

(identical to deploy/hailo_split_pipeline.py:_load_images).

Two output formats are supported:

  *.u8    : the raw 512*512*3 bytes, row-major HWC, RGB
  *.npy   : a uint8 numpy array of shape (512, 512, 3)

Use (in the venv-hailo python):

    /root/Semantic-sensing/venv-hailo/bin/python \
        cpp/tools/prep_images.py \
        --manifest /root/deploy_artifacts/data/val_manifest.json \
        --split full --n 2000 --out-dir /root/autodl-tmp/work/images_u8
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

SIZE = 512


def load_images(manifest_path, split, n, size=SIZE):
    """Load the raw 0..255 uint8 images exactly like the PyTorch reference."""
    from PIL import Image

    with open(manifest_path) as f:
        m = json.load(f)
    if split in m:
        items = m[split][:n]
    else:
        items = m["full"][:n]
    out = np.empty((len(items), size, size, 3), np.uint8)
    for i, it in enumerate(items):
        img = Image.open(it["image"]).convert("RGB").resize((size, size),
                                                           Image.BILINEAR)
        out[i] = np.asarray(img, np.uint8)
    return out, items


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--manifest", default="/root/deploy_artifacts/data/val_manifest.json")
    ap.add_argument("--split", default="full", choices=["subset", "full"])
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--format", default="u8", choices=["u8", "npy"],
                    help="u8: raw 512*512*3 bytes; npy: uint8 (512,512,3) array")
    args = ap.parse_args()

    images, items = load_images(args.manifest, args.split, args.n)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"loaded {len(images)} image(s) from {args.manifest}[{args.split}]")

    names = []
    for i, (im, it) in enumerate(zip(images, items)):
        name = it.get("name", f"img_{i:06d}")
        names.append(name)
        if args.format == "u8":
            path = os.path.join(args.out_dir, name + ".u8")
            with open(path, "wb") as f:
                f.write(im.tobytes())
        else:
            path = os.path.join(args.out_dir, name + ".npy")
            np.save(path, im)
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(images)}", flush=True)

    # a small index file so the C++ side and the package builder agree on the
    # image order (the C++ sender sorts the directory listing lexicographically).
    index = {
        "count": len(names),
        "names": names,
        "size": SIZE,
        "format": args.format,
        "split": args.split,
    }
    with open(os.path.join(args.out_dir, "index.json"), "w") as f:
        json.dump(index, f, indent=2)

    print(f"wrote {len(names)} file(s) to {args.out_dir}")
    print(f"format: {args.format}, size: {SIZE}x{SIZE}x3")


if __name__ == "__main__":
    sys.exit(main())
