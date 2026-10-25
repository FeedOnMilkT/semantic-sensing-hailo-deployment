#!/usr/bin/env python3
"""Does DFC's COMPILER keep a Floor layer, or does it drop it?

WHY THIS MATTERS
----------------
The rounding bug was observed in the DFC EMULATOR (InferenceContext.SDK_QUANTIZED):
`Floor` returned the input unchanged, so the hyper-latent was never rounded and the
z stream cost 52x more bits.

But the emulator is not the hardware.  The question that decides whether the fix is
needed ON THE BOARD is: does the compiler still emit a Floor/elementwise layer into
the HEF, or does it fold the op away?

  * a Floor LAYER in the optimized HAR  -> the chip may still round correctly, and the
    emulator is the thing that is broken (the fix would be "not needed", but the host
    split remains the safe choice).
  * NO Floor layer                        -> the op was eliminated at compile time, so
    the REAL board will not round either, and the host-side rounding is REQUIRED.

Usage (hailo-dfc env):
  python scripts/check_floor_layer.py --har sender_bottleneck_optimized.har
"""
from __future__ import annotations

import argparse
import json
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--har", required=True)
    a = ap.parse_args(argv)

    from hailo_sdk_client import ClientRunner

    runner = ClientRunner(har=a.har)
    hn = runner.get_hn_dict()
    layers = hn.get("layers", [])
    items = layers.items() if isinstance(layers, dict) else \
        ((l.get("name"), l) for l in layers)

    print(f"har: {a.har}")
    print(f"layers: {len(items)}")

    by_type = {}
    names = []
    for name, l in items:
        t = l.get("type", "?") if isinstance(l, dict) else "?"
        by_type[t] = by_type.get(t, 0) + 1
        names.append((name, t))

    print("\nlayer types:")
    for t, c in sorted(by_type.items(), key=lambda kv: -kv[1]):
        print(f"  {c:4d}  {t}")

    # look for anything rounding-related
    print("\nrounding-related layers:")
    hit = False
    for name, t in names:
        ln = str(name).lower()
        if t.lower() in ("floor", "ceil", "round") or "floor" in ln or "round" in ln:
            print(f"  {t:24s} {name}")
            hit = True
    if not hit:
        print("  (none)")

    # the elementwise/act layers are where a Floor would live
    print("\nall layer types containing 'elem'/'act'/'floor'/'ceil'/'round'/'quant':")
    keys = ("elem", "act", "floor", "ceil", "round", "quant")
    sel = [(n, t) for n, t in names if any(k in t.lower() for k in keys)]
    if sel:
        for n, t in sel:
            print(f"  {t:28s} {n}")
    else:
        print("  (none)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
