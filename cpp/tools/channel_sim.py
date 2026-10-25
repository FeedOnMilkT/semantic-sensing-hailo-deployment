#!/usr/bin/env python3
"""A minimal "noisy channel" simulator for the RTFF bitstreams.

WHAT THIS IS FOR
----------------
The codec has NO error protection: the RTFF container is a bare rANS stream.
A single flipped bit in the rANS payload desynchronises the decoder, and every
symbol after the flip decodes to garbage (or the decoder raises).  The
container header is checked first, so a flipped bit in the header is detected
as a hard error rather than a silent desync.

This script demonstrates that:

  * ber = 0        -> the stream passes through untouched (bit-identical copy);
  * ber > 0        -> random bits are flipped, and the receiver gets garbage;
  * drop_rate > 0  -> whole packets are dropped (the receiver sees a missing
                      file / a truncated packet).

It is a DEMO ONLY -- there is no FEC, no CRC and no retransmission.

Usage (system python is fine, numpy optional):

    python3 cpp/tools/channel_sim.py \
        --in-dir  /root/autodl-tmp/deploy_artifacts/package/sender_out/bitstreams \
        --out-dir /root/autodl-tmp/work/chan_ber1e3 \
        --ber 1e-3 --seed 0

    # drop every 10th packet
    python3 cpp/tools/channel_sim.py --in-dir ... --out-dir ... --drop-rate 0.1
"""
from __future__ import annotations

import argparse
import os
import random
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--in-dir", required=True,
                    help="directory holding the *.rtff bitstreams")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ber", type=float, default=0.0,
                    help="bit error rate (0 = passthrough)")
    ap.add_argument("--drop-rate", type=float, default=0.0,
                    help="probability that a whole packet is dropped")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ext", default=".rtff")
    ap.add_argument("--report", default=None,
                    help="optional JSON report path")
    args = ap.parse_args()

    if not 0.0 <= args.ber < 1.0:
        raise SystemExit("--ber must be in [0, 1)")
    if not 0.0 <= args.drop_rate <= 1.0:
        raise SystemExit("--drop-rate must be in [0, 1]")

    rng = random.Random(args.seed)
    names = sorted(n for n in os.listdir(args.in_dir) if n.endswith(args.ext))
    if not names:
        raise SystemExit(f"no {args.ext} files in {args.in_dir}")

    os.makedirs(args.out_dir, exist_ok=True)

    n_total = n_dropped = n_flipped_bits = n_corrupted = 0
    n_bytes = 0
    for name in names:
        src = os.path.join(args.in_dir, name)
        with open(src, "rb") as f:
            data = bytearray(f.read())
        n_total += 1
        n_bytes += len(data)

        if rng.random() < args.drop_rate:
            n_dropped += 1
            continue  # the packet simply does not arrive

        if args.ber > 0.0:
            n_bits = len(data) * 8
            n_flip = 0
            # Bernoulli bit flips; expected count is ber * n_bits.
            for bit in range(n_bits):
                if rng.random() < args.ber:
                    data[bit >> 3] ^= (1 << (bit & 7))
                    n_flip += 1
            if n_flip:
                n_corrupted += 1
                n_flipped_bits += n_flip

        with open(os.path.join(args.out_dir, name), "wb") as f:
            f.write(data)

    print(f"input packets   : {n_total} ({n_bytes} bytes)")
    print(f"dropped         : {n_dropped} ({100.0 * n_dropped / n_total:.2f}%)")
    print(f"corrupted       : {n_corrupted} packets, "
          f"{n_flipped_bits} flipped bits "
          f"({n_flipped_bits / max(n_bytes * 8, 1):.3e} of all bits)")
    print(f"output dir      : {args.out_dir}")
    print()
    print("NOTE: the codec has no FEC/CRC.  A single flipped bit in the rANS")
    print("      payload desynchronises the decoder; the whole image after the")
    print("      flip is garbage.  Use --ber 0 for a clean channel.")

    if args.report:
        import json
        with open(args.report, "w") as f:
            json.dump({
                "input_packets": n_total,
                "input_bytes": n_bytes,
                "dropped": n_dropped,
                "corrupted_packets": n_corrupted,
                "flipped_bits": n_flipped_bits,
                "ber": args.ber,
                "drop_rate": args.drop_rate,
                "seed": args.seed,
            }, f, indent=2)
        print(f"report          : {args.report}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
