#!/usr/bin/env python3
"""Rigorous alternating latency benchmark for the two TensorRT receiver engines.

Each engine is measured in several alternating rounds so that GPU clock /
thermal drift affects both engines equally, and the per-round medians are
reported.  The final number quoted in the README is the pooled median over
all rounds.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from receiver_trt import DEFAULT_SENDER, ReceiverPipeline  # noqa: E402


def stats(v):
    a = np.asarray(v, np.float64)
    return {"mean": float(a.mean()), "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)), "min": float(a.min()),
            "max": float(a.max()), "std": float(a.std())}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", nargs="+", required=True)
    ap.add_argument("--sender-dir", default=DEFAULT_SENDER)
    ap.add_argument("--n", type=int, default=50, help="bitstreams per round")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--out-json", default=None)
    a = ap.parse_args(argv)

    with open(os.path.join(a.sender_dir, "manifest.json")) as f:
        images = json.load(f)["images"][:a.n]
    bit_dir = os.path.join(a.sender_dir, "bitstreams")
    payloads = []
    for it in images:
        with open(os.path.join(bit_dir, it["file"]), "rb") as f:
            payloads.append(f.read())

    pipes = {e: ReceiverPipeline(e, warmup=a.warmup) for e in a.engines}
    dec = {e: [] for e in a.engines}
    inf = {e: [] for e in a.engines}
    post = {e: [] for e in a.engines}
    rounds = []

    for r in range(a.rounds):
        for e in a.engines:
            pipe = pipes[e]
            d_i, i_i, p_i = [], [], []
            for payload in payloads:
                res = pipe.infer_payload(payload)
                d_i.append(res.decode_seconds * 1e3)
                i_i.append(res.infer_seconds * 1e3)
                p_i.append(res.postprocess_seconds * 1e3)
            dec[e].extend(d_i)
            inf[e].extend(i_i)
            post[e].extend(p_i)
            rounds.append({"round": r, "engine": e,
                           "inference_median_ms": float(np.median(i_i)),
                           "decode_median_ms": float(np.median(d_i)),
                           "postprocess_median_ms": float(np.median(p_i))})
            print(f"[bench] round {r} {os.path.basename(e):24s} "
                  f"infer median={np.median(i_i):.3f} ms  "
                  f"decode median={np.median(d_i):.3f} ms  "
                  f"post median={np.median(p_i):.3f} ms", flush=True)

    for e in a.engines:
        pipes[e].close()

    out = {"n_per_round": len(payloads), "rounds": a.rounds,
           "per_round": rounds, "engines": {}}
    for e in a.engines:
        out["engines"][e] = {
            "n_samples": len(inf[e]),
            "decode_ms": stats(dec[e]),
            "inference_ms": stats(inf[e]),
            "postprocess_ms": stats(post[e]),
            "end_to_end_ms": stats(np.asarray(dec[e]) + np.asarray(inf[e]) + np.asarray(post[e])),
            "inference_round_medians": [r["inference_median_ms"] for r in rounds
                                        if r["engine"] == e],
        }
        s = out["engines"][e]
        print(f"[bench] POOLED {os.path.basename(e):24s} "
              f"infer: median={s['inference_ms']['median']:.3f} "
              f"mean={s['inference_ms']['mean']:.3f} "
              f"p95={s['inference_ms']['p95']:.3f} ms | "
              f"round medians={['%.3f' % x for x in s['inference_round_medians']]}",
              flush=True)

    if a.out_json:
        with open(a.out_json, "w") as f:
            json.dump(out, f, indent=1)
        print(f"[bench] -> {a.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
