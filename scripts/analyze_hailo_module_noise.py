#!/usr/bin/env python3
"""Run Hailo SDK layer-noise analysis on a copied, real-calibrated HAR."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if __name__ == "__main__":
    from hailo_deploy.logging import reexec_if_unlogged

    reexec_if_unlogged(Path(__file__), sys.argv[1:])

from hailo_deploy.io.calib import load_loader, map_hn_inputs
from hailo_deploy.io.hash import sha256
from hailo_deploy.io.handoff import load_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff-root", type=Path, required=True)
    parser.add_argument("--module", required=True)
    parser.add_argument("--quantized-har", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-calib", type=int, default=1024)
    parser.add_argument(
        "--expect-checkpoint-sha",
        default=None,
        help="optional expected checkpoint SHA-256; defaults to handoff identity",
    )
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    if args.n_calib != 1024:
        raise ValueError("formal noise analysis requires the same 1024 calibration samples")
    identity = load_json(args.handoff_root / "identity.json")
    checkpoint_sha = identity.get("checkpoint", {}).get("sha256")
    expected = args.expect_checkpoint_sha or checkpoint_sha
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("handoff has no valid checkpoint SHA-256")
    if checkpoint_sha != expected:
        raise ValueError("handoff checkpoint mismatch")
    # Older handoffs used ``calib``; the formal ConvBN handoff uses the
    # clearer ``calibration`` spelling.  Accept either without changing the
    # split or silently falling back to another source.
    calibration_root = args.handoff_root / "calib"
    if not calibration_root.is_dir():
        calibration_root = args.handoff_root / "calibration"
    if not calibration_root.is_dir():
        raise FileNotFoundError(
            f"missing calibration directory under {args.handoff_root} "
            "(expected calib or calibration)"
        )
    meta = load_json(calibration_root / args.module / "meta.json")
    if meta.get("dataset", {}).get("val_indices") != list(range(1024)):
        raise ValueError("calibration split mismatch")
    if not args.quantized_har.is_file():
        raise FileNotFoundError(args.quantized_har)
    loader = load_loader(calibration_root / args.module / "load_calib.py")
    arrays = loader.load_calib(limit=args.n_calib)
    from hailo_sdk_client import ClientRunner

    args.out_dir.mkdir(parents=True)
    runner = ClientRunner(har=str(args.quantized_har))
    # The training handoff calibration loaders deliberately yield NHWC, which
    # is exactly the Hailo HN input contract.  Do not transpose it again.
    feed, mapping = map_hn_inputs(runner, arrays)
    runner.analyze_noise(feed, data_count=args.n_calib, batch_size=1, analyze_mode="simple")
    annotated = args.out_dir / f"{args.module}_noise_annotated.har"
    runner.save_har(str(annotated))
    result = {"kind": "hailo_layer_noise_analysis", "status": "PASS", "module": args.module,
              "n_calib": args.n_calib, "input_mapping": mapping,
              "source_quantized_har": {"path": str(args.quantized_har), "sha256": sha256(args.quantized_har)},
              "annotated_har": {"path": str(annotated), "sha256": sha256(annotated), "bytes": annotated.stat().st_size},
              "notes": ["Noise analysis is stored on a copy; the formal quantized HAR is unchanged.",
                        "Open the annotated HAR in Hailo Profiler to inspect per-layer SNR."]}
    (args.out_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
