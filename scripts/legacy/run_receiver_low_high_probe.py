#!/usr/bin/env python3
"""Run the diagnostic low/high receiver cut through parse, NATIVE and dummy compile.

Random calibration is used solely to expose DFC compiler feasibility and context
count.  It is deliberately not a quantized-accuracy measurement.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if __name__ == "__main__":
    from run_hailo_logged import reexec_if_unlogged

    reexec_if_unlogged(Path(__file__), sys.argv[1:])

from hailo_deploy import compile as cmpl
from hailo_deploy import config
from hailo_deploy import emulate, parse


def _module(name: str) -> config.Module:
    return config.Module(
        name=name,
        risk="diagnostic",
        desc="Stage4/5 low/high cut; dummy calibration proves compile feasibility only",
        onnx_stem=name,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--only", nargs="+", choices=["receiver_low_path", "receiver_high_path"],
                        default=["receiver_low_path", "receiver_high_path"])
    parser.add_argument("--n-calib", type=int, default=8)
    parser.add_argument("--opt-level", type=int, default=0)
    parser.add_argument("--skip-compile", action="store_true",
                        help="stop after parse and CPU/FP32 NATIVE parity")
    args = parser.parse_args()
    if args.n_calib <= 0 or args.opt_level not in range(5):
        parser.error("--n-calib must be positive and --opt-level must be 0..4")
    if not args.onnx_dir.is_dir():
        parser.error(f"ONNX directory not found: {args.onnx_dir}")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error(f"refusing to overwrite non-empty probe directory: {args.out_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # These are process-local config overrides; no frozen demo profile is changed.
    config.ONNX_DIR = args.onnx_dir
    config.HAR_DIR = args.out_dir / "har"
    config.HEF_DIR = args.out_dir / "hef"
    config.RESULTS_DIR = args.out_dir / "results"
    config.REPORTS_DIR = args.out_dir / "reports"
    config.ensure_dirs()

    results = []
    for name in args.only:
        module = _module(name)
        parsed = parse.parse_module(module, config.ONNX_DIR, config.HAR_DIR)
        native = None
        compiled = None
        if parsed.status == "PASS" and parsed.har_path:
            native = emulate.native_check(module, Path(parsed.har_path),
                                          config.ONNX_DIR / module.onnx_name)
        if not args.skip_compile and native is not None and native.status == "PASS":
            compiled = cmpl.compile_module(
                module, Path(parsed.har_path), config.ONNX_DIR / module.onnx_name,
                n_calib=args.n_calib, opt_level=args.opt_level)
        results.append({
            "name": name,
            "parse": parsed.as_dict(),
            "native": native.as_dict() if native else None,
            "compile": compiled.as_dict() if compiled else None,
        })
        status = (compiled.status if compiled is not None else
                  native.status if native is not None else parsed.status)
        print(f"[{status}] {name}")

    result = {
        "kind": "diagnostic_low_high_cut",
        "warning": "dummy random calibration; no accuracy claim is valid",
        "hw_arch": config.HW_ARCH,
        "onnx_dir": str(config.ONNX_DIR),
        "n_calib": args.n_calib,
        "optimization_level": args.opt_level,
        "skip_compile": args.skip_compile,
        "modules": results,
    }
    result_path = config.RESULTS_DIR / "probe.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"result: {result_path}")
    if args.skip_compile:
        successful = all(item["native"] and item["native"]["status"] == "PASS" for item in results)
    else:
        successful = all(item["compile"] and item["compile"]["status"] == "PASS" for item in results)
    return 0 if successful else 1


if __name__ == "__main__":
    raise SystemExit(main())
