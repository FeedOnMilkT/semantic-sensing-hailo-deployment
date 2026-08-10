#!/usr/bin/env python3
"""Optimize and compile a split receiver stage with real ADE20K activations.

This is deliberately separate from run_fused_compile.py: that entry point uses
random calibration for feasibility only, while this one consumes the validated
feature calibration handoff under /local/shared_with_docker/calib.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if __name__ == "__main__":
    from hailo_deploy.logging import reexec_if_unlogged

    reexec_if_unlogged(Path(__file__), sys.argv[1:])

from hailo_deploy import compile as cmpl
from hailo_deploy import config, config_fused
from hailo_deploy.gates.compile_real import compile_quantized
from hailo_deploy.gates.optimize import build_model_script, optimize_with_calib
from hailo_deploy.gates.quantized_check import ort_quantized_check
from hailo_deploy.io.calib import load_calib_handoff, map_hn_inputs
from hailo_deploy.io.hash import sha256


def _run(args: argparse.Namespace) -> int:
    module = config.MODULES_BY_NAME[args.module]
    if args.calib_root is None:
        args.calib_root = Path("/local/shared_with_docker/calib") / args.module
    har_path = config.HAR_DIR / module.har_name
    onnx_path = config.ONNX_DIR / module.onnx_name
    result_path = config.RESULTS_DIR / f"{args.tag}.json"
    quantized_har = config.HAR_DIR / f"{args.tag}_quantized.har"
    hef_path = config.HEF_DIR / f"{args.tag}.hef"
    auto_alls = config.HEF_DIR / f"{args.tag}.auto.alls"
    alls_path = config_fused.alls_for(module.name)

    print(f"module: {module.name}; tag: {args.tag}")

    result = {
        "name": module.name,
        "tag": args.tag,
        "status": "FAIL",
        "calibration": {"root": str(args.calib_root), "count": args.n_calib},
        "optimization_level": args.opt_level,
        "compression_level": 0,
        "model_script": str(alls_path) if alls_path else None,
        "extra_model_script": str(args.extra_alls) if args.extra_alls else None,
        "resumed_quantized_har": (
            str(args.resume_quantized_har) if args.resume_quantized_har else None),
        "input_mapping": None,
        "quantized_check": None,
        "quantized_har": None,
        "hef": None,
        "auto_alls": None,
        "num_contexts": None,
        "error": None,
    }
    try:
        from hailo_sdk_client import ClientRunner

        arrays, metadata = load_calib_handoff(args.calib_root, args.n_calib)
        if metadata["onnx"]["sha256"] != sha256(onnx_path):
            raise ValueError("calibration ONNX hash does not match deployed ONNX")
        runner = ClientRunner(har=str(args.resume_quantized_har or har_path))
        calib, mapping = map_hn_inputs(runner, arrays)
        result["input_mapping"] = mapping
        result["calibration"].update({
            "checkpoint": metadata["checkpoint"],
            "onnx_sha256": metadata["onnx"]["sha256"],
            "codec": metadata["codec"],
            "roundtrip_alignment": metadata["roundtrip_alignment"],
        })

        alls_paths = []
        if alls_path is not None:
            alls_paths.append(alls_path)
            result["model_script_sha256"] = sha256(alls_path)
        if args.extra_alls is not None:
            if not args.extra_alls.is_file():
                raise FileNotFoundError(args.extra_alls)
            alls_paths.append(args.extra_alls)
            result["extra_model_script_sha256"] = sha256(args.extra_alls)
        script = build_model_script(
            opt_level=args.opt_level, n_calib=args.n_calib, alls_paths=alls_paths or None
        )
        if args.resume_quantized_har is None:
            optimize_with_calib(runner, calib, script, quantized_har=quantized_har)
        else:
            if not args.resume_quantized_har.is_file():
                raise FileNotFoundError(args.resume_quantized_har)
            quantized_har = args.resume_quantized_har
        result["quantized_har"] = {
            "path": str(quantized_har), "sha256": sha256(quantized_har),
            "bytes": quantized_har.stat().st_size,
        }

        result["quantized_check"] = ort_quantized_check(
            runner, onnx_path, arrays, args.eval_samples)

        compiled = compile_quantized(
            runner, hef_path=hef_path, auto_alls=auto_alls
        )
        result["hef"] = compiled["hef"]
        result["num_contexts"] = compiled["num_contexts"]
        result["auto_alls"] = compiled.get("auto_alls")
        if "auto_alls" in compiled and "error" in compiled["auto_alls"]:
            result["auto_alls_error"] = compiled["auto_alls"]["error"]
        result["status"] = "PASS"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    finally:
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"result: {result_path}")
    return 0 if result["status"] == "PASS" else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", required=True,
                        choices=["receiver_stage4", "receiver_stage5"])
    parser.add_argument("--calib-root", type=Path, default=None)
    parser.add_argument("--n-calib", type=int, required=True)
    parser.add_argument("--opt-level", type=int, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--eval-samples", type=int, default=4)
    parser.add_argument("--extra-alls", type=Path, default=None)
    parser.add_argument(
        "--resume-quantized-har", type=Path, default=None,
        help="reuse an existing quantized HAR and continue with validation/compile",
    )
    args = parser.parse_args()
    if args.n_calib <= 0 or args.opt_level not in range(5):
        parser.error("--n-calib must be positive and --opt-level must be 0..4")

    config_fused.activate()
    config.ensure_dirs()
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
