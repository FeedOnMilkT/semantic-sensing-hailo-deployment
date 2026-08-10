#!/usr/bin/env python3
"""Single-module real PTQ: optimize → quantized check → compile.

Covers fullmodule handoff PTQ (formal 1024-calib + held-out golden), and the
lighter stage45 / fused calib-root modes that compare against ONNX Runtime.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    from hailo_deploy.logging import reexec_if_unlogged

    reexec_if_unlogged(Path(__file__), sys.argv[1:])

from hailo_deploy.artifacts.layout import commit_or_fail, refuse_existing, stage_dir
from hailo_deploy.compile import num_contexts
from hailo_deploy.gates.compile_real import compile_quantized
from hailo_deploy.gates.optimize import build_model_script, optimize_with_calib
from hailo_deploy.gates.quantized_check import heldout_quantized_check, ort_quantized_check
from hailo_deploy.io.calib import load_calib_handoff, load_loader, map_hn_inputs
from hailo_deploy.io.hash import sha256
from hailo_deploy.io.handoff import FORMAL_CALIBRATION_COUNT, validate_handoff


def _artifact_record(path: Path, published_root: Path) -> dict[str, Any]:
    return {
        "path": str(published_root / path.name),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def run_fullmodule_ptq(args: argparse.Namespace) -> dict[str, Any]:
    refuse_existing(args.out_dir)
    if args.n_calib != FORMAL_CALIBRATION_COUNT:
        raise ValueError("formal full-module PTQ requires exactly --n-calib 1024")
    if not args.parse_har.is_file():
        raise FileNotFoundError(args.parse_har)

    result: dict[str, Any] = {
        "kind": "full_module_real_ptq",
        "schema": "ptq.v1",
        "status": "FAIL",
        "module": args.module,
        "n_calib": args.n_calib,
        "optimization_level": args.opt_level,
        "handoff_root": str(args.handoff_root),
        "parse_har": {
            "path": str(args.parse_har),
            "sha256": sha256(args.parse_har),
            "bytes": args.parse_har.stat().st_size,
        },
        "error": None,
    }
    temporary = stage_dir(args.out_dir)
    try:
        identity, onnx_entry, meta, onnx, calibration_base, calibration_indices, golden_indices = (
            validate_handoff(args.handoff_root, args.module)
        )
        loader = load_loader(calibration_base / args.module / "load_calib.py")
        arrays = loader.load_calib(limit=args.n_calib)
        if set(arrays) != set(meta["onnx"]["input_names"]):
            raise ValueError(
                f"loader inputs {sorted(arrays)} != ONNX inputs {meta['onnx']['input_names']}"
            )
        from hailo_sdk_client import ClientRunner

        runner = ClientRunner(har=str(args.parse_har))
        calibration, input_mapping = map_hn_inputs(runner, arrays)
        alls_paths = [args.extra_alls] if args.extra_alls is not None else None
        if args.extra_alls is not None and not args.extra_alls.is_file():
            raise FileNotFoundError(args.extra_alls)
        script = build_model_script(
            opt_level=args.opt_level, n_calib=args.n_calib, alls_paths=alls_paths
        )
        model_script = temporary / "model_script.alls"
        model_script.write_text(script, encoding="utf-8")
        quantized_har = temporary / f"{args.module}_l{args.opt_level}_quantized.har"
        optimize_with_calib(runner, calibration, script, quantized_har=quantized_har)
        quantized_check = heldout_quantized_check(
            runner, args.module, args.handoff_root, meta, golden_indices
        )
        hef = temporary / f"{args.module}_l{args.opt_level}.hef"
        compiled_har = temporary / f"{args.module}_l{args.opt_level}_compiled.har"
        auto_alls = temporary / f"{args.module}_l{args.opt_level}.auto.alls"
        compiled = compile_quantized(
            runner, hef_path=hef, compiled_har=compiled_har, auto_alls=auto_alls
        )
        result.update(
            {
                "status": "PASS",
                "identity": identity,
                "onnx": {
                    "path": str(onnx),
                    "sha256": sha256(onnx),
                    "manifest": onnx_entry,
                },
                "calibration": {
                    "meta": str(calibration_base / args.module / "meta.json"),
                    "meta_sha256": sha256(calibration_base / args.module / "meta.json"),
                    "input_mapping": input_mapping,
                    "val_indices": calibration_indices,
                },
                "model_script": _artifact_record(model_script, args.out_dir),
                "quantized_har": _artifact_record(quantized_har, args.out_dir),
                "held_out_quantized_tensor_check": quantized_check,
                "hef": _artifact_record(hef, args.out_dir),
                "compiled_har": _artifact_record(compiled_har, args.out_dir),
                "auto_alls": (
                    _artifact_record(auto_alls, args.out_dir)
                    if "path" in compiled.get("auto_alls", {})
                    else compiled.get("auto_alls")
                ),
                "num_contexts": compiled["num_contexts"],
            }
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    finally:
        (temporary / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
        )
    final = commit_or_fail(temporary, args.out_dir, result["status"] == "PASS")
    if result["status"] != "PASS":
        print(json.dumps(result, indent=2), file=sys.stderr)
    else:
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "module": args.module,
                    "out": str(final),
                    "contexts": result["num_contexts"],
                },
                indent=2,
            )
        )
    return result


def run_calib_root_ptq(args: argparse.Namespace) -> dict[str, Any]:
    """Stage45 / fused-style PTQ from a single module calib root + parse HAR."""
    refuse_existing(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    if not args.parse_har.is_file() or not args.onnx.is_file():
        raise FileNotFoundError("parse HAR and ONNX are required")

    result: dict[str, Any] = {
        "kind": "module_calib_root_ptq",
        "schema": "ptq.v1",
        "status": "FAIL",
        "module": args.module,
        "n_calib": args.n_calib,
        "optimization_level": args.opt_level,
        "error": None,
    }
    try:
        arrays, metadata = load_calib_handoff(args.calib_root, args.n_calib)
        if args.expect_onnx_sha and metadata.get("onnx", {}).get("sha256") != sha256(args.onnx):
            raise ValueError("calibration ONNX hash does not match deployed ONNX")
        if args.expect_checkpoint_sha and metadata.get("checkpoint", {}).get("sha256") != args.expect_checkpoint_sha:
            raise ValueError("calibration checkpoint SHA does not match expected")

        from hailo_sdk_client import ClientRunner

        runner = ClientRunner(har=str(args.parse_har))
        calibration, mapping = map_hn_inputs(runner, arrays)
        alls_paths = []
        if args.alls is not None:
            alls_paths.append(args.alls)
        if args.extra_alls is not None:
            alls_paths.append(args.extra_alls)
        script = build_model_script(
            opt_level=args.opt_level,
            n_calib=args.n_calib,
            context_switch=args.context_switch,
            alls_paths=alls_paths or None,
        )
        level_tag = f"l{args.opt_level}"
        model_script = args.out_dir / f"{args.module}_{level_tag}.model_script.alls"
        model_script.write_text(script, encoding="utf-8")
        quantized_har = args.out_dir / f"{args.module}_{level_tag}_quantized.har"
        optimize_with_calib(runner, calibration, script, quantized_har=quantized_har)
        quantized = ort_quantized_check(runner, args.onnx, arrays, args.eval_samples)
        compiled: dict[str, Any] = {}
        if not args.skip_compile:
            compiled = compile_quantized(
                runner,
                hef_path=args.out_dir / f"{args.module}_{level_tag}.hef",
                compiled_har=args.out_dir / f"{args.module}_{level_tag}_compiled.har",
                auto_alls=args.out_dir / f"{args.module}_{level_tag}.auto.alls",
            )
        result.update(
            {
                "status": "PASS",
                "input_mapping": mapping,
                "calibration": {
                    "root": str(args.calib_root),
                    "sample_count": args.n_calib,
                    "metadata": {
                        k: metadata.get(k)
                        for k in (
                            "schema", "segment", "calibration_mode", "checkpoint", "onnx",
                            "codec", "roundtrip_alignment", "source_bootstrap",
                            "quantized_upstream", "tensor_mapping", "storage", "statistics",
                        )
                        if k in metadata
                    },
                },
                "model_script": {
                    "path": str(model_script),
                    "sha256": sha256(model_script),
                },
                "quantized_har": {
                    "path": str(quantized_har),
                    "sha256": sha256(quantized_har),
                    "bytes": quantized_har.stat().st_size,
                },
                "quantized_check": quantized,
                **compiled,
                "compile": "SKIP by --skip-compile" if args.skip_compile else "PASS",
            }
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    (args.out_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    full = sub.add_parser("fullmodule", help="Formal handoff PTQ (1024 calib + golden)")
    full.add_argument("--handoff-root", type=Path, required=True)
    full.add_argument("--module", required=True)
    full.add_argument("--parse-har", type=Path, required=True)
    full.add_argument("--out-dir", type=Path, required=True)
    full.add_argument("--n-calib", type=int, default=1024)
    full.add_argument("--opt-level", type=int, default=2, choices=range(5))
    full.add_argument("--extra-alls", type=Path, default=None)

    calib = sub.add_parser("calib-root", help="Stage45/fused calib-root PTQ")
    calib.add_argument("--module", required=True)
    calib.add_argument("--onnx", type=Path, required=True)
    calib.add_argument("--parse-har", type=Path, required=True)
    calib.add_argument("--calib-root", type=Path, required=True)
    calib.add_argument("--out-dir", type=Path, required=True)
    calib.add_argument("--n-calib", type=int, required=True)
    calib.add_argument("--opt-level", type=int, required=True, choices=range(5))
    calib.add_argument("--eval-samples", type=int, default=4)
    calib.add_argument("--alls", type=Path, default=None)
    calib.add_argument("--extra-alls", type=Path, default=None)
    calib.add_argument("--context-switch", choices=["allowed", "enabled"], default=None)
    calib.add_argument("--skip-compile", action="store_true")
    calib.add_argument("--expect-onnx-sha", action="store_true")
    calib.add_argument("--expect-checkpoint-sha", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "fullmodule":
        result = run_fullmodule_ptq(args)
    else:
        result = run_calib_root_ptq(args)
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
