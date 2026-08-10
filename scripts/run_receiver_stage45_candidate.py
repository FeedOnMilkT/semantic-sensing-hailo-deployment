#!/usr/bin/env python3
"""Run the gated Hailo feasibility flow for a receiver_stage45 GFA candidate.

The input activation contract is intentionally identical to the frozen fused
baseline: x_h [N,64,64,128] and x_l [N,32,32,256] in NHWC.  Candidate weights
are never calibrated from RGB images.  This runner is a hardware gate only;
it does not make an ADE20K accuracy claim before receiver adaptation.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if __name__ == "__main__":
    from hailo_deploy.logging import reexec_if_unlogged

    reexec_if_unlogged(Path(__file__), sys.argv[1:])

from hailo_deploy import compile as cmpl
from hailo_deploy import config, emulate, parse
from hailo_deploy.gates.optimize import build_model_script, optimize_with_calib
from hailo_deploy.gates.quantized_check import ort_quantized_check
from hailo_deploy.io.calib import load_calib_handoff, map_hn_inputs
from hailo_deploy.io.hash import sha256
from hailo_deploy.onnx_io import signature


def _dummy_compile(har_path: Path, onnx_path: Path, out_dir: Path, n_calib: int, opt_level: int,
                   context_switch: str | None, skip_compile: bool) -> dict:
    from hailo_sdk_client import ClientRunner

    result = {"status": "FAIL", "n_calib": n_calib, "optimization_level": opt_level,
              "warning": "random calibration; feasibility only", "error": None}
    try:
        runner = ClientRunner(har=str(har_path))
        notes: list[str] = []
        calib = cmpl._random_calib(runner, signature(onnx_path), n_calib, config.SEED, notes)
        script = f"model_optimization_flavor(optimization_level={opt_level}, compression_level=0)\n"
        if context_switch is not None:
            script += f"context_switch_param(mode={context_switch})\n"
        runner.load_model_script(script)
        runner.optimize(calib)
        if skip_compile:
            result.update({"status": "PASS", "compile": "SKIP by --skip-compile", "notes": notes})
            return result
        hef_path = out_dir / "dummy_receiver_stage45.hef"
        hef_path.write_bytes(runner.compile())
        auto_path = out_dir / "dummy_receiver_stage45.auto.alls"
        try:
            runner.save_autogen_allocation_script(str(auto_path))
        except Exception as exc:
            result["auto_alls_error"] = f"{type(exc).__name__}: {exc}"
        result.update({
            "status": "PASS", "hef": {"path": str(hef_path), "bytes": hef_path.stat().st_size},
            "auto_alls": str(auto_path) if auto_path is not None and auto_path.is_file() else None,
            "num_contexts": cmpl.num_contexts(runner), "notes": notes,
        })
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    return result


def _real_level0(har_path: Path, onnx_path: Path, calib_root: Path, out_dir: Path,
                 n_calib: int, opt_level: int, eval_samples: int, expected_checkpoint_sha: str,
                 context_switch: str | None, skip_compile: bool) -> dict:
    from hailo_sdk_client import ClientRunner

    result = {"status": "FAIL", "n_calib": n_calib, "optimization_level": opt_level,
              "quantized_har": None, "hef": None, "error": None}
    try:
        arrays, metadata = load_calib_handoff(calib_root, n_calib)
        if metadata["checkpoint"]["sha256"] != expected_checkpoint_sha:
            raise ValueError("calibration checkpoint SHA does not match candidate checkpoint")
        expected = {"x_h": [1, 128, 64, 64], "x_l": [1, 256, 32, 32]}
        if dict(signature(onnx_path).inputs) != expected:
            raise ValueError(f"candidate input contract changed: {signature(onnx_path).inputs}")
        runner = ClientRunner(har=str(har_path))
        calib, mapping = map_hn_inputs(runner, arrays)
        level_tag = f"l{opt_level}"
        script = build_model_script(
            opt_level=opt_level, n_calib=n_calib, context_switch=context_switch
        )
        model_script_path = out_dir / f"receiver_stage45_{level_tag}.model_script.alls"
        model_script_path.write_text(script, encoding="utf-8")
        quantized_har = out_dir / f"receiver_stage45_{level_tag}_quantized.har"
        optimize_with_calib(runner, calib, script, quantized_har=quantized_har)
        quantized = ort_quantized_check(runner, onnx_path, arrays, eval_samples)
        hef_path = None
        auto_path = None
        compiled_har = None
        if not skip_compile:
            hef_path = out_dir / f"receiver_stage45_{level_tag}.hef"
            hef_path.write_bytes(runner.compile())
            compiled_har = out_dir / f"receiver_stage45_{level_tag}_compiled.har"
            runner.save_har(str(compiled_har))
            auto_path = out_dir / f"receiver_stage45_{level_tag}.auto.alls"
            try:
                runner.save_autogen_allocation_script(str(auto_path))
            except Exception as exc:
                result["auto_alls_error"] = f"{type(exc).__name__}: {exc}"
        result.update({
            "status": "PASS", "input_mapping": mapping,
            "calibration": {
                "root": str(calib_root), "sample_count": n_calib,
                "source_onnx_sha256": metadata["onnx"]["sha256"],
                "candidate_onnx_sha256": sha256(onnx_path),
                "checkpoint": metadata["checkpoint"], "codec": metadata["codec"],
                "roundtrip_alignment": metadata["roundtrip_alignment"],
            },
            "quantized_har": {"path": str(quantized_har), "bytes": quantized_har.stat().st_size},
            "model_script": {"path": str(model_script_path), "sha256": sha256(model_script_path)},
            "quantized_check": quantized,
            "compiled_har": (
                {"path": str(compiled_har), "bytes": compiled_har.stat().st_size}
                if compiled_har is not None else None
            ),
            "hef": ({"path": str(hef_path), "bytes": hef_path.stat().st_size}
                    if hef_path is not None else None),
            "auto_alls": (
                str(auto_path)
                if auto_path is not None and auto_path.is_file()
                else None
            ),
            "num_contexts": cmpl.num_contexts(runner) if hef_path is not None else None,
            "compile": "SKIP by --skip-compile" if skip_compile else "PASS",
        })
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--calib-root", type=Path, default=Path(
        "/local/shared_with_docker/calib/receiver_stage45"))
    parser.add_argument("--n-calib", type=int, default=128)
    parser.add_argument("--eval-samples", type=int, default=4)
    parser.add_argument("--opt-level", type=int, default=0)
    parser.add_argument("--dummy-calib", type=int, default=8)
    parser.add_argument("--context-switch", choices=["allowed", "enabled"], default=None,
                        help="compile boundary only; use enabled after a bounded single-context probe")
    parser.add_argument("--skip-compile", action="store_true",
                        help="quantize and run emulator only; preserves a bounded placement probe separately")
    parser.add_argument("--native-only", action="store_true",
                        help="stop after Parse + SDK_NATIVE; run with CUDA_VISIBLE_DEVICES='' for CPU parity")
    parser.add_argument("--skip-native", action="store_true",
                        help="skip SDK_NATIVE only after a separate CPU-native run has passed")
    args = parser.parse_args()
    if not args.onnx.is_file() or not args.candidate_manifest.is_file():
        parser.error("--onnx and --candidate-manifest must exist")
    if args.out_dir.exists():
        parser.error(f"refusing to overwrite candidate artifact directory: {args.out_dir}")
    if args.n_calib <= 0 or args.dummy_calib <= 0 or args.eval_samples < 0 or args.opt_level not in range(5):
        parser.error("invalid calibration, evaluation, or optimization-level value")
    if args.native_only and args.skip_native:
        parser.error("--native-only and --skip-native are mutually exclusive")

    candidate = json.loads(args.candidate_manifest.read_text(encoding="utf-8"))
    if candidate["onnx"]["sha256"] != sha256(args.onnx):
        parser.error("candidate manifest ONNX SHA does not match --onnx")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    config.ONNX_DIR = args.onnx.parent
    config.HAR_DIR = args.out_dir / "har"
    config.HEF_DIR = args.out_dir / "hef"
    config.RESULTS_DIR = args.out_dir / "results"
    config.REPORTS_DIR = args.out_dir / "reports"
    config.ensure_dirs()
    module = config.Module("receiver_stage45", "high", "GFA-mode receiver feasibility candidate")
    result = {
        "kind": "receiver_stage45_gfa_candidate_gate",
        "status": "FAIL", "gfa_mode": candidate["gfa_mode"],
        "hw_arch": "hailo8", "dfc_version": importlib.metadata.version("hailo-dataflow-compiler"),
        "onnx": {"path": str(args.onnx), "sha256": sha256(args.onnx)},
        "candidate_manifest": str(args.candidate_manifest), "parse": None, "native": None,
        "context_switch": args.context_switch,
        "fp_optimized": {"status": "SKIP", "reason": "no graph-transforming .alls command"},
        "dummy_compile": None, "real_level0": None,
    }
    parsed = parse.parse_module(module, config.ONNX_DIR, config.HAR_DIR)
    result["parse"] = parsed.as_dict()
    if parsed.status == "PASS" and parsed.har_path:
        native_pass = False
        if args.skip_native:
            result["native"] = {"status": "SKIP", "reason": "separate CPU-native gate required"}
            native_pass = True
        else:
            native = emulate.native_check(module, Path(parsed.har_path), args.onnx)
            result["native"] = native.as_dict()
            native_pass = native.status == "PASS"
        if native_pass and not args.native_only:
            result["dummy_compile"] = _dummy_compile(
                Path(parsed.har_path), args.onnx, config.HEF_DIR, args.dummy_calib, args.opt_level,
                args.context_switch, args.skip_compile)
            result["real_level0"] = _real_level0(
                Path(parsed.har_path), args.onnx, args.calib_root, config.HEF_DIR,
                args.n_calib, args.opt_level, args.eval_samples,
                candidate["checkpoint"]["sha256"], args.context_switch, args.skip_compile)
            if result["real_level0"]["status"] == "PASS":
                result["status"] = "PASS"
        elif native_pass:
            result["status"] = "PASS"
    result_path = config.RESULTS_DIR / "gate.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"result: {result_path}")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
