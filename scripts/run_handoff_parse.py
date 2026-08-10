#!/usr/bin/env python3
"""Parse and NATIVE-check one immutable training handoff for Hailo-8.

This is deliberately independent of the old GDN demo configuration: all ONNX
paths and checkpoint identities come from the supplied handoff directory.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

# Gate 3 must compare the Hailo NATIVE graph in CPU/FP32.  Letting this
# process inherit the PTQ GPU selection enables XLA/TF32 and creates false
# mismatches; PTQ is launched later in separate GPU-enabled processes.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hailo_deploy.emulate import native_check
from hailo_deploy.io.hash import sha256
from hailo_deploy.io.handoff import validate_handoff
from hailo_deploy.onnx_io import signature


MODULES = (
    "sender_backbone", "bn_align_h", "bn_g_a", "bn_h_a", "bn_h_s", "bn_g_s",
    "receiver_stage4", "receiver_stage5", "receiver_dappm", "receiver_head",
)


def op_types(runner) -> list[str]:
    layers = runner.get_hn_dict().get("layers", {})
    return sorted({str(layer.get("type", "?")) for layer in layers.values()})


def parse_one(onnx: Path, name: str, destination: Path) -> dict:
    from hailo_sdk_client import ClientRunner

    try:
        runner = ClientRunner(hw_arch="hailo8")
        try:
            runner.translate_onnx_model(str(onnx), name)
            used_shapes = False
        except Exception as first_error:
            sig = signature(onnx)
            runner = ClientRunner(hw_arch="hailo8")
            try:
                runner.translate_onnx_model(
                    str(onnx), name,
                    net_input_shapes={input_name: shape for input_name, shape in sig.inputs},
                )
                used_shapes = True
            except Exception:
                return {
                    "name": name, "status": "FAIL", "onnx": str(onnx),
                    "error": f"clean parse: {type(first_error).__name__}: {first_error}\n"
                             f"static-shape retry:\n{traceback.format_exc()}",
                }
        runner.save_har(str(destination))
        native = native_check(SimpleNamespace(name=name), destination, onnx).as_dict()
        return {
            "name": name,
            "status": "PASS" if native["status"] == "PASS" else "FAIL",
            "onnx": {"path": str(onnx), "sha256": sha256(onnx)},
            "parse_har": {"path": str(destination), "sha256": sha256(destination),
                          "bytes": destination.stat().st_size},
            "used_net_input_shapes": used_shapes,
            "hn_op_types": op_types(runner),
            "native": native,
        }
    except Exception as error:
        return {"name": name, "status": "FAIL", "onnx": str(onnx),
                "error": f"{type(error).__name__}: {error}\n{traceback.format_exc()}"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--modules", nargs="*", choices=MODULES, default=MODULES)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.out_dir}")
    temporary = args.out_dir.parent / f".{args.out_dir.name}.tmp-{__import__('os').getpid()}"
    temporary.mkdir(parents=True)
    results: list[dict] = []
    try:
        for name in args.modules:
            # The common validator checks immutable model/calibration/golden
            # identities before DFC receives any input from this candidate.
            _identity, _item, _meta, onnx, _calib, _indices, _golden = validate_handoff(args.handoff_root, name)
            result = parse_one(onnx, name, temporary / f"{name}_hailo_model.har")
            results.append(result)
            print(f"[{result['status']}] {name}", flush=True)
        status = "PASS" if all(item["status"] == "PASS" for item in results) else "FAIL"
        report = {"kind": "handoff_parse_native", "status": status,
                  "handoff_root": str(args.handoff_root), "modules": results}
        (temporary / "parse_results.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                                                        encoding="utf-8")
        if status != "PASS":
            raise RuntimeError("one or more modules failed parse/NATIVE validation")
        __import__('os').replace(temporary, args.out_dir)
        print(json.dumps({"status": "PASS", "out": str(args.out_dir)}, indent=2))
        return 0
    except Exception:
        failed = args.out_dir.parent / f"{args.out_dir.name}.FAILED"
        if failed.exists():
            shutil.rmtree(temporary, ignore_errors=True)
            raise FileExistsError(failed)
        __import__('os').replace(temporary, failed)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
