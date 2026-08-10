#!/usr/bin/env python3
"""Three-HEF ConvBN semantic-codec orchestration with strict chain contracts.

This thin entry point reuses the demo library for Hailo parse/NATIVE/PTQ.  It
has three explicit modes:

* ``plan`` validates ONNX/calibration identity and writes a frozen *intent*
  contract.  It performs no Hailo operation.
* ``parse-native`` runs only ONNX -> HAR -> SDK_NATIVE parity.  It does not
  calibrate, optimize, compile, AdaRound, or QFT.
* ``ptq`` is deliberately guarded by ``--execute-ptq``.  It permits bootstrap
  calibration only for ``analysis``.  ``hyper`` and ``synthesis_tail`` refuse
  it and require a later ``quantized_upstream_chain`` calibration handoff.

The CPU rANS boundaries make the calibration dependency directional:
analysis PTQ -> materialize z handoff -> hyper PTQ -> materialize y handoff ->
synthesis-tail PTQ.  The materializer is a separate explicit tool so no raw
activation is accidentally treated as a final board interface.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    from hailo_deploy.logging import reexec_if_unlogged
    reexec_if_unlogged(Path(__file__), sys.argv[1:])

from hailo_deploy import config, emulate, parse
from hailo_deploy.artifacts.layout import commit_or_fail, refuse_existing, stage_dir, write_json
from hailo_deploy.io.calib import load_calib_handoff
from hailo_deploy.io.hash import sha256


_SEGMENTS = ("analysis", "hyper", "synthesis_tail")
_FORMAL_MODE = {
    "analysis": "bootstrap_fp_codec",
    "hyper": "quantized_analysis_z_rans",
    "synthesis_tail": "quantized_analysis_hyper_y_rans",
}


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    value = _json(path)
    if value.get("schema") != "semantic_sensing.segmented_onnx.v1":
        raise ValueError(f"unsupported segments manifest: {value.get('schema')!r}")
    entries = {entry.get("id"): entry for entry in value.get("segments", [])}
    if tuple(entries) != _SEGMENTS and set(entries) != set(_SEGMENTS):
        raise ValueError(f"segments manifest must contain exactly {_SEGMENTS}, got {sorted(entries)}")
    for segment, entry in entries.items():
        onnx = Path(entry.get("onnx", ""))
        if not onnx.is_file():
            raise FileNotFoundError(f"{segment}: missing ONNX {onnx}")
        if sha256(onnx) != entry.get("sha256"):
            raise ValueError(f"{segment}: ONNX checksum mismatch")
    return value, entries


def _plan_config(path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    value = _json(path)
    if value.get("schema") != "semantic_sensing.segmented_quantization_plan.v1":
        raise ValueError(f"unsupported chain config: {value.get('schema')!r}")
    cfg_ids = [entry.get("id") for entry in value.get("segments", [])]
    if cfg_ids != list(_SEGMENTS):
        raise ValueError(f"chain config segment order must be {list(_SEGMENTS)}, got {cfg_ids}")
    expected_model = value.get("model", {})
    actual_model = manifest.get("model", {})
    for key in ("split", "bottleneck", "activation", "gfa_mode"):
        if expected_model.get(key) != actual_model.get(key):
            raise ValueError(f"model contract mismatch at {key}: {expected_model.get(key)!r} != {actual_model.get(key)!r}")
    if expected_model.get("image_nchw") != manifest.get("net_input_nchw"):
        raise ValueError("static image shape differs between config and ONNX manifest")
    return value


def _calibration_contract(
    calib_root: Path, entries: dict[str, dict[str, Any]], n_calib: int
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for segment in _SEGMENTS:
        root = calib_root / segment
        arrays, meta = load_calib_handoff(root, min(1, n_calib))
        if int(meta.get("sample_count", 0)) < n_calib:
            raise ValueError(f"{segment}: needs {n_calib} calib samples, has {meta.get('sample_count')}")
        if meta.get("segment") != segment:
            raise ValueError(f"{segment}: calibration meta has segment={meta.get('segment')!r}")
        if meta.get("onnx", {}).get("sha256") != entries[segment].get("sha256"):
            raise ValueError(f"{segment}: calibration/ONNX identity mismatch")
        expected_inputs = entries[segment].get("input_names")
        if set(arrays) != set(expected_inputs):
            raise ValueError(f"{segment}: loader inputs {sorted(arrays)} != ONNX {expected_inputs}")
        result[segment] = {
            "root": str(root),
            "mode": meta.get("calibration_mode"),
            "sample_count": int(meta["sample_count"]),
            "meta_sha256": sha256(root / "meta.json"),
            "input_names": sorted(arrays),
        }
    return result


def _intent(
    manifest_path: Path, config_path: Path, manifest: dict[str, Any], entries: dict[str, dict[str, Any]],
    calib: dict[str, dict[str, Any]], n_calib: int,
) -> dict[str, Any]:
    return {
        "schema": "semantic_sensing.segmented_interface_intent.v1",
        "status": "INTENT_ONLY",
        "reason_not_frozen": "Quantization scale/zero-point parameters exist only after final candidate PTQ and compilation.",
        "segments_manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
        "chain_config": {"path": str(config_path), "sha256": sha256(config_path)},
        "checkpoint": manifest["checkpoint"],
        "model": manifest["model"],
        "n_calib": n_calib,
        "hef_graphs": [
            {
                "id": segment,
                "deployment": entries[segment]["deployment"],
                "onnx": entries[segment]["onnx"],
                "onnx_sha256": entries[segment]["sha256"],
                "inputs": entries[segment]["input_names"],
                "outputs": entries[segment]["output_names"],
                "calibration": calib[segment],
            }
            for segment in _SEGMENTS
        ],
        "codec_boundaries": manifest["cpu_codec_boundaries"],
        "formal_order": [
            "PTQ analysis with bootstrap_fp_codec calibration",
            "materialize hyper calibration with pipelines/materialize_segmented_chain_calib.py from quantized analysis output + real z rANS",
            "PTQ hyper with quantized_analysis_z_rans calibration",
            "materialize synthesis_tail calibration with pipelines/materialize_segmented_chain_calib.py from quantized analysis + hyper outputs + real y rANS",
            "PTQ synthesis_tail with quantized_analysis_hyper_y_rans calibration",
            "freeze final HEF I/O quantization parameters from the selected candidate only",
        ],
        "transport_rule": "Only rANS byte streams traverse sender/receiver. No raw FP32 tensor is an inter-HEF deployment interface.",
        "context_rule": "DFC may partition each graph into contexts automatically; record count per HEF, do not force one context.",
    }


def _validated(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any], dict[str, dict[str, Any]]]:
    manifest_path = args.segments_manifest.resolve()
    config_path = args.chain_config.resolve()
    manifest, entries = _manifest(manifest_path)
    plan = _plan_config(config_path, manifest)
    calibration = _calibration_contract(args.calib_root.resolve(), entries, args.n_calib)
    return manifest, entries, plan, calibration


def run_plan(args: argparse.Namespace) -> int:
    manifest, entries, _plan, calibration = _validated(args)
    refuse_existing(args.out_root)
    temporary = stage_dir(args.out_root)
    try:
        write_json(temporary / "interface_intent.json", _intent(
            args.segments_manifest.resolve(), args.chain_config.resolve(), manifest, entries, calibration, args.n_calib
        ))
        write_json(temporary / "pipeline_status.json", {
            "schema": "semantic_sensing.segmented_pipeline_status.v1",
            "status": "PASS",
            "mode": "plan",
            "quantization_executed": False,
            "compile_executed": False,
            "native_executed": False,
        })
        final = commit_or_fail(temporary, args.out_root, True)
    except BaseException:
        commit_or_fail(temporary, args.out_root, False)
        raise
    print(json.dumps({"status": "PASS", "mode": "plan", "out": str(final), "ptq": "NOT RUN"}, indent=2))
    return 0


def run_parse_native(args: argparse.Namespace) -> int:
    manifest, entries, _plan, calibration = _validated(args)
    refuse_existing(args.out_root)
    temporary = stage_dir(args.out_root)
    result: dict[str, Any] = {
        "schema": "semantic_sensing.segmented_parse_native.v1",
        "status": "FAIL",
        "quantization_executed": False,
        "compile_executed": False,
        "segments": {},
        "error": None,
    }
    try:
        har_dir = temporary / "har"
        har_dir.mkdir()
        for segment in _SEGMENTS:
            entry = entries[segment]
            onnx = Path(entry["onnx"])
            module = config.Module(segment, entry["hailo_risk"], entry["risk_note"], onnx_stem=onnx.stem)
            parsed = parse.parse_module(module, onnx.parent, har_dir)
            native = None
            if parsed.status == "PASS" and not args.skip_native:
                native = emulate.native_check(module, Path(parsed.har_path), onnx, tol=args.native_tol)
            result["segments"][segment] = {
                "parse": parsed.as_dict(),
                "native": native.as_dict() if native is not None else "SKIP by --skip-native",
                "calibration_contract": calibration[segment],
            }
        statuses = []
        for segment_result in result["segments"].values():
            statuses.append(segment_result["parse"]["status"] == "PASS")
            native = segment_result["native"]
            statuses.append(native == "SKIP by --skip-native" or native["status"] == "PASS")
        result["status"] = "PASS" if all(statuses) else "FAIL"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    write_json(temporary / "results" / "parse_native.json", result)
    write_json(temporary / "interface_intent.json", _intent(
        args.segments_manifest.resolve(), args.chain_config.resolve(), manifest, entries, calibration, args.n_calib
    ))
    write_json(temporary / "pipeline_status.json", {
        "schema": "semantic_sensing.segmented_pipeline_status.v1",
        "status": result["status"],
        "mode": "parse-native",
        "quantization_executed": False,
        "compile_executed": False,
        "native_executed": not args.skip_native,
    })
    final = commit_or_fail(temporary, args.out_root, result["status"] == "PASS")
    print(json.dumps({"status": result["status"], "mode": "parse-native", "out": str(final),
                      "ptq": "NOT RUN"}, indent=2))
    return 0 if result["status"] == "PASS" else 1


def run_ptq(args: argparse.Namespace) -> int:
    if not args.execute_ptq:
        raise PermissionError("PTQ is disabled by default; pass --execute-ptq only after final candidate selection")
    manifest, entries, _plan, calibration = _validated(args)
    segment = args.segment
    expected_mode = _FORMAL_MODE[segment]
    actual_mode = calibration[segment]["mode"]
    if actual_mode != expected_mode:
        raise ValueError(
            f"{segment} needs calibration_mode={expected_mode!r}, got {actual_mode!r}; "
            "bootstrap tensors must not be used for downstream final PTQ"
        )
    parse_har = args.parse_run / "har" / f"{segment}_hailo_model.har"
    if not parse_har.is_file():
        raise FileNotFoundError(parse_har)
    output = args.out_root / "ptq" / segment
    if output.exists():
        raise FileExistsError(output)
    from pipelines.module_ptq import run_calib_root_ptq

    run_args = argparse.Namespace(
        module=segment,
        onnx=Path(entries[segment]["onnx"]),
        parse_har=parse_har,
        calib_root=args.calib_root / segment,
        out_dir=output,
        n_calib=args.n_calib,
        opt_level=args.opt_level,
        eval_samples=args.eval_samples,
        alls=args.alls,
        extra_alls=args.extra_alls,
        context_switch="allowed",
        skip_compile=args.skip_compile,
        expect_onnx_sha=True,
        expect_checkpoint_sha=manifest["checkpoint"]["sha256"],
    )
    result = run_calib_root_ptq(run_args)
    return 0 if result.get("status") == "PASS" else 1


def _artifact_from_result(result: dict[str, Any], key: str, segment: str) -> dict[str, Any]:
    record = result.get(key)
    if not isinstance(record, dict):
        raise ValueError(f"{segment}: missing {key} record; PTQ must compile before interface freeze")
    path = Path(record.get("path", ""))
    if not path.is_file() or sha256(path) != record.get("sha256"):
        raise ValueError(f"{segment}: {key} identity mismatch")
    return {"path": str(path), "sha256": record["sha256"], "bytes": path.stat().st_size}


def run_freeze_interface(args: argparse.Namespace) -> int:
    manifest, entries = _manifest(args.segments_manifest.resolve())
    _plan_config(args.chain_config.resolve(), manifest)
    refuse_existing(args.out_root)
    graphs = []
    for segment in _SEGMENTS:
        result_path = args.ptq_root / "ptq" / segment / "result.json"
        result = _json(result_path)
        if result.get("status") != "PASS" or result.get("module") != segment:
            raise ValueError(f"{segment}: PTQ result is not a PASS artifact: {result_path}")
        calibration = result.get("calibration", {})
        meta = calibration.get("metadata", {})
        if meta.get("onnx", {}).get("sha256") != entries[segment]["sha256"]:
            raise ValueError(f"{segment}: PTQ calibration did not prove ONNX identity")
        graphs.append({
            "id": segment,
            "deployment": entries[segment]["deployment"],
            "onnx": {"path": entries[segment]["onnx"], "sha256": entries[segment]["sha256"]},
            "hef": _artifact_from_result(result, "hef", segment),
            "quantized_har": _artifact_from_result(result, "quantized_har", segment),
            "compiled_har": _artifact_from_result(result, "compiled_har", segment),
            "num_contexts": result.get("num_contexts"),
            "input_mapping": result.get("input_mapping"),
            "calibration": calibration,
            "model_script": result.get("model_script"),
            "auto_alls": result.get("auto_alls"),
        })
    frozen = {
        "schema": "semantic_sensing.segmented_hef_interface.v1",
        "status": "FROZEN",
        "segments_manifest": {"path": str(args.segments_manifest.resolve()), "sha256": sha256(args.segments_manifest)},
        "chain_config": {"path": str(args.chain_config.resolve()), "sha256": sha256(args.chain_config)},
        "checkpoint": manifest["checkpoint"],
        "model": manifest["model"],
        "hef_graphs": graphs,
        "hyper_deployment": {
            "sender": "hyper HEF exact copy of graph id=hyper",
            "receiver": "hyper HEF exact copy of graph id=hyper",
            "sha256": next(graph["hef"]["sha256"] for graph in graphs if graph["id"] == "hyper"),
        },
        "cpu_codec_boundaries": manifest["cpu_codec_boundaries"],
        "transport_rule": "Only rANS byte streams cross endpoints; no raw FP32 or raw UINT8 tensor is a board interface.",
        "context_rule": "num_contexts is recorded per HEF; contexts are DFC scheduling partitions, not cross-codec graphs.",
    }
    temporary = stage_dir(args.out_root)
    try:
        write_json(temporary / "interface_manifest.json", frozen)
        write_json(temporary / "pipeline_status.json", {
            "schema": "semantic_sensing.segmented_pipeline_status.v1",
            "status": "PASS",
            "mode": "freeze-interface",
            "quantization_executed": True,
            "compile_executed": True,
        })
        final = commit_or_fail(temporary, args.out_root, True)
    except BaseException:
        commit_or_fail(temporary, args.out_root, False)
        raise
    print(json.dumps({"status": "PASS", "mode": "freeze-interface", "out": str(final)}, indent=2))
    return 0



def _base(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--segments-manifest", type=Path, required=True)
    parser.add_argument("--chain-config", type=Path,
                        default=ROOT / "configs" / "segmented_convbn_chain.json")
    parser.add_argument("--calib-root", type=Path, required=True)
    parser.add_argument("--n-calib", type=int, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    plan = sub.add_parser("plan", help="validate identities and write interface intent; no Hailo execution")
    _base(plan)
    plan.add_argument("--out-root", type=Path, required=True)

    native = sub.add_parser("parse-native", help="parse + SDK_NATIVE only; never optimize or compile")
    _base(native)
    native.add_argument("--out-root", type=Path, required=True)
    native.add_argument("--skip-native", action="store_true")
    native.add_argument("--native-tol", type=float, default=1e-3)

    ptq = sub.add_parser("ptq", help="guarded real PTQ; not used until final candidate selection")
    _base(ptq)
    ptq.add_argument("--segment", choices=_SEGMENTS, required=True)
    ptq.add_argument("--parse-run", type=Path, required=True)
    ptq.add_argument("--out-root", type=Path, required=True)
    ptq.add_argument("--opt-level", type=int, choices=range(5), required=True)
    ptq.add_argument("--eval-samples", type=int, default=4)
    ptq.add_argument("--alls", type=Path, default=None)
    ptq.add_argument("--extra-alls", type=Path, default=None)
    ptq.add_argument("--skip-compile", action="store_true")
    ptq.add_argument("--execute-ptq", action="store_true")
    freeze = sub.add_parser(
        "freeze-interface",
        help="freeze selected PTQ HEF/HAR identities into one deployment contract",
    )
    freeze.add_argument("--segments-manifest", type=Path, required=True)
    freeze.add_argument(
        "--chain-config", type=Path,
        default=ROOT / "configs" / "segmented_convbn_chain.json",
    )
    freeze.add_argument("--ptq-root", type=Path, required=True)
    freeze.add_argument("--out-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "freeze-interface":
        return run_freeze_interface(args)
    if args.n_calib <= 0:
        raise ValueError("--n-calib must be positive")
    if args.mode == "plan":
        return run_plan(args)
    if args.mode == "parse-native":
        return run_parse_native(args)
    return run_ptq(args)


if __name__ == "__main__":
    raise SystemExit(main())
