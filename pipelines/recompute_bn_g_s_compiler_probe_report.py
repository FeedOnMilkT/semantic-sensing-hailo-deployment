#!/usr/bin/env python3
"""Recompute a compiler-probe report without rerunning SDK, DFC, or HailoRT."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BOARD_EVAL = ROOT / "board_eval"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BOARD_EVAL))

from compiler_probe_metrics import (  # noqa: E402
    code_comparison,
    dequantize_sdk_codes,
    sdk_codes_to_integer,
    summarize_code_comparisons,
)
from evaluate_bn_g_s_vstream_parity import (  # noqa: E402
    atomic_json,
    comparison,
    sha256_file,
    summarize_comparisons,
)


def _single_compile_result(root: Path) -> Path:
    matches = sorted((root / "probes").glob("*/compile_result.json"))
    if len(matches) != 1:
        raise ValueError(f"expected one probe compile_result.json, got {matches}")
    return matches[0]


def _probe_contract(compile_result: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if compile_result.get("status") != "PASS":
        raise ValueError("probe compile_result is not PASS")
    probe = compile_result.get("probe", {})
    logical = probe.get("logical_name")
    if not logical:
        source_layer = probe.get("source_layer")
        if not isinstance(source_layer, str) or "/" not in source_layer:
            raise ValueError(f"cannot recover probe logical name: {probe}")
        logical = source_layer.rsplit("/", 1)[-1]
    contract = compile_result.get("mapped_outputs", {}).get(logical)
    if not isinstance(contract, dict):
        raise ValueError(f"missing mapped output for probe {logical}")
    if contract.get("format_type") not in {"UINT8", "UINT16"}:
        raise ValueError(f"unsupported probe format: {contract}")
    quantization = contract.get("quantization", {})
    if not {"qp_scale", "qp_zp"}.issubset(quantization):
        raise ValueError(f"probe HEF quantization is incomplete: {contract}")
    return str(logical), contract


def _indices(keys: list[str], prefix: str, logical: str) -> set[int]:
    suffix = f"__{logical}"
    result = set()
    for key in keys:
        if key.startswith(prefix) and key.endswith(suffix):
            result.add(int(key[len(prefix):-len(suffix)]))
    return result


def recompute(
    references_path: Path,
    board_tensors_path: Path,
    compile_result_path: Path,
    out_path: Path,
) -> dict[str, Any]:
    for path in (references_path, board_tensors_path, compile_result_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if out_path.exists():
        raise FileExistsError(f"refusing to overwrite {out_path}")
    if out_path.resolve() in {
        references_path.resolve(),
        board_tensors_path.resolve(),
        compile_result_path.resolve(),
    }:
        raise ValueError("corrected report path must be distinct from every input")

    compile_result = json.loads(compile_result_path.read_text(encoding="utf-8"))
    logical, contract = _probe_contract(compile_result)
    format_type = str(contract["format_type"])
    expected_shape = tuple(int(x) for x in contract["shape_hwc"])
    quantization = contract["quantization"]

    with np.load(references_path, allow_pickle=False) as references, np.load(
        board_tensors_path, allow_pickle=False
    ) as board:
        reference_prefix = f"sdk_code__" if any(
            key.startswith("sdk_code__") and key.endswith(f"__{logical}")
            for key in references.files
        ) else "sdk__"
        reference_indices = _indices(references.files, reference_prefix, logical)
        raw_indices = _indices(board.files, "probe_raw__", logical)
        float_indices = _indices(board.files, "probe_float__", logical)
        if not reference_indices or reference_indices != raw_indices or raw_indices != float_indices:
            raise ValueError(
                "probe sample sets disagree: "
                f"reference={sorted(reference_indices)}, raw={sorted(raw_indices)}, "
                f"float={sorted(float_indices)}"
            )

        float_rows = []
        code_rows = []
        sample_rows = []
        for index in sorted(reference_indices):
            sdk_code_float = np.asarray(
                references[f"{reference_prefix}{index}__{logical}"]
            )
            board_raw = np.asarray(board[f"probe_raw__{index}__{logical}"])
            board_float = np.asarray(board[f"probe_float__{index}__{logical}"])
            if (
                tuple(sdk_code_float.shape) != expected_shape
                or tuple(board_raw.shape) != expected_shape
                or tuple(board_float.shape) != expected_shape
            ):
                raise ValueError(f"{index}: probe tensor shape contract changed")
            sdk_codes = sdk_codes_to_integer(sdk_code_float, format_type)
            sdk_dequantized = dequantize_sdk_codes(
                sdk_code_float, format_type, quantization
            )
            float_row = comparison(board_float, sdk_dequantized)
            code_row = code_comparison(board_raw, sdk_codes)
            float_rows.append(float_row)
            code_rows.append(code_row)
            sample_rows.append({
                "val_index": index,
                "sdk_code_range": [int(sdk_codes.min()), int(sdk_codes.max())],
                "board_raw_code_range": [int(board_raw.min()), int(board_raw.max())],
                "board_float_vs_sdk_dequantized": float_row,
                "board_raw_vs_sdk_codes": code_row,
            })

    source_qhar = compile_result.get("source_quantized_har", {})
    if (
        source_qhar.get("sha256_before")
        and source_qhar.get("sha256_after")
        and source_qhar["sha256_before"] != source_qhar["sha256_after"]
    ):
        raise ValueError("compile result says the immutable source qHAR changed")
    report = {
        "kind": "bn_g_s_compiler_probe_corrected_report_v1",
        "status": "PASS",
        "probe": {
            "logical_name": logical,
            "source_layer": compile_result.get("probe", {}).get("source_layer"),
            "shape_hwc": list(expected_shape),
            "format_type": format_type,
        },
        "correction": {
            "old_interpretation": "SDK internal float32 values treated as dequantized floats",
            "correct_interpretation": "SDK internal float32 values are integer quantization codes",
            "dequantization": "(sdk_code - qp_zp) * qp_scale",
            "quantization_source": "probe HEF mapped output in compile_result.json",
            "code_nmse_denominator": "sum(square(sdk_code))",
        },
        "probe_hef_output_contract": contract,
        "sample_count": len(sample_rows),
        "sample_indices": [row["val_index"] for row in sample_rows],
        "board_float_vs_sdk_dequantized": summarize_comparisons(float_rows),
        "board_raw_vs_sdk_codes": summarize_code_comparisons(code_rows),
        "samples": sample_rows,
        "inputs": {
            "references": {
                "path": str(references_path),
                "sha256": sha256_file(references_path),
                "probe_key_prefix": reference_prefix,
            },
            "board_tensors": {
                "path": str(board_tensors_path),
                "sha256": sha256_file(board_tensors_path),
            },
            "compile_result": {
                "path": str(compile_result_path),
                "sha256": sha256_file(compile_result_path),
            },
        },
        "policy": {
            "read_only_inputs": True,
            "old_reports_overwritten": False,
            "sdk_inference_run": False,
            "compile_run": False,
            "board_run": False,
            "diagnostic_only": True,
            "full976_not_run": True,
            "semantic_ranking": False,
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(out_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--references", type=Path)
    parser.add_argument("--board-tensors", type=Path)
    parser.add_argument("--compile-result", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    root = args.experiment_root.resolve()
    references = (args.references or root / "sdk_reference8/references.npz").resolve()
    board_tensors = (args.board_tensors or root / "board_tensors.npz").resolve()
    compile_result = (args.compile_result or _single_compile_result(root)).resolve()
    out = (args.out or root / "corrected_report.json").resolve()
    report = recompute(references, board_tensors, compile_result, out)
    print(json.dumps({
        "status": report["status"],
        "probe": report["probe"]["logical_name"],
        "sample_count": report["sample_count"],
        "out": str(out),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
