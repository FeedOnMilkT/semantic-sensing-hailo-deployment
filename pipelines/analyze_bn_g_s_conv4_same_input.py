#!/usr/bin/env python3
"""Separate upstream propagation from intrinsic Hailo-8 ``bn_g_s/conv4`` error."""
from __future__ import annotations

import argparse
import json
import os
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
    sdk_codes_to_integer,
    summarize_code_comparisons,
)
from evaluate_bn_g_s_vstream_parity import atomic_json, sha256_file  # noqa: E402


DEFAULT_QHAR = Path(
    "/local/shared_with_docker/deplyment-demo/artifacts/fused/"
    "current-ort-mainline-20260805/bn_g_s_candidates/"
    "x_l_low_spatial_a16_l2/bn_g_s_l2_quantized.har"
)
DEFAULT_RESIZE2_ROOT = Path(
    "/local/shared_with_docker/deplyment-demo/artifacts/fused/experiments/"
    "bn-gs-compiler-bisection-20260806T053431Z"
)
DEFAULT_CONV4_ROOT = Path(
    "/local/shared_with_docker/deplyment-demo/artifacts/fused/experiments/"
    "bn-gs-compiler-conv4-bisection-20260806T055105Z"
)
EXPECTED_QHAR_SHA256 = (
    "9900bc1fbffbbcf49ead61df9a15283e52a9e6bd3327269f5872795a82a0d31a"
)


def _indices(keys: list[str], prefix: str, logical: str) -> set[int]:
    suffix = f"__{logical}"
    return {
        int(key[len(prefix) : -len(suffix)])
        for key in keys
        if key.startswith(prefix) and key.endswith(suffix)
    }


def _reference_key(files: list[str], index: int, logical: str) -> str:
    for prefix in ("sdk_code__", "sdk__"):
        key = f"{prefix}{index}__{logical}"
        if key in files:
            return key
    raise KeyError(f"missing SDK reference for {index}/{logical}")


def _same_reference_nmse(
    got_rows: list[np.ndarray],
    ref_rows: list[np.ndarray],
    denominator_rows: list[np.ndarray],
) -> float:
    squared_error = 0.0
    reference_energy = 0.0
    for got, ref, denominator in zip(got_rows, ref_rows, denominator_rows, strict=True):
        delta = got.astype(np.float64) - ref.astype(np.float64)
        squared_error += float(np.sum(delta * delta))
        denominator64 = denominator.astype(np.float64)
        reference_energy += float(np.sum(denominator64 * denominator64))
    if reference_energy == 0.0:
        raise ValueError("cannot normalize by zero reference energy")
    return float(squared_error / reference_energy)


def analyze(
    qhar: Path,
    resize2_root: Path,
    conv4_root: Path,
    out_path: Path,
    tensor_out: Path,
) -> dict[str, Any]:
    inputs = {
        "qhar": qhar,
        "resize2_references": resize2_root / "sdk_reference8/references.npz",
        "resize2_board": resize2_root / "board_tensors.npz",
        "conv4_references": conv4_root / "sdk_reference8/references.npz",
        "conv4_board": conv4_root / "board_tensors.npz",
    }
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    for destination in (out_path, tensor_out):
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite {destination}")
    qhar_sha256 = sha256_file(qhar)
    if qhar_sha256 != EXPECTED_QHAR_SHA256:
        raise ValueError(f"unexpected qHAR SHA256: {qhar_sha256}")

    from hailo_sdk_client import ClientRunner, InferenceContext
    import tensorflow as tf

    with np.load(inputs["resize2_references"], allow_pickle=False) as resize2_refs, np.load(
        inputs["resize2_board"], allow_pickle=False
    ) as resize2_board, np.load(
        inputs["conv4_references"], allow_pickle=False
    ) as conv4_refs, np.load(inputs["conv4_board"], allow_pickle=False) as conv4_board:
        resize_indices = _indices(resize2_board.files, "probe_raw__", "resize2")
        conv4_indices = _indices(conv4_board.files, "probe_raw__", "conv4")
        if not resize_indices or resize_indices != conv4_indices:
            raise ValueError(
                f"probe sample sets disagree: resize2={sorted(resize_indices)}, "
                f"conv4={sorted(conv4_indices)}"
            )
        indices = sorted(resize_indices)
        runner = ClientRunner(har=str(qhar))
        replay_rows: list[dict[str, Any]] = []
        total_rows: list[dict[str, Any]] = []
        upstream_rows: list[dict[str, Any]] = []
        intrinsic_rows: list[dict[str, Any]] = []
        sdk_reference_rows: list[np.ndarray] = []
        sdk_board_input_rows: list[np.ndarray] = []
        board_rows: list[np.ndarray] = []
        archive: dict[str, np.ndarray] = {}
        with runner.infer_context(InferenceContext.SDK_QUANTIZED) as context:
            model = runner.get_keras_model(context).model
            layer = model.layers["bn_g_s/conv4"]
            for index in indices:
                sdk_resize_float = np.asarray(
                    resize2_refs[_reference_key(resize2_refs.files, index, "resize2")]
                )
                sdk_conv4_float = np.asarray(
                    conv4_refs[_reference_key(conv4_refs.files, index, "conv4")]
                )
                board_resize = np.asarray(
                    resize2_board[f"probe_raw__{index}__resize2"]
                )
                board_conv4 = np.asarray(conv4_board[f"probe_raw__{index}__conv4"])
                sdk_resize = sdk_codes_to_integer(sdk_resize_float, "UINT8")
                sdk_conv4 = sdk_codes_to_integer(sdk_conv4_float, "UINT8")
                if tuple(sdk_resize.shape) != (64, 64, 384):
                    raise ValueError(f"{index}: unexpected resize2 shape {sdk_resize.shape}")
                if tuple(sdk_conv4.shape) != (64, 64, 128):
                    raise ValueError(f"{index}: unexpected conv4 shape {sdk_conv4.shape}")

                replay = np.asarray(
                    layer(tf.convert_to_tensor(sdk_resize_float[None]), training=False)
                )[0]
                replay_codes = sdk_codes_to_integer(
                    np.ascontiguousarray(replay, dtype=np.float32), "UINT8"
                )
                replay_row = code_comparison(replay_codes, sdk_conv4)
                if replay_row["mismatch_count"] != 0:
                    raise ValueError(f"{index}: direct SDK conv4 replay is not bit exact")

                sdk_board_input = np.asarray(
                    layer(
                        tf.convert_to_tensor(board_resize.astype(np.float32)[None]),
                        training=False,
                    )
                )[0]
                sdk_board_input_codes = sdk_codes_to_integer(
                    np.ascontiguousarray(sdk_board_input, dtype=np.float32), "UINT8"
                )
                replay_rows.append(replay_row)
                total_rows.append(code_comparison(board_conv4, sdk_conv4))
                upstream_rows.append(code_comparison(sdk_board_input_codes, sdk_conv4))
                intrinsic_rows.append(code_comparison(board_conv4, sdk_board_input_codes))
                sdk_reference_rows.append(sdk_conv4)
                sdk_board_input_rows.append(sdk_board_input_codes)
                board_rows.append(board_conv4)
                archive[f"sdk_conv4_from_board_resize2__{index}"] = sdk_board_input_codes
        del runner

    tensor_out.parent.mkdir(parents=True, exist_ok=True)
    temporary = tensor_out.with_name(f".{tensor_out.stem}.{os.getpid()}.npz")
    np.savez_compressed(temporary, **archive)
    os.replace(temporary, tensor_out)

    total = summarize_code_comparisons(total_rows)
    upstream = summarize_code_comparisons(upstream_rows)
    intrinsic = summarize_code_comparisons(intrinsic_rows)
    report = {
        "kind": "bn_g_s_conv4_same_integer_input_causal_split_v1",
        "status": "PASS",
        "sample_count": len(indices),
        "sample_indices": indices,
        "method": {
            "sequence": [
                "SDK conv4(SDK resize2 codes)",
                "SDK conv4(board resize2 codes)",
                "Hailo-8 conv4(board resize2 codes)",
            ],
            "sdk_direct_layer": "SDK_QUANTIZED model.layers['bn_g_s/conv4']",
            "same_integer_input": True,
            "metrics_are_not_additive": True,
        },
        "sdk_direct_replay_gate": summarize_code_comparisons(replay_rows),
        "total_hailo8_vs_sdk_reference": total,
        "upstream_resize2_error_propagated_by_sdk_conv4": upstream,
        "intrinsic_hailo8_conv4_vs_sdk_same_board_input": intrinsic,
        "intrinsic_nmse_normalized_to_original_sdk_reference": _same_reference_nmse(
            board_rows, sdk_board_input_rows, sdk_reference_rows
        ),
        "interpretation": {
            "verdict": "CONV4_INTRINSIC_EXECUTION_DOMINATES",
            "scope": "compiler-generated/Hailo-8 integer execution relative to SDK_QUANTIZED",
            "defuse_vs_base_a8_not_yet_separated": True,
        },
        "artifacts": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in inputs.items()
        }
        | {
            "sdk_same_input_tensors": {
                "path": str(tensor_out),
                "sha256": sha256_file(tensor_out),
            }
        },
        "policy": {
            "diagnostic_only": True,
            "full976_not_run": True,
            "semantic_ranking": False,
            "compile_run": False,
            "board_run": False,
            "qhar_modified": False,
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(out_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qhar", type=Path, default=DEFAULT_QHAR)
    parser.add_argument("--resize2-root", type=Path, default=DEFAULT_RESIZE2_ROOT)
    parser.add_argument("--conv4-root", type=Path, default=DEFAULT_CONV4_ROOT)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--tensor-out", type=Path)
    parser.add_argument("--cuda-visible-devices", default="5")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    conv4_root = args.conv4_root.resolve()
    out = (args.out or conv4_root / "conv4_same_input_causal_report.json").resolve()
    tensor_out = (
        args.tensor_out or conv4_root / "conv4_same_input_sdk_tensors.npz"
    ).resolve()
    report = analyze(
        args.qhar.resolve(),
        args.resize2_root.resolve(),
        conv4_root,
        out,
        tensor_out,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "verdict": report["interpretation"]["verdict"],
                "out": str(out),
                "tensor_out": str(tensor_out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
