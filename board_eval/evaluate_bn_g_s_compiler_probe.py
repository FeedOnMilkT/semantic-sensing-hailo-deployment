#!/usr/bin/env python3
"""Board-side evaluator for one compile-only ``bn_g_s`` side-output probe."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from compiler_probe_metrics import (
    code_comparison,
    dequantize_sdk_codes,
    sdk_codes_to_integer,
    summarize_code_comparisons,
)
from evaluate_bn_g_s_vstream_parity import (
    atomic_json,
    compare_codes,
    comparison,
    enum_name,
    quant_row,
    run_mode,
    sha256_array,
    sha256_file,
    summarize_codes,
    summarize_comparisons,
)
from hybrid_ort_runtime_runner import cpu_temperature_c


FINAL_SHAPES = {
    "x_h_hat": (64, 64, 128),
    "x_l_hat": (32, 32, 256),
}
FINAL_VSTREAM_NAMES = {
    "x_h_hat": "bn_g_s/conv5",
    "x_l_hat": "bn_g_s/conv3",
}


def map_outputs(infos: list[Any], probe_spec: dict[str, Any] | None) -> dict[str, Any]:
    expected = {
        logical: {
            "vstream_name": FINAL_VSTREAM_NAMES[logical],
            "shape_hwc": shape,
        }
        for logical, shape in FINAL_SHAPES.items()
    }
    if probe_spec is not None:
        expected[probe_spec["logical_name"]] = {
            "vstream_name": probe_spec["vstream_name"],
            "shape_hwc": tuple(probe_spec["expected_shape_hwc"]),
        }
    mapped: dict[str, Any] = {}
    for logical, contract in expected.items():
        matches = [
            info
            for info in infos
            if info.name == contract["vstream_name"]
            and tuple(info.shape) == tuple(contract["shape_hwc"])
        ]
        if len(matches) != 1:
            actual = [(info.name, tuple(info.shape)) for info in infos]
            raise ValueError(f"cannot map {logical} as {contract}: {actual}")
        mapped[logical] = matches[0]
    if len(mapped) != len(infos):
        raise ValueError(f"unexpected extra HEF outputs: {[info.name for info in infos]}")
    return mapped


def expected_dtype(info: Any) -> np.dtype:
    name = enum_name(info.format.type)
    if name == "UINT8":
        return np.dtype(np.uint8)
    if name == "UINT16":
        return np.dtype(np.uint16)
    raise ValueError(f"unsupported compiled output format: {name}")


def run_hef(
    path: Path,
    expected_sha: str,
    samples: list[dict[str, Any]],
    probe_spec: dict[str, Any] | None,
) -> dict[str, Any]:
    from hailo_platform import (
        ConfigureParams,
        FormatType,
        HailoStreamInterface,
        HEF,
        VDevice,
    )

    if sha256_file(path) != expected_sha:
        raise ValueError(f"HEF hash mismatch: {path}")
    hef = HEF(str(path))
    input_infos = list(hef.get_input_vstream_infos())
    output_infos = list(hef.get_output_vstream_infos())
    if len(input_infos) != 1 or tuple(input_infos[0].shape) != (16, 16, 192):
        raise ValueError(f"unexpected HEF input contract: {path}")
    mapped = map_outputs(output_infos, probe_spec)
    expected_formats = {"x_h_hat": "UINT8", "x_l_hat": "UINT16"}
    if probe_spec is not None:
        expected_formats[probe_spec["logical_name"]] = probe_spec[
            "expected_format_type"
        ]
    for logical, format_type in expected_formats.items():
        if enum_name(mapped[logical].format.type) != format_type:
            raise ValueError(
                f"{logical}: {enum_name(mapped[logical].format.type)} != {format_type}"
            )

    temperature_start = cpu_temperature_c()
    vdevice = VDevice()
    try:
        configure = ConfigureParams.create_from_hef(
            hef, interface=HailoStreamInterface.PCIe
        )
        groups = vdevice.configure(hef, configure)
        if len(groups) != 1:
            raise ValueError("expected exactly one configured network group")
        network_group = groups[0]
        float_outputs, float_meta = run_mode(
            network_group,
            samples,
            input_infos[0].name,
            FormatType.FLOAT32,
            FormatType.FLOAT32,
            None,
        )
        raw_outputs, raw_meta = run_mode(
            network_group,
            samples,
            input_infos[0].name,
            FormatType.FLOAT32,
            FormatType.AUTO,
            None,
        )
    finally:
        vdevice.release()

    logical_float: list[dict[str, np.ndarray]] = []
    logical_raw: list[dict[str, np.ndarray]] = []
    for position in range(len(samples)):
        floats: dict[str, np.ndarray] = {}
        raws: dict[str, np.ndarray] = {}
        for logical, info in mapped.items():
            float_value = np.ascontiguousarray(float_outputs[position][info.name], dtype=np.float32)
            raw_value = np.ascontiguousarray(raw_outputs[position][info.name])
            if raw_value.dtype != expected_dtype(info):
                raise ValueError(
                    f"{logical}: actual AUTO dtype {raw_value.dtype} != {expected_dtype(info)}"
                )
            if tuple(float_value.shape) != tuple(info.shape) or raw_value.shape != float_value.shape:
                raise ValueError(f"{logical}: runtime output shape mismatch")
            floats[logical] = float_value
            raws[logical] = raw_value
        logical_float.append(floats)
        logical_raw.append(raws)
    return {
        "static": {
            "hef": {"path": str(path), "sha256": expected_sha},
            "inputs": [quant_row(info) for info in input_infos],
            "outputs": [quant_row(info) for info in output_infos],
            "mapped_vstreams": {logical: info.name for logical, info in mapped.items()},
            "mapped_outputs": {logical: quant_row(info) for logical, info in mapped.items()},
        },
        "float_outputs": logical_float,
        "raw_outputs": logical_raw,
        "mode_parameters": {"float32": float_meta, "auto": raw_meta},
        "temperature_c": {"start": temperature_start, "end": cpu_temperature_c()},
    }


def summarize_paths(store: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return {name: summarize_comparisons(rows) for name, rows in store.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tensor-out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.tensor_out.exists():
        raise FileExistsError("refusing to overwrite an existing compiler-probe result")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("kind") != "bn_g_s_compiler_probe_manifest_v2":
        raise ValueError("unexpected compiler-probe manifest kind")
    probe_spec = manifest["probe"]
    probe_logical = probe_spec["logical_name"]
    root = args.manifest.parent
    references_path = root / manifest["references"]["filename"]
    if sha256_file(references_path) != manifest["references"]["sha256"]:
        raise ValueError("SDK reference bundle hash mismatch")
    references = np.load(references_path, allow_pickle=False)
    for key, expected in manifest["references"]["arrays"].items():
        if key not in references or sha256_array(references[key]) != expected:
            raise ValueError(f"SDK reference array mismatch: {key}")

    bundle_result = args.bundle_root / "result.json"
    if sha256_file(bundle_result) != manifest["golden_bundle"]["result_sha256"]:
        raise ValueError("frozen y_hat bundle result mismatch")
    bundle = json.loads(bundle_result.read_text(encoding="utf-8"))
    if bundle.get("checkpoint_sha256") != manifest["checkpoint_sha256"]:
        raise ValueError("checkpoint mismatch")
    rows = {int(row["val_index"]): row for row in bundle["rows"]}
    samples: list[dict[str, Any]] = []
    for index in [int(x) for x in manifest["sample_indices"]]:
        value = np.ascontiguousarray(
            np.load(args.bundle_root / "latent_y_hat" / f"{index}.npy", allow_pickle=False),
            dtype=np.float32,
        )
        digest = sha256_array(value)
        if digest != rows[index]["sha256"]:
            raise ValueError(f"frozen y_hat hash mismatch for {index}")
        samples.append({"index": index, "y_hat": value})

    formal = run_hef(
        root / manifest["formal"]["hef_filename"],
        manifest["formal"]["hef_sha256"],
        samples,
        None,
    )
    time.sleep(1.0)
    probe = run_hef(
        root / manifest["probe"]["hef_filename"],
        manifest["probe"]["hef_sha256"],
        samples,
        probe_spec,
    )

    formal_vs_sdk_rows = {name: [] for name in FINAL_SHAPES}
    probe_vs_sdk_rows = {name: [] for name in FINAL_SHAPES}
    probe_float_vs_sdk_dequantized_rows = []
    probe_code_vs_sdk_code_rows = []
    represent_float_rows = {name: [] for name in FINAL_SHAPES}
    represent_code_rows = {name: [] for name in FINAL_SHAPES}
    archive: dict[str, np.ndarray] = {}
    sample_reports = []
    for position, sample in enumerate(samples):
        index = sample["index"]
        sample_report = {"val_index": index, "outputs": {}}
        for logical in FINAL_SHAPES:
            sdk = references[f"sdk__{index}__{logical}"]
            formal_float = formal["float_outputs"][position][logical]
            probe_float = probe["float_outputs"][position][logical]
            formal_raw = formal["raw_outputs"][position][logical]
            probe_raw = probe["raw_outputs"][position][logical]
            formal_vs_sdk_rows[logical].append(comparison(formal_float, sdk))
            probe_vs_sdk_rows[logical].append(comparison(probe_float, sdk))
            represent_float_rows[logical].append(comparison(probe_float, formal_float))
            represent_code_rows[logical].append(compare_codes(probe_raw, formal_raw))
            archive[f"formal_float__{index}__{logical}"] = formal_float
            archive[f"formal_raw__{index}__{logical}"] = formal_raw
            archive[f"probe_float__{index}__{logical}"] = probe_float
            archive[f"probe_raw__{index}__{logical}"] = probe_raw
            sample_report["outputs"][logical] = {
                "formal_vs_sdk": formal_vs_sdk_rows[logical][-1],
                "probe_vs_sdk": probe_vs_sdk_rows[logical][-1],
                "probe_vs_formal": represent_float_rows[logical][-1],
                "probe_vs_formal_codes": represent_code_rows[logical][-1],
            }
        code_key = manifest["references"]["probe_code_key_template"].format(
            index=index
        )
        sdk_code_float = references[code_key]
        probe_contract = probe["static"]["mapped_outputs"][probe_logical]
        sdk_codes = sdk_codes_to_integer(
            sdk_code_float, probe_contract["format_type"]
        )
        sdk_dequantized = dequantize_sdk_codes(
            sdk_code_float,
            probe_contract["format_type"],
            probe_contract["quantization"],
        )
        probe_float = probe["float_outputs"][position][probe_logical]
        probe_raw = probe["raw_outputs"][position][probe_logical]
        probe_float_vs_sdk_dequantized_rows.append(
            comparison(probe_float, sdk_dequantized)
        )
        probe_code_vs_sdk_code_rows.append(code_comparison(probe_raw, sdk_codes))
        archive[f"sdk_dequantized__{index}__{probe_logical}"] = sdk_dequantized
        archive[f"probe_float__{index}__{probe_logical}"] = probe_float
        archive[f"probe_raw__{index}__{probe_logical}"] = probe_raw
        sample_report["outputs"][probe_logical] = {
            "board_float_vs_sdk_dequantized": probe_float_vs_sdk_dequantized_rows[-1],
            "board_raw_vs_sdk_codes": probe_code_vs_sdk_code_rows[-1],
        }
        sample_reports.append(sample_report)

    represent_float = summarize_paths(represent_float_rows)
    represent_codes = {
        name: summarize_codes(rows) for name, rows in represent_code_rows.items()
    }
    represent_pass = all(
        represent_float[name]["max_abs"]["max"] <= 1e-6
        and represent_codes[name]["mismatch_count"] == 0
        for name in FINAL_SHAPES
    )

    args.tensor_out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.tensor_out.with_name(f".{args.tensor_out.stem}.{os.getpid()}.npz")
    np.savez_compressed(temporary, **archive)
    os.replace(temporary, args.tensor_out)
    tensor_sha = sha256_file(args.tensor_out)
    report = {
        "kind": "bn_g_s_compiler_probe_board_result_v2",
        "status": "PASS",
        "probe": probe_spec,
        "sample_count": len(samples),
        "sample_indices": [sample["index"] for sample in samples],
        "formal_vs_sdk": summarize_paths(formal_vs_sdk_rows),
        "probe_vs_sdk": summarize_paths(probe_vs_sdk_rows),
        "probe_intermediate_corrected": {
            "logical_name": probe_logical,
            "sdk_representation": "integer quantization codes stored in float32",
            "dequantization_source": "probe HEF output qp_scale/qp_zp",
            "probe_hef_output_contract": probe["static"]["mapped_outputs"][probe_logical],
            "board_float_vs_sdk_dequantized": summarize_comparisons(
                probe_float_vs_sdk_dequantized_rows
            ),
            "board_raw_vs_sdk_codes": summarize_code_comparisons(
                probe_code_vs_sdk_code_rows
            ),
        },
        "representativeness_gate": {
            "status": "PASS" if represent_pass else "MISMATCH",
            "float32_probe_vs_formal": represent_float,
            "raw_codes_probe_vs_formal": represent_codes,
            "max_abs_tolerance": 1e-6,
            "required_raw_mismatch_count": 0,
        },
        "runtime_contract": {
            "formal": formal["static"],
            "probe": probe["static"],
            "formal_modes": formal["mode_parameters"],
            "probe_modes": probe["mode_parameters"],
        },
        "temperature_c": {"formal": formal["temperature_c"], "probe": probe["temperature_c"]},
        "samples": sample_reports,
        "board_tensors": {
            "path": str(args.tensor_out),
            "sha256": tensor_sha,
            "bytes": args.tensor_out.stat().st_size,
            "array_count": len(archive),
        },
        "policy": {
            "diagnostic_only": True,
            "full976_not_run": True,
            "semantic_ranking": False,
            "bn_g_s_only": True,
        },
    }
    atomic_json(args.out, report)
    print(json.dumps({
        "status": "PASS",
        "representativeness": report["representativeness_gate"]["status"],
        "out": str(args.out),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
