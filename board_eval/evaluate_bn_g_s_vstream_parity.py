#!/usr/bin/env python3
"""Gate-8 raw/FLOAT32 VStream parity probe for frozen bn_g_s HEFs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from bn_g_s_branch_attribution_runtime import tensor_drift
from hybrid_ort_runtime_runner import OrtModuleEngine, cpu_temperature_c


EXPECTED_OUTPUTS = {
    "x_h_hat": (64, 64, 128),
    "x_l_hat": (32, 32, 256),
}


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def enum_name(value: object) -> str:
    return str(getattr(value, "name", value))


def quant_row(info: object) -> dict:
    quant = info.quant_info
    return {
        "name": info.name,
        "shape_hwc": [int(x) for x in info.shape],
        "format_type": enum_name(info.format.type),
        "format_order": enum_name(info.format.order),
        "quantization": {
            "qp_scale": float(quant.qp_scale),
            "qp_zp": float(quant.qp_zp),
            "limvals_min": float(quant.limvals_min),
            "limvals_max": float(quant.limvals_max),
        },
    }


def map_outputs(infos: list[object]) -> dict[str, object]:
    mapped: dict[str, object] = {}
    for logical, shape in EXPECTED_OUTPUTS.items():
        matches = [info for info in infos if tuple(info.shape) == shape]
        if len(matches) != 1:
            actual = [(info.name, tuple(info.shape)) for info in infos]
            raise ValueError(f"cannot map {logical} by shape {shape}: {actual}")
        mapped[logical] = matches[0]
    return mapped


def summarize_scalars(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("cannot summarize an empty sequence")
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "min": float(array.min()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def comparison(candidate: np.ndarray, reference: np.ndarray) -> dict:
    got = np.asarray(candidate, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    if got.shape != ref.shape:
        raise ValueError(f"comparison shape mismatch: {got.shape} != {ref.shape}")
    row = tensor_drift(got, ref)
    delta = got - ref
    row.update({
        "sse": float(np.sum(delta * delta)),
        "reference_energy": float(np.sum(ref * ref)),
    })
    return row


def summarize_comparisons(rows: list[dict]) -> dict:
    sse = float(sum(row["sse"] for row in rows))
    energy = float(sum(row["reference_energy"] for row in rows))
    return {
        "aggregate_nmse": None if energy == 0.0 else float(sse / energy),
        **{
            metric: summarize_scalars([float(row[metric]) for row in rows])
            for metric in ("cosine", "mae", "rmse", "max_abs")
        },
    }


def compare_codes(candidate: np.ndarray, reference: np.ndarray) -> dict:
    got = np.asarray(candidate)
    ref = np.asarray(reference)
    if got.shape != ref.shape or got.dtype != ref.dtype:
        raise ValueError(
            f"raw code contract mismatch: {got.shape}/{got.dtype} != {ref.shape}/{ref.dtype}"
        )
    delta = got.astype(np.int64) - ref.astype(np.int64)
    mismatch = int(np.count_nonzero(delta))
    return {
        "element_count": int(got.size),
        "mismatch_count": mismatch,
        "agreement": float(1.0 - mismatch / got.size),
        "max_abs_code_diff": int(np.max(np.abs(delta))),
    }


def summarize_codes(rows: list[dict]) -> dict:
    total = int(sum(row["element_count"] for row in rows))
    mismatch = int(sum(row["mismatch_count"] for row in rows))
    return {
        "element_count": total,
        "mismatch_count": mismatch,
        "agreement": float(1.0 - mismatch / total),
        "max_abs_code_diff": int(max(row["max_abs_code_diff"] for row in rows)),
    }


def official_quantize(value: np.ndarray, info: object) -> np.ndarray:
    from hailo_platform.pyhailort.pyhailort import HailoRTTransformUtils

    source = np.ascontiguousarray(value, dtype=np.float32)
    dtype = np.uint8 if enum_name(info.format.type) == "UINT8" else np.uint16
    result = np.empty(source.shape, dtype=dtype)
    HailoRTTransformUtils.quantize_input_buffer(
        source, result, int(source.size), info.quant_info
    )
    return result


def formula_quantize(value: np.ndarray, info: object) -> np.ndarray:
    quant = info.quant_info
    scale, zp = float(quant.qp_scale), float(quant.qp_zp)
    qmin = int(round(float(quant.limvals_min) / scale + zp))
    qmax = int(round(float(quant.limvals_max) / scale + zp))
    dtype = np.uint8 if enum_name(info.format.type) == "UINT8" else np.uint16
    return np.clip(np.rint(np.asarray(value, dtype=np.float64) / scale + zp), qmin, qmax).astype(dtype)


def official_dequantize(value: np.ndarray, info: object) -> np.ndarray:
    from hailo_platform.pyhailort.pyhailort import HailoRTTransformUtils

    source = np.ascontiguousarray(value)
    result = np.empty(source.shape, dtype=np.float32)
    HailoRTTransformUtils.dequantize_output_buffer(
        source, result, int(source.size), info.quant_info
    )
    return result


def formula_dequantize(value: np.ndarray, info: object) -> np.ndarray:
    quant = info.quant_info
    return (
        (np.asarray(value, dtype=np.float32) - float(quant.qp_zp))
        * float(quant.qp_scale)
    ).astype(np.float32)


def resolved_params(params: dict) -> dict:
    result = {}
    for name, value in params.items():
        user = value.user_buffer_format
        result[name] = {
            "format_type": enum_name(user.type),
            "format_order": enum_name(user.order),
        }
    return result


def run_mode(network_group: object, samples: list[dict], input_name: str,
             in_format: object, out_format: object, raw_inputs: list[np.ndarray] | None) -> tuple[list[dict], dict]:
    from hailo_platform import InferVStreams, InputVStreamParams, OutputVStreamParams

    in_params = InputVStreamParams.make_from_network_group(network_group, format_type=in_format)
    out_params = OutputVStreamParams.make_from_network_group(network_group, format_type=out_format)
    outputs: list[dict] = []
    timings: list[float] = []
    with InferVStreams(network_group, in_params, out_params) as pipeline:
        with network_group.activate():
            for position, sample in enumerate(samples):
                source = sample["y_hat"] if raw_inputs is None else raw_inputs[position]
                start = time.perf_counter_ns()
                raw = pipeline.infer({input_name: np.ascontiguousarray(source)[None, ...]})
                timings.append((time.perf_counter_ns() - start) / 1e6)
                outputs.append({name: np.asarray(value[0] if value.ndim == 4 else value)
                                for name, value in raw.items()})
    return outputs, {
        "input_params": resolved_params(in_params),
        "output_params": resolved_params(out_params),
        "infer_wall_ms": summarize_scalars(timings),
    }


def ref_key(kind: str, index: int, branch: str, arm: str | None = None) -> str:
    fields = [kind]
    if arm is not None:
        fields.append(arm)
    fields.extend([str(index), branch])
    return "__".join(fields)


def add_comparison(store: dict, label: str, branch: str,
                   candidate: np.ndarray, reference: np.ndarray) -> dict:
    row = comparison(candidate, reference)
    store.setdefault(label, {}).setdefault(branch, []).append(row)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--raw-out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.raw_out.exists():
        raise FileExistsError("refusing to overwrite an existing parity artifact")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    root = args.manifest.parent
    references_path = root / manifest["references"]["filename"]
    if sha256_file(references_path) != manifest["references"]["sha256"]:
        raise ValueError("host reference bundle hash mismatch")
    bundle_result_path = args.bundle_root / "result.json"
    if sha256_file(bundle_result_path) != manifest["golden_bundle"]["result_sha256"]:
        raise ValueError("golden bundle result hash mismatch")
    bundle = json.loads(bundle_result_path.read_text(encoding="utf-8"))
    if bundle.get("checkpoint_sha256") != manifest["checkpoint_sha256"]:
        raise ValueError("checkpoint mismatch")

    indices = [int(x) for x in manifest["sample_indices"]]
    rows_by_index = {int(row["val_index"]): row for row in bundle["rows"]}
    samples: list[dict] = []
    for index in indices:
        path = args.bundle_root / "latent_y_hat" / f"{index}.npy"
        value = np.ascontiguousarray(np.load(path, allow_pickle=False), dtype=np.float32)
        digest = sha256_array(value)
        if digest != rows_by_index[index]["sha256"]:
            raise ValueError(f"golden y_hat hash mismatch for {index}")
        samples.append({"index": index, "y_hat": value, "sha256": digest})

    references = np.load(references_path, allow_pickle=False)
    for key, expected in manifest["references"]["arrays"].items():
        if key not in references or sha256_array(references[key]) != expected:
            raise ValueError(f"host reference array mismatch: {key}")

    fp_spec = {
        "engine": "ort_cpu_fp32", "name": "bn_g_s",
        "onnx_filename": manifest["onnx"]["filename"],
        "input_names": ["latent_y_hat"], "in_shapes_nchw": [[1, 192, 16, 16]],
        "output_names": ["x_h_hat", "x_l_hat"],
        "out_shapes_nchw": [[1, 128, 64, 64], [1, 256, 32, 32]],
    }
    if sha256_file(root / fp_spec["onnx_filename"]) != manifest["onnx"]["sha256"]:
        raise ValueError("bn_g_s ONNX hash mismatch")
    board_ort = OrtModuleEngine(root, fp_spec, int(manifest.get("ort_threads", 4)))
    ort_rows: dict[str, list[dict]] = {name: [] for name in EXPECTED_OUTPUTS}
    board_ort_outputs: list[dict] = []
    try:
        for sample in samples:
            got, _timing = board_ort.infer({"latent_y_hat": sample["y_hat"]})
            board_ort_outputs.append(got)
            for branch in EXPECTED_OUTPUTS:
                host = references[ref_key("ort", sample["index"], branch)]
                ort_rows[branch].append(comparison(got[branch], host))
    finally:
        board_ort.close()

    from hailo_platform import (ConfigureParams, FormatType, HailoStreamInterface,
                                HEF, VDevice)

    raw_archive: dict[str, np.ndarray] = {}
    arm_reports: dict[str, dict] = {}
    progress = args.out.with_suffix(".progress.json")
    for arm_position, arm in enumerate(manifest["arms"], start=1):
        name = arm["name"]
        hef_path = root / arm["hef_filename"]
        if sha256_file(hef_path) != arm["hef_sha256"]:
            raise ValueError(f"{name}: HEF hash mismatch")
        hef = HEF(str(hef_path))
        input_infos = list(hef.get_input_vstream_infos())
        output_infos = list(hef.get_output_vstream_infos())
        if len(input_infos) != 1 or tuple(input_infos[0].shape) != (16, 16, 192):
            raise ValueError(f"{name}: unexpected input contract")
        input_info = input_infos[0]
        mapped = map_outputs(output_infos)
        raw_inputs = [official_quantize(sample["y_hat"], input_info) for sample in samples]
        input_formula_rows = [compare_codes(
            raw_inputs[position], formula_quantize(sample["y_hat"], input_info)
        ) for position, sample in enumerate(samples)]

        temperature_start = cpu_temperature_c()
        vdevice = VDevice()
        try:
            configure = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
            groups = vdevice.configure(hef, configure)
            if len(groups) != 1:
                raise ValueError(f"{name}: expected one configured network group")
            network_group = groups[0]
            float_float, ff_meta = run_mode(
                network_group, samples, input_info.name,
                FormatType.FLOAT32, FormatType.FLOAT32, None,
            )
            float_raw, fr_meta = run_mode(
                network_group, samples, input_info.name,
                FormatType.FLOAT32, FormatType.AUTO, None,
            )
            raw_float, rf_meta = run_mode(
                network_group, samples, input_info.name,
                FormatType.AUTO, FormatType.FLOAT32, raw_inputs,
            )
            raw_raw, rr_meta = run_mode(
                network_group, samples, input_info.name,
                FormatType.AUTO, FormatType.AUTO, raw_inputs,
            )
        finally:
            vdevice.release()

        comparisons: dict[str, dict[str, list[dict]]] = {}
        raw_code_rows = {branch: [] for branch in EXPECTED_OUTPUTS}
        output_formula_rows = {branch: [] for branch in EXPECTED_OUTPUTS}
        sample_reports = []
        for position, sample in enumerate(samples):
            sample_report = {"val_index": sample["index"], "outputs": {}}
            for branch, info in mapped.items():
                stream = info.name
                ff = float_float[position][stream]
                fr = float_raw[position][stream]
                rf = raw_float[position][stream]
                rr = raw_raw[position][stream]
                expected_dtype = np.uint8 if enum_name(info.format.type) == "UINT8" else np.uint16
                if fr.dtype != expected_dtype or rr.dtype != expected_dtype:
                    raise ValueError(
                        f"{name}/{branch}: AUTO dtype {fr.dtype}/{rr.dtype} != {expected_dtype}"
                    )
                fr_deq = official_dequantize(fr, info)
                rr_deq = official_dequantize(rr, info)
                fr_formula = formula_dequantize(fr, info)
                raw_code_rows[branch].append(compare_codes(fr, rr))
                output_formula_rows[branch].append(comparison(fr_deq, fr_formula))
                board_fp = board_ort_outputs[position][branch]
                host_ort = references[ref_key("ort", sample["index"], branch)]
                sdk = references[ref_key("sdk", sample["index"], branch, name)]
                paths = {
                    "board_float_vs_board_ort": (ff, board_fp),
                    "board_float_vs_host_ort": (ff, host_ort),
                    "board_float_vs_sdk": (ff, sdk),
                    "float_raw_dequant_vs_board_float": (fr_deq, ff),
                    "raw_float_vs_board_float": (rf, ff),
                    "raw_raw_dequant_vs_raw_float": (rr_deq, rf),
                    "raw_raw_dequant_vs_sdk": (rr_deq, sdk),
                }
                sample_report["outputs"][branch] = {
                    label: add_comparison(comparisons, label, branch, got, ref)
                    for label, (got, ref) in paths.items()
                }
                sample_report["outputs"][branch]["float_raw_vs_raw_raw_codes"] = raw_code_rows[branch][-1]
                raw_archive[ref_key("float_input_raw", sample["index"], branch, name)] = np.ascontiguousarray(fr)
                raw_archive[ref_key("raw_input_raw", sample["index"], branch, name)] = np.ascontiguousarray(rr)
            sample_reports.append(sample_report)

        summarized = {
            label: {branch: summarize_comparisons(rows) for branch, rows in by_branch.items()}
            for label, by_branch in comparisons.items()
        }
        transform_max = max(
            summarized["float_raw_dequant_vs_board_float"][branch]["max_abs"]["max"]
            for branch in EXPECTED_OUTPUTS
        )
        input_code_summary = summarize_codes(input_formula_rows)
        raw_path_summary = {branch: summarize_codes(rows) for branch, rows in raw_code_rows.items()}
        arm_reports[name] = {
            "hef": {"path": str(hef_path), "sha256": arm["hef_sha256"]},
            "vstreams": {
                "inputs": [quant_row(info) for info in input_infos],
                "outputs": [quant_row(info) for info in output_infos],
            },
            "mode_parameters": {
                "float_input_float_output": ff_meta,
                "float_input_raw_output": fr_meta,
                "raw_input_float_output": rf_meta,
                "raw_input_raw_output": rr_meta,
            },
            "comparisons": summarized,
            "input_helper_vs_formula_codes": input_code_summary,
            "float_input_raw_vs_raw_input_raw_codes": raw_path_summary,
            "output_helper_vs_formula_dequant": {
                branch: summarize_comparisons(rows)
                for branch, rows in output_formula_rows.items()
            },
            "gates": {
                "hailort_float_output_equals_official_raw_dequant": {
                    "status": "PASS" if transform_max <= 1e-6 else "MISMATCH",
                    "max_abs": float(transform_max), "tolerance": 1e-6,
                },
                "official_input_quantizer_equals_affine_formula": {
                    "status": "PASS" if input_code_summary["mismatch_count"] == 0 else "MISMATCH",
                    **input_code_summary,
                },
                "float_input_auto_quant_equals_official_raw_input": {
                    "status": "PASS" if all(
                        row["mismatch_count"] == 0 for row in raw_path_summary.values()
                    ) else "MISMATCH",
                    "per_output": raw_path_summary,
                },
            },
            "temperature_c": {"start": temperature_start, "end": cpu_temperature_c()},
            "samples": sample_reports,
        }
        atomic_json(progress, {
            "status": "RUNNING" if arm_position < len(manifest["arms"]) else "PASS",
            "completed_arms": arm_position, "total_arms": len(manifest["arms"]),
            "last_arm": name, "temperature_c": cpu_temperature_c(),
        })
        print(json.dumps({"arm": name, "progress": f"{arm_position}/{len(manifest['arms'])}"}), flush=True)

    raw_temp = args.raw_out.with_name(f".{args.raw_out.stem}.{os.getpid()}.npz")
    args.raw_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(raw_temp, **raw_archive)
    os.replace(raw_temp, args.raw_out)
    raw_digest = sha256_file(args.raw_out)
    ort_summary = {branch: summarize_comparisons(rows) for branch, rows in ort_rows.items()}
    report = {
        "kind": "bn_g_s_hailort_raw_float_vstream_parity",
        "status": "PASS", "sample_count": len(samples), "val_indices": indices,
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "policy": {
            "diagnostic_only": True, "semantic_ranking": False,
            "full976_not_run": True, "bn_g_s_only": True,
            "model_topology_unchanged": True,
        },
        "hailort_api": {
            "raw_format_selector": "FormatType.AUTO",
            "quantized_argument": "unused in HailoRT 4.23 Python API",
            "transform_implementation": "HailoRTTransformUtils from board HailoRT",
        },
        "board_ort_vs_host_ort": ort_summary,
        "board_ort_gate": {
            "status": "PASS" if max(
                ort_summary[branch]["max_abs"]["max"] for branch in EXPECTED_OUTPUTS
            ) <= 1e-5 else "MISMATCH",
            "tolerance": 1e-5,
        },
        "arms": arm_reports,
        "raw_outputs": {
            "path": str(args.raw_out), "sha256": raw_digest,
            "bytes": args.raw_out.stat().st_size,
            "array_count": len(raw_archive),
        },
    }
    atomic_json(args.out, report)
    print(json.dumps({"status": "PASS", "out": str(args.out),
                      "raw_sha256": raw_digest}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
