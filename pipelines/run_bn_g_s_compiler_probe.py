#!/usr/bin/env python3
"""Compile-only ``bn_g_s`` side-output localization campaign.

The campaign is deliberately split into three separately authorized phases:

``prepare``
    Validate immutable sources and produce eight SDK_QUANTIZED references,
    including one selected internal tensor.  Stop before compilation.
``compile``
    Require ``--allow-compile`` and compile one HEF from the existing formal
    quantized HAR after adding exactly one ``output_layer`` command.  Never
    parse or optimize.
``board``
    Require ``--allow-board`` and run the probe plus the immutable formal HEF
    on the same eight frozen tensors.  The original conv3/conv5 outputs form a
    representativeness gate for the diagnostic HEF.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from board_eval.deploy import stream_upload  # noqa: E402
from hailo_deploy.artifacts.layout import (  # noqa: E402
    commit_or_fail,
    refuse_existing,
    stage_dir,
    write_json,
)
from hailo_deploy.gates.compile_real import compile_quantized  # noqa: E402
from hailo_deploy.gates.compiler_probe import (  # noqa: E402
    hn_sha256,
    load_compile_only_probe,
    optimized_params_sha256,
    validate_output_layer_command,
)
from hailo_deploy.io.hash import sha256  # noqa: E402
from hailo_deploy.onnx_io import run_onnxruntime  # noqa: E402
from pipelines.run_bn_g_s_clip_feasibility import (  # noqa: E402
    CHECKPOINT_SHA,
    ONNX,
    ONNX_SHA,
    REMOTE_PYTHON,
    SSH_TARGET,
    atomic_write,
    launch_remote,
    poll_remote,
    ssh,
    ssh_options,
    ssh_readonly,
)
import pipelines.run_bn_g_s_vstream_parity as parity  # noqa: E402
from board_eval.compiler_probe_metrics import sdk_codes_to_integer  # noqa: E402


CANONICAL = Path(
    "/local/shared_with_docker/deplyment-demo/artifacts/fused/"
    "current-ort-mainline-20260805"
)
FORMAL_DIR = CANONICAL / "bn_g_s_candidates/x_l_low_spatial_a16_l2"
FORMAL_QHAR = FORMAL_DIR / "bn_g_s_l2_quantized.har"
FORMAL_QHAR_SHA = "9900bc1fbffbbcf49ead61df9a15283e52a9e6bd3327269f5872795a82a0d31a"
FORMAL_COMPILED_HAR = FORMAL_DIR / "bn_g_s_l2_compiled.har"
FORMAL_COMPILED_HAR_SHA = "4a2a9da9c0a970cf5f35aa7d9324aabe877452a5627e33c36b76fd7b07b2fadf"
FORMAL_HEF = FORMAL_DIR / "bn_g_s_l2.hef"
FORMAL_HEF_SHA = "cadbdd1114ac3d75c6006d95804cfa6d86025490fc110cb900e91e51181a7856"
FORMAL_MODEL_SCRIPT = FORMAL_DIR / "model_script.alls"
FORMAL_MODEL_SCRIPT_SHA = "0224eb7746979b9a811adc01e913d74afedf6ddfd6f87806ec91d0c3dbbfc10f"
FORMAL_AUTO_ALLS = FORMAL_DIR / "bn_g_s_l2.auto.alls"
FORMAL_AUTO_ALLS_SHA = "48b49862cf6e4fb5219c15780843949619bb673e674c5a06c5033082e40e82e9"

GOLDEN = parity.GOLDEN
GOLDEN_RESULT_SHA = parity.GOLDEN_RESULT_SHA
REMOTE_GOLDEN = parity.REMOTE_GOLDEN
SAMPLE_INDICES = parity.SAMPLE_INDICES
CPU_MODULES = parity.CPU_MODULES

@dataclass(frozen=True)
class ProbeSpec:
    key: str
    name: str
    logical: str
    layer: str
    shape_hwc: tuple[int, int, int]
    format_type: str

    @property
    def script(self) -> str:
        return f"probe_{self.logical}_out = output_layer({self.layer})\n"


PROBE_SPECS = {
    "resize2": ProbeSpec(
        key="resize2",
        name="p0_resize2",
        logical="resize2",
        layer="bn_g_s/resize2",
        shape_hwc=(64, 64, 384),
        format_type="UINT8",
    ),
    "conv4": ProbeSpec(
        key="conv4",
        name="p1_conv4",
        logical="conv4",
        layer="bn_g_s/conv4",
        shape_hwc=(64, 64, 128),
        format_type="UINT8",
    ),
}
DEFAULT_PROBE = "conv4"
PROBE = PROBE_SPECS[DEFAULT_PROBE]
FINAL_SHAPES = {
    "x_h_hat": (64, 64, 128),
    "x_l_hat": (32, 32, 256),
}
EXPECTED_LAYERS = {
    "bn_g_s/conv2": {
        "shape": (32, 32, 128),
        "precision": "a8_w8_a16",
        "onnx_end": "/g_s_trunk/g_s_trunk.1/g_s_trunk.1.1/"
        "g_s_trunk.1.1.2/Relu",
    },
    "bn_g_s/conv3": {
        "shape": (32, 32, 256),
        "precision": "a16_w16_a16",
        "onnx_end": "/head_l/Conv",
    },
    "bn_g_s/concat1": {
        "shape": (32, 32, 384),
        "precision": "a8_w8_a8",
        "onnx_end": "/Concat",
    },
    "bn_g_s/resize2": {
        "shape": (64, 64, 384),
        "precision": "a8_w8_a8",
        "onnx_end": "/head_h/head_h.0/Resize",
    },
    "bn_g_s/conv4": {
        "shape": (64, 64, 128),
        "precision": "a8_w8_a8",
        "onnx_end": "/head_h/head_h.1/head_h.1.2/Relu",
    },
    "bn_g_s/conv5": {
        "shape": (64, 64, 128),
        "precision": "a8_w8_a8",
        "onnx_end": "/head_h/head_h.2/Conv",
    },
}


def select_probe(key: str) -> ProbeSpec:
    global PROBE
    PROBE = PROBE_SPECS[key]
    return PROBE


def _source_files() -> dict[Path, str]:
    return {
        FORMAL_QHAR: FORMAL_QHAR_SHA,
        FORMAL_COMPILED_HAR: FORMAL_COMPILED_HAR_SHA,
        FORMAL_HEF: FORMAL_HEF_SHA,
        FORMAL_MODEL_SCRIPT: FORMAL_MODEL_SCRIPT_SHA,
        FORMAL_AUTO_ALLS: FORMAL_AUTO_ALLS_SHA,
        ONNX: ONNX_SHA,
        GOLDEN / "result.json": GOLDEN_RESULT_SHA,
    }


def _quant_info(info: Any) -> dict[str, float]:
    quant = info.quant_info
    return {
        "qp_scale": float(quant.qp_scale),
        "qp_zp": float(quant.qp_zp),
        "limvals_min": float(quant.limvals_min),
        "limvals_max": float(quant.limvals_max),
    }


def hef_static(path: Path) -> dict[str, Any]:
    from hailo_platform import HEF

    hef = HEF(str(path))

    def row(info: Any) -> dict[str, Any]:
        return {
            "name": info.name,
            "shape_hwc": [int(x) for x in info.shape],
            "format_type": str(info.format.type.name),
            "format_order": str(info.format.order.name),
            "quantization": _quant_info(info),
        }

    return {
        "path": str(path),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "network_groups": list(hef.get_network_group_names()),
        "inputs": [row(info) for info in hef.get_input_vstream_infos()],
        "outputs": [row(info) for info in hef.get_output_vstream_infos()],
    }


def _map_static_outputs(
    rows: list[dict[str, Any]], *, include_probe: bool
) -> dict[str, dict[str, Any]]:
    expected: dict[str, tuple[str, tuple[int, int, int]]] = {
        "x_h_hat": ("bn_g_s/conv5", FINAL_SHAPES["x_h_hat"]),
        "x_l_hat": ("bn_g_s/conv3", FINAL_SHAPES["x_l_hat"]),
    }
    if include_probe:
        expected[PROBE.logical] = (PROBE.layer, PROBE.shape_hwc)
    mapped: dict[str, dict[str, Any]] = {}
    for logical, (name, shape) in expected.items():
        matches = [
            row
            for row in rows
            if row["name"] == name and tuple(row["shape_hwc"]) == shape
        ]
        if len(matches) != 1:
            raise ValueError(f"cannot map {logical} as {name}/{shape}: {rows}")
        mapped[logical] = matches[0]
    if len(mapped) != len(rows):
        raise ValueError(f"unexpected extra HEF outputs: {rows}")
    return mapped


def _allocation_facts(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    return {
        "sha256": sha256(path),
        "resize1_input_format_conversion": "resize1_input_reshape = format_conversion" in text,
        "resize1_output_format_conversion": "resize1_output_reshape = format_conversion" in text,
        "resize2_input_format_conversion": "resize2_input_reshape = format_conversion" in text,
        "resize2_output_format_conversion": "resize2_output_reshape = format_conversion" in text,
        "conv3_spatial_defuse_8": "defuse(conv3, 8, defuse_type=SPATIAL)" in text,
        "resize2_defuse_2": "defuse(resize2, 2)" in text,
        "conv4_spatial_defuse_8": "defuse(conv4, 8, defuse_type=SPATIAL)" in text,
        "conv5_defuse_4": "defuse(conv5, 4)" in text,
        "context_count_in_script": text.count(" = context(["),
    }


def _validate_golden() -> dict[str, Any]:
    result = json.loads((GOLDEN / "result.json").read_text(encoding="utf-8"))
    if result.get("status") != "PASS" or result.get("sample_count") != 32:
        raise ValueError("frozen y_hat bundle is incomplete")
    if result.get("checkpoint_sha256") != CHECKPOINT_SHA:
        raise ValueError("frozen y_hat bundle checkpoint mismatch")
    if tuple(result.get("val_indices", ())[:8]) != SAMPLE_INDICES:
        raise ValueError("frozen y_hat bundle first-eight contract changed")
    rows = {int(row["val_index"]): row for row in result["rows"]}
    tensors = []
    for index in SAMPLE_INDICES:
        path = GOLDEN / "latent_y_hat" / f"{index}.npy"
        value = np.load(path, allow_pickle=False)
        if value.shape != (16, 16, 192) or value.dtype != np.float32:
            raise ValueError(f"unexpected y_hat contract for {index}: {value.shape}/{value.dtype}")
        digest = parity.array_sha(value)
        if digest != rows[index]["sha256"]:
            raise ValueError(f"frozen y_hat hash mismatch for {index}")
        tensors.append({"val_index": index, "path": str(path), "sha256": digest})
    return {
        "result": {"path": str(GOLDEN / "result.json"), "sha256": GOLDEN_RESULT_SHA},
        "sample_indices": list(SAMPLE_INDICES),
        "tensors": tensors,
    }


def validate_sources() -> dict[str, Any]:
    for path, expected in _source_files().items():
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"immutable source mismatch: {path}: {actual} != {expected}")

    from hailo_sdk_client import ClientRunner

    runner = ClientRunner(har=str(FORMAL_QHAR))
    if str(runner.state) != "quantized_model":
        raise ValueError(f"formal source is not quantized_model: {runner.state}")
    hn = runner.get_hn_dict()
    if hn.get("net_params", {}).get("output_layers_order") != [
        "bn_g_s/conv5",
        "bn_g_s/conv3",
    ]:
        raise ValueError("formal output layer order changed")
    layer_facts: dict[str, Any] = {}
    for name, expected in EXPECTED_LAYERS.items():
        layer = hn.get("layers", {}).get(name)
        if layer is None:
            raise ValueError(f"missing HN layer: {name}")
        shapes = {tuple(shape[1:]) for shape in layer.get("output_shapes", [])}
        if expected["shape"] not in shapes:
            raise ValueError(f"{name}: expected output shape {expected['shape']}, got {shapes}")
        precision = layer.get("quantization_params", {}).get("precision_mode")
        if precision != expected["precision"]:
            raise ValueError(f"{name}: precision {precision} != {expected['precision']}")
        originals = list(layer.get("original_names") or [])
        if expected["onnx_end"] not in originals:
            raise ValueError(f"{name}: missing ONNX endpoint {expected['onnx_end']}")
        layer_facts[name] = {
            "type": layer.get("type"),
            "inputs": list(layer.get("input") or []),
            "outputs": list(layer.get("output") or []),
            "output_shapes_nhwc": [list(shape) for shape in layer.get("output_shapes", [])],
            "precision_mode": precision,
            "original_names": originals,
        }
    parsed = validate_output_layer_command(runner, PROBE.script)
    numerical_sha = optimized_params_sha256(runner)
    source_hn_sha = hn_sha256(runner)
    del runner

    guard_runner = ClientRunner(har=str(FORMAL_QHAR))
    in_memory_guard = load_compile_only_probe(guard_runner, PROBE.script)
    del guard_runner
    if sha256(FORMAL_QHAR) != FORMAL_QHAR_SHA:
        raise ValueError("formal quantized HAR changed during validation")

    formal_static = hef_static(FORMAL_HEF)
    formal_outputs = _map_static_outputs(formal_static["outputs"], include_probe=False)
    if set(formal_outputs) != {"x_h_hat", "x_l_hat"}:
        raise ValueError(f"formal HEF output contract changed: {formal_outputs}")
    if formal_outputs["x_h_hat"]["format_type"] != "UINT8":
        raise ValueError("formal x_h output is not UINT8")
    if formal_outputs["x_l_hat"]["format_type"] != "UINT16":
        raise ValueError("formal x_l output is not UINT16")
    if len(formal_static["inputs"]) != 1 or formal_static["inputs"][0]["shape_hwc"] != [16, 16, 192]:
        raise ValueError("formal HEF input contract changed")

    return {
        "status": "PASS",
        "policy": {
            "diagnostic_only": True,
            "full976_not_run": True,
            "semantic_ranking": False,
            "source_quantized_har_immutable": True,
            "parse_called": False,
            "optimize_called": False,
        },
        "source": {
            "checkpoint_sha256": CHECKPOINT_SHA,
            "onnx": {"path": str(ONNX), "sha256": ONNX_SHA},
            "quantized_har": {"path": str(FORMAL_QHAR), "sha256": FORMAL_QHAR_SHA},
            "compiled_har": {
                "path": str(FORMAL_COMPILED_HAR),
                "sha256": FORMAL_COMPILED_HAR_SHA,
            },
            "formal_hef": {"path": str(FORMAL_HEF), "sha256": FORMAL_HEF_SHA},
            "model_script": {
                "path": str(FORMAL_MODEL_SCRIPT),
                "sha256": FORMAL_MODEL_SCRIPT_SHA,
            },
            "auto_alls": {
                "path": str(FORMAL_AUTO_ALLS),
                **_allocation_facts(FORMAL_AUTO_ALLS),
            },
        },
        "versions": {
            "dfc": parity.package_version("hailo-dataflow-compiler"),
            "hailort_host": parity.package_version("hailort"),
            "model_zoo": parity.package_version("hailo-model-zoo"),
            "hw_arch": "hailo8",
            "hailort_board_expected": "4.23.0",
        },
        "probe": {
            "key": PROBE.key,
            "name": PROBE.name,
            "logical_name": PROBE.logical,
            "source_layer": PROBE.layer,
            "script": PROBE.script,
            "parsed_commands": parsed,
            "shape_hwc": list(PROBE.shape_hwc),
            "expected_format_type": PROBE.format_type,
        },
        "quantized_model": {
            "state": "quantized_model",
            "hn_sha256": source_hn_sha,
            "optimized_params_sha256": numerical_sha,
            "output_layers_order": hn["net_params"]["output_layers_order"],
            "layers": layer_facts,
            "in_memory_script_guard": in_memory_guard,
        },
        "formal_hef_static": formal_static,
        "golden": _validate_golden(),
    }


def _as_numpy(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _as_numpy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_numpy(item) for item in value]
    return np.asarray(value.numpy() if hasattr(value, "numpy") else value)


def _map_by_shape(values: Any, expected: dict[str, tuple[int, ...]]) -> dict[str, np.ndarray]:
    if isinstance(values, dict):
        arrays = [np.asarray(value) for value in values.values()]
    elif isinstance(values, (list, tuple)):
        arrays = [np.asarray(value) for value in values]
    else:
        arrays = [np.asarray(values)]
    result: dict[str, np.ndarray] = {}
    for logical, shape in expected.items():
        matches = [value for value in arrays if tuple(value.shape[1:]) == shape]
        if len(matches) != 1:
            raise ValueError(
                f"cannot map {logical} by batched NHWC shape {shape}: "
                f"{[tuple(value.shape) for value in arrays]}"
            )
        result[logical] = np.ascontiguousarray(matches[0][0], dtype=np.float32)
    return result


def make_sdk_references(root: Path) -> tuple[Path, dict[str, Any]]:
    from hailo_sdk_client import ClientRunner, InferenceContext

    arrays: dict[str, np.ndarray] = {}
    final_metric_rows = {name: [] for name in FINAL_SHAPES}
    runner = ClientRunner(har=str(FORMAL_QHAR))
    simulation_notes = {
        "api": "get_keras_model(SDK_QUANTIZED).model.set_output_interal_layers",
        "internal_layers": [PROBE.layer],
        "set_output_interal_layers_called": False,
        "intermediate_representation": "integer quantization codes stored in float32",
    }
    with runner.infer_context(InferenceContext.SDK_QUANTIZED) as context:
        simulation = runner.get_keras_model(context)
        simulation.model.set_output_interal_layers([PROBE.layer])
        simulation_notes["set_output_interal_layers_called"] = True
        try:
            for index in SAMPLE_INDICES:
                y_hwc = np.ascontiguousarray(
                    np.load(GOLDEN / "latent_y_hat" / f"{index}.npy", allow_pickle=False),
                    dtype=np.float32,
                )
                raw = _as_numpy(simulation.model(y_hwc[None, ...], training=False))
                if not isinstance(raw, list) or len(raw) != 2:
                    raise ValueError(f"unexpected internal-output structure: {type(raw)}/{raw}")
                final_values, internal_values = raw
                final = _map_by_shape(final_values, FINAL_SHAPES)
                internal = _map_by_shape(
                    internal_values, {PROBE.logical: PROBE.shape_hwc}
                )

                ort = run_onnxruntime(
                    ONNX,
                    {"latent_y_hat": y_hwc.transpose(2, 0, 1)[None, ...]},
                )
                for logical, value in final.items():
                    arrays[f"sdk__{index}__{logical}"] = value
                    ort_hwc = np.ascontiguousarray(
                        np.asarray(ort[logical])[0].transpose(1, 2, 0), dtype=np.float32
                    )
                    arrays[f"ort__{index}__{logical}"] = ort_hwc
                    parity.add_metric(final_metric_rows, logical, ort_hwc, value)
                sdk_codes = internal[PROBE.logical]
                sdk_codes_to_integer(sdk_codes, PROBE.format_type)
                arrays[f"sdk_code__{index}__{PROBE.logical}"] = sdk_codes
        finally:
            simulation.model.reset_output_interal_layers()
    del runner

    destination = root / "sdk_reference8" / "references.npz"
    destination.parent.mkdir(parents=True, exist_ok=False)
    temporary = destination.with_name(f".{destination.stem}.{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, destination)
    report = {
        "kind": "bn_g_s_compiler_probe_sdk_quantized_reference8",
        "status": "PASS",
        "sample_count": len(SAMPLE_INDICES),
        "sample_indices": list(SAMPLE_INDICES),
        "source_quantized_har_sha256": FORMAL_QHAR_SHA,
        "probe": {
            "key": PROBE.key,
            "logical_name": PROBE.logical,
            "source_layer": PROBE.layer,
            "shape_hwc": list(PROBE.shape_hwc),
            "format_type": PROBE.format_type,
            "reference_key_template": f"sdk_code__{{index}}__{PROBE.logical}",
            "reference_representation": "integer quantization codes stored in float32",
        },
        "simulation": simulation_notes,
        "final_outputs_vs_ort": {
            name: parity.summarize_metric_rows(rows)
            for name, rows in final_metric_rows.items()
        },
        "bundle": {
            "path": str(destination),
            "sha256": sha256(destination),
            "bytes": destination.stat().st_size,
            "arrays": {key: parity.array_sha(value) for key, value in arrays.items()},
        },
        "policy": {
            "diagnostic_only": True,
            "full976_not_run": True,
            "semantic_ranking": False,
        },
    }
    atomic_write(root / "sdk_reference8" / "report.json", report)
    return destination, report


def prepare(root: Path, cuda_visible_devices: str) -> dict[str, Any]:
    refuse_existing(root)
    root.mkdir(parents=True)
    (root / "logs").mkdir()
    (root / "model_scripts").mkdir()
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    os.environ["HAILO_SDK_LOG_DIR"] = str(root / "logs")
    source = validate_sources()
    atomic_write(root / "source_validation.json", source)
    script_path = root / "model_scripts" / f"{PROBE.name}.alls"
    script_path.write_text(PROBE.script, encoding="utf-8")
    references_path, references = make_sdk_references(root)
    if sha256(FORMAL_QHAR) != FORMAL_QHAR_SHA:
        raise ValueError("formal quantized HAR changed during SDK reference generation")
    report = {
        "kind": "bn_g_s_compiler_probe_precompile",
        "status": "WAITING_FOR_COMPILE_APPROVAL",
        "created_unix_seconds": time.time(),
        "out_root": str(root),
        "probe": {
            "key": PROBE.key,
            "name": PROBE.name,
            "logical_name": PROBE.logical,
            "source_layer": PROBE.layer,
            "shape_hwc": list(PROBE.shape_hwc),
            "format_type": PROBE.format_type,
        },
        "source_validation": {
            "path": str(root / "source_validation.json"),
            "sha256": sha256(root / "source_validation.json"),
        },
        "sdk_reference8": {
            "path": str(references_path),
            "sha256": references["bundle"]["sha256"],
            "report": str(root / "sdk_reference8" / "report.json"),
        },
        "probe_script": {
            "path": str(script_path),
            "sha256": sha256(script_path),
            "text": PROBE.script,
        },
        "commands_after_approval": {
            "compile": shlex.join([
                sys.executable,
                "pipelines/run_bn_g_s_compiler_probe.py",
                "--phase",
                "compile",
                "--out-root",
                str(root),
                "--probe",
                PROBE.key,
                "--cuda-visible-devices",
                cuda_visible_devices,
                "--allow-compile",
            ]),
            "board": shlex.join([
                sys.executable,
                "pipelines/run_bn_g_s_compiler_probe.py",
                "--phase",
                "board",
                "--out-root",
                str(root),
                "--probe",
                PROBE.key,
                "--allow-board",
            ]),
        },
        "risk_register": [
            "output_layer adds host egress and can change allocator placement",
            f"{PROBE.logical} adds a {PROBE.shape_hwc} {PROBE.format_type} side output",
            "SDK_QUANTIZED internal output is integer code values stored in float32",
            "the probe HEF qp_scale/qp_zp must dequantize those codes before FLOAT32 comparison",
            "per-feature quantization must be handled by HailoRT FLOAT32 conversion",
            "probe HEF is representative only if its conv3/conv5 outputs match the formal HEF",
        ],
        "policy": {
            "compile_started": False,
            "board_started": False,
            "parse_called": False,
            "optimize_called": False,
            "diagnostic_only": True,
            "full976_not_run": True,
        },
    }
    atomic_write(root / "precompile_report.json", report)
    atomic_write(root / "campaign_state.json", {
        "kind": "bn_g_s_compiler_probe_state",
        "status": "WAITING_FOR_COMPILE_APPROVAL",
        "stage": "prepared",
        "out_root": str(root),
        "probe_key": PROBE.key,
    })
    return report


def _same_quantization(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        left["format_type"] == right["format_type"]
        and left["format_order"] == right["format_order"]
        and left["quantization"] == right["quantization"]
    )


def _relocate_paths(value: Any, source: Path, destination: Path) -> Any:
    """Rewrite staged absolute paths before persisting the committed report."""

    if isinstance(value, dict):
        return {
            key: _relocate_paths(item, source, destination)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_relocate_paths(item, source, destination) for item in value]
    if isinstance(value, str):
        return value.replace(str(source), str(destination))
    return value


def compile_probe(root: Path, cuda_visible_devices: str, allowed: bool) -> dict[str, Any]:
    if not allowed:
        raise PermissionError("compile phase requires explicit --allow-compile")
    precompile_path = root / "precompile_report.json"
    if not precompile_path.is_file():
        raise FileNotFoundError("prepare phase has not completed")
    precompile = json.loads(precompile_path.read_text(encoding="utf-8"))
    if precompile.get("status") != "WAITING_FOR_COMPILE_APPROVAL":
        raise ValueError(f"unexpected precompile state: {precompile.get('status')}")
    if precompile.get("probe", {}).get("key") != PROBE.key:
        raise ValueError(
            f"prepared probe is {precompile.get('probe')}, requested {PROBE.key}"
        )
    script_path = root / "model_scripts" / f"{PROBE.name}.alls"
    if script_path.read_text(encoding="utf-8") != PROBE.script:
        raise ValueError("probe script text changed after prepare")
    source = validate_sources()
    if source["quantized_model"]["optimized_params_sha256"] != source[
        "quantized_model"
    ]["in_memory_script_guard"]["optimized_params_sha256_before"]:
        raise ValueError("source numerical hash is inconsistent")

    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    os.environ["HAILO_SDK_LOG_DIR"] = str(root / "logs")
    from hailo_sdk_client import ClientRunner

    destination = root / "probes" / PROBE.name
    refuse_existing(destination)
    temporary = stage_dir(destination)
    try:
        runner = ClientRunner(har=str(FORMAL_QHAR))
        qhar_before = sha256(FORMAL_QHAR)
        numerical_guard = load_compile_only_probe(runner, PROBE.script)
        (temporary / "probe.alls").write_text(PROBE.script, encoding="utf-8")
        result = compile_quantized(
            runner,
            hef_path=temporary / "bn_g_s_probe.hef",
            compiled_har=temporary / "bn_g_s_probe_compiled.har",
            auto_alls=temporary / "bn_g_s_probe.auto.alls",
        )
        del runner
        qhar_after = sha256(FORMAL_QHAR)
        if qhar_before != FORMAL_QHAR_SHA or qhar_after != FORMAL_QHAR_SHA:
            raise ValueError("formal quantized HAR changed during compile")

        probe_static = hef_static(temporary / "bn_g_s_probe.hef")
        mapped = _map_static_outputs(probe_static["outputs"], include_probe=True)
        if set(mapped) != {"x_h_hat", "x_l_hat", PROBE.logical}:
            raise ValueError(f"probe HEF did not preserve three outputs: {mapped}")
        expected_formats = {
            "x_h_hat": "UINT8",
            "x_l_hat": "UINT16",
            PROBE.logical: PROBE.format_type,
        }
        for logical, expected_format in expected_formats.items():
            if mapped[logical]["format_type"] != expected_format:
                raise ValueError(
                    f"{logical}: {mapped[logical]['format_type']} != {expected_format}"
                )
        formal_mapped = _map_static_outputs(
            source["formal_hef_static"]["outputs"], include_probe=False
        )
        original_static = {
            logical: {
                "status": "PASS" if _same_quantization(formal_mapped[logical], mapped[logical]) else "MISMATCH",
                "formal": formal_mapped[logical],
                "probe": mapped[logical],
            }
            for logical in FINAL_SHAPES
        }
        if any(row["status"] != "PASS" for row in original_static.values()):
            raise ValueError("probe compile changed an original output quantization contract")

        compile_report = {
            "kind": "bn_g_s_compiler_probe_compile",
            "status": "PASS",
            "source_quantized_har": {
                "path": str(FORMAL_QHAR),
                "sha256_before": qhar_before,
                "sha256_after": qhar_after,
            },
            "probe": {
                "key": PROBE.key,
                "name": PROBE.name,
                "logical_name": PROBE.logical,
                "source_layer": PROBE.layer,
                "shape_hwc": list(PROBE.shape_hwc),
                "format_type": PROBE.format_type,
                "script": PROBE.script,
                "script_sha256": sha256(temporary / "probe.alls"),
            },
            "numerical_guard": numerical_guard,
            "compile": result,
            "hef_static": probe_static,
            "mapped_outputs": mapped,
            "original_output_static_representativeness": original_static,
            "policy": {
                "parse_called": False,
                "optimize_called": False,
                "source_quantized_har_immutable": True,
                "one_probe_output_only": True,
            },
        }
        write_json(
            temporary / "compile_result.json",
            _relocate_paths(compile_report, temporary, destination),
        )
        committed = commit_or_fail(temporary, destination, True)
    except Exception:
        commit_or_fail(temporary, destination, False)
        raise

    final_hef = committed / "bn_g_s_probe.hef"
    references_report = json.loads(
        (root / "sdk_reference8" / "report.json").read_text(encoding="utf-8")
    )
    manifest = {
        "kind": "bn_g_s_compiler_probe_manifest_v2",
        "checkpoint_sha256": CHECKPOINT_SHA,
        "sample_indices": list(SAMPLE_INDICES),
        "golden_bundle": {"result_sha256": GOLDEN_RESULT_SHA},
        "references": {
            "filename": "references.npz",
            "sha256": references_report["bundle"]["sha256"],
            "arrays": references_report["bundle"]["arrays"],
            "probe_code_key_template": f"sdk_code__{{index}}__{PROBE.logical}",
            "probe_representation": "integer quantization codes stored in float32",
        },
        "formal": {
            "hef_filename": "hef/formal_bn_g_s.hef",
            "hef_sha256": FORMAL_HEF_SHA,
        },
        "probe": {
            "key": PROBE.key,
            "name": PROBE.name,
            "logical_name": PROBE.logical,
            "source_layer": PROBE.layer,
            "vstream_name": PROBE.layer,
            "hef_filename": f"hef/{PROBE.key}_probe.hef",
            "hef_sha256": sha256(final_hef),
            "expected_shape_hwc": list(PROBE.shape_hwc),
            "expected_format_type": PROBE.format_type,
        },
        "expected_outputs": {
            "x_h_hat": {"shape_hwc": list(FINAL_SHAPES["x_h_hat"]), "format_type": "UINT8"},
            "x_l_hat": {"shape_hwc": list(FINAL_SHAPES["x_l_hat"]), "format_type": "UINT16"},
            PROBE.logical: {
                "vstream_name": PROBE.layer,
                "shape_hwc": list(PROBE.shape_hwc),
                "format_type": PROBE.format_type,
            },
        },
        "policy": {
            "diagnostic_only": True,
            "full976_not_run": True,
            "semantic_ranking": False,
            "cpu_modules": CPU_MODULES,
            "source_quantized_har_sha256": FORMAL_QHAR_SHA,
        },
    }
    atomic_write(root / "board_manifest.json", manifest)
    postcompile = {
        "kind": "bn_g_s_compiler_probe_postcompile",
        "status": "WAITING_FOR_BOARD_APPROVAL",
        "probe": {
            "key": PROBE.key,
            "name": PROBE.name,
            "logical_name": PROBE.logical,
            "source_layer": PROBE.layer,
        },
        "compile_result": str(committed / "compile_result.json"),
        "probe_hef": {"path": str(final_hef), "sha256": sha256(final_hef)},
        "board_manifest": {
            "path": str(root / "board_manifest.json"),
            "sha256": sha256(root / "board_manifest.json"),
        },
        "original_output_static_representativeness": original_static,
        "next_command": shlex.join([
            sys.executable,
            "pipelines/run_bn_g_s_compiler_probe.py",
            "--phase",
            "board",
            "--out-root",
            str(root),
            "--probe",
            PROBE.key,
            "--allow-board",
        ]),
    }
    atomic_write(root / "postcompile_report.json", postcompile)
    atomic_write(root / "campaign_state.json", {
        "kind": "bn_g_s_compiler_probe_state",
        "status": "WAITING_FOR_BOARD_APPROVAL",
        "stage": "compiled",
        "out_root": str(root),
        "probe_key": PROBE.key,
    })
    return postcompile


def board_probe(root: Path, allowed: bool) -> dict[str, Any]:
    if not allowed:
        raise PermissionError("board phase requires explicit --allow-board")
    postcompile_path = root / "postcompile_report.json"
    if not postcompile_path.is_file():
        raise FileNotFoundError("compile phase has not completed")
    postcompile = json.loads(postcompile_path.read_text(encoding="utf-8"))
    if postcompile.get("status") != "WAITING_FOR_BOARD_APPROVAL":
        raise ValueError(f"unexpected postcompile state: {postcompile.get('status')}")
    if postcompile.get("probe", {}).get("key") != PROBE.key:
        raise ValueError(
            f"compiled probe is {postcompile.get('probe')}, requested {PROBE.key}"
        )
    manifest_path = root / "board_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    probe_hef = root / "probes" / PROBE.name / "bn_g_s_probe.hef"
    if sha256(probe_hef) != manifest["probe"]["hef_sha256"]:
        raise ValueError("local probe HEF identity mismatch")
    if sha256(FORMAL_HEF) != FORMAL_HEF_SHA:
        raise ValueError("formal HEF identity mismatch")

    prelaunch = ssh_readonly(
        "pgrep -af '[e]valuate_bn_g_s|[r]un_bn_g_s' || true; "
        "vcgencmd measure_temp 2>/dev/null || true"
    )
    if "evaluate_bn_g_s" in prelaunch or "run_bn_g_s" in prelaunch:
        raise RuntimeError(f"board has an active bn_g_s campaign:\n{prelaunch}")
    remote_root = f".local/share/semantic-sensing-board/hybrid-ort/{root.name}"
    if ssh_readonly(
        f"test ! -e {shlex.quote(remote_root)} && echo ABSENT || echo PRESENT"
    ).strip() != "ABSENT":
        raise FileExistsError(f"remote result root exists: {remote_root}")
    ssh(f"mkdir -p {shlex.quote(remote_root + '/hef')}")
    options = ssh_options()
    evaluator = ROOT / "board_eval/evaluate_bn_g_s_compiler_probe.py"
    dependencies = (
        ROOT / "board_eval/compiler_probe_metrics.py",
        ROOT / "board_eval/evaluate_bn_g_s_vstream_parity.py",
        ROOT / "board_eval/bn_g_s_branch_attribution_runtime.py",
        ROOT / "board_eval/hybrid_ort_runtime_runner.py",
        ROOT / "board_eval/fullmodule_runtime_runner.py",
    )
    for source in (evaluator, *dependencies):
        stream_upload(source, f"{remote_root}/{source.name}", options, SSH_TARGET)
    stream_upload(FORMAL_HEF, f"{remote_root}/hef/formal_bn_g_s.hef", options, SSH_TARGET)
    stream_upload(
        probe_hef,
        f"{remote_root}/hef/{PROBE.key}_probe.hef",
        options,
        SSH_TARGET,
    )
    references = root / "sdk_reference8" / "references.npz"
    stream_upload(references, f"{remote_root}/references.npz", options, SSH_TARGET)
    stream_upload(manifest_path, f"{remote_root}/board_manifest.json", options, SSH_TARGET)
    remote_golden_sha = ssh_readonly(
        f"sha256sum {shlex.quote(REMOTE_GOLDEN + '/result.json')} | cut -d' ' -f1"
    ).strip()
    if remote_golden_sha != GOLDEN_RESULT_SHA:
        raise ValueError("remote golden bundle identity mismatch")

    result_remote = f"{remote_root}/result.json"
    tensors_remote = f"{remote_root}/board_tensors.npz"
    job = launch_remote(
        remote_root,
        f"board_{PROBE.key}_probe_attempt0",
        [
            REMOTE_PYTHON,
            f"{remote_root}/{evaluator.name}",
            "--manifest",
            f"{remote_root}/board_manifest.json",
            "--bundle-root",
            REMOTE_GOLDEN,
            "--out",
            result_remote,
            "--tensor-out",
            tensors_remote,
        ],
        result_override=result_remote,
        progress_override=f"{remote_root}/result.progress.json",
    )
    atomic_write(root / "campaign_state.json", {
        "kind": "bn_g_s_compiler_probe_state",
        "status": "RUNNING",
        "stage": "board",
        "probe_key": PROBE.key,
        "remote_job": job,
        "board_prelaunch": prelaunch,
    })
    state = json.loads((root / "campaign_state.json").read_text(encoding="utf-8"))
    poll_remote(root, state, job, timeout=1800)
    board_result = root / "board_result.json"
    board_tensors = root / "board_tensors.npz"
    parity.remote_download(result_remote, board_result)
    parity.remote_download(tensors_remote, board_tensors)
    result = json.loads(board_result.read_text(encoding="utf-8"))
    if result.get("status") != "PASS" or result.get("sample_count") != len(SAMPLE_INDICES):
        raise RuntimeError("board probe result is incomplete")
    if sha256(board_tensors) != result["board_tensors"]["sha256"]:
        raise ValueError("downloaded board tensor archive hash mismatch")
    summary = {
        "kind": "bn_g_s_compiler_probe_summary",
        "status": "PASS",
        "probe": {
            "key": PROBE.key,
            "name": PROBE.name,
            "logical_name": PROBE.logical,
            "source_layer": PROBE.layer,
        },
        "verdict": (
            "PROBE_REPRESENTATIVE"
            if result["representativeness_gate"]["status"] == "PASS"
            else "PROBE_MAPPING_PERTURBED"
        ),
        "probe_vs_sdk": result["probe_vs_sdk"],
        "probe_intermediate_corrected": result["probe_intermediate_corrected"],
        "formal_vs_sdk": result["formal_vs_sdk"],
        "representativeness_gate": result["representativeness_gate"],
        "artifacts": {
            "board_result": {"path": str(board_result), "sha256": sha256(board_result)},
            "board_tensors": {"path": str(board_tensors), "sha256": sha256(board_tensors)},
        },
        "policy": {
            "diagnostic_only": True,
            "full976_not_run": True,
            "semantic_ranking": False,
        },
    }
    atomic_write(root / "summary.json", summary)
    atomic_write(root / "campaign_state.json", {
        "kind": "bn_g_s_compiler_probe_state",
        "status": "PASS",
        "stage": "complete",
        "probe_key": PROBE.key,
        "verdict": summary["verdict"],
    })
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "compile", "board"), default="prepare")
    parser.add_argument("--out-root", type=Path)
    parser.add_argument(
        "--probe", choices=tuple(PROBE_SPECS), default=DEFAULT_PROBE,
        help="exactly one compile-only side output (default: conv4)",
    )
    parser.add_argument("--cuda-visible-devices", default="2")
    parser.add_argument("--allow-compile", action="store_true")
    parser.add_argument("--allow-board", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    select_probe(args.probe)

    if args.validate_only:
        validation = validate_sources()
        print(json.dumps(validation, indent=2, sort_keys=True))
        return 0
    if args.out_root is None:
        parser.error("--out-root is required unless --validate-only is used")
    root = args.out_root.resolve()
    if args.phase == "prepare":
        report = prepare(root, args.cuda_visible_devices)
    elif args.phase == "compile":
        report = compile_probe(root, args.cuda_visible_devices, args.allow_compile)
    else:
        report = board_probe(root, args.allow_board)
    print(json.dumps({
        "status": report["status"],
        "phase": args.phase,
        "out_root": str(root),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
