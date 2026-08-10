#!/usr/bin/env python3
"""Strict graph-only rewrite probe for ``bn_g_s`` Resize -> conv4 -> ReLU.

The original high-resolution 3x3 convolution is evaluated as four stride-2
phase convolutions.  Their outputs are concatenated in ONNX DepthToSpace DCR
order and interleaved back to the original 64x64 tensor.  Weights and bias are
reused verbatim; there is no training, calibration, optimization, compilation,
or board execution in this probe.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

# NATIVE parity is intentionally CPU/FP32; GPU TF32 has caused false mismatches.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hailo_deploy.emulate import _build_native_feed, _match_outputs  # noqa: E402
from hailo_deploy.io.hash import sha256  # noqa: E402
from hailo_deploy.onnx_io import signature  # noqa: E402


SOURCE_ONNX = Path(
    "/local/shared_with_docker/hailo_full_candidate/"
    "convbn-gfa-gs-3a3ea630af6d2d547808137be678bd909f9789f5594b7bdc76c36b90/"
    "onnx/bn_g_s.onnx"
)
SOURCE_ONNX_SHA256 = "b292687ee569fe2f523ae3704cfca92bc84e4bc7e4041ea5c9fa41ac8c0e0c4f"
CHECKPOINT_SHA256 = "13dc6c4b3a3ea630af6d2d547808137be678bd909f9789f5594b7bdc76c36b90"
GOLDEN_ROOT = Path(
    "/local/shared_with_docker/deplyment-demo/artifacts/fused/experiments/"
    "bn-gs-input-clip-range-sweep-retry-20260806T025328Z/board/clip_yhat_bundle"
)
DEFAULT_OUT = Path(
    "/local/shared_with_docker/deplyment-demo/artifacts/fused/experiments/"
    "bn-gs-resize-conv-phase-rewrite-20260806T071000Z"
)
INDICES = tuple(range(1024, 1032))
ORT_EQUIVALENCE_TOL = 1e-5
NATIVE_TOL = 1e-3

RESIZE_NAME = "/head_h/head_h.0/Resize"
CONV_NAME = "/head_h/head_h.1/head_h.1.0/Conv"
RELU_NAME = "/head_h/head_h.1/head_h.1.2/Relu"
CONV_WEIGHT = "onnx::Conv_115"
CONV_BIAS = "onnx::Conv_116"


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def attrs(node: onnx.NodeProto) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for attr in node.attribute:
        value = helper.get_attribute_value(attr)
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if isinstance(value, tuple):
            value = list(value)
        result[attr.name] = value
    return result


def source_contract(model: onnx.ModelProto) -> dict[str, Any]:
    nodes = {node.name: node for node in model.graph.node}
    for name in (RESIZE_NAME, CONV_NAME, RELU_NAME):
        if name not in nodes:
            raise ValueError(f"required source node is absent: {name}")
    resize, conv, relu = nodes[RESIZE_NAME], nodes[CONV_NAME], nodes[RELU_NAME]
    resize_attrs = attrs(resize)
    conv_attrs = attrs(conv)
    initializers = {item.name: onnx.numpy_helper.to_array(item) for item in model.graph.initializer}
    scales_name = resize.input[2]
    scales = initializers.get(scales_name)
    weight = initializers.get(CONV_WEIGHT)
    bias = initializers.get(CONV_BIAS)
    expected_resize = {
        "coordinate_transformation_mode": "half_pixel",
        "mode": "linear",
        "nearest_mode": "floor",
    }
    for key, expected in expected_resize.items():
        if resize_attrs.get(key) != expected:
            raise ValueError(f"Resize {key}={resize_attrs.get(key)!r}, expected {expected!r}")
    if scales is None or not np.array_equal(scales, np.asarray([1, 1, 2, 2], np.float32)):
        raise ValueError(f"unexpected Resize scales: {scales}")
    expected_conv = {
        "kernel_shape": [3, 3], "pads": [1, 1, 1, 1],
        "strides": [1, 1], "dilations": [1, 1], "group": 1,
    }
    for key, expected in expected_conv.items():
        if conv_attrs.get(key) != expected:
            raise ValueError(f"conv4 {key}={conv_attrs.get(key)!r}, expected {expected!r}")
    if weight is None or weight.shape != (128, 384, 3, 3):
        raise ValueError(f"unexpected conv4 weight shape: {None if weight is None else weight.shape}")
    if bias is None or bias.shape != (128,):
        raise ValueError(f"unexpected conv4 bias shape: {None if bias is None else bias.shape}")
    if list(conv.input) != [resize.output[0], CONV_WEIGHT, CONV_BIAS]:
        raise ValueError("conv4 does not consume resize2 output with the expected parameters")
    if list(relu.input) != [conv.output[0]]:
        raise ValueError("conv4 ReLU is not directly connected")
    return {
        "resize": {
            "name": resize.name, "input": resize.input[0], "output": resize.output[0],
            "attributes": resize_attrs, "scales_name": scales_name,
            "scales": scales.tolist(),
        },
        "conv4": {
            "name": conv.name, "input": conv.input[0], "output": conv.output[0],
            "attributes": conv_attrs, "weight_name": CONV_WEIGHT,
            "weight_shape": list(weight.shape), "weight_sha256": array_sha(weight),
            "bias_name": CONV_BIAS, "bias_shape": list(bias.shape),
            "bias_sha256": array_sha(bias),
        },
        "relu": {"name": relu.name, "input": relu.input[0], "output": relu.output[0]},
    }


def rewrite_model(source: onnx.ModelProto) -> onnx.ModelProto:
    """Replace conv4+ReLU with four exact phase convolutions and D2S."""
    model = onnx.ModelProto()
    model.CopyFrom(source)
    nodes = list(model.graph.node)
    by_name = {node.name: index for index, node in enumerate(nodes)}
    resize_i, conv_i, relu_i = (by_name[RESIZE_NAME], by_name[CONV_NAME], by_name[RELU_NAME])
    if not (resize_i + 1 == conv_i and conv_i + 1 == relu_i):
        raise ValueError("expected Resize -> conv4 -> ReLU nodes to be consecutive")
    resize, conv, relu = nodes[resize_i], nodes[conv_i], nodes[relu_i]

    phase_nodes: list[onnx.NodeProto] = []
    phase_outputs: list[str] = []
    phase_contract: list[str] = []
    # DCR ordering is phase-major: (00 channels), (01), (10), (11).
    for oy in range(2):
        for ox in range(2):
            output = f"phase_conv_{oy}{ox}_output"
            phase_outputs.append(output)
            pads = [1 - oy, 1 - ox, oy, ox]
            phase_contract.append(f"phase ({oy},{ox}): pads={pads}, stride=2")
            phase_nodes.append(helper.make_node(
                "Conv", [resize.output[0], conv.input[1], conv.input[2]], [output],
                name=f"/head_h/phase_conv_{oy}{ox}/Conv",
                kernel_shape=[3, 3], pads=pads, strides=[2, 2],
                dilations=[1, 1], group=1,
            ))
    phase_nodes.extend([
        helper.make_node(
            "Concat", phase_outputs, ["phase_concat_output"], axis=1,
            name="/head_h/phase_concat/Concat",
        ),
        helper.make_node(
            "Relu", ["phase_concat_output"], ["phase_relu_output"],
            name="/head_h/phase_relu/Relu",
        ),
        helper.make_node(
            "DepthToSpace", ["phase_relu_output"], [relu.output[0]],
            blocksize=2, mode="DCR", name="/head_h/phase_d2s/DepthToSpace",
        ),
    ])
    del model.graph.node[:]
    model.graph.node.extend(nodes[:conv_i] + phase_nodes + nodes[relu_i + 1:])
    model.producer_name = "semantic-sensing-bn-g-s-phase-rewrite"
    model.producer_version = "1"
    del model.metadata_props[:]
    for key, value in (
        ("rewrite", "exact_conv4_four_stride2_phases_depth_to_space_dcr"),
        ("source_onnx_sha256", SOURCE_ONNX_SHA256),
        ("checkpoint_sha256", CHECKPOINT_SHA256),
        ("phase_contract", "; ".join(phase_contract)),
    ):
        prop = model.metadata_props.add()
        prop.key, prop.value = key, value
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model, full_check=True)
    return model


def add_conv4_output(model: onnx.ModelProto) -> onnx.ModelProto:
    debug = onnx.ModelProto()
    debug.CopyFrom(model)
    name = "/head_h/head_h.1/head_h.1.2/Relu_output_0"
    if name not in {output.name for output in debug.graph.output}:
        debug.graph.output.append(helper.make_tensor_value_info(
            name, TensorProto.FLOAT, [1, 128, 64, 64]
        ))
    onnx.checker.check_model(debug)
    return debug


def ort_session(model: onnx.ModelProto) -> ort.InferenceSession:
    options = ort.SessionOptions()
    # Keep the graph rewrite visible; don't let ORT rewrite both arms to another form.
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(
        model.SerializeToString(), sess_options=options,
        providers=["CPUExecutionProvider"],
    )


def metric(reference: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    delta = np.asarray(actual, np.float64) - np.asarray(reference, np.float64)
    absolute = np.abs(delta)
    return {
        "max_abs": float(absolute.max()), "mean_abs": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "nonzero_count": int(np.count_nonzero(delta)), "element_count": int(delta.size),
        "reference_sha256": array_sha(reference), "actual_sha256": array_sha(actual),
    }


def golden_input(index: int) -> np.ndarray:
    path = GOLDEN_ROOT / "latent_y_hat" / f"{index}.npy"
    value = np.load(path, allow_pickle=False)
    if value.shape != (16, 16, 192):
        raise ValueError(f"{index}: golden shape {value.shape} != (16,16,192)")
    return np.ascontiguousarray(value.transpose(2, 0, 1)[None], dtype=np.float32)


def ort_equivalence(source: onnx.ModelProto, candidate: onnx.ModelProto) -> dict[str, Any]:
    source_session = ort_session(add_conv4_output(source))
    candidate_session = ort_session(add_conv4_output(candidate))
    output_names = ["x_h_hat", "x_l_hat", "/head_h/head_h.1/head_h.1.2/Relu_output_0"]
    rows: list[dict[str, Any]] = []
    worst = 0.0
    for index in INDICES:
        inputs = {"latent_y_hat": golden_input(index)}
        source_values = source_session.run(output_names, inputs)
        candidate_values = candidate_session.run(output_names, inputs)
        per_output = {}
        for name, reference, actual in zip(output_names, source_values, candidate_values):
            row = metric(reference, actual)
            per_output[name] = row
            worst = max(worst, row["max_abs"])
        # Explicitly report the exact high-resolution one-pixel border.
        reference = source_values[2]
        actual = candidate_values[2]
        border = np.zeros(reference.shape, dtype=bool)
        border[:, :, 0, :] = True
        border[:, :, -1, :] = True
        border[:, :, :, 0] = True
        border[:, :, :, -1] = True
        per_output["conv4_relu_border"] = metric(reference[border], actual[border])
        per_output["conv4_relu_interior"] = metric(reference[:, :, 1:-1, 1:-1], actual[:, :, 1:-1, 1:-1])
        rows.append({"val_index": index, "outputs": per_output})
    return {
        "status": "PASS" if worst <= ORT_EQUIVALENCE_TOL else "FAIL",
        "sample_count": len(rows), "indices": list(INDICES),
        "all_tensor_tolerance": ORT_EQUIVALENCE_TOL, "worst_max_abs": worst,
        "rows": rows,
        "scope": "diagnostic golden8 tensor equivalence only; no semantic selection",
    }


def derivation_text() -> str:
    return """# Exact phase rewrite derivation

Let the resized tensor be `R` and the original 3x3 convolution use zero padding
one, weights `W[co,ci,ky,kx]`, and bias `b[co]`.  For output phase
`(oy,ox) in {0,1}^2`, define

```
P_oy,ox[n,m,co] = Conv_stride2(
    R, W, b,
    pads=[1-oy, 1-ox, oy, ox]
)[n,m,co].
```

ONNX convolution indexing gives the sampled high-resolution location
`(2n + oy + ky - 1, 2m + ox + kx - 1)`, exactly the location used by the
original stride-1 convolution at output `(2n+oy, 2m+ox)`.  The asymmetric pad
puts zeros at precisely the original convolution's out-of-image locations.
Consequently every phase tensor equals the corresponding checkerboard of the
original output, including all four borders and corners.

Concatenate phases in DCR order `(00,01,10,11)`, apply ReLU (a pointwise
operation, so it commutes with the permutation), then `DepthToSpace(block=2,
mode=DCR)`.  This restores the original output exactly over real arithmetic.
The transformation reuses the original weight and bias initializers verbatim;
it introduces no learned or fitted parameter.

## Why a single ordinary low-resolution convolution is not generally exact

For 2x bilinear half-pixel resize in one dimension, an interior even sample is
`R[2n] = 1/4 X[n-1] + 3/4 X[n]`, while `R[0] = X[0]` because Resize clamps its
sampling coordinate at the boundary.  The following high-resolution Conv,
however, treats `R[-1]` as zero.  A translation-invariant low-resolution
convolution with one fixed padding rule cannot simultaneously encode this edge
clamp and the subsequent high-resolution zero padding for arbitrary weights.
It needs explicit boundary correction.  The four-phase stride-2 construction
above retains the high-resolution Resize and therefore avoids this mismatch.
"""


def prepare(out: Path) -> dict[str, Any]:
    if out.exists():
        raise FileExistsError(f"refusing to overwrite artifact root: {out}")
    if not SOURCE_ONNX.is_file() or sha256(SOURCE_ONNX) != SOURCE_ONNX_SHA256:
        raise ValueError("immutable source ONNX is missing or its SHA256 changed")
    if not GOLDEN_ROOT.is_dir():
        raise FileNotFoundError(GOLDEN_ROOT)
    out.mkdir(parents=True)
    source = onnx.load(str(SOURCE_ONNX))
    contract = source_contract(source)
    candidate = rewrite_model(source)
    candidate_path = out / "bn_g_s_conv4_phase_stride2.onnx"
    onnx.save(candidate, str(candidate_path))
    (out / "derivation.md").write_text(derivation_text(), encoding="utf-8")
    equivalence = ort_equivalence(source, candidate)
    report = {
        "kind": "bn_g_s_resize_conv_phase_rewrite_preparse_v1",
        "status": equivalence["status"],
        "gate": "FP32_ORT_EQUIVALENCE",
        "source": {
            "onnx": str(SOURCE_ONNX), "onnx_sha256": sha256(SOURCE_ONNX),
            "checkpoint_sha256": CHECKPOINT_SHA256, "contract": contract,
        },
        "candidate": {
            "onnx": str(candidate_path), "onnx_sha256": sha256(candidate_path),
            "rewrite": "four stride-2 phase convs + concat + ReLU + DepthToSpace DCR",
            "weights": "original conv4 W/b reused verbatim in all four phases",
            "training": False, "calibration": False,
        },
        "equivalence": equivalence,
        "environment": {
            "python": platform.python_version(), "onnx": onnx.__version__,
            "onnxruntime": ort.__version__, "provider": "CPUExecutionProvider",
            "ort_graph_optimizations": "disabled", "CUDA_VISIBLE_DEVICES": "",
        },
        "restrictions": {
            "optimize": False, "compile": False, "board": False,
            "full976": False, "semantic_ranking": False,
        },
    }
    atomic_json(out / "preparse_report.json", report)
    atomic_json(out / "campaign_state.json", {
        "status": "READY_FOR_PARSE_NATIVE" if equivalence["status"] == "PASS" else "STOPPED_AT_ORT_EQUIVALENCE",
        "preparse_report": str(out / "preparse_report.json"),
        "candidate_onnx": str(candidate_path),
    })
    return report


def hn_facts(runner: Any) -> dict[str, Any]:
    hn = runner.get_hn_dict()
    layers = hn.get("layers", {})
    rows = []
    for name, spec in layers.items():
        originals = list(spec.get("original_names") or [])
        if any("phase_" in item or "DepthToSpace" in item for item in originals):
            rows.append({"name": name, "type": spec.get("type"), "original_names": originals})
    serialized = json.dumps(hn, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "layer_count": len(layers),
        "op_types": sorted({str(spec.get("type", "?")) for spec in layers.values()}),
        "phase_related_layers": rows,
        "phase_related_layer_count": len(rows),
        "hn_json_sha256": hashlib.sha256(serialized).hexdigest(),
    }


def native_on_golden8(runner: Any, onnx_path: Path) -> dict[str, Any]:
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    output_names = [output.name for output in session.get_outputs()]
    sig = signature(onnx_path)
    rows = []
    worst = 0.0
    notes: list[str] = []
    from hailo_sdk_client import InferenceContext

    with runner.infer_context(InferenceContext.SDK_NATIVE) as context:
        for index in INDICES:
            inputs = {"latent_y_hat": golden_input(index)}
            ort_values = session.run(output_names, inputs)
            references = dict(zip(output_names, ort_values))
            feed = _build_native_feed(runner, sig, inputs, notes)
            raw = runner.infer(context, feed)
            pairs = _match_outputs(runner, references, raw, notes)
            if set(pairs) != set(output_names):
                raise ValueError(f"{index}: SDK_NATIVE did not map all outputs: {set(pairs)}")
            per_output = {}
            for name, (reference, actual) in pairs.items():
                row = metric(reference, actual)
                per_output[name] = row
                worst = max(worst, row["max_abs"])
            rows.append({"val_index": index, "outputs": per_output})
    return {
        "status": "PASS" if worst <= NATIVE_TOL else "MISMATCH",
        "sample_count": len(rows), "indices": list(INDICES), "tolerance": NATIVE_TOL,
        "worst_max_abs": worst, "rows": rows, "mapping_notes": sorted(set(notes)),
    }


def parse_native(out: Path) -> dict[str, Any]:
    preparse_path = out / "preparse_report.json"
    if not preparse_path.is_file():
        raise FileNotFoundError("run --phase prepare first")
    preparse = json.loads(preparse_path.read_text(encoding="utf-8"))
    if preparse.get("status") != "PASS":
        raise RuntimeError("ORT equivalence gate did not pass; DFC is forbidden")
    candidate_path = out / "bn_g_s_conv4_phase_stride2.onnx"
    if sha256(candidate_path) != preparse["candidate"]["onnx_sha256"]:
        raise ValueError("candidate ONNX changed after the equivalence gate")
    har_path = out / "bn_g_s_conv4_phase_stride2_hailo_model.har"
    report_path = out / "parse_native_report.json"
    if har_path.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite existing parse/NATIVE artifacts")

    from hailo_sdk_client import ClientRunner

    runner = ClientRunner(hw_arch="hailo8")
    runner.translate_onnx_model(str(candidate_path), "bn_g_s_phase_stride2")
    facts = hn_facts(runner)
    runner.save_har(str(har_path))
    native = native_on_golden8(runner, candidate_path)
    report = {
        "kind": "bn_g_s_resize_conv_phase_rewrite_parse_native_v1",
        "status": "PASS" if native["status"] == "PASS" else "FAIL",
        "gates": {"ort_equivalence": "PASS", "dfc_parse": "PASS", "sdk_native": native["status"]},
        "target": {"hw_arch": "hailo8", "dataflow_compiler": package_version("hailo-dataflow-compiler")},
        "candidate": {"onnx": str(candidate_path), "onnx_sha256": sha256(candidate_path)},
        "har": {"path": str(har_path), "sha256": sha256(har_path)},
        "hn": facts, "sdk_native_golden8": native,
        "graph_distinctness": {
            "source_conv4_removed": True,
            "onnx_phase_conv_count": 4,
            "hn_phase_related_layer_count": facts["phase_related_layer_count"],
            "kept_as_distinct_hn_graph": facts["phase_related_layer_count"] > 0,
        },
        "restrictions": {
            "optimize": False, "compile": False, "board": False,
            "full976": False, "semantic_ranking": False,
        },
    }
    atomic_json(report_path, report)
    atomic_json(out / "campaign_state.json", {
        "status": "PASS_PARSE_NATIVE_ONLY" if report["status"] == "PASS" else "STOPPED_AT_NATIVE",
        "preparse_report": str(preparse_path), "parse_native_report": str(report_path),
        "explicit_stop": "No optimize, compile, or board execution is authorized for this lane.",
    })
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "parse-native", "all"), default="all")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out_root.resolve()
    if args.phase in ("prepare", "all"):
        report = prepare(out)
        print(json.dumps(report, indent=2, sort_keys=True))
        if report["status"] != "PASS":
            return 1
    if args.phase in ("parse-native", "all"):
        report = parse_native(out)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["status"] == "PASS" else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
