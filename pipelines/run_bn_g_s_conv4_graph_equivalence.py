#!/usr/bin/env python3
"""Pre-PTQ graph-equivalence probes for ``bn_g_s`` conv4.

This pipeline is intentionally limited to:

1. create independent ONNX rewrites of conv4;
2. ONNX checker + golden8 ORT parity against the immutable FP32 ONNX;
3. DFC translation and SDK_NATIVE parity for candidates that pass (2);
4. inspect the translated HN to determine whether DFC retained the rewrite.

It never calls optimize(), compile(), or a board runtime.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

# NATIVE must be CPU/FP32 in this project because GPU TF32 previously produced
# false mismatches.  Set this before importing the Hailo SDK.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper, shape_inference

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hailo_deploy import emulate  # noqa: E402
from hailo_deploy.io.hash import sha256  # noqa: E402
from hailo_deploy.io.metrics import metric, summarize  # noqa: E402


DEFAULT_SOURCE = Path(
    "/local/shared_with_docker/hailo_full_candidate/"
    "convbn-gfa-gs-3a3ea630af6d2d547808137be678bd909f9789f5594b7bdc76c36b90/"
    "onnx/bn_g_s.onnx"
)
DEFAULT_GOLDEN = DEFAULT_SOURCE.parent.parent / "evaluation/golden"
DEFAULT_OUT = Path(
    "/local/shared_with_docker/deplyment-demo/artifacts/fused/experiments/"
    "bn-gs-conv4-graph-equivalence-20260806T070529Z"
)

CONV4_NAME = "/head_h/head_h.1/head_h.1.0/Conv"
RELU4_NAME = "/head_h/head_h.1/head_h.1.2/Relu"
CONV5_NAME = "/head_h/head_h.2/Conv"
CONV4_WEIGHT = "onnx::Conv_115"
CONV4_BIAS = "onnx::Conv_116"
CONV4_INPUT = "/head_h/head_h.0/Resize_output_0"
CONV4_OUTPUT = "/head_h/head_h.1/head_h.1.0/Conv_output_0"
RELU4_OUTPUT = "/head_h/head_h.1/head_h.1.2/Relu_output_0"
PREFIX = "/graph_equiv/conv4"
GOLDEN_INDICES = tuple(range(1024, 1032))

# Tight numerical gate.  A mathematically equivalent re-association is allowed
# only when its complete golden8 FP32 drift remains below both limits.
ORT_MAX_ABS_TOL = 2e-5
ORT_RMSE_TOL = 2e-6
NATIVE_MAX_ABS_TOL = 1e-3


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    split_count: int
    rewrite: Callable[[onnx.ModelProto], list[str]]


def atomic_json(path: Path, payload: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite: {path}")
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _attributes(node: onnx.NodeProto) -> dict[str, Any]:
    return {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}


def _graph_parts(model: onnx.ModelProto) -> tuple[onnx.NodeProto, onnx.NodeProto, int, int]:
    nodes = list(model.graph.node)
    conv_i = next(i for i, node in enumerate(nodes) if node.name == CONV4_NAME)
    relu_i = next(i for i, node in enumerate(nodes) if node.name == RELU4_NAME)
    if relu_i != conv_i + 1:
        raise ValueError("formal conv4 and ReLU are no longer adjacent")
    conv, relu = nodes[conv_i], nodes[relu_i]
    if list(conv.input) != [CONV4_INPUT, CONV4_WEIGHT, CONV4_BIAS]:
        raise ValueError(f"formal conv4 inputs changed: {list(conv.input)}")
    if list(conv.output) != [CONV4_OUTPUT] or list(relu.output) != [RELU4_OUTPUT]:
        raise ValueError("formal conv4/ReLU outputs changed")
    return conv, relu, conv_i, relu_i


def _initializer_map(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    return {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}


def _replace_conv4(
    model: onnx.ModelProto,
    new_nodes: list[onnx.NodeProto],
    new_initializers: list[onnx.TensorProto],
) -> None:
    _, _, conv_i, relu_i = _graph_parts(model)
    old_nodes = list(model.graph.node)
    model.graph.ClearField("node")
    model.graph.node.extend(old_nodes[:conv_i] + new_nodes + old_nodes[relu_i + 1 :])

    retained = [
        item for item in model.graph.initializer
        if item.name not in {CONV4_WEIGHT, CONV4_BIAS}
    ]
    model.graph.ClearField("initializer")
    model.graph.initializer.extend(retained + new_initializers)


def _output_channel_split(model: onnx.ModelProto, count: int) -> list[str]:
    conv, _, _, _ = _graph_parts(model)
    arrays = _initializer_map(model)
    weight, bias = arrays[CONV4_WEIGHT], arrays[CONV4_BIAS]
    if weight.shape != (128, 384, 3, 3) or bias.shape != (128,):
        raise ValueError(f"unexpected conv4 parameters: W={weight.shape}, b={bias.shape}")
    if weight.shape[0] % count:
        raise ValueError("output channels are not divisible by split count")

    nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = []
    relu_outputs: list[str] = []
    created: list[str] = []
    for part, (w_chunk, b_chunk) in enumerate(
        zip(np.split(weight, count, axis=0), np.split(bias, count, axis=0))
    ):
        stem = f"{PREFIX}/oc{count}/part{part}"
        w_name, b_name = f"{stem}/weight", f"{stem}/bias"
        conv_out, relu_out = f"{stem}/conv_out", f"{stem}/relu_out"
        conv_name, relu_name = f"{stem}/Conv", f"{stem}/Relu"
        initializers.extend([
            numpy_helper.from_array(np.ascontiguousarray(w_chunk), w_name),
            numpy_helper.from_array(np.ascontiguousarray(b_chunk), b_name),
        ])
        nodes.append(helper.make_node(
            "Conv", [CONV4_INPUT, w_name, b_name], [conv_out],
            name=conv_name, **_attributes(conv),
        ))
        nodes.append(helper.make_node("Relu", [conv_out], [relu_out], name=relu_name))
        relu_outputs.append(relu_out)
        created.extend([conv_name, relu_name])

    concat_name = f"{PREFIX}/oc{count}/Concat"
    nodes.append(helper.make_node(
        "Concat", relu_outputs, [RELU4_OUTPUT], name=concat_name, axis=1,
    ))
    created.append(concat_name)
    _replace_conv4(model, nodes, initializers)
    return created


def _int64_initializer(name: str, values: list[int]) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(values, dtype=np.int64), name)


def _input_channel_split(model: onnx.ModelProto, count: int) -> list[str]:
    conv, _, _, _ = _graph_parts(model)
    arrays = _initializer_map(model)
    weight, bias = arrays[CONV4_WEIGHT], arrays[CONV4_BIAS]
    if weight.shape != (128, 384, 3, 3) or bias.shape != (128,):
        raise ValueError(f"unexpected conv4 parameters: W={weight.shape}, b={bias.shape}")
    if weight.shape[1] % count:
        raise ValueError("input channels are not divisible by split count")

    nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = []
    partials: list[str] = []
    created: list[str] = []
    width = weight.shape[1] // count
    axes_name = f"{PREFIX}/ic{count}/slice_axes"
    initializers.append(_int64_initializer(axes_name, [1]))
    for part in range(count):
        start, end = part * width, (part + 1) * width
        stem = f"{PREFIX}/ic{count}/part{part}"
        starts_name, ends_name = f"{stem}/starts", f"{stem}/ends"
        w_name = f"{stem}/weight"
        slice_out, conv_out = f"{stem}/slice_out", f"{stem}/conv_out"
        slice_name, conv_name = f"{stem}/Slice", f"{stem}/Conv"
        initializers.extend([
            _int64_initializer(starts_name, [start]),
            _int64_initializer(ends_name, [end]),
            numpy_helper.from_array(
                np.ascontiguousarray(weight[:, start:end, :, :]), w_name
            ),
        ])
        nodes.append(helper.make_node(
            "Slice", [CONV4_INPUT, starts_name, ends_name, axes_name], [slice_out],
            name=slice_name,
        ))
        nodes.append(helper.make_node(
            "Conv", [slice_out, w_name], [conv_out], name=conv_name,
            **_attributes(conv),
        ))
        partials.append(conv_out)
        created.extend([slice_name, conv_name])

    accumulator = partials[0]
    for part, partial in enumerate(partials[1:], start=1):
        output = f"{PREFIX}/ic{count}/sum{part}_out"
        name = f"{PREFIX}/ic{count}/AddPartial{part}"
        nodes.append(helper.make_node("Add", [accumulator, partial], [output], name=name))
        accumulator = output
        created.append(name)

    bias_name = f"{PREFIX}/ic{count}/bias"
    bias_output = f"{PREFIX}/ic{count}/bias_out"
    bias_add_name = f"{PREFIX}/ic{count}/AddBias"
    relu_name = f"{PREFIX}/ic{count}/Relu"
    initializers.append(numpy_helper.from_array(
        np.ascontiguousarray(bias.reshape(1, -1, 1, 1)), bias_name
    ))
    nodes.append(helper.make_node(
        "Add", [accumulator, bias_name], [bias_output], name=bias_add_name,
    ))
    nodes.append(helper.make_node("Relu", [bias_output], [RELU4_OUTPUT], name=relu_name))
    created.extend([bias_add_name, relu_name])
    _replace_conv4(model, nodes, initializers)
    return created


def _spatial_height_split(model: onnx.ModelProto, count: int) -> list[str]:
    """Tile conv4 along output height with the exact one-row input halo.

    Conv4 is a stride-1 3x3 convolution over a fixed 64x64 tensor with
    symmetric padding 1.  Each output-height tile therefore needs one input
    row of halo on both internal edges.  External padding is retained only on
    the first/last tile.  No partial sums or numerical re-association are
    introduced: every output element is evaluated by one convolution with the
    original weights and bias, then the tiles are concatenated in row order.
    """
    conv, _, _, _ = _graph_parts(model)
    attrs = _attributes(conv)
    arrays = _initializer_map(model)
    weight, bias = arrays[CONV4_WEIGHT], arrays[CONV4_BIAS]
    if weight.shape != (128, 384, 3, 3) or bias.shape != (128,):
        raise ValueError(f"unexpected conv4 parameters: W={weight.shape}, b={bias.shape}")
    expected = {
        "dilations": [1, 1],
        "group": 1,
        "kernel_shape": [3, 3],
        "pads": [1, 1, 1, 1],
        "strides": [1, 1],
    }
    if attrs != expected:
        raise ValueError(f"height tiling assumes canonical conv4 attributes: {attrs}")
    height = 64
    if height % count:
        raise ValueError("output height is not divisible by split count")

    nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = []
    relu_outputs: list[str] = []
    created: list[str] = []
    axes_name = f"{PREFIX}/ht{count}/slice_axes"
    initializers.append(_int64_initializer(axes_name, [2]))
    tile_height = height // count
    for part in range(count):
        out_start, out_end = part * tile_height, (part + 1) * tile_height
        input_start = max(0, out_start - 1)
        input_end = min(height, out_end + 1)
        stem = f"{PREFIX}/ht{count}/part{part}"
        starts_name, ends_name = f"{stem}/starts", f"{stem}/ends"
        w_name, b_name = f"{stem}/weight", f"{stem}/bias"
        slice_out, conv_out, relu_out = (
            f"{stem}/slice_out", f"{stem}/conv_out", f"{stem}/relu_out"
        )
        slice_name, conv_name, relu_name = (
            f"{stem}/Slice", f"{stem}/Conv", f"{stem}/Relu"
        )
        initializers.extend([
            _int64_initializer(starts_name, [input_start]),
            _int64_initializer(ends_name, [input_end]),
            numpy_helper.from_array(np.ascontiguousarray(weight), w_name),
            numpy_helper.from_array(np.ascontiguousarray(bias), b_name),
        ])
        nodes.append(helper.make_node(
            "Slice", [CONV4_INPUT, starts_name, ends_name, axes_name], [slice_out],
            name=slice_name,
        ))
        tile_attrs = dict(attrs)
        tile_attrs["pads"] = [
            1 if part == 0 else 0,
            1,
            1 if part == count - 1 else 0,
            1,
        ]
        nodes.append(helper.make_node(
            "Conv", [slice_out, w_name, b_name], [conv_out],
            name=conv_name, **tile_attrs,
        ))
        nodes.append(helper.make_node("Relu", [conv_out], [relu_out], name=relu_name))
        relu_outputs.append(relu_out)
        created.extend([slice_name, conv_name, relu_name])

    concat_name = f"{PREFIX}/ht{count}/Concat"
    nodes.append(helper.make_node(
        "Concat", relu_outputs, [RELU4_OUTPUT], name=concat_name, axis=2,
    ))
    created.append(concat_name)
    _replace_conv4(model, nodes, initializers)
    return created


def _bias_activation_separation(model: onnx.ModelProto) -> list[str]:
    conv, _, _, _ = _graph_parts(model)
    arrays = _initializer_map(model)
    weight, bias = arrays[CONV4_WEIGHT], arrays[CONV4_BIAS]
    stem = f"{PREFIX}/bias_relu_sep"
    w_name, b_name = f"{stem}/weight", f"{stem}/bias"
    conv_out, bias_out, identity_out = (
        f"{stem}/conv_out", f"{stem}/bias_out", f"{stem}/identity_out"
    )
    names = [f"{stem}/Conv", f"{stem}/AddBias", f"{stem}/Identity", f"{stem}/Relu"]
    nodes = [
        helper.make_node(
            "Conv", [CONV4_INPUT, w_name], [conv_out], name=names[0],
            **_attributes(conv),
        ),
        helper.make_node("Add", [conv_out, b_name], [bias_out], name=names[1]),
        helper.make_node("Identity", [bias_out], [identity_out], name=names[2]),
        helper.make_node("Relu", [identity_out], [RELU4_OUTPUT], name=names[3]),
    ]
    initializers = [
        numpy_helper.from_array(np.ascontiguousarray(weight), w_name),
        numpy_helper.from_array(np.ascontiguousarray(bias.reshape(1, -1, 1, 1)), b_name),
    ]
    _replace_conv4(model, nodes, initializers)
    return names


def _make_output_split(count: int) -> Callable[[onnx.ModelProto], list[str]]:
    return lambda model: _output_channel_split(model, count)


def _make_input_split(count: int) -> Callable[[onnx.ModelProto], list[str]]:
    return lambda model: _input_channel_split(model, count)


def _make_spatial_height_split(count: int) -> Callable[[onnx.ModelProto], list[str]]:
    return lambda model: _spatial_height_split(model, count)


CANDIDATES = (
    Candidate("output_channel_split_2", "output_channel", 2, _make_output_split(2)),
    Candidate("output_channel_split_4", "output_channel", 4, _make_output_split(4)),
    Candidate("input_channel_split_2", "input_channel", 2, _make_input_split(2)),
    Candidate("input_channel_split_4", "input_channel", 4, _make_input_split(4)),
    Candidate("spatial_height_split_2", "spatial_height", 2, _make_spatial_height_split(2)),
    Candidate("spatial_height_split_4", "spatial_height", 4, _make_spatial_height_split(4)),
    Candidate("bias_activation_separation", "bias_activation", 1, _bias_activation_separation),
)


def _ort_session(path: Path) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


def _run_ort(session: ort.InferenceSession, tensor: np.ndarray) -> dict[str, np.ndarray]:
    names = [item.name for item in session.get_outputs()]
    values = session.run(names, {"latent_y_hat": np.asarray(tensor, dtype=np.float32)})
    return dict(zip(names, values))


def _golden_input(root: Path, index: int) -> np.ndarray:
    path = root / str(index) / "bn_g_s/inputs/latent_y_hat.npy"
    if not path.is_file():
        raise FileNotFoundError(path)
    tensor = np.load(path, allow_pickle=False)
    if tensor.shape != (1, 192, 16, 16) or tensor.dtype != np.float32:
        raise ValueError(f"unexpected golden input {path}: {tensor.shape} {tensor.dtype}")
    return tensor


def _ort_equivalence(formal: Path, candidate: Path, golden: Path) -> dict[str, Any]:
    formal_session, candidate_session = _ort_session(formal), _ort_session(candidate)
    rows: dict[str, list[dict[str, float | None]]] = {"x_h_hat": [], "x_l_hat": []}
    bit_exact: dict[str, bool] = {"x_h_hat": True, "x_l_hat": True}
    per_sample: dict[str, Any] = {}
    for index in GOLDEN_INDICES:
        tensor = _golden_input(golden, index)
        ref, got = _run_ort(formal_session, tensor), _run_ort(candidate_session, tensor)
        sample: dict[str, Any] = {}
        for name in ("x_h_hat", "x_l_hat"):
            item = metric(ref[name], got[name])
            exact = bool(np.array_equal(ref[name], got[name]))
            item["bit_exact"] = exact
            rows[name].append(item)
            bit_exact[name] = bit_exact[name] and exact
            sample[name] = item
        per_sample[str(index)] = sample
    aggregate = {name: summarize(items) for name, items in rows.items()}
    passed = all(
        aggregate[name]["max_abs"]["max"] <= ORT_MAX_ABS_TOL
        and aggregate[name]["rmse"]["max"] <= ORT_RMSE_TOL
        for name in aggregate
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "gate": {"max_abs_lte": ORT_MAX_ABS_TOL, "rmse_lte": ORT_RMSE_TOL},
        "ort_graph_optimizations": "disabled",
        "indices": list(GOLDEN_INDICES),
        "all_samples_bit_exact": bit_exact,
        "aggregate": aggregate,
        "per_sample": per_sample,
    }


def _interesting_hn_layers(hn: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for name, layer in hn.get("layers", {}).items():
        originals = list(layer.get("original_names") or [])
        if any(PREFIX in original for original in originals):
            result.append({
                "name": name,
                "type": layer.get("type"),
                "original_names": originals,
                "input": layer.get("input"),
                "output": layer.get("output"),
                "activation": layer.get("params", {}).get("activation"),
                "input_shapes": layer.get("input_shapes"),
                "output_shapes": layer.get("output_shapes"),
            })
    return result


def _retention(candidate: Candidate, layers: list[dict[str, Any]]) -> dict[str, Any]:
    conv_layers = [
        row for row in layers
        if "conv" in str(row.get("type", "")).lower()
        and any("/Conv" in original for original in row["original_names"])
    ]
    if candidate.family in {"output_channel", "input_channel", "spatial_height"}:
        retained = len(conv_layers) >= candidate.split_count
        return {
            "status": "RETAINED" if retained else "REFUSED_TO_SINGLE_CONV",
            "effective_graph_optimization": retained,
            "expected_conv_branches": candidate.split_count,
            "observed_hn_conv_layers": len(conv_layers),
            "reason": (
                "DFC HN retained multiple conv4 branches"
                if retained else "DFC HN no longer contains the requested number of conv4 branches"
            ),
        }

    # A bias/ReLU separation is useful only if at least one non-convolution HN
    # layer still owns AddBias or Relu.  If the single conv owns every original
    # name, the parser immediately folded the rewrite back.
    separated_names = {f"{PREFIX}/bias_relu_sep/AddBias", f"{PREFIX}/bias_relu_sep/Relu"}
    separate_layers = [
        row for row in layers
        if not ("conv" in str(row.get("type", "")).lower())
        and separated_names.intersection(row["original_names"])
    ]
    retained = bool(separate_layers)
    return {
        "status": "RETAINED" if retained else "REFUSED_TO_SINGLE_CONV",
        "effective_graph_optimization": retained,
        "observed_hn_conv_layers": len(conv_layers),
        "observed_separate_bias_or_relu_layers": len(separate_layers),
        "reason": (
            "DFC HN retained a separate bias/activation layer"
            if retained else "DFC folded Conv/Add/Identity/ReLU back into one HN conv"
        ),
    }


def _parse(candidate: Candidate, onnx_path: Path, har_path: Path) -> tuple[Any, dict[str, Any]]:
    from hailo_sdk_client import ClientRunner

    runner = ClientRunner(hw_arch="hailo8")
    try:
        runner.translate_onnx_model(str(onnx_path), f"bn_g_s_ge_{candidate.name}")
    except Exception as exc:
        return None, {
            "status": "FAIL",
            "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        }
    runner.save_har(str(har_path))
    hn = runner.get_hn_dict()
    layers = _interesting_hn_layers(hn)
    return runner, {
        "status": "PASS",
        "har": str(har_path),
        "har_sha256": sha256(har_path),
        "hn_layer_count": len(hn.get("layers", {})),
        "interesting_hn_layers": layers,
        "rewrite_retention": _retention(candidate, layers),
    }


def _native_parity(runner: Any, onnx_path: Path, golden: Path) -> dict[str, Any]:
    from hailo_sdk_client import InferenceContext

    session = _ort_session(onnx_path)
    rows: dict[str, list[dict[str, float | None]]] = {"x_h_hat": [], "x_l_hat": []}
    per_sample: dict[str, Any] = {}
    notes: list[str] = []
    with runner.infer_context(InferenceContext.SDK_NATIVE) as context:
        for index in GOLDEN_INDICES:
            tensor = _golden_input(golden, index)
            reference = _run_ort(session, tensor)
            feed = np.transpose(tensor, (0, 2, 3, 1))
            native_result = runner.infer(context, feed)
            pairs = emulate._match_outputs(runner, reference, native_result, notes)
            if set(pairs) != set(reference):
                raise ValueError(f"native outputs did not map at index {index}: {set(pairs)}")
            sample: dict[str, Any] = {}
            for name, (ref, got) in pairs.items():
                item = metric(ref, got)
                rows[name].append(item)
                sample[name] = item
            per_sample[str(index)] = sample
    aggregate = {name: summarize(items) for name, items in rows.items()}
    passed = all(
        aggregate[name]["max_abs"]["max"] < NATIVE_MAX_ABS_TOL
        for name in aggregate
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "max_abs_lt": NATIVE_MAX_ABS_TOL,
        "execution": "SDK_NATIVE CPU/FP32; CUDA_VISIBLE_DEVICES is empty",
        "indices": list(GOLDEN_INDICES),
        "aggregate": aggregate,
        "per_sample": per_sample,
        "mapping_notes": sorted(set(notes)),
    }


def _node_inventory(model: onnx.ModelProto) -> list[dict[str, Any]]:
    return [
        {
            "name": node.name,
            "op_type": node.op_type,
            "inputs": list(node.input),
            "outputs": list(node.output),
        }
        for node in model.graph.node
        if PREFIX in node.name
    ]


def _make_candidate(source: Path, target: Path, candidate: Candidate) -> dict[str, Any]:
    model = copy.deepcopy(onnx.load(str(source)))
    created = candidate.rewrite(model)
    model = shape_inference.infer_shapes(model)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, str(target))
    reloaded = onnx.load(str(target))
    onnx.checker.check_model(reloaded, full_check=True)
    return {
        "path": str(target),
        "sha256": sha256(target),
        "size_bytes": target.stat().st_size,
        "created_node_names": created,
        "candidate_nodes": _node_inventory(reloaded),
        "onnx_checker": "PASS",
        "opset": [(item.domain, item.version) for item in reloaded.opset_import],
    }


def run(
    source: Path,
    golden: Path,
    out: Path,
    candidates: tuple[Candidate, ...] = CANDIDATES,
) -> dict[str, Any]:
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing out root: {out}")
    (out / "onnx").mkdir(parents=True)
    (out / "har").mkdir()
    (out / "reports").mkdir()

    formal_model = onnx.load(str(source))
    onnx.checker.check_model(formal_model, full_check=True)
    conv, relu, _, _ = _graph_parts(formal_model)
    conv5 = next(node for node in formal_model.graph.node if node.name == CONV5_NAME)
    params = _initializer_map(formal_model)
    report: dict[str, Any] = {
        "kind": "bn_g_s_conv4_graph_equivalence_pre_ptq_v1",
        "scope": "bn_g_s only; ONNX/ORT/DFC parse/SDK_NATIVE; no optimize/compile/board",
        "status": "RUNNING",
        "environment": {
            "target": "hailo8",
            "dfc": "3.33.0",
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "formal": {
            "path": str(source),
            "sha256": sha256(source),
            "onnx_checker": "PASS",
            "conv4": {
                "node": conv.name,
                "op_type": conv.op_type,
                "input": CONV4_INPUT,
                "weight_shape": list(params[CONV4_WEIGHT].shape),
                "bias_shape": list(params[CONV4_BIAS].shape),
                "attributes": _attributes(conv),
                "activation": relu.name,
            },
            "conv5": {"node": conv5.name, "op_type": conv5.op_type},
        },
        "golden8": {
            "root": str(golden),
            "indices": list(GOLDEN_INDICES),
            "role": "diagnostic/localization only; not semantic ranking or acceptance",
        },
        "gates": {
            "ort": {"max_abs_lte": ORT_MAX_ABS_TOL, "rmse_lte": ORT_RMSE_TOL},
            "native": {"max_abs_lt": NATIVE_MAX_ABS_TOL},
        },
        "candidates": {},
    }

    for candidate in candidates:
        item: dict[str, Any] = {
            "family": candidate.family,
            "split_count": candidate.split_count,
            "status": "RUNNING",
        }
        report["candidates"][candidate.name] = item
        candidate_dir = out / "onnx" / candidate.name
        candidate_dir.mkdir()
        onnx_path = candidate_dir / "bn_g_s.onnx"
        try:
            item["onnx"] = _make_candidate(source, onnx_path, candidate)
            item["ort_equivalence"] = _ort_equivalence(source, onnx_path, golden)
            if item["ort_equivalence"]["status"] != "PASS":
                item["status"] = "REJECTED_AT_ORT_EQUIVALENCE"
                continue
            har_path = out / "har" / f"{candidate.name}.har"
            runner, parse_result = _parse(candidate, onnx_path, har_path)
            item["dfc_parse"] = parse_result
            if runner is None:
                item["status"] = "REJECTED_AT_DFC_PARSE"
                continue
            item["sdk_native_parity"] = _native_parity(runner, onnx_path, golden)
            retention = parse_result["rewrite_retention"]["effective_graph_optimization"]
            native_pass = item["sdk_native_parity"]["status"] == "PASS"
            if not native_pass:
                item["status"] = "REJECTED_AT_SDK_NATIVE"
            elif not retention:
                item["status"] = "PASS_NUMERIC_BUT_GRAPH_REWRITE_FOLDED"
            else:
                item["status"] = "PASS_PRE_PTQ"
        except Exception as exc:
            item["status"] = "ERROR"
            item["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        atomic_json(out / "reports" / f"{candidate.name}.json", item)

    passing = [
        name for name, item in report["candidates"].items()
        if item["status"] == "PASS_PRE_PTQ"
    ]
    folded = [
        name for name, item in report["candidates"].items()
        if item["status"] == "PASS_NUMERIC_BUT_GRAPH_REWRITE_FOLDED"
    ]
    report["summary"] = {
        "pass_pre_ptq": passing,
        "numeric_pass_but_folded": folded,
        "rejected": {
            name: item["status"] for name, item in report["candidates"].items()
            if item["status"] not in {"PASS_PRE_PTQ", "PASS_NUMERIC_BUT_GRAPH_REWRITE_FOLDED"}
        },
        "interpretation": (
            "Only PASS_PRE_PTQ candidates are eligible for a later PTQ/compile/board A/B. "
            "This report contains no quantized or semantic-quality result."
        ),
        "estimated_followup_per_retained_candidate": {
            "representative_L2_PTQ": "approximately 10-20 minutes on an available GPU",
            "compile_and_topology_gate": "approximately 5-15 minutes",
            "golden8_board_diagnostic": "approximately 5-10 minutes",
            "full976_if_later_approved": "approximately 1.5-2 hours",
        },
    }
    report["status"] = "PASS" if passing else "NO_EFFECTIVE_PRE_PTQ_CANDIDATE"
    atomic_json(out / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--golden-root", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--candidate",
        action="append",
        choices=[item.name for item in CANDIDATES],
        help="Run only the named candidate; repeat to select more than one.",
    )
    args = parser.parse_args()
    selected = (
        tuple(item for item in CANDIDATES if item.name in args.candidate)
        if args.candidate
        else CANDIDATES
    )
    report = run(
        args.source.resolve(), args.golden_root.resolve(), args.out_root.resolve(), selected
    )
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    print(report["status"])
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
