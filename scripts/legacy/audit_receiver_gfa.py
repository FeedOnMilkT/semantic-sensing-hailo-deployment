#!/usr/bin/env python3
"""Audit GFA operator chains in a static receiver ONNX/HAR pair.

This tool is diagnostic only: it never changes ONNX, HAR, calibration, or model
scripts.  It writes a JSON evidence bundle that links ONNX topology/shape data
with the parsed Hailo HN layer metadata retained in a HAR.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def _shape_from_value_info(value_info: Any) -> list[Any]:
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        return []
    dims: list[Any] = []
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(dim.dim_value)
        elif dim.HasField("dim_param"):
            dims.append(dim.dim_param)
        else:
            dims.append(None)
    return dims


def _tensor_shapes(model: Any) -> dict[str, list[Any]]:
    shapes: dict[str, list[Any]] = {}
    for item in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        shapes[item.name] = _shape_from_value_info(item)
    for initializer in model.graph.initializer:
        shapes.setdefault(initializer.name, list(initializer.dims))
    return shapes


def _node_dict(index: int, node: Any, shapes: dict[str, list[Any]]) -> dict[str, Any]:
    return {
        "index": index,
        "name": node.name or f"node_{index}",
        "op_type": node.op_type,
        "inputs": [{"tensor": name, "shape": shapes.get(name)} for name in node.input],
        "outputs": [{"tensor": name, "shape": shapes.get(name)} for name in node.output],
    }


def _upstream_ops(tensor: str, producers: dict[str, int], nodes: list[Any], depth: int = 8) -> list[str]:
    """Return a bounded, nearest-first producer-op trace for a tensor."""
    trace: list[str] = []
    frontier = [(tensor, 0)]
    seen: set[str] = set()
    while frontier:
        name, level = frontier.pop(0)
        if name in seen or level >= depth or name not in producers:
            continue
        seen.add(name)
        node = nodes[producers[name]]
        trace.append(node.op_type)
        frontier.extend((item, level + 1) for item in node.input)
    return trace


def _onnx_audit(path: Path) -> dict[str, Any]:
    import onnx

    source = onnx.load(path)
    onnx.checker.check_model(source)
    try:
        model = onnx.shape_inference.infer_shapes(source)
        shape_inference_error = None
    except Exception as exc:  # Preserve a usable static audit if inference is partial.
        model = source
        shape_inference_error = f"{type(exc).__name__}: {exc}"
    shapes = _tensor_shapes(model)
    nodes = list(model.graph.node)
    producers = {output: index for index, node in enumerate(nodes) for output in node.output}
    consumers: dict[str, list[int]] = defaultdict(list)
    for index, node in enumerate(nodes):
        for input_name in node.input:
            consumers[input_name].append(index)

    selected_types = {"MatMul", "Softmax", "Reshape", "Transpose", "ReduceSum", "Div", "Mul"}
    relevant = [_node_dict(index, node, shapes) for index, node in enumerate(nodes)
                if node.op_type in selected_types]
    softmaxes = []
    grouped_divisions = []
    for index, node in enumerate(nodes):
        if node.op_type == "Softmax":
            item = _node_dict(index, node, shapes)
            item["upstream_ops"] = _upstream_ops(node.input[0], producers, nodes)
            item["downstream_ops"] = [nodes[i].op_type for i in consumers.get(node.output[0], [])]
            softmaxes.append(item)
        if node.op_type == "Div":
            item = _node_dict(index, node, shapes)
            left = _upstream_ops(node.input[0], producers, nodes)
            right = _upstream_ops(node.input[1], producers, nodes)
            if "ReduceSum" in left or "ReduceSum" in right:
                item["left_upstream_ops"] = left
                item["right_upstream_ops"] = right
                grouped_divisions.append(item)
    return {
        "onnx": str(path),
        "ir_version": model.ir_version,
        "opsets": [{"domain": item.domain, "version": item.version} for item in model.opset_import],
        "inputs": [{"name": item.name, "shape": shapes.get(item.name)} for item in model.graph.input],
        "outputs": [{"name": item.name, "shape": shapes.get(item.name)} for item in model.graph.output],
        "shape_inference_error": shape_inference_error,
        "node_count": len(nodes),
        "relevant_nodes": relevant,
        "token_softmax_candidates": softmaxes,
        "grouped_normalization_division_candidates": grouped_divisions,
    }


def _har_audit(path: Path, patterns: list[str]) -> dict[str, Any]:
    try:
        from hailo_sdk_client import ClientRunner
    except Exception as exc:
        return {"status": "UNAVAILABLE", "error": f"{type(exc).__name__}: {exc}"}
    try:
        runner = ClientRunner(har=str(path))
        layers = runner.get_hn_dict().get("layers", {})
        compiled = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
        selected = {}
        for name, layer in layers.items():
            original_names = list(layer.get("original_names") or [])
            text = " ".join([name, str(layer.get("type", "")), *original_names])
            if not compiled or any(pattern.search(text) for pattern in compiled):
                selected[name] = {
                    "type": layer.get("type"),
                    "original_names": original_names,
                    "input_shapes": layer.get("input_shapes"),
                    "output_shapes": layer.get("output_shapes"),
                }
        return {"status": "PASS", "har": str(path), "layer_count": len(layers), "selected_layers": selected}
    except Exception as exc:
        return {"status": "FAIL", "har": str(path), "error": f"{type(exc).__name__}: {exc}"}


def _first_failure(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"(?:receiver_stage\d+/)?(?:ew_[\w]+\d+).*", text)
    return {"log": str(path), "first_matching_lines": matches[:8]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--har", type=Path, default=None)
    parser.add_argument("--failure-log", type=Path, default=None)
    parser.add_argument("--hailo-pattern", action="append", default=["softmax", "ew_", "reduce", "normalization"])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.onnx.is_file():
        parser.error(f"ONNX not found: {args.onnx}")
    if args.har is not None and not args.har.is_file():
        parser.error(f"HAR not found: {args.har}")
    if args.failure_log is not None and not args.failure_log.is_file():
        parser.error(f"failure log not found: {args.failure_log}")

    result = {
        "status": "PASS",
        "onnx": _onnx_audit(args.onnx),
        "har": _har_audit(args.har, args.hailo_pattern) if args.har else None,
        "reported_failure": _first_failure(args.failure_log),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "out": str(args.out),
        "onnx_node_count": result["onnx"]["node_count"],
        "softmax_candidates": len(result["onnx"]["token_softmax_candidates"]),
        "grouped_division_candidates": len(result["onnx"]["grouped_normalization_division_candidates"]),
        "har_status": None if result["har"] is None else result["har"]["status"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
