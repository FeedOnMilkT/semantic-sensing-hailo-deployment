"""Freeze semantic names and actual HEF VStream quantization contracts."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from hailo_deploy.io.hash import sha256


def _enum(value: Any) -> str:
    name = getattr(value, "name", None)
    return str(name if name is not None else value)


def _artifact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size}


def extract_hef_io_contract(
    quantized_har: Path, hef: Path, entry: dict[str, Any]
) -> dict[str, Any]:
    """Map HEF streams to ONNX semantic names and record real scale/zero-point.

    Hailo output stream names are normally the producer layer, not the ONNX
    output name.  The quantized HAR's HN output layer is therefore the source
    of the exact mapping; shape fallback is deliberately not accepted.
    """
    from hailo_platform import HEF
    from hailo_sdk_client import ClientRunner

    quantized_har = quantized_har.resolve()
    hef = hef.resolve()
    runner = ClientRunner(har=str(quantized_har))
    layers = runner.get_hn_dict().get("layers", {})

    semantic_to_stream: dict[str, dict[str, str]] = {"inputs": {}, "outputs": {}}
    precision: dict[str, str | None] = {}
    for layer_name, layer in layers.items():
        layer_type = str(layer.get("type", "")).lower()
        originals = list(layer.get("original_names") or [])
        if "input" in layer_type:
            if len(originals) != 1:
                raise ValueError(f"{layer_name}: input needs one original name, got {originals}")
            semantic_to_stream["inputs"][originals[0]] = layer_name
            precision[originals[0]] = layer.get("quantization_params", {}).get("precision_mode")
        elif "output" in layer_type:
            producers = list(layer.get("input") or [])
            if len(originals) != 1 or len(producers) != 1:
                raise ValueError(
                    f"{layer_name}: output mapping needs one original and producer; "
                    f"got originals={originals} producers={producers}"
                )
            semantic_to_stream["outputs"][originals[0]] = producers[0]
            precision[originals[0]] = layer.get("quantization_params", {}).get("precision_mode")

    expected_inputs = set(entry.get("input_names", []))
    expected_outputs = set(entry.get("output_names", []))
    if set(semantic_to_stream["inputs"]) != expected_inputs:
        raise ValueError(
            f"{entry.get('id')}: HAR inputs {sorted(semantic_to_stream['inputs'])} "
            f"!= ONNX {sorted(expected_inputs)}"
        )
    if set(semantic_to_stream["outputs"]) != expected_outputs:
        raise ValueError(
            f"{entry.get('id')}: HAR outputs {sorted(semantic_to_stream['outputs'])} "
            f"!= ONNX {sorted(expected_outputs)}"
        )

    model = HEF(str(hef))
    raw_infos = {
        "inputs": {info.name: info for info in model.get_input_vstream_infos()},
        "outputs": {info.name: info for info in model.get_output_vstream_infos()},
    }
    contract: dict[str, dict[str, Any]] = {"inputs": {}, "outputs": {}}
    for direction in ("inputs", "outputs"):
        for semantic_name, stream_name in semantic_to_stream[direction].items():
            info = raw_infos[direction].get(stream_name)
            if info is None:
                raise ValueError(
                    f"{entry.get('id')}: HEF lacks {direction[:-1]} stream {stream_name!r} "
                    f"for {semantic_name!r}; has {sorted(raw_infos[direction])}"
                )
            quant = info.quant_info
            contract[direction][semantic_name] = {
                "hef_stream_name": stream_name,
                "shape_hwc": [int(value) for value in info.shape],
                "format_type": _enum(info.format.type),
                "format_order": _enum(info.format.order),
                "precision_mode": precision[semantic_name],
                "quantization": {
                    "qp_scale": float(quant.qp_scale),
                    "qp_zp": float(quant.qp_zp),
                    "limvals_min": float(quant.limvals_min),
                    "limvals_max": float(quant.limvals_max),
                },
            }

    return {
        "schema": "semantic_sensing.hef_io_quantization_contract.v1",
        "segment": entry.get("id"),
        "quantized_har": _artifact(quantized_har),
        "hef": _artifact(hef),
        "host_vstream_format": "FLOAT32",
        "host_conversion_rule": (
            "HailoRT dequantizes HEF outputs and quantizes HEF inputs with each stream's "
            "recorded qp_scale/qp_zp. Raw UINT8 handoff is forbidden by this contract."
        ),
        "inputs": contract["inputs"],
        "outputs": contract["outputs"],
    }


def codec_boundary_contracts(
    manifest: dict[str, Any], contracts: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Bind both CPU codec boundaries to exact producer/consumer contracts."""
    required = {"analysis", "hyper", "synthesis_tail"}
    if set(contracts) != required:
        raise ValueError(f"needs contracts {sorted(required)}, got {sorted(contracts)}")
    analysis, hyper, tail = (
        contracts["analysis"], contracts["hyper"], contracts["synthesis_tail"]
    )
    bindings = [
        {
            "id": "z",
            "producer": analysis["outputs"]["latent_z"],
            "consumer": hyper["inputs"]["latent_z_hat"],
            "cpu_operation": "entropy_bottleneck real rANS compress/decompress",
        },
        {
            "id": "y",
            "producer": analysis["outputs"]["latent_y"],
            "parameters": {
                "means": hyper["outputs"]["means"],
                "scales": hyper["outputs"]["scales"],
            },
            "consumer": tail["inputs"]["latent_y_hat"],
            "cpu_operation": "gaussian_conditional real rANS compress/decompress",
        },
    ]
    manifest_ids = {item.get("id") for item in manifest.get("cpu_codec_boundaries", [])}
    if manifest_ids != {"z", "y"}:
        raise ValueError(f"unexpected ONNX codec boundaries: {sorted(manifest_ids)}")
    for binding in bindings:
        binding.update(
            {
                "raw_integer_boundary": False,
                "requires_equal_qp_across_hefs": False,
                "alignment_rule": (
                    "The exact upstream HEF is executed with FLOAT32 VStreams, real rANS "
                    "reconstructs a float tensor, and the exact downstream HEF applies its own "
                    "frozen input quantization. The two qp pairs must be recorded, not forced equal."
                ),
            }
        )
    return bindings
