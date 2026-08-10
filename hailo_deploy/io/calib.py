"""Calibration handoff loaders and HN input mapping.

Contract: each module calibration directory provides ``load_calib.py`` (callable
``load_calib(limit=...)`` returning ``{onnx_name: NHWC float32}``) plus
``meta.json``. Pipelines must not invent an alternate activation source.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_loader(path: Path):
    """Import a handoff ``load_calib.py`` without polluting ``sys.modules`` names."""
    spec = importlib.util.spec_from_file_location(f"calib_{path.parent.name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import calibration loader {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_calib_handoff(root: Path, limit: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load ``limit`` NHWC calibration arrays + metadata from a module calib root."""
    loader_path, meta_path = root / "load_calib.py", root / "meta.json"
    if not loader_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(f"incomplete calibration handoff: {root}")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    sample_count = int(metadata.get("sample_count", 0))
    if limit > sample_count:
        raise ValueError(f"requested {limit} samples, only {sample_count} available")
    arrays = load_loader(loader_path).load_calib(limit=limit)
    return arrays, metadata


def map_hn_inputs(
    runner, arrays: dict[str, np.ndarray]
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Map ONNX-named NHWC arrays onto Hailo HN input layer names.

    Requires a unique ``original_names`` hit per ONNX input and full coverage of
    every HN input layer. Rejects non-float32 / non-4D / non-finite tensors.
    """
    layers = runner.get_hn_dict().get("layers", {})
    hn_inputs = {
        name: list(layer.get("original_names") or [])
        for name, layer in layers.items()
        if "input" in str(layer.get("type", "")).lower()
    }
    feed: dict[str, np.ndarray] = {}
    mapping: dict[str, str] = {}
    for onnx_name, value in arrays.items():
        choices = [name for name, originals in hn_inputs.items() if onnx_name in originals]
        if len(choices) != 1:
            raise ValueError(f"cannot map ONNX input {onnx_name!r} to HN inputs {hn_inputs}")
        hn_name = choices[0]
        if value.dtype != np.float32 or value.ndim != 4 or not np.isfinite(value).all():
            raise ValueError(f"invalid calibration for {onnx_name}: {value.shape}/{value.dtype}")
        feed[hn_name] = np.ascontiguousarray(value)
        mapping[onnx_name] = hn_name
    if set(feed) != set(hn_inputs):
        raise ValueError(f"calibration coverage {sorted(feed)} != HN inputs {sorted(hn_inputs)}")
    return feed, mapping
