"""Immutable training-side handoff validation for full-module PTQ."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hailo_deploy.io.hash import sha256

FORMAL_CALIBRATION_COUNT = 1024
FORMAL_GOLDEN_COUNT = 8


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError("none of the required handoff paths exist: " + ", ".join(map(str, paths)))


def validate_handoff(
    root: Path, module: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path, Path, list[int], list[int]]:
    identity = load_json(root / "identity.json")
    checkpoint_sha = identity.get("checkpoint", {}).get("sha256")
    if not isinstance(checkpoint_sha, str) or len(checkpoint_sha) != 64:
        raise ValueError("handoff has no valid checkpoint SHA-256")
    model = identity.get("model", {})
    if model.get("gfa_mode") != "group_softmax":
        raise ValueError("handoff is not group_softmax")
    if model.get("split") != "mid" or model.get("bottleneck") != "rtformer_fused":
        raise ValueError("handoff does not match the supported fused mid split")
    if model.get("activation") not in {"gdn", "convbn"}:
        raise ValueError(f"unsupported bottleneck activation: {model.get('activation')!r}")
    if identity.get("splits", {}).get("intersection_empty") is not True:
        raise ValueError("handoff does not prove calibration/test disjointness")
    report_path = root / "handoff_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    report = load_json(report_path)
    if report.get("status") != "PASS":
        raise ValueError(f"handoff_report is not PASS: {report.get('status')!r}")
    manifest_path = first_existing([root / "onnx" / "manifest.json", root / "onnx" / "onnx_manifest.json"])
    manifest = load_json(manifest_path)
    modules = {item["name"]: item for item in manifest.get("modules", [])}
    if module not in modules:
        raise ValueError(f"module {module!r} absent from ONNX manifest")
    item = modules[module]
    onnx = root / "onnx" / f"{module}.onnx"
    if not onnx.is_file() or sha256(onnx) != item.get("sha256"):
        raise ValueError(f"ONNX identity mismatch for {module}")
    calibration_base = first_existing([root / "calib", root / "calibration"])
    calib_root = calibration_base / module
    meta = load_json(calib_root / "meta.json")
    if meta.get("checkpoint", {}).get("sha256") != checkpoint_sha:
        raise ValueError(f"calibration checkpoint identity mismatch for {module}")
    if meta.get("onnx", {}).get("sha256") != item.get("sha256"):
        raise ValueError(f"calibration ONNX identity mismatch for {module}")
    calibration_indices = meta.get("dataset", {}).get("val_indices")
    if not isinstance(calibration_indices, list) or len(calibration_indices) != FORMAL_CALIBRATION_COUNT:
        raise ValueError(f"calibration sample count is not {FORMAL_CALIBRATION_COUNT} for {module}")
    if len(set(calibration_indices)) != len(calibration_indices):
        raise ValueError(f"calibration indices contain duplicates for {module}")
    if int(meta.get("sample_count", 0)) != len(calibration_indices):
        raise ValueError(f"calibration metadata count mismatch for {module}")
    if not (calib_root / "load_calib.py").is_file():
        raise FileNotFoundError(calib_root / "load_calib.py")
    golden_path = first_existing(
        [root / "evaluation" / "golden_manifest.json", root / "evaluation" / "manifest.json"]
    )
    golden = load_json(golden_path)
    if golden.get("checkpoint_sha256") != checkpoint_sha:
        raise ValueError("golden checkpoint identity mismatch")
    golden_indices = golden.get("golden_indices")
    if not isinstance(golden_indices, list) or len(golden_indices) != FORMAL_GOLDEN_COUNT:
        raise ValueError(f"golden held-out sample count is not {FORMAL_GOLDEN_COUNT}")
    if len(set(golden_indices)) != len(golden_indices):
        raise ValueError("golden held-out indices contain duplicates")
    if set(calibration_indices) & set(golden_indices):
        raise ValueError("calibration overlaps held-out golden samples")
    return identity, item, meta, onnx, calibration_base, calibration_indices, golden_indices
