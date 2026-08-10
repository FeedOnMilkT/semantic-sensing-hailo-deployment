"""Tensor drift metrics shared by PTQ gates and chain evaluation."""
from __future__ import annotations

from typing import Any

import numpy as np


def metric(reference: np.ndarray, actual: np.ndarray) -> dict[str, float | None]:
    ref = np.asarray(reference, dtype=np.float64)
    got = np.asarray(actual, dtype=np.float64)
    delta = got - ref
    denom = float(np.linalg.norm(ref.ravel()) * np.linalg.norm(got.ravel()))
    return {
        "cosine": None if denom == 0.0 else float(np.dot(ref.ravel(), got.ravel()) / denom),
        "mae": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "max_abs": float(np.max(np.abs(delta))),
    }


def summarize(items: list[dict[str, float | None]]) -> dict[str, dict[str, float | int | None]]:
    result: dict[str, dict[str, float | int | None]] = {}
    for name in ("cosine", "mae", "rmse", "max_abs"):
        values = np.asarray(
            [item[name] for item in items if item.get(name) is not None], dtype=np.float64
        )
        result[name] = {
            "count": int(values.size),
            "mean": float(values.mean()) if values.size else None,
            "min": float(values.min()) if values.size else None,
            "p50": float(np.percentile(values, 50)) if values.size else None,
            "p95": float(np.percentile(values, 95)) if values.size else None,
            "max": float(values.max()) if values.size else None,
        }
    return result


def accumulate_abs_error(
    accumulators: dict[str, dict[str, Any]],
    name: str,
    reference: np.ndarray,
    actual: np.ndarray,
) -> None:
    """Accumulate max/mean absolute error vs an ORT (or other FP) reference."""
    delta = np.abs(reference.astype(np.float32) - actual.astype(np.float32))
    item = accumulators.setdefault(
        name,
        {
            "max_abs_diff": 0.0,
            "absolute_error_sum": 0.0,
            "reference_abs_sum": 0.0,
            "element_count": 0,
        },
    )
    item["max_abs_diff"] = max(item["max_abs_diff"], float(delta.max()))
    item["absolute_error_sum"] += float(delta.astype(np.float64).sum())
    item["reference_abs_sum"] += float(np.abs(reference).astype(np.float64).sum())
    item["element_count"] += int(delta.size)


def finalize_abs_error(accumulators: dict[str, dict[str, Any]]) -> dict[str, dict[str, float | None]]:
    per_output: dict[str, dict[str, float | None]] = {}
    for name, item in accumulators.items():
        mae = item["absolute_error_sum"] / item["element_count"]
        ref_mae = item["reference_abs_sum"] / item["element_count"]
        per_output[name] = {
            "max_abs_diff": item["max_abs_diff"],
            "mean_abs_diff": mae,
            "reference_abs_mean": ref_mae,
            "relative_mean_abs_error": mae / ref_mae if ref_mae else None,
        }
    return per_output
