"""Pure NumPy metrics for SDK-code versus HailoRT compiler probes."""
from __future__ import annotations

from typing import Any

import numpy as np


_CODE_DTYPES = {
    "UINT8": np.dtype(np.uint8),
    "UINT16": np.dtype(np.uint16),
}


def sdk_codes_to_integer(value: np.ndarray, format_type: str) -> np.ndarray:
    """Validate SDK internal float32 code values and return their integer form."""

    if format_type not in _CODE_DTYPES:
        raise ValueError(f"unsupported probe code format: {format_type}")
    codes = np.asarray(value)
    if codes.dtype != np.float32:
        raise ValueError(f"SDK internal codes must be stored in float32, got {codes.dtype}")
    if not np.all(np.isfinite(codes)):
        raise ValueError("SDK internal codes contain non-finite values")
    rounded = np.rint(codes)
    if not np.array_equal(codes, rounded):
        count = int(np.count_nonzero(codes != rounded))
        raise ValueError(f"SDK internal tensor has {count} non-integer float32 values")
    dtype = _CODE_DTYPES[format_type]
    limit = np.iinfo(dtype)
    if float(rounded.min()) < limit.min or float(rounded.max()) > limit.max:
        raise ValueError(
            f"SDK internal codes exceed {format_type}: "
            f"[{float(rounded.min())}, {float(rounded.max())}]"
        )
    return np.ascontiguousarray(rounded, dtype=dtype)


def dequantize_sdk_codes(
    value: np.ndarray, format_type: str, quantization: dict[str, Any]
) -> np.ndarray:
    """Apply the probe HEF vstream scale/zero-point to SDK integer codes."""

    codes = sdk_codes_to_integer(value, format_type)
    scale = float(quantization["qp_scale"])
    zero_point = float(quantization["qp_zp"])
    if not np.isfinite(scale) or scale <= 0.0 or not np.isfinite(zero_point):
        raise ValueError(f"invalid HEF quantization parameters: {quantization}")
    return np.ascontiguousarray(
        (codes.astype(np.float32) - zero_point) * scale,
        dtype=np.float32,
    )


def code_comparison(board_raw: np.ndarray, sdk_codes: np.ndarray) -> dict[str, Any]:
    """Compare board raw codes to SDK codes, treating SDK as the reference."""

    got = np.asarray(board_raw)
    ref = np.asarray(sdk_codes)
    if got.shape != ref.shape:
        raise ValueError(f"code shape mismatch: {got.shape} != {ref.shape}")
    delta = got.astype(np.int64) - ref.astype(np.int64)
    delta64 = delta.astype(np.float64)
    reference64 = ref.astype(np.float64)
    count = int(delta.size)
    mismatch = int(np.count_nonzero(delta))
    sse = float(np.sum(delta64 * delta64))
    reference_energy = float(np.sum(reference64 * reference64))
    absolute_error_sum = float(np.sum(np.abs(delta64)))
    return {
        "element_count": count,
        "mismatch_count": mismatch,
        "agreement": float(1.0 - mismatch / count),
        "code_nmse": None if reference_energy == 0.0 else float(sse / reference_energy),
        "code_mae": float(absolute_error_sum / count),
        "max_abs_code_diff": int(np.max(np.abs(delta))),
        "squared_error_sum": sse,
        "reference_squared_sum": reference_energy,
        "absolute_error_sum": absolute_error_sum,
    }


def summarize_code_comparisons(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty code comparison")
    count = int(sum(int(row["element_count"]) for row in rows))
    mismatch = int(sum(int(row["mismatch_count"]) for row in rows))
    sse = float(sum(float(row["squared_error_sum"]) for row in rows))
    energy = float(sum(float(row["reference_squared_sum"]) for row in rows))
    absolute_error = float(sum(float(row["absolute_error_sum"]) for row in rows))
    return {
        "element_count": count,
        "mismatch_count": mismatch,
        "agreement": float(1.0 - mismatch / count),
        "code_nmse": None if energy == 0.0 else float(sse / energy),
        "code_mae": float(absolute_error / count),
        "max_abs_code_diff": int(max(int(row["max_abs_code_diff"]) for row in rows)),
    }
