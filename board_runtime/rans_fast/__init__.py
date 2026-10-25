"""Import helper for the compiled ``_rans_fast`` extension."""
from __future__ import annotations

try:
    from . import _rans_fast as _ext
except ImportError:  # pragma: no cover - missing build
    _ext = None


def available() -> bool:
    return _ext is not None


def encode_with_indexes(symbols, indexes, cdfs, cdfs_sizes, offsets):
    if _ext is None:
        raise ImportError(
            "rans_fast extension not built; run "
            "python src/codec/rans_fast/setup_ext.py build_ext --inplace"
        )
    return _ext.encode_with_indexes(symbols, indexes, cdfs, cdfs_sizes, offsets)


def decode_with_indexes(encoded, indexes, cdfs, cdfs_sizes, offsets):
    if _ext is None:
        raise ImportError(
            "rans_fast extension not built; run "
            "python src/codec/rans_fast/setup_ext.py build_ext --inplace"
        )
    return _ext.decode_with_indexes(encoded, indexes, cdfs, cdfs_sizes, offsets)
