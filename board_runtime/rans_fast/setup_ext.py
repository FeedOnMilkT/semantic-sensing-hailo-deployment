"""Build ``_rans_fast`` into ``src/codec/rans_fast/``.

::

    cd /workspace/semantic-sensing
    PYTHONPATH=/workspace/.python-pkgs:$PYTHONPATH \\
      python src/codec/rans_fast/setup_ext.py build_ext --inplace
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from setuptools import Extension, setup

HERE = Path(__file__).resolve().parent
CPP = HERE / "cpp"

_EXTRA = Path("/workspace/.python-pkgs")
if _EXTRA.is_dir() and str(_EXTRA) not in sys.path:
    sys.path.insert(0, str(_EXTRA))

import pybind11  # noqa: E402

# Build from the package directory so --inplace drops .so next to __init__.py
os.chdir(HERE)

ext = Extension(
    "_rans_fast",
    sources=["cpp/rans_fast.cpp"],
    include_dirs=["cpp", pybind11.get_include()],
    language="c++",
    extra_compile_args=["-O3", "-std=c++14", "-fvisibility=hidden"],
)

setup(
    name="semsense-rans-fast",
    version="0.1.0",
    ext_modules=[ext],
    zip_safe=False,
)
