"""Deployment gate helpers: parse / NATIVE / optimize / quantized check / compile."""

from hailo_deploy import compile as compile_mod
from hailo_deploy import emulate as native_mod
from hailo_deploy import parse as parse_mod

from .optimize import build_model_script, optimize_with_calib
from .quantized_check import heldout_quantized_check, ort_quantized_check
from .compile_real import compile_quantized

# Thin aliases so callers can ``from hailo_deploy.gates import parse, native``.
parse = parse_mod
native = native_mod
compile_probe = compile_mod

__all__ = [
    "parse",
    "native",
    "compile_probe",
    "build_model_script",
    "optimize_with_calib",
    "heldout_quantized_check",
    "ort_quantized_check",
    "compile_quantized",
]
