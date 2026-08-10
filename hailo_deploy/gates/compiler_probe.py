"""Compile-only side-output probes for an immutable quantized HAR.

The helpers in this module deliberately do not expose parse or optimize.  They
are used to prove that a diagnostic ``output_layer`` command leaves the source
HN and all optimized numerical parameters unchanged before compilation.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import numpy as np


def _update_array(digest: "hashlib._Hash", value: Any) -> None:
    array = np.ascontiguousarray(np.asarray(value))
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("utf-8"))
    digest.update(b"\0")
    digest.update(array.view(np.uint8))


def optimized_params_sha256(runner: Any) -> str:
    """Hash every Hailo-optimized numerical parameter deterministically.

    DFC 3.33 does not expose these post-optimization parameters through the
    public ``ClientRunner.get_params`` method (that method returns native
    parameters).  Reading the backend's existing ``ModelParams`` object is the
    smallest non-mutating way to guard this compile-only experiment.
    """

    params = runner._sdk_backend.get_params_hailo_optimized()  # noqa: SLF001
    mapping = getattr(params, "_params", None)  # noqa: SLF001
    if not isinstance(mapping, Mapping) or not mapping:
        raise ValueError("quantized HAR has no Hailo-optimized parameters")
    digest = hashlib.sha256()
    for key in sorted(mapping):
        digest.update(str(key).encode("utf-8"))
        digest.update(b"\0")
        _update_array(digest, mapping[key])
        digest.update(b"\0")
    return digest.hexdigest()


def hn_sha256(runner: Any) -> str:
    payload = json.dumps(
        runner.get_hn_dict(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parse_allocation_script(runner: Any, script: str) -> list[Any]:
    """Parse allocation-only commands against the runner's actual HN."""

    from hailo_sdk_client.sdk_backend.script_parser.model_script_parser import (
        ModelScriptParser,
    )
    from hailo_sdk_common.model_script_parser.model_script_modes import (
        ModelScriptModes,
    )

    parser = ModelScriptParser(
        runner.get_hn_model(), ModelScriptModes.ALLOCATION_ONLY_MODE
    )
    return list(parser.parse_script(script))


def validate_output_layer_command(runner: Any, script: str) -> list[str]:
    """Parse one output-layer allocation command against the actual HN."""

    commands = parse_allocation_script(runner, script)
    if len(commands) != 1 or "output_layer" not in str(commands[0]):
        raise ValueError(f"expected exactly one output_layer command, got {commands}")
    return [str(command) for command in commands]


def load_compile_only_script(runner: Any, script: str) -> dict[str, Any]:
    """Load allocation commands and prove numerical/HN state did not change."""

    state_before = str(runner.state)
    if state_before != "quantized_model":
        raise ValueError(f"probe source must be quantized_model, got {state_before}")
    commands = parse_allocation_script(runner, script)
    if not commands:
        raise ValueError("compile-only script parsed no allocation commands")
    parsed = [str(command) for command in commands]
    params_before = optimized_params_sha256(runner)
    hn_before = hn_sha256(runner)
    runner.load_model_script(script)
    params_after = optimized_params_sha256(runner)
    hn_after = hn_sha256(runner)
    state_after = str(runner.state)
    if state_after != state_before:
        raise ValueError(f"runner state changed: {state_before} -> {state_after}")
    if params_after != params_before:
        raise ValueError("output_layer command changed optimized numerical parameters")
    if hn_after != hn_before:
        raise ValueError("output_layer command changed the precompile HN")
    return {
        "state_before": state_before,
        "state_after": state_after,
        "optimized_params_sha256_before": params_before,
        "optimized_params_sha256_after": params_after,
        "hn_sha256_before": hn_before,
        "hn_sha256_after": hn_after,
        "parsed_commands": parsed,
        "numerical_state_unchanged": True,
    }


def load_compile_only_probe(runner: Any, script: str) -> dict[str, Any]:
    """Load one side-output command and prove numerical state did not change."""

    validate_output_layer_command(runner, script)
    return load_compile_only_script(runner, script)
