"""Real-calibration optimize helpers (Gate 5/6 precursor)."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def build_model_script(
    *,
    opt_level: int,
    n_calib: int,
    compression_level: int = 0,
    batch_size: int = 1,
    context_switch: str | None = None,
    alls_paths: list[Path] | None = None,
    extra_text: str | None = None,
) -> str:
    """Compose a deterministic model script for real PTQ."""
    script = (
        f"model_optimization_flavor(optimization_level={opt_level}, "
        f"compression_level={compression_level}, batch_size={batch_size})\n"
    )
    if context_switch is not None:
        script += f"context_switch_param(mode={context_switch})\n"
    script += (
        "model_optimization_config(calibration, "
        f"batch_size={batch_size}, calibset_size={n_calib})\n"
    )
    for path in alls_paths or []:
        text = path.read_text(encoding="utf-8")
        script += "\n" + text
        if not script.endswith("\n"):
            script += "\n"
    if extra_text:
        script += "\n" + extra_text
        if not script.endswith("\n"):
            script += "\n"
    return script


def optimize_with_calib(
    runner,
    calibration: Any,
    script: str,
    *,
    quantized_har: Path | None = None,
) -> None:
    """Load model script, optimize with real calibration, optionally save HAR."""
    runner.load_model_script(script)
    runner.optimize(calibration)
    if quantized_har is not None:
        runner.save_har(str(quantized_har))
