"""Real HEF compile from an already-quantized runner state."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from hailo_deploy.compile import num_contexts
from hailo_deploy.io.hash import sha256


def compile_quantized(
    runner,
    *,
    hef_path: Path,
    compiled_har: Path | None = None,
    auto_alls: Path | None = None,
) -> dict[str, Any]:
    """Compile the current quantized runner; optionally save HAR and auto.alls."""
    hef_path.write_bytes(runner.compile())
    result: dict[str, Any] = {
        "hef": {
            "path": str(hef_path),
            "sha256": sha256(hef_path),
            "bytes": hef_path.stat().st_size,
        },
        "num_contexts": num_contexts(runner),
    }
    if compiled_har is not None:
        runner.save_har(str(compiled_har))
        result["compiled_har"] = {
            "path": str(compiled_har),
            "sha256": sha256(compiled_har),
            "bytes": compiled_har.stat().st_size,
        }
    if auto_alls is not None:
        try:
            runner.save_autogen_allocation_script(str(auto_alls))
            result["auto_alls"] = {
                "path": str(auto_alls),
                "sha256": sha256(auto_alls),
                "bytes": auto_alls.stat().st_size,
            }
        except Exception as exc:
            result["auto_alls"] = {"error": f"{type(exc).__name__}: {exc}"}
    return result
