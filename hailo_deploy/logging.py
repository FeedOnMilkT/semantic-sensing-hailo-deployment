"""Run Hailo entry points with SDK logs isolated per run.

DFC Python loggers honour HAILO_SDK_LOG_DIR; the core allocator also writes
relative to the process working directory. This module sets both before the
child imports the Hailo SDK, then consolidates a minute-stamped log bundle.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from hailo_deploy.artifacts.sdk_logs import DEFAULT_LOG_ROOT, consolidate


def _timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M")


def _unique_run_dir(log_root: Path, stem: str) -> tuple[str, Path]:
    run_id = f"{stem}_{_timestamp()}"
    candidate = log_root / "runs" / run_id
    duplicate = 1
    while candidate.exists():
        candidate = log_root / "runs" / f"{run_id}_{duplicate:02d}"
        duplicate += 1
    candidate.mkdir(parents=True)
    return candidate.name, candidate


def _resolve_existing_paths(argv: list[str], original_cwd: Path) -> list[str]:
    """Keep common relative input paths valid after moving the child CWD."""
    resolved: list[str] = []
    for value in argv:
        path = Path(value)
        candidate = path if path.is_absolute() else original_cwd / path
        resolved.append(str(candidate.resolve()) if candidate.exists() else value)
    return resolved


def reexec_if_unlogged(script: Path, script_args: list[str]) -> None:
    """Re-exec a direct entry-point call through this launcher exactly once."""
    if os.environ.get("_HAILO_SDK_LOGGED") == "1":
        return
    wrapper = Path(__file__).resolve().parent.parent / "scripts" / "run_hailo_logged.py"
    if not wrapper.is_file():
        # Fallback: invoke this module as a script.
        command = [sys.executable, "-m", "hailo_deploy.logging", str(script.resolve()), *script_args]
    else:
        command = [sys.executable, str(wrapper), str(script.resolve()), *script_args]
    os.execvpe(sys.executable, command, os.environ.copy())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a Python Hailo entry point with SDK logs kept out of the project root."
    )
    parser.add_argument("script", type=Path, help="Python deployment entry point")
    parser.add_argument(
        "script_args",
        nargs=argparse.REMAINDER,
        help="arguments forwarded unchanged to the entry point",
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=DEFAULT_LOG_ROOT,
        help="SDK log root (default: persistent deployment artifact store)",
    )
    args = parser.parse_args(argv)

    original_cwd = Path.cwd()
    script = args.script if args.script.is_absolute() else original_cwd / args.script
    script = script.resolve()
    if not script.is_file():
        parser.error(f"entry point does not exist: {script}")
    if script.suffix != ".py":
        parser.error(f"entry point must be a .py file: {script}")

    log_root = args.log_root.resolve()
    run_id, run_dir = _unique_run_dir(log_root, script.stem)
    forwarded = _resolve_existing_paths(args.script_args, original_cwd)
    command = [sys.executable, str(script), *forwarded]
    env = os.environ.copy()
    env["HAILO_SDK_LOG_DIR"] = str(run_dir)
    env["_HAILO_SDK_LOGGED"] = "1"

    command_log = run_dir / "command.log"
    print(f"Hailo SDK log directory: {run_dir}")
    print("Command:", " ".join(command))
    with command_log.open("w", encoding="utf-8", buffering=1) as output:
        output.write(f"# cwd: {run_dir}\n")
        output.write(f"# command: {' '.join(command)}\n\n")
        process = subprocess.Popen(
            command,
            cwd=run_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            output.write(line)
        status = process.wait()

    bundle = consolidate(run_dir, log_root, run_id)
    print(f"Consolidated SDK log: {bundle}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
