#!/usr/bin/env python3
"""Formal all-module handoff pipeline: parse → per-module PTQ → HN chain eval."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
PIPELINES = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from hailo_deploy.artifacts.layout import refuse_existing, write_json
from hailo_deploy.config import FULLMODULE_NAMES

SUITE_PYTHON = Path("/local/workspace/hailo_virtualenv/bin/python")


def _python() -> str:
    if SUITE_PYTHON.is_file():
        return str(SUITE_PYTHON)
    return sys.executable


def run(command: list[str], log: Path, environment: dict[str, str]) -> None:
    with log.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n")
        handle.flush()
        subprocess.run(
            command, check=True, stdout=handle, stderr=subprocess.STDOUT, env=environment
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--cuda-visible-devices", default="1")
    parser.add_argument(
        "--modules",
        nargs="*",
        default=list(FULLMODULE_NAMES),
        choices=list(FULLMODULE_NAMES),
    )
    parser.add_argument("--opt-level", type=int, default=2, choices=range(5))
    parser.add_argument("--skip-chain-eval", action="store_true")
    args = parser.parse_args(argv)

    refuse_existing(args.out_root)
    args.out_root.mkdir(parents=True)
    logs, har_root, ptq_root = args.out_root / "logs", args.out_root / "har", args.out_root / "ptq"
    logs.mkdir()
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    environment["PYTHONPATH"] = ":".join(
        filter(
            None,
            [
                "/local/workspace/board_eval_ort_cuda_site",
                "/workspace/semantic-sensing",
                "/workspace/.board-eval-python",
                str(ROOT),
                environment.get("PYTHONPATH", ""),
            ],
        )
    )
    status: dict = {
        "kind": "formal_handoff_all_module_pipeline",
        "schema": "fullmodule_chain.v1",
        "status": "RUNNING",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "handoff_root": str(args.handoff_root),
        "out_root": str(args.out_root),
        "cuda_visible_devices": args.cuda_visible_devices,
        "modules": {},
        "error": None,
    }
    state_path = args.out_root / "pipeline_status.json"
    write_json(state_path, status)
    py = _python()
    try:
        parse_script = SCRIPTS / "run_handoff_parse.py"
        run(
            [
                py,
                str(parse_script),
                "--handoff-root",
                str(args.handoff_root),
                "--out-dir",
                str(har_root),
            ],
            logs / "parse_native.log",
            environment,
        )
        for name in args.modules:
            destination = ptq_root / name
            command = [
                py,
                str(PIPELINES / "module_ptq.py"),
                "fullmodule",
                "--handoff-root",
                str(args.handoff_root),
                "--module",
                name,
                "--parse-har",
                str(har_root / f"{name}_hailo_model.har"),
                "--out-dir",
                str(destination),
                "--n-calib",
                "1024",
                "--opt-level",
                str(args.opt_level),
            ]
            run(command, logs / f"ptq_{name}.log", environment)
            status["modules"][name] = {
                "status": "PASS",
                "result": str(destination / "result.json"),
            }
            write_json(state_path, status)
        if not args.skip_chain_eval:
            semantic_out = args.out_root / "evaluation" / "sdk_quantized_chain_full.json"
            semantic_out.parent.mkdir(parents=True, exist_ok=True)
            run(
                [
                    py,
                    str(SCRIPTS / "evaluate_fullmodule_hn_chain.py"),
                    "--handoff-root",
                    str(args.handoff_root),
                    "--ptq-root",
                    str(ptq_root),
                    "--out",
                    str(semantic_out),
                ],
                logs / "semantic_full.log",
                environment,
            )
            status["semantic_evaluation"] = str(semantic_out)
        status.update(
            {
                "status": "PASS",
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        write_json(state_path, status)
        print(json.dumps({"status": "PASS", "out": str(args.out_root)}, indent=2))
        return 0
    except Exception as error:
        status.update(
            {
                "status": "FAIL",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "error": f"{type(error).__name__}: {error}\n{traceback.format_exc()}",
            }
        )
        write_json(state_path, status)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
