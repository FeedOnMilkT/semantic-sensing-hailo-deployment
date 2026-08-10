#!/usr/bin/env python3
"""Gate-8 board orchestration: freeze → deploy → evaluate → optional tail metrics.

Board tools require ``--ssh-target`` (and usually Tailscale ProxyCommand / identity).
This orchestrator forwards those flags; it does not invent a plain-ssh path.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BOARD = ROOT / "board_eval"
sys.path.insert(0, str(ROOT))

from hailo_deploy.artifacts.layout import refuse_existing, write_json


def run(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n")
        handle.flush()
        subprocess.run(
            command, check=True, stdout=handle, stderr=subprocess.STDOUT, env=os.environ.copy()
        )


def _ssh_flags(args: argparse.Namespace) -> list[str]:
    flags: list[str] = []
    if args.ssh_target:
        flags += ["--ssh-target", args.ssh_target]
    if args.identity_file:
        flags += ["--identity-file", str(args.identity_file)]
    if args.proxy_command:
        flags += ["--proxy-command", args.proxy_command]
    if args.known_hosts:
        flags += ["--known-hosts", str(args.known_hosts)]
    return flags


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None,
                        help="Existing frozen manifest (skip freeze when set without freeze args)")
    parser.add_argument("--deployment", type=Path, default=None)
    parser.add_argument("--skip-deploy", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--tail-metrics", action="store_true",
                        help="Also run board_eval/tail_metrics.py (training container)")
    parser.add_argument("--ssh-target", default=None,
                        help="Required for deploy/eval unless skipped")
    parser.add_argument("--identity-file", type=Path, default=None)
    parser.add_argument("--proxy-command", default=None)
    parser.add_argument("--known-hosts", type=Path, default=None)
    parser.add_argument(
        "passthrough",
        nargs=argparse.REMAINDER,
        help="Optional: -- <freeze_manifest args> [tail -- <tail_metrics args>]",
    )
    args = parser.parse_args(argv)

    freeze_argv: list[str] = []
    tail_argv: list[str] = []
    tokens = list(args.passthrough or [])
    if tokens and tokens[0] == "--":
        tokens = tokens[1:]
    if "tail" in tokens:
        idx = tokens.index("tail")
        freeze_argv, rest = tokens[:idx], tokens[idx + 1 :]
        if rest[:1] == ["--"]:
            rest = rest[1:]
        tail_argv = rest
    else:
        freeze_argv = tokens

    refuse_existing(args.out_root)
    args.out_root.mkdir(parents=True)
    logs = args.out_root / "logs"
    logs.mkdir()
    status = {
        "kind": "board_gate8_pipeline",
        "schema": "board_gate8.v1",
        "status": "RUNNING",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "out_root": str(args.out_root),
        "steps": {},
        "error": None,
    }
    state_path = args.out_root / "pipeline_status.json"
    write_json(state_path, status)
    py = sys.executable

    try:
        manifest = args.manifest
        if freeze_argv:
            if "--out" in freeze_argv:
                idx = freeze_argv.index("--out")
                manifest = Path(freeze_argv[idx + 1])
                cmd = [py, str(BOARD / "freeze_manifest.py"), *freeze_argv]
            else:
                manifest = args.out_root / "manifest.json"
                cmd = [
                    py, str(BOARD / "freeze_manifest.py"), *freeze_argv,
                    "--out", str(manifest),
                ]
            run(cmd, logs / "freeze_manifest.log")
            status["steps"]["freeze_manifest"] = {"status": "PASS", "manifest": str(manifest)}
            write_json(state_path, status)
        if manifest is None:
            raise ValueError("provide --manifest or freeze args after --")

        deployment = args.deployment
        if not args.skip_deploy:
            if not args.ssh_target:
                raise ValueError("deploy requires --ssh-target")
            deployment = args.out_root / "deployment.json"
            cmd = [
                py, str(BOARD / "deploy.py"),
                "--manifest", str(manifest),
                "--out", str(deployment),
                *_ssh_flags(args),
            ]
            run(cmd, logs / "deploy.log")
            status["steps"]["deploy"] = {"status": "PASS", "deployment": str(deployment)}
            write_json(state_path, status)
        if deployment is None and not args.skip_eval:
            raise ValueError("provide --deployment or run deploy")

        eval_out = None
        if not args.skip_eval:
            if not args.ssh_target:
                raise ValueError("evaluate requires --ssh-target")
            eval_out = args.out_root / "evaluation" / "board_eval.json"
            eval_out.parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                py, str(BOARD / "evaluate_receiver_stage45.py"),
                "--manifest", str(manifest),
                "--deployment", str(deployment),
                "--out", str(eval_out),
                *_ssh_flags(args),
            ]
            run(cmd, logs / "evaluate.log")
            status["steps"]["evaluate"] = {"status": "PASS", "result": str(eval_out)}
            write_json(state_path, status)

        if args.tail_metrics:
            if not tail_argv:
                raise ValueError("--tail-metrics requires: -- ... tail -- <tail_metrics args>")
            cmd = [py, str(BOARD / "tail_metrics.py"), *tail_argv]
            run(cmd, logs / "tail_metrics.log")
            status["steps"]["tail_metrics"] = {"status": "PASS"}

        status.update({
            "status": "PASS",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "manifest": str(manifest),
            "deployment": str(deployment) if deployment else None,
            "evaluation": str(eval_out) if eval_out else None,
        })
        write_json(state_path, status)
        print(json.dumps({"status": "PASS", "out": str(args.out_root)}, indent=2))
        return 0
    except Exception as exc:
        status.update({
            "status": "FAIL",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "error": f"{type(exc).__name__}: {exc}",
        })
        write_json(state_path, status)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
