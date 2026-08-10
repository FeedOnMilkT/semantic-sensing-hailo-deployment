#!/usr/bin/env python3
"""Deploy and benchmark independent HEFs on a Pi Hailo-8 board.

The target runner loads one HEF at a time, warms it, and times resident
``InferVStreams.infer`` calls on zero tensors matching the HEF's vstream
contract.  SSH command startup and file upload are recorded separately and
never included in the reported target inference latency.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from board_eval.deploy import common_options, stream_upload
from hailo_deploy.io.hash import sha256


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def label_for(hef: Path) -> str:
    return hef.parent.name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hef", type=Path, action="append", required=True,
                        help="repeat once per HEF")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--remote-root", default=".local/share/semantic-sensing-board/latency")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--ssh-target", required=True)
    parser.add_argument("--identity-file", type=Path)
    parser.add_argument("--proxy-command")
    parser.add_argument("--known-hosts", type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite: {args.out}")
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be >=0 and iterations must be >0")
    hefs = [path.resolve() for path in args.hef]
    missing = [str(path) for path in hefs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing HEF(s): {missing}")
    labels = [label_for(path) for path in hefs]
    if len(set(labels)) != len(labels):
        raise ValueError(f"HEF parent names must be unique: {labels}")
    run_tag = args.run_tag or datetime.now(timezone.utc).strftime("hef_latency_%Y%m%dT%H%M%SZ")
    remote_dir = f"{args.remote_root.rstrip('/')}/{run_tag}"
    opts = common_options(args)
    runner = Path(__file__).with_name("generic_benchmark_runner.py")
    run(["ssh", *opts, args.ssh_target, f"mkdir -p {shlex.quote(remote_dir)}"])
    stream_upload(runner, f"{remote_dir}/{runner.name}", opts, args.ssh_target)

    rows: list[dict[str, object]] = []
    for label, hef in zip(labels, hefs):
        remote_hef = f"{remote_dir}/{label}.hef"
        row: dict[str, object] = {
            "module": label,
            "local_hef": {"path": str(hef), "sha256": sha256(hef), "bytes": hef.stat().st_size},
            "remote_hef": remote_hef,
        }
        try:
            stream_upload(hef, remote_hef, opts, args.ssh_target)
            remote_sha = run(["ssh", *opts, args.ssh_target, f"sha256sum {shlex.quote(remote_hef)}"]).stdout.split()[0]
            if remote_sha != row["local_hef"]["sha256"]:
                raise ValueError(f"remote HEF hash mismatch: {remote_sha}")
            start = time.perf_counter_ns()
            completed = run([
                "ssh", *opts, args.ssh_target,
                f"python3 {shlex.quote(remote_dir + '/' + runner.name)} --hef {shlex.quote(remote_hef)} "
                f"--warmup {args.warmup} --iterations {args.iterations}",
            ])
            row["ssh_command_ms"] = (time.perf_counter_ns() - start) / 1e6
            row["benchmark"] = json.loads(completed.stdout)
            row["status"] = row["benchmark"].get("status")
            if completed.stderr:
                row["remote_stderr"] = completed.stderr
        except Exception as exc:
            row.update({"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"})
        rows.append(row)

    report = {
        "kind": "pi_hailort_independent_hef_latency",
        "status": "PASS" if all(row["status"] == "PASS" for row in rows) else "PARTIAL_FAIL",
        "scope": [
            "Each HEF is timed independently with a resident HailoRT InferVStreams pipeline.",
            "infer_wall_ms excludes SSH command startup, upload, model load, CPU rANS, and inter-module transfers.",
            "Zero inputs are used only because arithmetic values do not affect fixed-shape Hailo latency.",
        ],
        "board": {"ssh_target": args.ssh_target, "remote_dir": remote_dir},
        "warmup": args.warmup,
        "iterations": args.iterations,
        "modules": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": report["status"], "out": str(args.out), "modules": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
