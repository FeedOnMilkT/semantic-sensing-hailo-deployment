#!/usr/bin/env python3
"""Wait for a complete PASS handoff and run its formal all-module pipeline once."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SUITE_PYTHON = Path("/local/workspace/hailo_virtualenv/bin/python")


def ready(candidate: Path) -> bool:
    report = candidate / "handoff_report.json"
    if not all((report.is_file(), (candidate / "identity.json").is_file(),
                (candidate / "onnx").is_dir(), (candidate / "evaluation").is_dir())):
        return False
    try:
        return json.loads(report.read_text(encoding="utf-8")).get("status") == "PASS"
    except json.JSONDecodeError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff-parent", type=Path, required=True)
    parser.add_argument("--artifacts-parent", type=Path, required=True)
    parser.add_argument("--ignore", action="append", default=[])
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--cuda-visible-devices", default="1")
    args = parser.parse_args()
    ignored = set(args.ignore)
    state_dir = args.artifacts_parent / "handoff_watch"
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / "status.json"
    while True:
        for candidate in sorted(args.handoff_parent.iterdir()):
            if not candidate.is_dir() or candidate.name in ignored or not ready(candidate):
                continue
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            out_root = args.artifacts_parent / "candidates" / f"{candidate.name}-fullmodules-{stamp}" / "fused"
            log = out_root.parent / "launcher.log"
            state = {"status": "LAUNCHED", "candidate": str(candidate), "out_root": str(out_root),
                     "log": str(log), "launched_at": datetime.now(timezone.utc).isoformat()}
            state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            command = [str(SUITE_PYTHON), str(ROOT / "launch_handoff_fullmodules.py"),
                       "--handoff-root", str(candidate), "--out-root", str(out_root),
                       "--cuda-visible-devices", args.cuda_visible_devices]
            with log.open("w", encoding="utf-8") as handle:
                handle.write("$ " + " ".join(command) + "\n")
                handle.flush()
                completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
            state["status"] = "PASS" if completed.returncode == 0 else "FAIL"
            state["completed_at"] = datetime.now(timezone.utc).isoformat()
            state["returncode"] = completed.returncode
            state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(json.dumps(state, indent=2), flush=True)
            return completed.returncode
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
