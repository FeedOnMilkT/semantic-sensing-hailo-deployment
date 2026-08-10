#!/usr/bin/env python3
"""Deploy and run the bounded two-arm Pi hybrid ORT/Hailo experiment."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEPLOY = ROOT / "board_eval" / "deploy_hybrid_ort_runtime.py"
EVALUATE = ROOT / "board_eval" / "evaluate_hybrid_ort_runtime.py"
ARMS = ("hs_cpu", "gs_only_hailo")


def atomic_write(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def run(command: list[str]) -> None:
    print(json.dumps({"command": shlex.join(command)}), flush=True)
    subprocess.run(command, check=True)


def ssh_options(args) -> list[str]:
    return [
        "-i", str(args.identity_file),
        "-o", "IdentitiesOnly=yes",
        "-o", f"ProxyCommand={args.proxy_command}",
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=25",
        "-o", "StrictHostKeyChecking=yes",
        "-o", f"UserKnownHostsFile={args.known_hosts}",
    ]


def board_temperature(args) -> float:
    result = subprocess.run(
        ["ssh", *ssh_options(args), args.ssh_target,
         "cat /sys/class/thermal/thermal_zone0/temp"],
        check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return float(result.stdout.strip()) / 1000.0


def wait_until_cool(args, state: dict, stage: str, target_c: float = 55.0) -> None:
    deadline = time.monotonic() + 600.0
    while True:
        temperature = board_temperature(args)
        state.update({"stage": stage, "waiting_for_cool": True, "board_temperature_c": temperature})
        atomic_write(args.out_root / "campaign_state.json", state)
        print(json.dumps({"stage": stage, "temperature_c": temperature, "target_c": target_c}), flush=True)
        if temperature <= target_c:
            state["waiting_for_cool"] = False
            atomic_write(args.out_root / "campaign_state.json", state)
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"board did not cool below {target_c:.1f}C within 600 seconds")
        time.sleep(5.0)


def evaluator_command(
    args, deployment: Path, arm: str, mode: str, output: Path,
    *, limit: int, warmup: int, interval: float, cosine_samples: int,
) -> list[str]:
    return [
        sys.executable, str(EVALUATE),
        "--deployment", str(deployment),
        "--handoff", str(args.handoff),
        "--arm", arm,
        "--mode", mode,
        "--out", str(output),
        "--limit", str(limit),
        "--warmup", str(warmup),
        "--interval-seconds", str(interval),
        "--cosine-samples", str(cosine_samples),
        "--ssh-target", args.ssh_target,
        "--identity-file", str(args.identity_file),
        "--proxy-command", args.proxy_command,
        "--known-hosts", str(args.known_hosts),
    ]


def build_summary(out_root: Path) -> dict:
    arms: dict[str, dict] = {}
    for arm in ARMS:
        latency = json.loads((out_root / arm / "latency8.json").read_text(encoding="utf-8"))
        accuracy = json.loads((out_root / arm / "accuracy64.json").read_text(encoding="utf-8"))
        arms[arm] = {
            "cpu_modules": accuracy["board_hello"]["engines"],
            "latency_ms": latency["timing_and_similarity"]["board_total_pipeline_ms"],
            "ort_stage_ms": {
                key: value for key, value in latency["timing_and_similarity"].items()
                if ".infer_wall_ms" in key
                and latency["board_hello"]["engines"].get(key.split(".", 1)[0]) == "ort_cpu_fp32"
            },
            "semantic64": accuracy["semantic"],
            "bitstream64": accuracy["bitstream"],
            "temperature64": {
                "start": accuracy["timing_and_similarity"].get("thermal_start_c"),
                "end": accuracy["timing_and_similarity"].get("thermal_end_c"),
                "cooldown_wait_ms": accuracy["timing_and_similarity"].get("cooldown_wait_ms"),
            },
        }
    return {
        "kind": "pi_hybrid_ort_two_arm_campaign_summary",
        "status": "PASS",
        "stress_test": False,
        "arms": arms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--codec-runtime", type=Path, required=True)
    parser.add_argument("--ssh-target", required=True)
    parser.add_argument("--identity-file", type=Path, required=True)
    parser.add_argument("--proxy-command", required=True)
    parser.add_argument("--known-hosts", type=Path, required=True)
    args = parser.parse_args()
    args.out_root = args.out_root.resolve()
    if args.out_root.exists():
        raise FileExistsError(f"refusing to overwrite campaign root: {args.out_root}")
    args.out_root.mkdir(parents=True)
    state = {
        "kind": "pi_hybrid_ort_two_arm_campaign_state",
        "status": "RUNNING",
        "pid": os.getpid(),
        "run_tag": args.run_tag,
        "started_unix_seconds": time.time(),
        "stage": "deploy",
        "completed": [],
    }
    state_path = args.out_root / "campaign_state.json"
    atomic_write(state_path, state)
    deployment = args.out_root / "deployment.json"
    try:
        run([
            sys.executable, str(DEPLOY),
            "--selection-manifest", str(args.selection_manifest.resolve()),
            "--onnx-manifest", str((args.handoff / "onnx" / "manifest.json").resolve()),
            "--codec-runtime", str(args.codec_runtime.resolve()),
            "--run-tag", args.run_tag,
            "--out", str(deployment),
            "--ort-threads", "4",
            "--ssh-target", args.ssh_target,
            "--identity-file", str(args.identity_file),
            "--proxy-command", args.proxy_command,
            "--known-hosts", str(args.known_hosts),
        ])
        state["completed"].append("deploy")
        atomic_write(state_path, state)

        for arm in ARMS:
            arm_root = args.out_root / arm
            arm_root.mkdir()
            stages = (
                ("preflight1", "preflight", 1, 0, 0.0, 8192),
                ("latency8", "latency", 8, 1, 1.0, 8192),
                ("accuracy64", "accuracy", 64, 0, 1.0, 0),
            )
            for label, mode, limit, warmup, interval, cosine_samples in stages:
                stage = f"{arm}:{label}"
                wait_until_cool(args, state, f"{stage}:cooldown")
                state.update({"stage": stage, "waiting_for_cool": False})
                atomic_write(state_path, state)
                run(evaluator_command(
                    args, deployment, arm, mode, arm_root / f"{label}.json",
                    limit=limit, warmup=warmup, interval=interval,
                    cosine_samples=cosine_samples,
                ))
                state["completed"].append(stage)
                atomic_write(state_path, state)

        summary = build_summary(args.out_root)
        atomic_write(args.out_root / "summary.json", summary)
        state.update({
            "status": "PASS",
            "stage": "complete",
            "finished_unix_seconds": time.time(),
        })
        atomic_write(state_path, state)
    except BaseException as exc:
        state.update({
            "status": "FAILED",
            "error": f"{type(exc).__name__}: {exc}",
            "finished_unix_seconds": time.time(),
        })
        atomic_write(state_path, state)
        raise


if __name__ == "__main__":
    main()
