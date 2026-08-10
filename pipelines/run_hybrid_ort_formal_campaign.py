#!/usr/bin/env python3
"""Run the detached, thermally bounded FP32/FP16 formal board campaign."""
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
DEPLOY = ROOT / "board_eval" / "deploy_hybrid_ort_formal_runtime.py"
REMOTE_EVALUATOR = "evaluate_hybrid_ort_local_dataset.py"
ARM_ORDER = (
    "hs_cpu_fp32",
    "gs_only_hailo_fp32",
    "hs_cpu_fp16",
    "gs_only_hailo_fp16",
)
STAGES = {
    "hs_cpu_fp32": (
        ("preflight1", "preflight", 1, 0, 0.0),
        ("accuracy976", "accuracy", 976, 0, 1.0),
        ("latency8", "latency", 8, 1, 1.0),
    ),
    "gs_only_hailo_fp32": (
        ("preflight1", "preflight", 1, 0, 0.0),
        ("accuracy976", "accuracy", 976, 0, 1.0),
        ("latency8", "latency", 8, 1, 1.0),
    ),
    "hs_cpu_fp16": (
        ("preflight1", "preflight", 1, 0, 0.0),
        ("latency8", "latency", 8, 1, 1.0),
        ("accuracy976", "accuracy", 976, 0, 1.0),
    ),
    "gs_only_hailo_fp16": (
        ("preflight1", "preflight", 1, 0, 0.0),
        ("latency8", "latency", 8, 1, 1.0),
        ("accuracy976", "accuracy", 976, 0, 1.0),
    ),
}


def atomic_write(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def run(command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess:
    print(json.dumps({"command": shlex.join(command)}), flush=True)
    return subprocess.run(
        command, check=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        text=capture,
    )


def ssh_options(args) -> list[str]:
    return [
        "-i", str(args.identity_file),
        "-o", "IdentitiesOnly=yes",
        "-o", f"ProxyCommand={args.proxy_command}",
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=25",
        "-o", "ServerAliveInterval=10",
        "-o", "ServerAliveCountMax=3",
        "-o", "StrictHostKeyChecking=yes",
        "-o", f"UserKnownHostsFile={args.known_hosts}",
    ]


def ssh(args, remote_command: str, *, retries: int = 3) -> subprocess.CompletedProcess[str]:
    last_error: BaseException | None = None
    for attempt in range(1, retries + 1):
        try:
            return subprocess.run(
                ["ssh", *ssh_options(args), args.ssh_target, remote_command],
                check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError as exc:
            last_error = exc
            print(json.dumps({
                "ssh_retry": attempt,
                "returncode": exc.returncode,
                "stderr": exc.stderr[-1000:] if exc.stderr else "",
            }), flush=True)
            if attempt != retries:
                time.sleep(5.0 * attempt)
    assert last_error is not None
    raise last_error


def board_temperature(args) -> float:
    result = ssh(args, "cat /sys/class/thermal/thermal_zone0/temp")
    return float(result.stdout.strip()) / 1000.0


def wait_until_cool(args, state: dict, stage: str, target_c: float = 55.0) -> None:
    deadline = time.monotonic() + 900.0
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
            raise TimeoutError(f"board did not cool below {target_c:.1f} C within 900 seconds")
        time.sleep(5.0)


def launch_remote_stage(
    args, deployment: dict, arm: str, label: str, mode: str,
    limit: int, warmup: int, interval: float,
) -> tuple[str, str, str]:
    remote_dir = deployment["remote_dir"]
    dataset_root = deployment["dataset"]["remote_root"]
    result = f"{remote_dir}/results/{arm}_{label}.json"
    progress = f"{remote_dir}/results/{arm}_{label}.progress.json"
    log = f"{remote_dir}/logs/{arm}_{label}.log"
    pid_file = f"{remote_dir}/logs/{arm}_{label}.pid"
    manifest = f"{remote_dir}/runtime_manifest_{arm}.json"
    evaluator = f"{remote_dir}/{REMOTE_EVALUATOR}"
    command = [
        deployment["remote_ort_python"], evaluator,
        "--manifest", manifest,
        "--dataset-root", dataset_root,
        "--out", result,
        "--mode", mode,
        "--limit", str(limit),
        "--warmup", str(warmup),
        "--interval-seconds", str(interval),
        "--progress-every", "10",
    ]
    remote = (
        f"mkdir -p {shlex.quote(remote_dir + '/results')} {shlex.quote(remote_dir + '/logs')} || exit $?; "
        f"test ! -e {shlex.quote(result)} || exit 9; "
        f"test ! -e {shlex.quote(pid_file)} || exit 9; "
        f"nohup setsid {shlex.join(command)} > {shlex.quote(log)} 2>&1 < /dev/null & "
        f"pid=$!; echo $pid > {shlex.quote(pid_file)}; echo $pid"
    )
    launched = ssh(args, remote)
    pid = launched.stdout.strip().splitlines()[-1]
    if not pid.isdigit():
        raise RuntimeError(f"invalid remote PID for {arm}:{label}: {launched.stdout!r}")
    print(json.dumps({
        "remote_stage": f"{arm}:{label}", "remote_pid": int(pid),
        "result": result, "log": log,
    }), flush=True)
    return result, progress, pid_file


def poll_remote_stage(
    args, state: dict, stage: str, result: str, progress: str, pid_file: str,
) -> None:
    progress_code = (
        "import json,sys; m=json.load(open(sys.argv[1])); "
        "print(json.dumps({'sample_count':m.get('sample_count'),"
        "'fixed19':m.get('semantic',{}).get('fixed_19_large_object_macro_iou')}))"
    )
    while True:
        remote = (
            f"if test -f {shlex.quote(result)}; then echo COMPLETE; "
            f"elif test -f {shlex.quote(pid_file)} && kill -0 $(cat {shlex.quote(pid_file)}) 2>/dev/null; "
            f"then echo RUNNING; else echo FAILED; fi; "
            f"if test -f {shlex.quote(progress)}; then "
            f"python3 -c {shlex.quote(progress_code)} "
            f"{shlex.quote(progress)}; fi"
        )
        probe = ssh(args, remote)
        lines = probe.stdout.strip().splitlines()
        status = lines[0] if lines else "UNKNOWN"
        progress_record = None
        if len(lines) > 1:
            try:
                progress_record = json.loads(lines[-1])
            except json.JSONDecodeError:
                progress_record = {"raw": lines[-1]}
        state.update({
            "stage": stage,
            "remote_stage_status": status,
            "remote_stage_progress": progress_record,
            "last_poll_unix_seconds": time.time(),
        })
        atomic_write(args.out_root / "campaign_state.json", state)
        print(json.dumps({"stage": stage, "status": status, "progress": progress_record}), flush=True)
        if status == "COMPLETE":
            return
        if status != "RUNNING":
            raise RuntimeError(f"remote stage {stage} stopped without a result")
        time.sleep(20.0)


def download_json(args, remote: str, local: Path) -> dict:
    local.parent.mkdir(parents=True, exist_ok=True)
    temporary = local.with_name(f".{local.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        completed = subprocess.run(
            ["ssh", *ssh_options(args), args.ssh_target, "cat", remote],
            stdout=handle, stderr=subprocess.PIPE,
        )
    if completed.returncode:
        temporary.unlink(missing_ok=True)
        raise subprocess.CalledProcessError(completed.returncode, completed.args, stderr=completed.stderr)
    payload = json.loads(temporary.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS":
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"remote result is not PASS: {remote}")
    os.replace(temporary, local)
    return payload


def compact_arm(out_root: Path, arm: str) -> dict:
    accuracy = json.loads((out_root / arm / "accuracy976.json").read_text(encoding="utf-8"))
    latency = json.loads((out_root / arm / "latency8.json").read_text(encoding="utf-8"))
    timing = latency["timing"]
    engines = accuracy["board_hello"]["engines"]
    ort_stages = {
        key: value for key, value in timing.items()
        if key.endswith(".infer_wall_ms") and engines.get(key.split(".", 1)[0], "").startswith("ort_cpu")
    }
    return {
        "cpu_modules": [name for name, engine in engines.items() if engine.startswith("ort_cpu")],
        "engines": engines,
        "semantic976": accuracy["semantic"],
        "bitstream976": accuracy["bitstream"],
        "pipeline_latency_ms": timing["board_total_pipeline_ms"],
        "ort_stage_latency_ms": ort_stages,
        "temperature_c": {
            "accuracy": accuracy["timing"].get("thermal_end_c"),
            "latency": latency["timing"].get("thermal_end_c"),
        },
        "session_io_types": accuracy["board_hello"]["ort"]["session_io_types"],
    }


def delta(fp16: dict, fp32: dict) -> dict:
    sem16, sem32 = fp16["semantic976"], fp32["semantic976"]
    bit16, bit32 = fp16["bitstream976"], fp32["bitstream976"]
    lat16, lat32 = fp16["pipeline_latency_ms"], fp32["pipeline_latency_ms"]
    return {
        "fixed19_iou": sem16["fixed_19_large_object_macro_iou"] - sem32["fixed_19_large_object_macro_iou"],
        "fixed19_iou_points": 100.0 * (sem16["fixed_19_large_object_macro_iou"] - sem32["fixed_19_large_object_macro_iou"]),
        "mIoU": sem16["mIoU"] - sem32["mIoU"],
        "mIoU_points": 100.0 * (sem16["mIoU"] - sem32["mIoU"]),
        "pixel_accuracy": sem16["pixel_accuracy"] - sem32["pixel_accuracy"],
        "mean_bpp": bit16["mean_bpp"] - bit32["mean_bpp"],
        "pipeline_latency_mean_ms": lat16["mean"] - lat32["mean"],
        "pipeline_latency_p50_ms": lat16["p50"] - lat32["p50"],
        "pipeline_latency_p95_ms": lat16["p95"] - lat32["p95"],
        "pipeline_latency_mean_percent": 100.0 * (lat16["mean"] / lat32["mean"] - 1.0),
    }


def build_summary(out_root: Path) -> dict:
    arms = {arm: compact_arm(out_root, arm) for arm in ARM_ORDER}
    return {
        "kind": "pi_hybrid_ort_fp32_fp16_formal_campaign_summary",
        "status": "PASS",
        "primary_metric": "fixed_19_large_object_macro_iou",
        "sample_count_per_accuracy_arm": 976,
        "stress_test": False,
        "arms": arms,
        "fp16_minus_same_boundary_fp32": {
            "hs_cpu": delta(arms["hs_cpu_fp16"], arms["hs_cpu_fp32"]),
            "gs_only_hailo": delta(arms["gs_only_hailo_fp16"], arms["gs_only_hailo_fp32"]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--codec-runtime", type=Path, required=True)
    parser.add_argument("--remote-seed", required=True)
    parser.add_argument("--remote-dataset-root", required=True)
    parser.add_argument("--ssh-target", required=True)
    parser.add_argument("--identity-file", type=Path, required=True)
    parser.add_argument("--proxy-command", required=True)
    parser.add_argument("--known-hosts", type=Path, required=True)
    args = parser.parse_args()
    args.out_root = args.out_root.resolve()
    if args.out_root.exists():
        raise FileExistsError(f"refusing to overwrite campaign root: {args.out_root}")
    args.out_root.mkdir(parents=True)
    estimate = {
        "kind": "pi_hybrid_ort_fp32_fp16_formal_campaign_estimate",
        "created_unix_seconds": time.time(),
        "sample_count": 976,
        "arms": list(ARM_ORDER),
        "serial_idle_seconds_per_accuracy_sample": 1.0,
        "estimated_seconds": {"low": 6600, "high": 8100},
        "estimated_wall_clock": "1h50m to 2h15m",
        "assumptions": [
            "The 976-sample compressed dataset upload takes 10-20 minutes and occurs once.",
            "Each serial accuracy arm takes about 21-22 minutes including the 1 second idle interval.",
            "Preflight, latency probes, cooldown, deployment, and summary take 15-25 minutes.",
        ],
    }
    atomic_write(args.out_root / "estimate.json", estimate)
    state = {
        "kind": "pi_hybrid_ort_fp32_fp16_formal_campaign_state",
        "status": "RUNNING",
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "process_session_id": os.getsid(0),
        "run_tag": args.run_tag,
        "started_unix_seconds": time.time(),
        "stage": "deploy_and_dataset_upload",
        "completed": [],
        "schedule": {arm: [stage[0] for stage in STAGES[arm]] for arm in ARM_ORDER},
    }
    state_path = args.out_root / "campaign_state.json"
    atomic_write(state_path, state)
    deployment_path = args.out_root / "deployment.json"
    try:
        run([
            sys.executable, str(DEPLOY),
            "--selection-manifest", str(args.selection_manifest.resolve()),
            "--onnx-manifest", str((args.handoff / "onnx" / "manifest.json").resolve()),
            "--codec-runtime", str(args.codec_runtime.resolve()),
            "--evaluation-root", str((args.handoff / "evaluation").resolve()),
            "--run-tag", args.run_tag,
            "--out", str(deployment_path),
            "--fp16-root", str(args.out_root / "fp16_onnx"),
            "--ort-threads", "4",
            "--remote-seed", args.remote_seed,
            "--remote-dataset-root", args.remote_dataset_root,
            "--ssh-target", args.ssh_target,
            "--identity-file", str(args.identity_file),
            "--proxy-command", args.proxy_command,
            "--known-hosts", str(args.known_hosts),
        ])
        deployment = json.loads(deployment_path.read_text(encoding="utf-8"))
        state["completed"].append("deploy_and_dataset_upload")
        state["remote_dir"] = deployment["remote_dir"]
        state["remote_dataset_root"] = deployment["dataset"]["remote_root"]
        atomic_write(state_path, state)

        for arm in ARM_ORDER:
            arm_root = args.out_root / arm
            arm_root.mkdir()
            for label, mode, limit, warmup, interval in STAGES[arm]:
                stage = f"{arm}:{label}"
                wait_until_cool(args, state, f"{stage}:cooldown")
                state.update({"stage": stage, "waiting_for_cool": False, "remote_stage_progress": None})
                atomic_write(state_path, state)
                result, progress, pid_file = launch_remote_stage(
                    args, deployment, arm, label, mode, limit, warmup, interval
                )
                poll_remote_stage(args, state, stage, result, progress, pid_file)
                download_json(args, result, arm_root / f"{label}.json")
                state["completed"].append(stage)
                state["remote_stage_status"] = "COMPLETE"
                atomic_write(state_path, state)

        summary = build_summary(args.out_root)
        atomic_write(args.out_root / "summary.json", summary)
        state.update({
            "status": "PASS",
            "stage": "complete",
            "finished_unix_seconds": time.time(),
            "elapsed_seconds": time.time() - state["started_unix_seconds"],
        })
        atomic_write(state_path, state)
    except BaseException as exc:
        state.update({
            "status": "FAILED",
            "error": f"{type(exc).__name__}: {exc}",
            "finished_unix_seconds": time.time(),
            "elapsed_seconds": time.time() - state["started_unix_seconds"],
        })
        atomic_write(state_path, state)
        raise


if __name__ == "__main__":
    main()
