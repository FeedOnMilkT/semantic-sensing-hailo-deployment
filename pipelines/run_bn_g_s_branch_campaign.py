#!/usr/bin/env python3
"""Detached full-976 orchestrator for current-topology ``bn_g_s`` experiments.

The campaign first performs a same-y_hat four-way branch splice, then evaluates
the frozen bn_g_s HEF candidates without changing any other engine boundary.
Deployment comparison uses only fixed-19 large-object macro IoU on the immutable
976-sample held-out set.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from board_eval.deploy import stream_upload  # noqa: E402


CPU_MODULES = ("bn_align_h", "bn_g_a", "bn_h_a", "bn_h_s")
HAILO_MODULES = (
    "sender_backbone", "bn_g_s", "receiver_stage4", "receiver_stage5",
    "receiver_dappm", "receiver_head",
)
CANDIDATE_ROOT = Path(
    "/local/shared_with_docker/deplyment-demo/artifacts/fused/"
    "current-ort-mainline-20260805/bn_g_s_candidates"
)
CANDIDATES = (
    (
        "a8_l2",
        "A8 baseline",
        CANDIDATE_ROOT / "a8_l2/bn_g_s_l2.hef",
        "6dc2590175cfa9ca78ce1e7abeafab69b9b16010027fe5fbc84a0ff44e1c6c5b",
    ),
    (
        "x_l_low_spatial_a16_l2",
        "A16 on x_l low-spatial 256x32x32 output",
        CANDIDATE_ROOT / "x_l_low_spatial_a16_l2/bn_g_s_l2.hef",
        "cadbdd1114ac3d75c6006d95804cfa6d86025490fc110cb900e91e51181a7856",
    ),
    (
        "x_h_high_spatial_a16_l2",
        "A16 on x_h high-spatial 128x64x64 output",
        CANDIDATE_ROOT / "x_h_high_spatial_a16_l2/bn_g_s_l2.hef",
        "9ba443ee23a04c286c3215c894fbe0581473f064fa06e23d54bac62f20af198e",
    ),
    (
        "both_outputs_a16_l2",
        "A16 boundary treatment on both bn_g_s outputs",
        CANDIDATE_ROOT / "both_outputs_a16_l2/bn_g_s_l2.hef",
        "bd997f19aa20717620ac0f2e0d416231545052dae8a35e77b2de15a78076b5c3",
    ),
    (
        "x_l_low_spatial_a16_qft_e8",
        "QFT candidate with x_l low-spatial A16",
        CANDIDATE_ROOT / "x_l_low_spatial_a16_qft_e8/bn_g_s_l2.hef",
        "2b0bd40a9b989e87471378b3154751799c65726e509153ced7fa3c191e750b51",
    ),
    (
        "x_l_low_spatial_a16_adaround_l4",
        "AdaRound candidate with x_l low-spatial A16",
        CANDIDATE_ROOT / "x_l_low_spatial_a16_adaround_l4/bn_g_s_l4.hef",
        "fe764a1ed89a339407dd57f0159ef5e7405d687e41223e9cbbbce6298e599cfc",
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def ssh_options(args) -> list[str]:
    return [
        "-i", str(args.identity_file), "-o", "IdentitiesOnly=yes",
        "-o", f"ProxyCommand={args.proxy_command}", "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=25", "-o", "ServerAliveInterval=10",
        "-o", "ServerAliveCountMax=3", "-o", "StrictHostKeyChecking=yes",
        "-o", f"UserKnownHostsFile={args.known_hosts}",
    ]


def ssh(args, command: str, *, retries: int = 3) -> subprocess.CompletedProcess[str]:
    error: subprocess.CalledProcessError | None = None
    for attempt in range(1, retries + 1):
        try:
            return subprocess.run(
                ["ssh", *ssh_options(args), args.ssh_target, command], check=True,
                text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError as exc:
            error = exc
            print(json.dumps({
                "ssh_retry": attempt, "returncode": exc.returncode,
                "stderr": (exc.stderr or "")[-1000:],
            }), flush=True)
            if attempt < retries:
                time.sleep(5 * attempt)
    assert error is not None
    raise error


def board_temperature(args) -> float:
    return float(ssh(args, "cat /sys/class/thermal/thermal_zone0/temp").stdout.strip()) / 1000.0


def wait_until_cool(args, state: dict, stage: str) -> None:
    deadline = time.monotonic() + 1800.0
    while True:
        temperature = board_temperature(args)
        state.update({
            "stage": f"{stage}:cooldown", "waiting_for_cool": True,
            "board_temperature_c": temperature,
        })
        atomic_write(args.out_root / "campaign_state.json", state)
        if temperature <= 60.0:
            state["waiting_for_cool"] = False
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("board did not cool to 60 C within 30 minutes")
        time.sleep(10.0)


def launch_stage(args, *, label: str, evaluator: str, manifest: str, limit: int,
                 interval: float) -> tuple[str, str, str, str]:
    stage_root = f"{args.remote_root}/experiment/{label}"
    result = f"{stage_root}/result.json"
    progress = f"{stage_root}/result.progress.json"
    log = f"{stage_root}/run.log"
    pid_file = f"{stage_root}/run.pid"
    command = [
        args.remote_ort_python, f"{args.remote_root}/{evaluator}",
        "--manifest", f"{args.remote_root}/{manifest}",
        "--dataset-root", args.remote_dataset_root,
        "--out", result, "--limit", str(limit),
        "--interval-seconds", str(interval), "--progress-every", "10",
    ]
    if evaluator == "evaluate_hybrid_ort_local_dataset.py":
        command[command.index("--limit"):command.index("--interval-seconds")] = [
            "--mode", "accuracy", "--limit", str(limit), "--warmup", "0",
        ]
    remote = (
        f"mkdir -p {shlex.quote(stage_root)} || exit $?; "
        f"test ! -e {shlex.quote(result)} || exit 9; "
        f"test ! -e {shlex.quote(pid_file)} || exit 9; "
        f"nohup setsid {shlex.join(command)} > {shlex.quote(log)} 2>&1 < /dev/null & "
        f"pid=$!; echo $pid > {shlex.quote(pid_file)}; echo $pid"
    )
    launched = ssh(args, remote)
    pid = launched.stdout.strip().splitlines()[-1]
    if not pid.isdigit():
        raise RuntimeError(f"invalid remote PID: {launched.stdout!r}")
    return result, progress, pid_file, log


def poll_stage(args, state: dict, label: str, result: str, progress: str,
               pid_file: str, log: str) -> None:
    progress_code = (
        "import json,sys; d=json.load(open(sys.argv[1])); "
        "print(d.get('sample_count', 0))"
    )
    while True:
        probe = ssh(args, (
            f"if test -f {shlex.quote(result)}; then echo COMPLETE; "
            f"elif test -f {shlex.quote(pid_file)} && "
            f"kill -0 $(cat {shlex.quote(pid_file)}) 2>/dev/null; then echo RUNNING; "
            f"else echo FAILED; fi; "
            f"if test -f {shlex.quote(progress)}; then "
            f"python3 -c {shlex.quote(progress_code)} "
            f"{shlex.quote(progress)}; fi"
        ))
        lines = probe.stdout.strip().splitlines()
        status = lines[0] if lines else "UNKNOWN"
        count = int(lines[-1]) if len(lines) > 1 and lines[-1].isdigit() else None
        state.update({
            "stage": label, "remote_stage_status": status,
            "remote_stage_sample_count": count, "last_poll_unix_seconds": time.time(),
        })
        atomic_write(args.out_root / "campaign_state.json", state)
        print(json.dumps({"stage": label, "status": status, "samples": count}), flush=True)
        if status == "COMPLETE":
            return
        if status != "RUNNING":
            tail = ssh(args, f"tail -80 {shlex.quote(log)} 2>/dev/null || true").stdout
            raise RuntimeError(f"remote stage {label} failed:\n{tail}")
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
    data = json.loads(temporary.read_text(encoding="utf-8"))
    if data.get("status") != "PASS":
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"remote result is not PASS: {remote}")
    os.replace(temporary, local)
    return data


def validate_base_manifest(manifest: dict) -> None:
    if manifest.get("kind") != "semantic_sensing_pi_hybrid_ort_runtime":
        raise ValueError("base manifest is not a hybrid runtime")
    if tuple(manifest.get("cpu_modules", ())) != CPU_MODULES:
        raise ValueError("base manifest violates permanent ORT CPU policy")
    if tuple(manifest.get("hailo_modules", ())) != HAILO_MODULES:
        raise ValueError("base manifest violates permanent Hailo policy")
    engines = {spec["name"]: spec["engine"] for spec in manifest["modules"]}
    if any(engines[name] != "ort_cpu_fp32" for name in CPU_MODULES):
        raise ValueError("base manifest is not the FP32 policy tier")
    if any(engines[name] != "hailort_float32_vstream" for name in HAILO_MODULES):
        raise ValueError("base manifest has an unexpected non-Hailo mainline module")


def build_summary(branch: dict, candidates: dict[str, dict], descriptions: dict[str, str]) -> dict:
    branch_semantic = branch["semantic_by_splice"]
    metric = "fixed_19_large_object_macro_iou"
    baseline = branch_semantic["B00_fp_high_fp_low"][metric]
    branch_rows = {}
    for arm, semantic in branch_semantic.items():
        branch_rows[arm] = {
            "fixed19_large_object_macro_iou": semantic[metric],
            "delta_fixed19_points_vs_B00": 100.0 * (semantic[metric] - baseline),
        }
    candidate_rows = {}
    for name, result in candidates.items():
        semantic = result["semantic"]
        candidate_rows[name] = {
            "description": descriptions[name],
            "fixed19_large_object_macro_iou": semantic[metric],
            "mean_bpp": result["bitstream"]["mean_bpp"],
        }
    return {
        "kind": "bn_g_s_current_topology_full976_summary",
        "status": "AWAITING_REVIEW",
        "sample_count": 976,
        "selection_metric": metric,
        "ade20k_150_class_miou": "NOT_USED",
        "automatic_metric_rejection": False,
        "branch_attribution": branch_rows,
        "candidate_screen": candidate_rows,
        "next_step_after_review": "select candidates directly from full976 fixed19 results",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--remote-seed", required=True)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--remote-dataset-root", required=True)
    parser.add_argument("--remote-ort-python", required=True)
    parser.add_argument("--ssh-target", required=True)
    parser.add_argument("--identity-file", type=Path, required=True)
    parser.add_argument("--proxy-command", required=True)
    parser.add_argument("--known-hosts", type=Path, required=True)
    parser.add_argument("--branch-limit", type=int, default=976)
    parser.add_argument("--screen-limit", type=int, default=976)
    args = parser.parse_args()
    if args.branch_limit != 976 or args.screen_limit != 976:
        parser.error("deployment comparison requires exactly 976 samples for both stages")
    args.out_root = args.out_root.resolve()
    if args.out_root.exists():
        raise FileExistsError(f"refusing to overwrite campaign root: {args.out_root}")
    args.out_root.mkdir(parents=True)

    base = json.loads(args.base_manifest.read_text(encoding="utf-8"))
    validate_base_manifest(base)
    artifact_rows = []
    for name, description, path, expected_sha in CANDIDATES:
        if not path.is_file() or sha256(path) != expected_sha:
            raise ValueError(f"candidate artifact hash mismatch: {name} {path}")
        artifact_rows.append({
            "name": name, "description": description, "path": str(path),
            "sha256": expected_sha, "bytes": path.stat().st_size,
        })
    atomic_write(args.out_root / "frozen_artifacts.json", {
        "run_tag": args.run_tag,
        "permanent_cpu_modules": CPU_MODULES,
        "semantic_branch_contract": {
            "x_h_hat": "128x64x64 high-spatial-resolution",
            "x_l_hat": "256x32x32 low-spatial-resolution",
        },
        "candidates": artifact_rows,
    })
    atomic_write(args.out_root / "estimate.json", {
        "branch_samples": args.branch_limit, "candidate_samples_each": args.screen_limit,
        "candidate_count": len(CANDIDATES), "stress_test": False,
        "estimated_wall_clock": "3-5 hours including cooldown and board setup",
    })
    state = {
        "kind": "bn_g_s_current_topology_campaign_state", "status": "RUNNING",
        "pid": os.getpid(), "ppid": os.getppid(), "session_id": os.getsid(0),
        "run_tag": args.run_tag, "started_unix_seconds": time.time(),
        "stage": "artifact_and_topology_preflight", "completed": [],
    }
    state_path = args.out_root / "campaign_state.json"
    atomic_write(state_path, state)

    try:
        # A hard-link clone keeps every previously frozen artifact immutable and
        # avoids another large board upload. New manifests/results use new names.
        ssh(args, (
            f"test -d {shlex.quote(args.remote_seed)} || exit 4; "
            f"test ! -e {shlex.quote(args.remote_root)} || exit 9; "
            f"mkdir -p {shlex.quote(args.remote_root)}; "
            f"cp -al {shlex.quote(args.remote_seed + '/.')} {shlex.quote(args.remote_root + '/')}"
        ))
        upload_options = ssh_options(args)
        for source in (
            ROOT / "board_eval" / "bn_g_s_branch_attribution_runtime.py",
            ROOT / "board_eval" / "evaluate_bn_g_s_branch_attribution.py",
        ):
            stream_upload(source, f"{args.remote_root}/{source.name}", upload_options, args.ssh_target)

        manifests_root = args.out_root / "runtime_manifests"
        manifests_root.mkdir()
        branch_manifest = copy.deepcopy(base)
        branch_manifest["arm"] = "bn_g_s_same_y_hat_branch_attribution"
        branch_manifest["branch_attribution"] = {
            "bn_g_s_fp32_onnx_filename": "onnx/bn_g_s.onnx",
            "semantic_output_contract": {
                "x_h_hat": [1, 128, 64, 64], "x_l_hat": [1, 256, 32, 32],
            },
        }
        branch_manifest_path = manifests_root / "branch_attribution.json"
        atomic_write(branch_manifest_path, branch_manifest)
        stream_upload(
            branch_manifest_path, f"{args.remote_root}/runtime_manifest_branch_attribution.json",
            upload_options, args.ssh_target,
        )

        candidate_manifests: dict[str, Path] = {}
        for name, description, path, expected_sha in CANDIDATES:
            remote_hef = f"hef_candidates/{name}.hef"
            stream_upload(path, f"{args.remote_root}/{remote_hef}", upload_options, args.ssh_target)
            candidate = copy.deepcopy(base)
            candidate["arm"] = f"bn_g_s_screen_{name}"
            candidate["bn_g_s_candidate_description"] = description
            gs = next(spec for spec in candidate["modules"] if spec["name"] == "bn_g_s")
            gs.update({
                "hef_filename": remote_hef, "hef_sha256": expected_sha,
                "hef_bytes": path.stat().st_size,
            })
            local_manifest = manifests_root / f"candidate_{name}.json"
            atomic_write(local_manifest, candidate)
            remote_manifest = f"runtime_manifest_candidate_{name}.json"
            stream_upload(
                local_manifest, f"{args.remote_root}/{remote_manifest}",
                upload_options, args.ssh_target,
            )
            candidate_manifests[name] = local_manifest

        # Remote artifact integrity is checked independently of upload success.
        checks = []
        for name, _description, _path, expected_sha in CANDIDATES:
            checks.append(
                f"test $(sha256sum {shlex.quote(args.remote_root + '/hef_candidates/' + name + '.hef')} "
                f"| cut -d' ' -f1) = {shlex.quote(expected_sha)}"
            )
        ssh(args, " && ".join(checks))
        state["completed"].append("artifact_and_topology_preflight")
        state["remote_root"] = args.remote_root
        atomic_write(state_path, state)

        results_root = args.out_root / "results"
        results_root.mkdir()
        stages = (
            ("branch_preflight1", "evaluate_bn_g_s_branch_attribution.py",
             "runtime_manifest_branch_attribution.json", 1, 0.0),
            ("branch_attribution", "evaluate_bn_g_s_branch_attribution.py",
             "runtime_manifest_branch_attribution.json", args.branch_limit, 1.0),
        )
        branch_result = None
        for label, evaluator, manifest, limit, interval in stages:
            wait_until_cool(args, state, label)
            result, progress, pid_file, log = launch_stage(
                args, label=label, evaluator=evaluator, manifest=manifest,
                limit=limit, interval=interval,
            )
            poll_stage(args, state, label, result, progress, pid_file, log)
            data = download_json(args, result, results_root / f"{label}.json")
            state["completed"].append(label)
            atomic_write(state_path, state)
            if label == "branch_attribution":
                branch_result = data
        assert branch_result is not None

        candidate_results: dict[str, dict] = {}
        descriptions: dict[str, str] = {}
        for name, description, _path, _sha in CANDIDATES:
            label = f"candidate_{name}"
            wait_until_cool(args, state, label)
            result, progress, pid_file, log = launch_stage(
                args, label=label, evaluator="evaluate_hybrid_ort_local_dataset.py",
                manifest=f"runtime_manifest_candidate_{name}.json",
                limit=args.screen_limit, interval=1.0,
            )
            poll_stage(args, state, label, result, progress, pid_file, log)
            candidate_results[name] = download_json(
                args, result, results_root / f"{label}.json"
            )
            descriptions[name] = description
            state["completed"].append(label)
            atomic_write(state_path, state)

        summary = build_summary(branch_result, candidate_results, descriptions)
        atomic_write(args.out_root / "summary.json", summary)
        state.update({
            "status": "AWAITING_REVIEW", "stage": "awaiting_user_candidate_selection",
            "finished_unix_seconds": time.time(),
            "elapsed_seconds": time.time() - state["started_unix_seconds"],
        })
        atomic_write(state_path, state)
    except BaseException as exc:
        state.update({
            "status": "FAILED", "error": f"{type(exc).__name__}: {exc}",
            "finished_unix_seconds": time.time(),
            "elapsed_seconds": time.time() - state["started_unix_seconds"],
        })
        atomic_write(state_path, state)
        raise


if __name__ == "__main__":
    main()
