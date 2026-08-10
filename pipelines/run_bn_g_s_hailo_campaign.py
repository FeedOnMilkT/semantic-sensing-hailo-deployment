#!/usr/bin/env python3
"""Detached, resume-observable bn_g_s Host/Board comparison campaign.

Eight existing, identity-compatible bn_g_s HEFs are evaluated in two lanes:

* one SDK_QUANTIZED screen64 job per available GPU;
* one thermally bounded all-Hailo board lane, serializing candidates because
  the Raspberry Pi has one Hailo-8 device.

Only hard artifact/runtime failures stop an individual arm. Accuracy never
selects or rejects a candidate; the campaign emits a comparison table for
manual review before any 976-sample run.
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
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from board_eval.deploy import stream_upload
from hailo_deploy.io.hash import sha256


EVALUATOR = ROOT / "scripts" / "evaluate_fullmodule_hn_chain.py"
REMOTE_RUNNER = ROOT / "board_eval" / "hybrid_ort_runtime_runner.py"
REMOTE_BASE_RUNNER = ROOT / "board_eval" / "fullmodule_runtime_runner.py"
REMOTE_EVALUATOR = ROOT / "board_eval" / "evaluate_hybrid_ort_local_dataset.py"
DEFAULT_SPEC = ROOT / "configs" / "bn_g_s_hailo_campaign_20260805.json"
SEQUENCE = (
    "sender_backbone", "bn_align_h", "bn_g_a", "bn_h_a", "bn_h_s", "bn_g_s",
    "receiver_stage4", "receiver_stage5", "receiver_dappm", "receiver_head",
)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def canonical_fingerprint(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def artifact_record(path: Path, expected_sha: str | None = None) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = sha256(path)
    if expected_sha is not None and digest != expected_sha:
        raise ValueError(f"SHA mismatch for {path}: {digest} != {expected_sha}")
    return {"path": str(path), "sha256": digest, "bytes": path.stat().st_size}


def normalize_result_artifact(result: dict[str, Any], key: str) -> dict[str, Any] | None:
    record = result.get(key)
    if not isinstance(record, dict) or not record.get("path"):
        return None
    return artifact_record(Path(record["path"]), record.get("sha256"))


def candidate_identity(result: dict[str, Any]) -> tuple[str | None, str | None]:
    checkpoint = result.get("identity", {}).get("checkpoint", {}).get("sha256")
    onnx_sha = result.get("onnx", {}).get("sha256")
    metadata = result.get("calibration", {}).get("metadata", {})
    checkpoint = checkpoint or metadata.get("checkpoint", {}).get("sha256")
    onnx_sha = onnx_sha or metadata.get("onnx", {}).get("sha256")
    return checkpoint, onnx_sha


def gpu_snapshot() -> list[dict[str, int]]:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"],
        check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    rows = []
    for line in completed.stdout.splitlines():
        index, memory, utilization = [int(item.strip()) for item in line.split(",")]
        rows.append({"index": index, "memory_used_mib": memory,
                     "utilization_percent": utilization})
    return rows


class Campaign:
    def __init__(self, spec_path: Path) -> None:
        self.spec_path = spec_path.resolve()
        self.spec = load_json(self.spec_path)
        self.out = Path(self.spec["out_root"]).resolve()
        self.handoff = Path(self.spec["handoff_root"]).resolve()
        self.base_selection_path = Path(self.spec["base_selection_manifest"]).resolve()
        self.base_selection = load_json(self.base_selection_path)
        self.state_path = self.out / "campaign_state.json"
        self.state: dict[str, Any] = {}
        self.host_processes: dict[str, dict[str, Any]] = {}
        self.pending_host: list[dict[str, Any]] = []
        self.free_gpus: list[int] = []
        self.candidates: list[dict[str, Any]] = []

    def write_state(self, **updates: Any) -> None:
        self.state.update(updates)
        self.state["updated_unix_seconds"] = time.time()
        atomic_json(self.state_path, self.state)

    def validate(self, *, require_out_absent: bool) -> dict[str, Any]:
        if self.spec.get("schema") != "semantic_sensing.bn_g_s_hailo_campaign.v1":
            raise ValueError("unsupported campaign schema")
        if require_out_absent and self.out.exists():
            raise FileExistsError(f"refusing existing out root: {self.out}")
        if not self.handoff.is_dir() or not self.base_selection_path.is_file():
            raise FileNotFoundError("handoff or base selection is absent")
        if self.base_selection.get("module_order") != list(SEQUENCE):
            raise ValueError("base selection is not the frozen ten-module sequence")
        identity = load_json(self.handoff / "identity.json")
        checkpoint = identity.get("checkpoint", {}).get("sha256")
        onnx_manifest = load_json(self.handoff / "onnx" / "manifest.json")
        onnx_by_name = {item["name"]: item for item in onnx_manifest["modules"]}
        if set(onnx_by_name) != set(SEQUENCE):
            raise ValueError("handoff ONNX manifest is not the ten-module contract")
        if checkpoint != self.base_selection.get("checkpoint_sha256"):
            raise ValueError("base selection checkpoint differs from handoff")
        expected_onnx = onnx_by_name["bn_g_s"]["sha256"]
        seen_names: set[str] = set()
        audited = []
        for configured in self.spec["candidates"]:
            name = str(configured["name"])
            if name in seen_names:
                raise ValueError(f"duplicate candidate {name}")
            seen_names.add(name)
            result_path = Path(configured["result_json"]).resolve()
            result = load_json(result_path)
            if result.get("schema") != "ptq.v1" or result.get("status") != "PASS":
                raise ValueError(f"{name}: result is not ptq.v1 PASS")
            if result.get("module") != "bn_g_s":
                raise ValueError(f"{name}: result module is not bn_g_s")
            result_checkpoint, result_onnx = candidate_identity(result)
            if result_checkpoint not in (None, checkpoint):
                raise ValueError(f"{name}: checkpoint mismatch")
            if result_onnx not in (None, expected_onnx):
                raise ValueError(f"{name}: ONNX mismatch")
            hef = normalize_result_artifact(result, "hef")
            quantized_har = normalize_result_artifact(result, "quantized_har")
            if hef is None or quantized_har is None:
                raise ValueError(f"{name}: missing HEF or quantized HAR")
            if int(result.get("num_contexts", 0)) != 1:
                raise ValueError(f"{name}: only single-context HEFs are allowed")
            audited.append({
                **configured,
                "result_path": str(result_path),
                "result": result,
                "result_json": artifact_record(result_path),
                "hef": hef,
                "quantized_har": quantized_har,
                "checkpoint_sha256": checkpoint,
                "onnx_sha256": expected_onnx,
                "n_calib": int(result.get("n_calib", 0)),
                "optimization_level": int(result.get("optimization_level", -1)),
                "num_contexts": int(result["num_contexts"]),
            })
        snapshot = gpu_snapshot()
        scheduler = self.spec["scheduler"]
        requested = set(int(item) for item in scheduler["gpu_ids"])
        eligible = [
            row["index"] for row in snapshot
            if row["index"] in requested
            and row["memory_used_mib"] < int(scheduler["gpu_memory_used_max_mib"])
            and row["utilization_percent"] < int(scheduler["gpu_utilization_max_percent"])
        ]
        if not eligible:
            raise RuntimeError("no eligible GPU for SDK screen64")
        board_seed = self.spec["board_seed"]
        deployment = load_json(Path(board_seed["deployment_json"]))
        runtime_manifest = load_json(Path(board_seed["runtime_manifest"]))
        if deployment.get("status") != "PASS":
            raise ValueError("board seed deployment is not PASS")
        if deployment.get("checkpoint_sha256") != checkpoint:
            raise ValueError("board seed checkpoint mismatch")
        if runtime_manifest.get("kind") != "semantic_sensing_pi_hybrid_ort_runtime":
            raise ValueError("board seed is not a hybrid runtime manifest")
        for path in (EVALUATOR, REMOTE_RUNNER, REMOTE_BASE_RUNNER, REMOTE_EVALUATOR):
            if not path.is_file():
                raise FileNotFoundError(path)
        self.candidates = audited
        self.free_gpus = eligible
        return {
            "spec_sha256": sha256(self.spec_path),
            "base_selection_sha256": sha256(self.base_selection_path),
            "checkpoint_sha256": checkpoint,
            "bn_g_s_onnx_sha256": expected_onnx,
            "gpu_snapshot": snapshot,
            "eligible_gpus": eligible,
            "candidate_count": len(audited),
            "board_seed_remote_dir": deployment["remote_dir"],
            "board_dataset_remote_root": deployment["dataset"]["remote_root"],
        }

    def freeze_manifests(self) -> None:
        manifest_root = self.out / "manifests"
        for candidate in self.candidates:
            name = candidate["name"]
            selection = copy.deepcopy(self.base_selection)
            selection["candidate"] = f"bn_g_s_{name}"
            selection["policy"] = {
                "kind": "diagnostic_bn_g_s_override",
                "description": candidate["description"],
                "accuracy_gate": "record_only",
                "production_boundary_change": False,
            }
            record = selection["modules"]["bn_g_s"]
            result = candidate["result"]
            record.update({
                "result_json": candidate["result_json"],
                "n_calib": candidate["n_calib"],
                "optimization_level": candidate["optimization_level"],
                "num_contexts": candidate["num_contexts"],
                "quantized_har": candidate["quantized_har"],
                "hef": candidate["hef"],
            })
            for key in ("model_script", "compiled_har", "auto_alls"):
                normalized = normalize_result_artifact(result, key)
                if normalized is not None:
                    record[key] = normalized
            contract = {key: value for key, value in selection.items()
                        if key != "candidate_fingerprint"}
            selection["candidate_fingerprint"] = canonical_fingerprint(contract)
            path = manifest_root / name / "selection_manifest.json"
            atomic_json(path, selection)
            candidate["selection_manifest"] = str(path)
            candidate["candidate_fingerprint"] = selection["candidate_fingerprint"]

    def host_command(self, candidate: dict[str, Any], output: Path, progress: Path) -> list[str]:
        scheduler = self.spec["scheduler"]
        return [
            sys.executable, str(EVALUATOR),
            "--handoff-root", str(self.handoff),
            "--selection-manifest", candidate["selection_manifest"],
            "--out", str(output),
            "--limit", str(scheduler["host_limit"]),
            "--progress-out", str(progress),
            "--checkpoint-every", str(scheduler["host_checkpoint_every"]),
        ]

    def launch_host_job(self, candidate: dict[str, Any], gpu: int) -> None:
        name = candidate["name"]
        root = self.out / "host" / name
        root.mkdir(parents=True, exist_ok=True)
        output = root / "screen64.json"
        progress = root / "screen64.progress.json"
        log_path = root / "screen64.log"
        sdk_logs = root / "sdk_logs"
        tmp_dir = root / "tmp"
        sdk_logs.mkdir(exist_ok=True)
        tmp_dir.mkdir(exist_ok=True)
        command = self.host_command(candidate, output, progress)
        environment = os.environ.copy()
        xla_flags = environment.get("XLA_FLAGS", "")
        if "--xla_gpu_autotune_level=" not in xla_flags:
            xla_flags = (xla_flags + " --xla_gpu_autotune_level=0").strip()
        environment.update({
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "TF_FORCE_GPU_ALLOW_GROWTH": "true",
            "XLA_FLAGS": xla_flags,
            "HAILO_SDK_LOG_DIR": str(sdk_logs),
            "_HAILO_SDK_LOGGED": "1",
            "TMPDIR": str(tmp_dir),
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
        })
        handle = log_path.open("w", encoding="utf-8")
        handle.write("$ " + shlex.join(command) + "\n")
        handle.flush()
        process = subprocess.Popen(
            command, cwd=ROOT, env=environment, stdout=handle,
            stderr=subprocess.STDOUT, start_new_session=True,
        )
        self.host_processes[name] = {
            "process": process, "handle": handle, "gpu": gpu,
            "output": output, "progress": progress, "log": log_path,
        }
        self.state["candidates"][name]["host"].update({
            "status": "RUNNING", "pid": process.pid, "gpu": gpu,
            "output": str(output), "progress": str(progress), "log": str(log_path),
        })
        self.write_state(stage="HOST_AND_BOARD_RUNNING")

    def fill_host_queue(self) -> None:
        while self.pending_host and self.free_gpus:
            self.launch_host_job(self.pending_host.pop(0), self.free_gpus.pop(0))

    def refresh_host_jobs(self) -> None:
        completed: list[str] = []
        for name, job in list(self.host_processes.items()):
            progress = job["progress"]
            if progress.is_file():
                try:
                    self.state["candidates"][name]["host"]["progress_record"] = load_json(progress)
                except Exception as exc:
                    self.state["candidates"][name]["host"]["progress_error"] = str(exc)
            return_code = job["process"].poll()
            if return_code is None:
                continue
            job["handle"].close()
            passed = False
            if return_code == 0 and job["output"].is_file():
                try:
                    passed = load_json(job["output"]).get("status") == "PASS"
                except Exception:
                    passed = False
            self.state["candidates"][name]["host"].update({
                "status": "PASS" if passed else "FAIL",
                "return_code": return_code,
                "finished_unix_seconds": time.time(),
            })
            self.free_gpus.append(int(job["gpu"]))
            self.free_gpus.sort()
            completed.append(name)
        for name in completed:
            self.host_processes.pop(name)
        self.fill_host_queue()
        self.write_state()

    def ssh_options(self) -> list[str]:
        value = self.spec["ssh"]
        return [
            "-i", value["identity_file"], "-o", "IdentitiesOnly=yes",
            "-o", f"ProxyCommand={value['proxy_command']}",
            "-o", "BatchMode=yes", "-o", "ConnectTimeout=25",
            "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=3",
            "-o", "StrictHostKeyChecking=yes",
            "-o", f"UserKnownHostsFile={value['known_hosts']}",
        ]

    def ssh(self, remote_command: str, *, retries: int = 3) -> subprocess.CompletedProcess[str]:
        last: subprocess.CalledProcessError | None = None
        for attempt in range(1, retries + 1):
            try:
                return subprocess.run(
                    ["ssh", *self.ssh_options(), self.spec["ssh"]["target"], remote_command],
                    check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
            except subprocess.CalledProcessError as exc:
                last = exc
                if attempt != retries:
                    time.sleep(5 * attempt)
        assert last is not None
        raise last

    def board_temperature(self) -> float:
        value = self.ssh("cat /sys/class/thermal/thermal_zone0/temp").stdout.strip()
        return float(value) / 1000.0

    def wait_board_cool(self, candidate: str, stage: str) -> None:
        scheduler = self.spec["scheduler"]
        target = float(scheduler["board_resume_temperature_c"])
        deadline = time.monotonic() + float(scheduler["board_cooldown_timeout_seconds"])
        while True:
            self.refresh_host_jobs()
            temperature = self.board_temperature()
            self.state["board"].update({
                "candidate": candidate, "stage": f"{stage}:cooldown",
                "temperature_c": temperature, "waiting_for_cool": temperature > target,
            })
            self.write_state(stage="HOST_AND_BOARD_RUNNING")
            if temperature <= target:
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(f"board did not cool below {target:.1f} C")
            time.sleep(5)

    def setup_board(self) -> tuple[dict[str, Any], dict[str, Any], str]:
        board_seed = self.spec["board_seed"]
        deployment = load_json(Path(board_seed["deployment_json"]))
        seed_manifest = load_json(Path(board_seed["runtime_manifest"]))
        remote_seed = deployment["remote_dir"]
        remote_work = board_seed["remote_work_root"]
        disk = self.ssh("df -Pk . | tail -1").stdout.strip()
        self.ssh(
            f"test -d {shlex.quote(remote_seed)} && "
            f"test ! -e {shlex.quote(remote_work)} && "
            f"cp -a {shlex.quote(remote_seed)} {shlex.quote(remote_work)}"
        )
        target = self.spec["ssh"]["target"]
        options = self.ssh_options()
        for source, destination in (
            (REMOTE_RUNNER, "hybrid_ort_runtime_runner.py"),
            (REMOTE_BASE_RUNNER, "fullmodule_runtime_runner.py"),
            (REMOTE_EVALUATOR, "evaluate_hybrid_ort_local_dataset.py"),
        ):
            stream_upload(source, f"{remote_work}/{destination}", options, target)
        # The formal seed used ORT for four bottleneck modules. Materialize
        # their selected C1 HEFs once so every candidate below is all-Hailo.
        for name in SEQUENCE:
            if name == "bn_g_s":
                continue
            record = self.base_selection["modules"][name]["hef"]
            source = Path(record["path"])
            artifact_record(source, record["sha256"])
            stream_upload(source, f"{remote_work}/hef/{name}.hef", options, target)
        self.state["board"].update({
            "status": "RUNNING", "remote_seed": remote_seed,
            "remote_work_root": remote_work, "remote_disk_snapshot": disk,
        })
        self.write_state(stage="HOST_AND_BOARD_RUNNING")
        return deployment, seed_manifest, remote_work

    def all_hailo_manifest(
        self, seed: dict[str, Any], candidate: dict[str, Any]
    ) -> dict[str, Any]:
        manifest = copy.deepcopy(seed)
        manifest["arm"] = f"bn_g_s_{candidate['name']}_all_hailo"
        manifest["selection_candidate_fingerprint"] = candidate["candidate_fingerprint"]
        manifest["cpu_modules"] = []
        manifest["hailo_modules"] = list(SEQUENCE)
        manifest["precision_policy"] = "candidate_bn_g_s_with_frozen_C1_other_hefs"
        by_name = {item["name"]: item for item in manifest["modules"]}
        onnx_manifest = load_json(self.handoff / "onnx" / "manifest.json")
        onnx_by_name = {item["name"]: item for item in onnx_manifest["modules"]}
        modules = []
        for name in SEQUENCE:
            source = onnx_by_name[name]
            selected = self.base_selection["modules"][name]
            hef = selected["hef"]
            contexts = selected.get("num_contexts")
            if name == "bn_g_s":
                hef = candidate["hef"]
                contexts = candidate["num_contexts"]
            modules.append({
                "name": name,
                "input_names": source["input_names"],
                "output_names": source["output_names"],
                "in_shapes_nchw": source["in_shapes_nchw"],
                "out_shapes_nchw": source["out_shapes_nchw"],
                "engine": "hailort_float32_vstream",
                "hef_filename": f"hef/{name}.hef",
                "hef_sha256": hef["sha256"],
                "hef_bytes": hef["bytes"],
                "num_contexts": contexts,
            })
        manifest["modules"] = modules
        if tuple(item["name"] for item in modules) != SEQUENCE or set(by_name) != set(SEQUENCE):
            raise ValueError("seed runtime module sequence mismatch")
        return manifest

    def launch_remote_board_stage(
        self, remote_work: str, manifest_name: str, candidate: str, stage: dict[str, Any],
        deployment: dict[str, Any],
    ) -> dict[str, str]:
        label = stage["label"]
        result = f"{remote_work}/results/{candidate}_{label}.json"
        progress = f"{remote_work}/results/{candidate}_{label}.progress.json"
        log = f"{remote_work}/logs/{candidate}_{label}.log"
        pid_file = f"{remote_work}/logs/{candidate}_{label}.pid"
        command = [
            deployment["remote_ort_python"],
            f"{remote_work}/evaluate_hybrid_ort_local_dataset.py",
            "--manifest", f"{remote_work}/{manifest_name}",
            "--dataset-root", deployment["dataset"]["remote_root"],
            "--out", result,
            "--mode", str(stage["mode"]),
            "--limit", str(stage["limit"]),
            "--warmup", str(stage["warmup"]),
            "--interval-seconds", str(stage["interval_seconds"]),
            "--progress-every", "8",
        ]
        launch_body = (
            f"nohup setsid {shlex.join(command)} > {shlex.quote(log)} 2>&1 < /dev/null & "
            f"pid=$!; echo $pid > {shlex.quote(pid_file)}; echo $pid"
        )
        remote = (
            f"mkdir -p {shlex.quote(remote_work + '/results')} "
            f"{shlex.quote(remote_work + '/logs')} && "
            f"test ! -e {shlex.quote(result)} && test ! -e {shlex.quote(pid_file)} && "
            f"sh -c {shlex.quote(launch_body)}"
        )
        completed = self.ssh(remote)
        pid = completed.stdout.strip().splitlines()[-1]
        if not pid.isdigit():
            raise RuntimeError(f"invalid remote PID: {completed.stdout!r}")
        return {"result": result, "progress": progress, "log": log,
                "pid_file": pid_file, "pid": pid}

    def fetch_remote_text(self, path: str) -> str:
        return self.ssh(f"cat {shlex.quote(path)}").stdout

    def poll_remote_board_stage(
        self, candidate: str, label: str, remote: dict[str, str]
    ) -> dict[str, Any]:
        poll_seconds = int(self.spec["scheduler"]["board_poll_seconds"])
        while True:
            self.refresh_host_jobs()
            command = (
                f"if test -f {shlex.quote(remote['result'])}; then echo COMPLETE; "
                f"elif test -f {shlex.quote(remote['pid_file'])} && "
                f"kill -0 $(cat {shlex.quote(remote['pid_file'])}) 2>/dev/null; "
                f"then echo RUNNING; else echo FAILED; fi; "
                f"if test -f {shlex.quote(remote['progress'])}; then "
                f"cat {shlex.quote(remote['progress'])}; fi"
            )
            probe = self.ssh(command)
            lines = probe.stdout.splitlines()
            status = lines[0].strip() if lines else "UNKNOWN"
            progress = None
            if len(lines) > 1:
                try:
                    progress = json.loads("\n".join(lines[1:]))
                except json.JSONDecodeError:
                    progress = {"raw_tail": "\n".join(lines[-8:])}
            self.state["board"].update({
                "candidate": candidate, "stage": label,
                "remote_pid": int(remote["pid"]), "remote_stage_status": status,
                "progress_record": progress,
            })
            self.state["candidates"][candidate]["board"]["stages"][label] = {
                "status": status, "remote_pid": int(remote["pid"]),
                "progress_record": progress,
            }
            self.write_state(stage="HOST_AND_BOARD_RUNNING")
            if status == "COMPLETE":
                result = json.loads(self.fetch_remote_text(remote["result"]))
                return result
            if status == "FAILED":
                try:
                    log_tail = self.ssh(f"tail -100 {shlex.quote(remote['log'])}").stdout
                except Exception as exc:
                    log_tail = f"failed to fetch remote log: {exc}"
                raise RuntimeError(f"remote board stage failed: {log_tail}")
            time.sleep(poll_seconds)

    def run_board_lane(self) -> None:
        try:
            deployment, seed, remote_work = self.setup_board()
        except Exception as exc:
            self.state["board"].update({"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"})
            for candidate in self.candidates:
                self.state["candidates"][candidate["name"]]["board"]["status"] = "SKIP_BOARD_SETUP_FAIL"
            self.write_state()
            return
        target = self.spec["ssh"]["target"]
        options = self.ssh_options()
        manifest_name = "runtime_manifest_bn_g_s_candidate.json"
        for candidate in self.candidates:
            name = candidate["name"]
            board_state = self.state["candidates"][name]["board"]
            board_state["status"] = "RUNNING"
            candidate_root = self.out / "board" / name
            candidate_root.mkdir(parents=True, exist_ok=True)
            try:
                stream_upload(Path(candidate["hef"]["path"]),
                              f"{remote_work}/hef/bn_g_s.hef", options, target)
                manifest = self.all_hailo_manifest(seed, candidate)
                local_manifest = candidate_root / "runtime_manifest.json"
                atomic_json(local_manifest, manifest)
                stream_upload(local_manifest, f"{remote_work}/{manifest_name}", options, target)
                remote_sha = self.ssh(
                    f"sha256sum {shlex.quote(remote_work + '/hef/bn_g_s.hef')} | cut -d' ' -f1"
                ).stdout.strip()
                if remote_sha != candidate["hef"]["sha256"]:
                    raise ValueError(f"remote bn_g_s HEF SHA mismatch: {remote_sha}")
                board_state["remote_hef_sha256"] = remote_sha
                for stage in self.spec["scheduler"]["board_stages"]:
                    label = stage["label"]
                    self.wait_board_cool(name, label)
                    launched = self.launch_remote_board_stage(
                        remote_work, manifest_name, name, stage, deployment
                    )
                    result = self.poll_remote_board_stage(name, label, launched)
                    local_result = candidate_root / f"{label}.json"
                    atomic_json(local_result, result)
                    try:
                        (candidate_root / f"{label}.log").write_text(
                            self.fetch_remote_text(launched["log"]), encoding="utf-8"
                        )
                    except Exception:
                        pass
                    passed = result.get("status") == "PASS"
                    board_state["stages"][label] = {
                        "status": "PASS" if passed else "FAIL", "result": str(local_result),
                        "remote_result": launched["result"],
                    }
                    self.write_state()
                    if not passed:
                        raise RuntimeError(f"{name}:{label} result is not PASS")
                board_state["status"] = "PASS"
            except Exception as exc:
                board_state.update({"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"})
            self.write_state()
        self.state["board"]["status"] = "PASS" if all(
            self.state["candidates"][item["name"]]["board"]["status"] == "PASS"
            for item in self.candidates
        ) else "COMPLETE_WITH_FAILURES"
        self.write_state()

    def wait_host_lane(self) -> None:
        while self.host_processes or self.pending_host:
            self.refresh_host_jobs()
            if self.host_processes:
                time.sleep(10)

    @staticmethod
    def semantic_summary(result: dict[str, Any], *, host: bool) -> dict[str, Any]:
        semantic = result.get("semantic", {})
        if host:
            semantic = semantic.get("sdk_quantized_chain", {})
            bpp = result.get("bitstream", {}).get("sdk_quantized", {}).get("mean")
        else:
            bpp = result.get("bitstream", {}).get("mean_bpp")
        return {
            "mIoU": semantic.get("mIoU"),
            "fixed_19_large_object_macro_iou": semantic.get("fixed_19_large_object_macro_iou"),
            "pixel_accuracy": semantic.get("pixel_acc", semantic.get("pixel_accuracy")),
            "bpp": bpp,
        }

    def build_summary(self) -> dict[str, Any]:
        rows: dict[str, Any] = {}
        for candidate in self.candidates:
            name = candidate["name"]
            row: dict[str, Any] = {
                "description": candidate["description"],
                "result_json": candidate["result_json"],
                "hef": candidate["hef"],
                "n_calib": candidate["n_calib"],
                "optimization_level": candidate["optimization_level"],
                "host_status": self.state["candidates"][name]["host"]["status"],
                "board_status": self.state["candidates"][name]["board"]["status"],
            }
            host_path = self.out / "host" / name / "screen64.json"
            board_accuracy = self.out / "board" / name / "accuracy64.json"
            board_latency = self.out / "board" / name / "latency8.json"
            if host_path.is_file():
                row["host_screen64"] = self.semantic_summary(load_json(host_path), host=True)
            if board_accuracy.is_file():
                row["board_accuracy64"] = self.semantic_summary(load_json(board_accuracy), host=False)
            if board_latency.is_file():
                timing = load_json(board_latency).get("timing", {})
                row["board_latency8"] = {
                    "bn_g_s_infer_wall_ms": timing.get("bn_g_s.infer_wall_ms"),
                    "bn_g_s_activation_plus_infer_ms": timing.get("bn_g_s.activation_plus_infer_ms"),
                    "board_total_pipeline_ms": timing.get("board_total_pipeline_ms"),
                    "thermal_start_c": timing.get("thermal_start_c"),
                    "thermal_end_c": timing.get("thermal_end_c"),
                }
            rows[name] = row
        return {
            "schema": "semantic_sensing.bn_g_s_hailo_campaign_summary.v1",
            "status": "COMPLETE",
            "selection": None,
            "selection_policy": "manual_review_only",
            "formal_976": "not_started_by_design",
            "gate_policy": self.spec["gate_policy"],
            "candidates": rows,
        }

    def run(self) -> None:
        preflight = self.validate(require_out_absent=True)
        self.out.mkdir(parents=True, exist_ok=False)
        self.state = {
            "schema": "semantic_sensing.bn_g_s_hailo_campaign_state.v1",
            "status": "RUNNING", "stage": "SETUP", "pid": os.getpid(),
            "started_unix_seconds": time.time(), "out_root": str(self.out),
            "spec": {"path": str(self.spec_path), "sha256": sha256(self.spec_path)},
            "preflight": preflight,
            "gate_policy": self.spec["gate_policy"],
            "board": {"status": "PENDING"},
            "candidates": {
                item["name"]: {
                    "description": item["description"],
                    "host": {"status": "PENDING"},
                    "board": {"status": "PENDING", "stages": {}},
                }
                for item in self.candidates
            },
        }
        atomic_json(self.out / "campaign_manifest.json", {
            "schema": "semantic_sensing.bn_g_s_hailo_campaign_manifest.v1",
            "spec": self.state["spec"], "preflight": preflight,
            "gate_policy": self.spec["gate_policy"],
            "candidate_order": [item["name"] for item in self.candidates],
        })
        self.write_state()
        try:
            self.freeze_manifests()
            self.pending_host = list(self.candidates)
            self.fill_host_queue()
            self.run_board_lane()
            self.wait_host_lane()
            summary = self.build_summary()
            atomic_json(self.out / "summary.json", summary)
            failures = [name for name, item in self.state["candidates"].items()
                        if item["host"]["status"] != "PASS" or item["board"]["status"] != "PASS"]
            self.write_state(
                status="COMPLETE" if not failures else "COMPLETE_WITH_FAILURES",
                stage="COMPLETE", failures=failures,
                summary=str(self.out / "summary.json"),
                finished_unix_seconds=time.time(),
            )
        except BaseException as exc:
            for job in self.host_processes.values():
                if job["process"].poll() is None:
                    job["process"].terminate()
                job["handle"].close()
            self.write_state(
                status="FAILED_INFRASTRUCTURE", stage="FAILED",
                error=f"{type(exc).__name__}: {exc}", finished_unix_seconds=time.time(),
            )
            raise


def print_status(out_root: Path) -> None:
    state = load_json(out_root / "campaign_state.json")
    pid = state.get("pid")
    state["local_orchestrator_alive"] = bool(
        isinstance(pid, int) and Path(f"/proc/{pid}").exists()
    )
    print(json.dumps(state, indent=2, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    sub = value.add_subparsers(dest="command", required=True)
    for name in ("validate", "plan", "run"):
        item = sub.add_parser(name)
        item.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
        if name == "run":
            item.add_argument("--execute", action="store_true")
    status = sub.add_parser("status")
    status.add_argument("--out-root", type=Path, required=True)
    return value


def main() -> int:
    args = parser().parse_args()
    if args.command == "status":
        print_status(args.out_root.resolve())
        return 0
    campaign = Campaign(args.spec)
    if args.command in {"validate", "plan"}:
        preflight = campaign.validate(require_out_absent=True)
        print(json.dumps({
            "status": "PASS", "out_root": str(campaign.out), "preflight": preflight,
            "host_lane": "one screen64 job per eligible GPU",
            "board_lane": "serial all-Hailo preflight1/latency8/accuracy64",
            "candidates": [item["name"] for item in campaign.candidates],
            "automatic_selection": False,
        }, indent=2, sort_keys=True))
        return 0
    if not args.execute:
        raise PermissionError("run requires --execute")
    campaign.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
