#!/usr/bin/env python3
"""Detached, resumable F0/F1 bn_g_s input-clipping feasibility campaign."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from board_eval.deploy import stream_upload  # noqa: E402
from hailo_deploy.emulate import _build_native_feed, _match_outputs  # noqa: E402
from hailo_deploy.gates.compile_real import compile_quantized  # noqa: E402
from hailo_deploy.io.hash import sha256  # noqa: E402
from hailo_deploy.io.metrics import metric, summarize  # noqa: E402
from hailo_deploy.onnx_io import nhwc_to_nchw, run_onnxruntime, signature  # noqa: E402


CHECKPOINT_SHA = "13dc6c4b3a3ea630af6d2d547808137be678bd909f9789f5594b7bdc76c36b90"
ONNX_SHA = "b292687ee569fe2f523ae3704cfca92bc84e4bc7e4041ea5c9fa41ac8c0e0c4f"
ONNX = Path(
    "/local/shared_with_docker/hailo_full_candidate/"
    "convbn-gfa-gs-3a3ea630af6d2d547808137be678bd909f9789f5594b7bdc76c36b90/"
    "onnx/bn_g_s.onnx"
)
CALIB = Path(
    "/local/shared_with_docker/hailo_full_candidate/"
    "convbn-gfa-gs-ade-train4096-20260730/calibration/bn_g_s"
)
BASE_MANIFEST = Path(
    "/local/shared_with_docker/deplyment-demo/artifacts/fused/"
    "board-hybrid-ort-formal-20260805T030725Z/runtime_manifests/"
    "runtime_manifest_gs_only_hailo_fp32.json"
)
REMOTE_SEED = ".local/share/semantic-sensing-board/hybrid-ort/board-hybrid-ort-formal-20260805T030725Z"
REMOTE_DATASET = ".local/share/semantic-sensing-board/datasets/convbn-gfa-gs-eval976-13dc6c4b3a3ea630"
REMOTE_PYTHON = os.environ.get(
    "HAILO_REMOTE_PYTHON",
    ".local/share/semantic-sensing-board/venv-ort128/bin/python",
)
SSH_TARGET = os.environ.get("HAILO_SSH_TARGET", "user@hailo-board")
IDENTITY = Path(
    os.environ.get("HAILO_SSH_IDENTITY_FILE", "~/.ssh/id_ed25519")
).expanduser()
KNOWN_HOSTS = Path(
    os.environ.get("HAILO_SSH_KNOWN_HOSTS", "~/.ssh/known_hosts")
).expanduser()
PROXY = os.environ.get("HAILO_SSH_PROXY_COMMAND")


def atomic_write(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def run(command: list[str], *, cwd: Path = ROOT, env: dict | None = None) -> None:
    print(json.dumps({"command": shlex.join(command), "cwd": str(cwd)}), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def ssh_options() -> list[str]:
    options = [
        "-i", str(IDENTITY), "-o", "IdentitiesOnly=yes",
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=25", "-o", "ServerAliveInterval=10",
        "-o", "ServerAliveCountMax=3", "-o", "StrictHostKeyChecking=yes",
        "-o", f"UserKnownHostsFile={KNOWN_HOSTS}",
    ]
    if PROXY:
        options.extend(["-o", f"ProxyCommand={PROXY}"])
    return options


def ssh(command: str) -> str:
    completed = subprocess.run(
        ["ssh", *ssh_options(), SSH_TARGET, command], check=True,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60,
    )
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="", flush=True)
    return completed.stdout


def ssh_readonly(command: str, *, attempts: int = 3, timeout_seconds: float = 35.0) -> str:
    """Retry only commands that are explicitly read-only and safe to repeat."""
    errors = []
    for attempt in range(1, attempts + 1):
        try:
            completed = subprocess.run(
                ["ssh", *ssh_options(), SSH_TARGET, command], check=True,
                text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                timeout=timeout_seconds,
            )
            if completed.stderr:
                print(completed.stderr, file=sys.stderr, end="", flush=True)
            return completed.stdout
        except (subprocess.SubprocessError, OSError) as exc:
            errors.append(f"attempt {attempt}/{attempts}: {type(exc).__name__}: {exc}")
            print(errors[-1], file=sys.stderr, flush=True)
            if attempt != attempts:
                time.sleep(3 * attempt)
    raise RuntimeError("bounded read-only SSH retries exhausted: " + " | ".join(errors))


def update_state(root: Path, state: dict, **updates) -> None:
    state.update(updates)
    state.update({"pid": os.getpid(), "ppid": os.getppid(), "pgid": os.getpgid(0),
                  "session_id": os.getsid(0)})
    state["updated_unix_seconds"] = time.time()
    atomic_write(root / "campaign_state.json", state)


def exact_percentiles() -> tuple[float, float, dict]:
    files = sorted((CALIB / "latent_y_hat").glob("*.npy"))
    if len(files) != 4096:
        raise ValueError(f"expected 4096 calibration tensors, got {len(files)}")
    arrays = np.stack([np.load(path, allow_pickle=False) for path in files], axis=0)
    if arrays.dtype != np.float32 or arrays.shape != (4096, 16, 16, 192):
        raise ValueError(f"unexpected calibration batch {arrays.shape}/{arrays.dtype}")
    values = np.quantile(arrays, [0.001, 0.999], method="linear")
    low, high = float(values[0]), float(values[1])
    clipped = np.count_nonzero((arrays < low) | (arrays > high))
    audit = {
        "method": "numpy.quantile linear over every activation value",
        "sample_count": 4096, "element_count": int(arrays.size),
        "q0_1": low, "q99_9": high,
        "minimum": float(arrays.min()), "maximum": float(arrays.max()),
        "clipped_element_count": int(clipped),
        "clipped_fraction": float(clipped / arrays.size),
        "ideal_uint8_step": float((high - low) / 255.0),
    }
    return low, high, audit


def parse_once(root: Path, state: dict, env: dict) -> Path:
    parse_root = root / "parse"
    har = parse_root / "har" / "bn_g_s_hailo_model.har"
    result = parse_root / "results" / "pipeline.json"
    if not (har.is_file() and result.is_file()):
        if parse_root.exists() and any(parse_root.iterdir()):
            raise RuntimeError("incomplete parse directory exists; refusing implicit overwrite")
        run([
            sys.executable, "pipelines/parse_native.py", "--profile", "fused", "--only", "bn_g_s",
            "--onnx-dir", str(ONNX.parent), "--artifacts-dir", str(parse_root),
        ], env=env)
    if not har.is_file() or not result.is_file():
        raise RuntimeError("parse/NATIVE did not produce required artifacts")
    cpu_result = parse_root / "results" / "native_cpu_fp32.json"
    if not cpu_result.is_file():
        cpu_env = dict(env)
        cpu_env["CUDA_VISIBLE_DEVICES"] = ""
        run([
            sys.executable, "pipelines/check_native_har_cpu.py", "--har", str(har),
            "--onnx", str(ONNX), "--out", str(cpu_result),
        ], env=cpu_env)
    cpu = json.loads(cpu_result.read_text(encoding="utf-8"))
    if cpu.get("status") != "PASS":
        raise RuntimeError(f"CPU/FP32 NATIVE recheck did not pass: {cpu}")
    update_state(root, state, stage="parse_native_cpu_pass", parse_har_sha256=sha256(har),
                 native_cpu_max_abs_diff=cpu.get("max_abs_diff"))
    return har


def write_scripts(root: Path, low: float, high: float, f1_mode: str) -> dict[str, Path]:
    scripts = root / "model_scripts"
    scripts.mkdir(exist_ok=True)
    common = (
        "# Fixed actual x_l low-spatial output A16 policy.\n"
        "quantization_param(bn_g_s/conv3, precision_mode=a16_w16)\n"
        "quantization_param(bn_g_s/output_layer1, precision_mode=a16_w16)\n"
    )
    paths = {"F0_minmax": scripts / "F0_minmax.alls", "F1_q0p1_q99p9": scripts / "F1_q0p1_q99p9.alls"}
    if not paths["F0_minmax"].exists():
        paths["F0_minmax"].write_text(common, encoding="utf-8")
    if not paths["F1_q0p1_q99p9"].exists():
        if f1_mode == "manual_activation_clipping":
            control = (
                "pre_quantization_optimization(activation_clipping, "
                "layers=[bn_g_s/input_layer1], mode=manual, "
                f"clipping_values=[{low:.17g}, {high:.17g}])\n"
            )
        elif f1_mode == "first_consumer_force_range_in":
            control = (
                "# DFC 3.33 rejects activation_clipping on the activation-less input layer.\n"
                "# The exact first/only consumer is bn_g_s/conv1 input index 0.\n"
                "quantization_param([bn_g_s/conv1], "
                f"force_range_in=[{low:.17g}, {high:.17g}], force_range_index=0)\n"
            )
        else:
            raise ValueError(f"unsupported F1 range-control mode: {f1_mode}")
        paths["F1_q0p1_q99p9"].write_text(common + control, encoding="utf-8")
    return paths


def hardlink(source: Path, destination: Path) -> dict:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    os.link(source, destination)
    source_stat = source.stat()
    destination_stat = destination.stat()
    if (source_stat.st_dev, source_stat.st_ino) != (destination_stat.st_dev, destination_stat.st_ino):
        raise RuntimeError(f"hardlink identity mismatch: {source} -> {destination}")
    return {
        "source": str(source), "destination": str(destination),
        "sha256": sha256(destination), "device": destination_stat.st_dev,
        "inode": destination_stat.st_ino,
    }


def prepare_continuation(root: Path, source: Path) -> None:
    """Reuse frozen PASS artifacts without mutating the prior campaign root."""
    lineage = root / "continuation_lineage.json"
    if lineage.is_file():
        return
    if any((root / relative).exists() for relative in ("parse", "ptq", "references", "range_audit.json")):
        raise RuntimeError("partial continuation layout exists; refusing implicit overwrite")
    links = []
    mappings = (
        ("parse/har/bn_g_s_hailo_model.har", "parse/har/bn_g_s_hailo_model.har"),
        ("parse/results/pipeline.json", "parse/results/pipeline.json"),
        ("parse/results/native_cpu_fp32.json", "parse/results/native_cpu_fp32.json"),
        ("parse/reports/pipeline.md", "parse/reports/pipeline.md"),
        ("ptq/F0_minmax/bn_g_s_l2.model_script.alls", "ptq/F0_minmax/bn_g_s_l2.model_script.alls"),
        ("ptq/F0_minmax/bn_g_s_l2_quantized.har", "ptq/F0_minmax/bn_g_s_l2_quantized.har"),
        ("ptq/F0_minmax/result.json", "ptq/F0_minmax/result.json"),
        ("range_audit.json", "range_audit.json"),
        ("precompile_report.json", "references/previous_precompile_report.json"),
    )
    for source_relative, destination_relative in mappings:
        links.append(hardlink(source / source_relative, root / destination_relative))
    atomic_write(lineage, {
        "kind": "bn_g_s_clip_force_range_continuation_lineage",
        "source_root": str(source),
        "source_policy": "immutable; reused files are verified hardlinks",
        "reused_arm": "F0_minmax",
        "new_arm": "F1_q0p1_q99p9",
        "links": links,
    })


def optimize_arm(root: Path, arm: str, script: Path, har: Path, env: dict) -> dict:
    out = root / "ptq" / arm
    result = out / "result.json"
    if result.is_file():
        data = json.loads(result.read_text(encoding="utf-8"))
        if data.get("status") != "PASS":
            raise RuntimeError(f"existing PTQ result is not PASS: {result}")
        return data
    if out.exists():
        raise RuntimeError(f"incomplete PTQ arm exists; refusing overwrite: {out}")
    run([
        sys.executable, "pipelines/module_ptq.py", "calib-root", "--module", "bn_g_s",
        "--onnx", str(ONNX), "--parse-har", str(har), "--calib-root", str(CALIB),
        "--out-dir", str(out), "--n-calib", "4096", "--opt-level", "2",
        "--eval-samples", "8", "--extra-alls", str(script), "--skip-compile",
        "--expect-onnx-sha", "--expect-checkpoint-sha", CHECKPOINT_SHA,
    ], env=env)
    data = json.loads(result.read_text(encoding="utf-8"))
    if data.get("status") != "PASS":
        raise RuntimeError(f"PTQ failed for {arm}: {data.get('error')}")
    return data


def precompile_report(root: Path, har: Path, audit: dict, results: dict, started: float) -> dict:
    report_path = root / "precompile_report.json"
    if report_path.is_file():
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        if payload.get("status") != "CONTINUE_TO_COMPILE":
            raise RuntimeError("existing precompile report is not a continue decision")
        return payload
    payload = {
        "kind": "bn_g_s_clip_feasibility_precompile",
        "status": "CONTINUE_TO_COMPILE", "created_unix_seconds": time.time(),
        "elapsed_seconds": time.monotonic() - started,
        "identity": {
            "checkpoint_sha256": CHECKPOINT_SHA, "onnx_sha256": sha256(ONNX),
            "parse_har_sha256": sha256(har), "calibration_meta_sha256": sha256(CALIB / "meta.json"),
            "calibration_sample_count": 4096, "dfc": "3.33.0", "hailort_host": "4.23.0",
            "hw_arch": "hailo8", "optimization_level": 2, "compression_level": 0,
            "batch_size": 1, "x_h_format": "UINT8", "x_l_format": "UINT16",
        },
        "native_parity": {
            "gpu_probe": json.loads((root / "parse/results/pipeline.json").read_text(encoding="utf-8")),
            "cpu_fp32_recheck": json.loads((root / "parse/results/native_cpu_fp32.json").read_text(encoding="utf-8")),
            "interpretation": "GPU TF32 probe mismatch is superseded by mandatory CPU/FP32 same-HAR recheck",
        },
        "range": audit,
        "range_control_substitution": {
            "original_rejected_command": (
                "pre_quantization_optimization(activation_clipping, "
                "layers=[bn_g_s/input_layer1], mode=manual, clipping_values=[LOW, HIGH])"
            ),
            "original_error": "Layer bn_g_s/input_layer1 has no activation",
            "static_graph_check": {
                "input_layer": "bn_g_s/input_layer1",
                "input_shape_nhwc": [-1, 16, 16, 192],
                "exact_first_and_only_consumer": "bn_g_s/conv1",
                "consumer_input_index": 0,
            },
            "replacement_command": (
                "quantization_param([bn_g_s/conv1], force_range_in=[LOW, HIGH], "
                "force_range_index=0)"
            ),
            "experimental_intent_unchanged": True,
        },
        "continuation_lineage": json.loads(
            (root / "continuation_lineage.json").read_text(encoding="utf-8")
        ) if (root / "continuation_lineage.json").is_file() else None,
        "arms": {
            name: {
                "model_script": data["model_script"], "quantized_har": data["quantized_har"],
                "sdk_quantized_calibration_tail8": data["quantized_check"],
                "compile": data["compile"],
            } for name, data in results.items()
        },
        "continue_reason": "both arms optimized and SDK_QUANTIZED checks passed; compile next",
        "policy": {"diagnostic_only": True, "full976_not_run": True},
    }
    atomic_write(report_path, payload)
    return payload


def compile_arm(root: Path, arm: str) -> dict:
    out = root / "ptq" / arm
    result_path = out / "compile_result.json"
    if result_path.is_file():
        data = json.loads(result_path.read_text(encoding="utf-8"))
        if data.get("status") != "PASS":
            raise RuntimeError(f"existing compile result is not PASS: {arm}")
        return data
    from hailo_sdk_client import ClientRunner
    quantized = out / "bn_g_s_l2_quantized.har"
    runner = ClientRunner(har=str(quantized))
    compiled = compile_quantized(
        runner, hef_path=out / "bn_g_s_l2.hef",
        compiled_har=out / "bn_g_s_l2_compiled.har",
        auto_alls=out / "bn_g_s_l2.auto.alls",
    )
    data = {"status": "PASS", **compiled}
    atomic_write(result_path, data)
    return data


def hef_static(path: Path) -> dict:
    from hailo_platform import HEF
    hef = HEF(str(path))
    def info(value) -> dict:
        quant = value.quant_info
        return {
            "name": value.name, "shape_hwc": list(value.shape),
            "format": value.format.type.name,
            "quantization": {
                "qp_scale": float(quant.qp_scale), "qp_zp": float(quant.qp_zp),
                "limvals_min": float(quant.limvals_min), "limvals_max": float(quant.limvals_max),
            },
        }
    return {
        "sha256": sha256(path), "bytes": path.stat().st_size,
        "network_groups": list(hef.get_network_group_names()),
        "inputs": [info(x) for x in hef.get_input_vstream_infos()],
        "outputs": [info(x) for x in hef.get_output_vstream_infos()],
    }


def validate_static(root: Path, low: float, high: float, compiled: dict) -> dict:
    rows = {name: hef_static(Path(data["hef"]["path"])) for name, data in compiled.items()}
    for name, row in rows.items():
        if len(row["network_groups"]) != 1 or len(row["inputs"]) != 1 or len(row["outputs"]) != 2:
            raise ValueError(f"{name}: HEF topology mismatch")
        inp = row["inputs"][0]
        if inp["shape_hwc"] != [16, 16, 192] or inp["format"] != "UINT8":
            raise ValueError(f"{name}: HEF input contract mismatch")
        formats = {(tuple(x["shape_hwc"]), x["format"]) for x in row["outputs"]}
        if formats != {((64, 64, 128), "UINT8"), ((32, 32, 256), "UINT16")}:
            raise ValueError(f"{name}: actual output shape/format mismatch: {formats}")
    f0q = rows["F0_minmax"]["inputs"][0]["quantization"]
    f1q = rows["F1_q0p1_q99p9"]["inputs"][0]["quantization"]
    tolerance = max(2.0 * f1q["qp_scale"], 1e-4)
    if abs(f1q["limvals_min"] - low) > tolerance or abs(f1q["limvals_max"] - high) > tolerance:
        raise ValueError(f"F1 clipping range did not reach HEF: requested={[low, high]} actual={f1q}")
    if not f1q["qp_scale"] < f0q["qp_scale"]:
        raise ValueError("F1 input quantization scale did not improve")
    atomic_write(root / "hef_static_report.json", {"status": "PASS", "arms": rows})
    return rows


def launch_remote(remote_root: str, label: str, command: list[str], *,
                  result_override: str | None = None,
                  progress_override: str | None = None) -> dict:
    stage = f"{remote_root}/{label}"
    result = result_override or f"{stage}/result.json"
    progress = progress_override or f"{stage}/progress.json"
    log = f"{stage}/run.log"
    pidfile = f"{stage}/run.pid"
    remote = (
        f"mkdir -p {shlex.quote(stage)}; test ! -e {shlex.quote(pidfile)} || exit 9; "
        f"nohup setsid {shlex.join(command)} >{shlex.quote(log)} 2>&1 </dev/null & "
        f"pid=$!; echo $pid >{shlex.quote(pidfile)}; sleep 1; "
        f"ps -o pid=,ppid=,pgid=,sid=,stat=,args= -p $pid"
    )
    probe = ssh(remote).strip()
    pid = probe.split()[0]
    return {"label": label, "pid": int(pid), "pid_file": pidfile, "result": result,
            "progress": progress, "log": log, "ps": probe}


def poll_remote(root: Path, state: dict, job: dict, timeout: float = 1800.0) -> None:
    deadline = time.monotonic() + timeout
    while True:
        output = ssh(
            f"if test -f {shlex.quote(job['result'])}; then echo COMPLETE; "
            f"elif kill -0 $(cat {shlex.quote(job['pid_file'])}) 2>/dev/null; then echo RUNNING; "
            "else echo FAILED; fi; "
            f"test ! -f {shlex.quote(job['progress'])} || cat {shlex.quote(job['progress'])}"
        )
        lines = output.splitlines()
        status = lines[0] if lines else "UNKNOWN"
        update_state(root, state, stage=f"remote_{job['label']}", remote_job=job,
                     remote_stage_status=status, remote_probe=output[-2000:])
        if status == "COMPLETE":
            return
        if status != "RUNNING" or time.monotonic() > deadline:
            tail = ssh(f"tail -100 {shlex.quote(job['log'])} 2>/dev/null || true")
            raise RuntimeError(f"remote {job['label']} {status}:\n{tail}")
        time.sleep(10)


def download_remote_tree(remote_path: str, local_parent: Path) -> Path:
    local_parent.mkdir(parents=True, exist_ok=True)
    archive = local_parent / ".remote_bundle.tar.gz"
    remote_parent, name = remote_path.rsplit("/", 1)
    with archive.open("wb") as handle:
        completed = subprocess.run(
            ["ssh", *ssh_options(), SSH_TARGET,
             f"tar -C {shlex.quote(remote_parent)} -czf - {shlex.quote(name)}"],
            stdout=handle, stderr=subprocess.PIPE,
        )
    if completed.returncode:
        raise RuntimeError(completed.stderr.decode("utf-8", errors="replace"))
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(local_parent)
    archive.unlink()
    return local_parent / name


def sdk32(root: Path, bundle: Path) -> dict:
    report_path = root / "sdk32.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") != "PASS" or report.get("sample_count") != 32:
            raise RuntimeError("existing SDK32 report is incomplete")
        return report
    from hailo_sdk_client import ClientRunner, InferenceContext
    sig = signature(ONNX)
    report = {"kind": "bn_g_s_clip_sdk_quantized_same_yhat", "status": "PASS", "sample_count": 32,
              "policy": {"diagnostic_only": True, "semantic_ranking": False}, "arms": {}}
    bundle_result = json.loads((bundle / "result.json").read_text(encoding="utf-8"))
    for arm in ("F0_minmax", "F1_q0p1_q99p9"):
        runner = ClientRunner(har=str(root / "ptq" / arm / "bn_g_s_l2_quantized.har"))
        per_output = {"x_h_hat": [], "x_l_hat": []}
        notes: list[str] = []
        with runner.infer_context(InferenceContext.SDK_QUANTIZED) as ctx:
            for index in bundle_result["val_indices"]:
                y_hwc = np.load(bundle / "latent_y_hat" / f"{index}.npy", allow_pickle=False)
                sample = {"latent_y_hat": y_hwc.transpose(2, 0, 1)[None].astype(np.float32)}
                reference = run_onnxruntime(ONNX, sample)
                feed = _build_native_feed(runner, sig, sample, notes)
                raw = runner.infer(ctx, feed)
                pairs = _match_outputs(runner, reference, raw, notes)
                for name, (ref, got) in pairs.items():
                    per_output[name].append(metric(ref, got))
        report["arms"][arm] = {"outputs": {
            name: summarize(rows) for name, rows in per_output.items()
        }, "notes": sorted(set(notes))}
    atomic_write(report_path, report)
    return report


def board(root: Path, state: dict, static: dict) -> dict:
    run_tag = root.name
    remote_root = f".local/share/semantic-sensing-board/hybrid-ort/{run_tag}"
    board_leftovers = ssh_readonly(
        "pgrep -af '[c]apture_bn_g_s_yhat_bundle.py|[e]valuate_bn_g_s_clip_bundle.py|"
        "[r]un_board_hybrid_ort.py|[r]un_board_formal_eval.py' || true"
    ).strip()
    update_state(root, state, board_prelaunch_processes=board_leftovers)
    if board_leftovers:
        raise RuntimeError(
            "board experiment process exists at prelaunch; refusing to kill or overlap: "
            + board_leftovers
        )
    existing = ssh(f"test ! -e {shlex.quote(remote_root)} && echo ABSENT || echo PRESENT").strip()
    if existing == "ABSENT":
        ssh(
            f"test -d {shlex.quote(REMOTE_SEED)}; mkdir -p {shlex.quote(remote_root)}; "
            f"cp -al {shlex.quote(REMOTE_SEED + '/.')} {shlex.quote(remote_root + '/')}"
        )
    opts = ssh_options()
    for source in (
        ROOT / "board_eval/capture_bn_g_s_yhat_bundle.py",
        ROOT / "board_eval/evaluate_bn_g_s_clip_bundle.py",
        ROOT / "board_eval/bn_g_s_branch_attribution_runtime.py",
    ):
        stream_upload(source, f"{remote_root}/{source.name}", opts, SSH_TARGET)
    base_remote = f"{remote_root}/runtime_manifest_base.json"
    stream_upload(BASE_MANIFEST, base_remote, opts, SSH_TARGET)

    local_bundle = root / "board" / "clip_yhat_bundle"
    remote_bundle = f"{remote_root}/clip_yhat_bundle"
    if not (local_bundle / "result.json").is_file():
        if ssh(f"test -f {shlex.quote(remote_bundle + '/result.json')} && echo PASS || true").strip() != "PASS":
            capture_job_root = f"{remote_root}/clip_yhat_capture_job"
            active = ssh(
                f"test -f {shlex.quote(capture_job_root + '/run.pid')} && "
                f"kill -0 $(cat {shlex.quote(capture_job_root + '/run.pid')}) 2>/dev/null && echo ACTIVE || true"
            ).strip()
            if active != "ACTIVE":
                job = launch_remote(remote_root, "clip_yhat_capture_job", [
                    REMOTE_PYTHON, f"{remote_root}/capture_bn_g_s_yhat_bundle.py",
                    "--manifest", base_remote, "--dataset-root", REMOTE_DATASET,
                    "--out-root", remote_bundle, "--limit", "32",
                ], result_override=remote_bundle + "/result.json",
                   progress_override=remote_bundle + "/progress.json")
            else:
                pid = int(ssh(f"cat {shlex.quote(capture_job_root + '/run.pid')}").strip())
                job = {"label": "clip_yhat_bundle", "pid": pid,
                       "pid_file": capture_job_root + "/run.pid", "result": remote_bundle + "/result.json",
                       "progress": remote_bundle + "/progress.json", "log": capture_job_root + "/run.log"}
            poll_remote(root, state, job)
        local_bundle = download_remote_tree(remote_bundle, root / "board")

    sdk_report = sdk32(root, local_bundle)
    candidate_dir = f"{remote_root}/clip_candidates"
    ssh(f"mkdir -p {shlex.quote(candidate_dir)}")
    arms = []
    for arm in ("F0_minmax", "F1_q0p1_q99p9"):
        hef = root / "ptq" / arm / "bn_g_s_l2.hef"
        destination = f"{candidate_dir}/{arm}.hef"
        stream_upload(hef, destination, opts, SSH_TARGET)
        input_q = static[arm]["inputs"][0]["quantization"]
        arms.append({
            "name": arm,
            "input_quantization": input_q,
            "module_spec": {
                "engine": "hailort_float32_vstream", "hef_filename": f"clip_candidates/{arm}.hef",
                "input_names": ["latent_y_hat"], "name": "bn_g_s",
                "in_shapes_nchw": [[1, 192, 16, 16]],
                "output_names": ["x_h_hat", "x_l_hat"],
                "out_shapes_nchw": [[1, 128, 64, 64], [1, 256, 32, 32]],
            },
        })
    manifest = {
        "kind": "bn_g_s_clip_bundle_board_manifest", "checkpoint_sha256": CHECKPOINT_SHA,
        "ort_threads": 4,
        "fp32": {
            "engine": "ort_cpu_fp32", "name": "bn_g_s", "onnx_filename": "onnx/bn_g_s.onnx",
            "input_names": ["latent_y_hat"], "in_shapes_nchw": [[1, 192, 16, 16]],
            "output_names": ["x_h_hat", "x_l_hat"],
            "out_shapes_nchw": [[1, 128, 64, 64], [1, 256, 32, 32]],
        },
        "arms": arms,
    }
    local_manifest = root / "board" / "board_clip_manifest.json"
    atomic_write(local_manifest, manifest)
    remote_manifest = f"{remote_root}/board_clip_manifest.json"
    stream_upload(local_manifest, remote_manifest, opts, SSH_TARGET)
    result_remote = f"{remote_root}/board_clip32/result.json"
    if ssh(f"test -f {shlex.quote(result_remote)} && echo PASS || true").strip() != "PASS":
        # A failed detached launch is retained as evidence. Each campaign
        # resume gets a fresh stage directory while writing the same canonical
        # result path, so no stale PID file is deleted or overwritten.
        launch_label = f"board_clip32_attempt{int(state.get('resume_count', 0))}"
        job = launch_remote(remote_root, launch_label, [
            REMOTE_PYTHON, f"{remote_root}/evaluate_bn_g_s_clip_bundle.py",
            "--manifest", remote_manifest, "--bundle-root", remote_bundle,
            "--out", result_remote,
        ], result_override=result_remote,
           progress_override=f"{remote_root}/board_clip32/result.progress.json")
        poll_remote(root, state, job)
    output = root / "board" / "board_clip32.json"
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("wb") as handle:
        completed = subprocess.run(["ssh", *opts, SSH_TARGET, "cat", result_remote], stdout=handle)
    if completed.returncode:
        raise RuntimeError("failed to download board result")
    os.replace(temporary, output)
    board_report = json.loads(output.read_text(encoding="utf-8"))
    return {"remote_root": remote_root, "sdk32": sdk_report, "board32": board_report}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--continuation-source", type=Path)
    parser.add_argument(
        "--f1-mode",
        choices=("manual_activation_clipping", "first_consumer_force_range_in"),
        default="manual_activation_clipping",
    )
    args = parser.parse_args()
    root = args.out_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.continuation_source is not None:
        prepare_continuation(root, args.continuation_source.resolve())
    started = time.monotonic()
    state_path = root / "campaign_state.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state.setdefault("process_history", []).append({
            key: state.get(key) for key in ("pid", "ppid", "pgid", "session_id", "stage", "status")
        })
        state["resume_count"] = int(state.get("resume_count", 0)) + 1
        state["resumed_unix_seconds"] = time.time()
    else:
        state = {"kind": "bn_g_s_clip_feasibility_state", "status": "RUNNING",
                 "started_unix_seconds": time.time(), "resume_count": 0,
                 "process_history": []}
    state.update({"pid": os.getpid(), "ppid": os.getppid(), "pgid": os.getpgid(0),
                  "session_id": os.getsid(0)})
    update_state(root, state, status="RUNNING", stage="preflight", error=None)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    try:
        if sha256(ONNX) != ONNX_SHA:
            raise ValueError("frozen ONNX SHA mismatch")
        meta = json.loads((CALIB / "meta.json").read_text(encoding="utf-8"))
        if meta["checkpoint"]["sha256"] != CHECKPOINT_SHA or meta["onnx"]["sha256"] != ONNX_SHA:
            raise ValueError("calibration identity mismatch")
        try:
            remote_processes = ssh_readonly(
                "pgrep -af '[c]apture_bn_g_s_yhat_bundle.py|[e]valuate_bn_g_s_clip_bundle.py|"
                "[r]un_board_hybrid_ort.py|[r]un_board_formal_eval.py' || true",
                attempts=2,
            )
            board_temperature_c = float(
                ssh_readonly("cat /sys/class/thermal/thermal_zone0/temp", attempts=2).strip()
            ) / 1000.0
            update_state(root, state, remote_preflight_status="PASS",
                         remote_preflight_processes=remote_processes,
                         board_temperature_c=board_temperature_c)
        except Exception as exc:
            # Host PTQ/compile are independent of board reachability. The board
            # stage performs a mandatory retried connectivity and idle check
            # immediately before it creates any remote artifact or process.
            update_state(root, state, remote_preflight_status="DEFERRED_TO_BOARD_STAGE",
                         remote_preflight_error=f"{type(exc).__name__}: {exc}")
        har = parse_once(root, state, env)
        range_audit_path = root / "range_audit.json"
        if range_audit_path.is_file():
            audit = json.loads(range_audit_path.read_text(encoding="utf-8"))
            low, high = float(audit["q0_1"]), float(audit["q99_9"])
        else:
            low, high, audit = exact_percentiles()
            atomic_write(range_audit_path, audit)
        scripts = write_scripts(root, low, high, args.f1_mode)
        results = {}
        for arm in ("F0_minmax", "F1_q0p1_q99p9"):
            update_state(root, state, stage=f"optimize_{arm}")
            results[arm] = optimize_arm(root, arm, scripts[arm], har, env)
        precompile_already_reported = (root / "precompile_report.json").is_file()
        report = precompile_report(root, har, audit, results, started)
        update_state(root, state, stage="precompile_report_written", precompile_report_sha256=sha256(root / "precompile_report.json"))
        print(json.dumps({"PRECOMPILE_REPORT": report}, sort_keys=True), flush=True)
        # A finite notification window lets the supervising Codex session send
        # the mandated collaboration message; loss of that session cannot stop
        # the detached campaign from continuing.
        if not precompile_already_reported:
            time.sleep(90)
        compiled = {}
        for arm in ("F0_minmax", "F1_q0p1_q99p9"):
            update_state(root, state, stage=f"compile_{arm}")
            compiled[arm] = compile_arm(root, arm)
        static = validate_static(root, low, high, compiled)
        update_state(root, state, stage="board_correctness")
        correctness = board(root, state, static)
        final = {
            "kind": "bn_g_s_clip_feasibility_summary", "status": "PASS",
            "elapsed_seconds": time.monotonic() - started,
            "range": audit, "precompile": report, "hef_static": static,
            "sdk32": correctness["sdk32"], "board32": correctness["board32"],
            "remote_root": correctness["remote_root"],
            "policy": {"diagnostic_only": True, "full976_not_run": True,
                       "semantic_ranking": False},
        }
        atomic_write(root / "summary.json", final)
        update_state(root, state, status="PASS", stage="complete",
                     elapsed_seconds=final["elapsed_seconds"])
        return 0
    except Exception as exc:
        import traceback
        error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        atomic_write(root / "failure.json", {"status": "FAIL", "error": error,
                                               "elapsed_seconds": time.monotonic() - started})
        update_state(root, state, status="FAIL", stage="failed", error=error)
        print(error, file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
