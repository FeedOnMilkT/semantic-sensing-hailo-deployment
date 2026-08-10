#!/usr/bin/env python3
"""Resume-safe formal PTQ for the approved three-graph/four-candidate DAG.

The one common Sender is reused by SHA.  Only the physical hyper-synthesis
graph branches into AdaRound and QFT4.  Each branch materializes its own real
rANS y_hat calibration before the W8/A16 receiver variants are optimized.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hailo_deploy.io.hash import sha256
from hailo_deploy.io.interface_contract import codec_boundary_contracts, extract_hef_io_contract

MODULE_PTQ = ROOT / "pipelines" / "module_ptq.py"
MATERIALIZE = ROOT / "pipelines" / "materialize_segmented_chain_calib.py"
CHAIN_CONFIG = ROOT / "configs" / "segmented_convbn_chain.json"
QFT4_ALLS = ROOT / "configs" / "segmented_candidates" / "qft4_3276.alls"
A16_ALLS = ROOT / "configs" / "segmented_candidates" / "synthesis_tail_validated_gs_a16.alls"

BRANCHES = {
    "hyper_adaround": {"hyper_alls": None},
    "hyper_qft4": {"hyper_alls": str(QFT4_ALLS)},
}
CANDIDATES = {
    "A_hyper_adaround_receiver_w8": {"branch": "hyper_adaround", "tail_alls": None},
    "B_hyper_qft4_receiver_w8": {"branch": "hyper_qft4", "tail_alls": None},
    "C_hyper_adaround_receiver_a16": {"branch": "hyper_adaround", "tail_alls": str(A16_ALLS)},
    "D_hyper_qft4_receiver_a16": {"branch": "hyper_qft4", "tail_alls": str(A16_ALLS)},
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--student-ckpt", type=Path, required=True)
    parser.add_argument("--n-calib", type=int, default=3276)
    parser.add_argument("--gpus", nargs="+", required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def artifact(record: dict[str, Any], key: str, module: str) -> Path:
    item = record.get(key)
    if not isinstance(item, dict):
        raise ValueError(f"{module}: missing {key}")
    path = Path(item.get("path", ""))
    if not path.is_file() or sha256(path) != item.get("sha256"):
        raise ValueError(f"{module}: invalid {key} artifact {path}")
    return path.resolve()


def pass_result(directory: Path, module: str) -> dict[str, Any] | None:
    path = directory / "result.json"
    if not path.is_file():
        return None
    value = load_json(path)
    status = value.get("status")
    if status in {"FAIL", "FAILED"}:
        return None
    if status != "PASS" or value.get("module") != module:
        raise ValueError(f"ambiguous non-PASS result at {path}")
    artifact(value, "quantized_har", module)
    artifact(value, "hef", module)
    return value


def completed_result(directory: Path, module: str) -> bool:
    """Report progress without masking a child failure with a parse error."""
    path = directory / "result.json"
    if not path.is_file():
        return False
    value = load_json(path)
    return value.get("status") == "PASS" and value.get("module") == module


def handoff_meta(directory: Path, mode: str, n_calib: int) -> dict[str, Any] | None:
    path = directory / "meta.json"
    if not path.is_file():
        return None
    value = load_json(path)
    if value.get("calibration_mode") != mode or int(value.get("sample_count", 0)) != n_calib:
        raise ValueError(f"invalid handoff at {path}")
    return value


def record_for_run(record: dict[str, Any], module: str) -> dict[str, Any]:
    return {
        "module": module,
        "result": {"path": str((Path(record["quantized_har"]["path"]).parent / "result.json").resolve())},
        "quantized_har": record["quantized_har"],
        "hef": record["hef"],
        "compiled_har": record.get("compiled_har"),
        "model_script": record.get("model_script"),
        "num_contexts": record.get("num_contexts"),
        "calibration": record.get("calibration"),
    }


class Supervisor:
    def __init__(self, args: argparse.Namespace) -> None:
        if not args.execute:
            raise PermissionError("pass --execute only for the user-approved four-candidate run")
        if args.n_calib != 3276:
            raise ValueError("approved optimization set is exactly 3276 samples")
        if args.poll_seconds <= 0 or not args.gpus:
            raise ValueError("positive --poll-seconds and at least one GPU are required")
        if len(set(args.gpus)) != len(args.gpus):
            raise ValueError("--gpus contains duplicates")
        self.args = args
        self.source = args.source_root.resolve()
        self.out = args.out_root.resolve()
        self.manifest_path = self.source / "onnx" / "segments_manifest.json"
        self.bootstrap = self.source / "bootstrap_calib_3276"
        self.parse_har = self.source / "parse_native_cpu" / "har"
        self.sender_dir = self.source / "lanes" / "adaround_core" / "ptq" / "analysis"
        self.manifest = load_json(self.manifest_path)
        self.entries = {entry["id"]: entry for entry in self.manifest.get("segments", [])}
        self.state_path = self.out / "pipeline_state.json"
        self.logs = self.out / "logs"
        if set(self.entries) != {"analysis", "hyper", "synthesis_tail"}:
            raise ValueError("segments manifest is not the three-graph contract")
        if sha256(args.student_ckpt) != self.manifest["checkpoint"]["sha256"]:
            raise ValueError("checkpoint differs from ONNX manifest")
        self.sender = pass_result(self.sender_dir, "analysis")
        if self.sender is None:
            raise FileNotFoundError("common Sender PASS result is missing")
        sender_script = Path(self.sender["model_script"]["path"]).read_text(encoding="utf-8")
        if "post_quantization_optimization(finetune" in sender_script:
            raise ValueError("common Sender contains explicit QFT and cannot be reused")
        sender_meta = self.sender.get("calibration", {}).get("metadata", {})
        if sender_meta.get("onnx", {}).get("sha256") != self.entries["analysis"]["sha256"]:
            raise ValueError("common Sender ONNX identity mismatch")
        if sender_meta.get("checkpoint", {}).get("sha256") != self.manifest["checkpoint"]["sha256"]:
            raise ValueError("common Sender checkpoint identity mismatch")

    def gpu(self, index: int) -> str:
        return self.args.gpus[index % len(self.args.gpus)]

    def state(self, stage: str, **extra: Any) -> None:
        atomic_json(self.state_path, {
            "schema": "semantic_sensing.segmented_joint4_state.v1",
            "status": "RUNNING" if stage not in {"COMPLETE", "FAILED"} else stage,
            "stage": stage,
            "updated_unix_seconds": time.time(),
            "n_calib": self.args.n_calib,
            "source_root": str(self.source),
            "out_root": str(self.out),
            "gpus": self.args.gpus,
            "completed_candidates": sum(
                completed_result(self.tail_dir(name), "synthesis_tail")
                for name in CANDIDATES
            ),
            "total_candidates": 4,
            **extra,
        })

    def setup(self) -> None:
        identity = {
            "schema": "semantic_sensing.segmented_joint4_run.v1",
            "checkpoint": self.manifest["checkpoint"],
            "segments_manifest": {"path": str(self.manifest_path), "sha256": sha256(self.manifest_path)},
            "chain_config": {"path": str(CHAIN_CONFIG), "sha256": sha256(CHAIN_CONFIG)},
            "bootstrap_analysis_meta": {
                "path": str(self.bootstrap / "analysis" / "meta.json"),
                "sha256": sha256(self.bootstrap / "analysis" / "meta.json"),
            },
            "n_calib": self.args.n_calib,
            "common_sender": record_for_run(self.sender, "analysis"),
            "candidates": CANDIDATES,
        }
        if self.out.exists():
            if not self.args.resume:
                raise FileExistsError(f"refusing existing out root without --resume: {self.out}")
            existing = load_json(self.out / "run_manifest.json")
            for key in ("checkpoint", "segments_manifest", "chain_config", "bootstrap_analysis_meta", "n_calib"):
                if existing.get(key) != identity.get(key):
                    raise ValueError(f"resume identity mismatch at {key}")
        else:
            self.out.mkdir(parents=True)
            atomic_json(self.out / "run_manifest.json", identity)
        self.logs.mkdir(exist_ok=True)

    def env(self, gpu: str) -> dict[str, str]:
        value = os.environ.copy()
        value["CUDA_VISIBLE_DEVICES"] = gpu
        value["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
        flags = value.get("XLA_FLAGS", "")
        if "--xla_gpu_autotune_level=" not in flags:
            flags = (flags + " --xla_gpu_autotune_level=0").strip()
        value["XLA_FLAGS"] = flags
        return value

    def run_jobs(self, stage: str, jobs: list[tuple[str, list[str], str]]) -> list[str]:
        """Run jobs in parallel; never terminate siblings on a peer failure.

        Returns labels that exited non-zero. Successful PASS artifacts are left
        on disk for freeze/resume. Raises only after every job has finished.
        """
        active: dict[str, tuple[subprocess.Popen[Any], Any, str, float, list[str]]] = {}
        finished: dict[str, dict[str, Any]] = {}
        for label, command, gpu in jobs:
            log = (self.logs / f"{label}.log").open("ab")
            process = subprocess.Popen(command, env=self.env(gpu), stdout=log, stderr=subprocess.STDOUT)
            active[label] = (process, log, gpu, time.time(), command)
        try:
            while active:
                snapshot = dict(finished)
                for label, (process, log, gpu, started, command) in list(active.items()):
                    code = process.poll()
                    entry = {
                        "pid": process.pid, "gpu": gpu, "returncode": code,
                        "elapsed_seconds": time.time() - started, "command": command,
                    }
                    snapshot[label] = entry
                    if code is not None:
                        log.close()
                        del active[label]
                        finished[label] = entry
                self.state(stage, jobs=snapshot)
                if active:
                    time.sleep(self.args.poll_seconds)
        finally:
            # Only reached with live children on KeyboardInterrupt / unexpected exit.
            for process, log, _gpu, _started, _command in active.values():
                if process.poll() is None:
                    process.terminate()
                log.close()
        failed = [label for label, entry in finished.items() if entry.get("returncode") not in (0, None)]
        if failed:
            detail = ", ".join(f"{label} rc={finished[label]['returncode']}" for label in failed)
            raise RuntimeError(f"{len(failed)} job(s) failed after all peers finished: {detail}")
        return failed

    def materialize_command(self, target: str, out: Path, hyper_har: Path | None = None) -> list[str]:
        command = [
            sys.executable, str(MATERIALIZE), "--target", target,
            "--segments-manifest", str(self.manifest_path),
            "--bootstrap-calib-root", str(self.bootstrap),
            "--analysis-quantized-har", str(artifact(self.sender, "quantized_har", "analysis")),
            "--student-ckpt", str(self.args.student_ckpt.resolve()),
            "--n-calib", str(self.args.n_calib), "--out-root", str(out),
        ]
        if hyper_har is not None:
            command += ["--hyper-quantized-har", str(hyper_har)]
        return command

    def ptq_command(self, segment: str, calib: Path, out: Path, alls: Path | None) -> list[str]:
        command = [
            sys.executable, str(MODULE_PTQ), "calib-root", "--module", segment,
            "--onnx", self.entries[segment]["onnx"],
            "--parse-har", str(self.parse_har / f"{segment}_hailo_model.har"),
            "--calib-root", str(calib), "--out-dir", str(out),
            "--n-calib", str(self.args.n_calib), "--opt-level", "4",
            "--eval-samples", "4", "--context-switch", "allowed",
            "--expect-onnx-sha", "--expect-checkpoint-sha", self.manifest["checkpoint"]["sha256"],
        ]
        if alls is not None:
            command += ["--extra-alls", str(alls)]
        return command

    def hyper_dir(self, branch: str) -> Path:
        return self.out / "hyper" / branch / "ptq"

    def tail_dir(self, candidate: str) -> Path:
        return self.out / "candidates" / candidate / "ptq" / "synthesis_tail"

    def validate_parent(self, meta: dict[str, Any], key: str, expected: Path) -> None:
        item = meta.get("quantized_upstream", {}).get(key)
        if not isinstance(item, dict) or item.get("sha256") != sha256(expected):
            raise ValueError(f"handoff does not bind {key} to {expected}")

    def freeze_candidates(self) -> list[str]:
        sender_qhar = artifact(self.sender, "quantized_har", "analysis")
        sender_hef = artifact(self.sender, "hef", "analysis")
        sender_contract = extract_hef_io_contract(sender_qhar, sender_hef, self.entries["analysis"])
        frozen: list[str] = []
        for candidate, spec in CANDIDATES.items():
            branch = spec["branch"]
            hyper = pass_result(self.hyper_dir(branch), "hyper")
            tail = pass_result(self.tail_dir(candidate), "synthesis_tail")
            if hyper is None or tail is None:
                continue
            hyper_qhar = artifact(hyper, "quantized_har", "hyper")
            tail_qhar = artifact(tail, "quantized_har", "synthesis_tail")
            hyper_meta = hyper.get("calibration", {}).get("metadata", {})
            tail_meta = tail.get("calibration", {}).get("metadata", {})
            if hyper_meta.get("calibration_mode") != "quantized_analysis_z_rans":
                raise ValueError(f"{candidate}: Hyper calibration mode is not the formal chain mode")
            self.validate_parent(hyper_meta, "analysis_quantized_har", sender_qhar)
            if tail_meta.get("calibration_mode") != "quantized_analysis_hyper_y_rans":
                raise ValueError(f"{candidate}: Receiver calibration mode is not the formal chain mode")
            self.validate_parent(tail_meta, "analysis_quantized_har", sender_qhar)
            self.validate_parent(tail_meta, "hyper_quantized_har", hyper_qhar)
            contracts = {
                "analysis": sender_contract,
                "hyper": extract_hef_io_contract(hyper_qhar, artifact(hyper, "hef", "hyper"), self.entries["hyper"]),
                "synthesis_tail": extract_hef_io_contract(
                    tail_qhar, artifact(tail, "hef", "synthesis_tail"), self.entries["synthesis_tail"]
                ),
            }
            record = {
                "schema": "semantic_sensing.segmented_candidate_interface.v1",
                "status": "FROZEN",
                "candidate": candidate,
                "policy": spec,
                "checkpoint": self.manifest["checkpoint"],
                "analysis": record_for_run(self.sender, "analysis"),
                "hyper": record_for_run(hyper, "hyper"),
                "synthesis_tail": record_for_run(tail, "synthesis_tail"),
                "hef_io_quantization": contracts,
                "codec_boundaries": codec_boundary_contracts(self.manifest, contracts),
                "hyper_deployment": {
                    "sender_hef_sha256": hyper["hef"]["sha256"],
                    "receiver_hef_sha256": hyper["hef"]["sha256"],
                    "exact_copy_required": True,
                },
            }
            atomic_json(self.out / "candidates" / candidate / "interface_manifest.json", record)
            frozen.append(candidate)
        return frozen

    def run(self) -> None:
        self.setup()
        sender_qhar = artifact(self.sender, "quantized_har", "analysis")
        hyper_calib = self.out / "calibration" / "hyper"
        stage_failures: list[str] = []
        if handoff_meta(hyper_calib, "quantized_analysis_z_rans", self.args.n_calib) is None:
            try:
                self.run_jobs("MATERIALIZE_HYPER_CALIBRATION", [(
                    "materialize_hyper", self.materialize_command("hyper", hyper_calib), self.gpu(0)
                )])
            except RuntimeError as exc:
                stage_failures.append(str(exc))
                raise
        hyper_jobs = []
        for index, (branch, spec) in enumerate(BRANCHES.items()):
            if pass_result(self.hyper_dir(branch), "hyper") is None:
                hyper_jobs.append((
                    f"ptq_{branch}",
                    self.ptq_command("hyper", hyper_calib, self.hyper_dir(branch), spec["hyper_alls"]),
                    self.gpu(index),
                ))
        if hyper_jobs:
            try:
                self.run_jobs("PTQ_TWO_HYPER_BRANCHES", hyper_jobs)
            except RuntimeError as exc:
                stage_failures.append(str(exc))
                # Hyper is shared by candidates; cannot continue without both branches if both requested.
                # Still freeze whatever already PASSed below after attempting tails that have hyper.
                if pass_result(self.hyper_dir("hyper_adaround"), "hyper") is None and \
                   pass_result(self.hyper_dir("hyper_qft4"), "hyper") is None:
                    raise

        tail_materialize_jobs = []
        for index, branch in enumerate(BRANCHES):
            hyper = pass_result(self.hyper_dir(branch), "hyper")
            if hyper is None:
                continue
            tail_calib = self.out / "calibration" / "synthesis_tail" / branch
            if handoff_meta(tail_calib, "quantized_analysis_hyper_y_rans", self.args.n_calib) is None:
                tail_materialize_jobs.append((
                    f"materialize_tail_{branch}",
                    self.materialize_command(
                        "synthesis_tail", tail_calib, artifact(hyper, "quantized_har", "hyper")
                    ),
                    self.gpu(index + 2),
                ))
        if tail_materialize_jobs:
            try:
                self.run_jobs("MATERIALIZE_TWO_TAIL_CALIBRATIONS", tail_materialize_jobs)
            except RuntimeError as exc:
                stage_failures.append(str(exc))

        tail_jobs = []
        for index, (candidate, spec) in enumerate(CANDIDATES.items()):
            if pass_result(self.tail_dir(candidate), "synthesis_tail") is None:
                if pass_result(self.hyper_dir(spec["branch"]), "hyper") is None:
                    continue
                calib = self.out / "calibration" / "synthesis_tail" / spec["branch"]
                if handoff_meta(calib, "quantized_analysis_hyper_y_rans", self.args.n_calib) is None:
                    continue
                tail_jobs.append((
                    f"ptq_{candidate}",
                    self.ptq_command("synthesis_tail", calib, self.tail_dir(candidate), spec["tail_alls"]),
                    self.gpu(index),
                ))
        tail_error: str | None = None
        if tail_jobs:
            try:
                self.run_jobs("PTQ_FOUR_RECEIVERS", tail_jobs)
            except RuntimeError as exc:
                tail_error = str(exc)
                stage_failures.append(tail_error)

        self.state("FREEZE_IO_CONTRACTS")
        frozen = self.freeze_candidates()
        completed = len(frozen)
        if completed == 4 and not stage_failures:
            self.state("COMPLETE", completed_candidates=4, total_candidates=4, frozen_candidates=frozen)
            return
        if completed == 0:
            self.state(
                "FAILED",
                completed_candidates=0,
                frozen_candidates=[],
                error="; ".join(stage_failures) or "no candidates produced PASS artifacts",
            )
            raise RuntimeError("; ".join(stage_failures) or "no candidates produced PASS artifacts")
        self.state(
            "COMPLETE_PARTIAL",
            completed_candidates=completed,
            total_candidates=4,
            frozen_candidates=frozen,
            error="; ".join(stage_failures) if stage_failures else None,
        )
        # Partial success is a usable handoff for evaluation; still exit non-zero so monitors notice.
        raise RuntimeError(
            f"partial success: froze {completed}/4 candidates ({', '.join(frozen)}); "
            + ("; ".join(stage_failures) if stage_failures else "missing candidates")
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    supervisor = Supervisor(args)
    try:
        supervisor.run()
    except BaseException as exc:
        if supervisor.out.exists():
            current = {}
            if supervisor.state_path.is_file():
                try:
                    current = load_json(supervisor.state_path)
                except Exception:
                    current = {}
            # Preserve COMPLETE_PARTIAL handoff for the evaluation supervisor.
            if current.get("status") != "COMPLETE_PARTIAL":
                supervisor.state("FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
