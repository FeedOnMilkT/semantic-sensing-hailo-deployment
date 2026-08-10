#!/usr/bin/env python3
"""Wait for frozen candidates, then evaluate them as GPU shards.

Tolerates COMPLETE_PARTIAL quantization: evaluates only FROZEN interfaces.
Does not permanently die on a transient FAILED quantization status when at
least one candidate is already frozen. Sibling eval shards are never
terminated when a peer fails.
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
EVALUATOR = ROOT / "scripts" / "evaluate_segmented_hn_chain.py"
CANDIDATES = (
    "A_hyper_adaround_receiver_w8",
    "B_hyper_qft4_receiver_w8",
    "C_hyper_adaround_receiver_a16",
    "D_hyper_qft4_receiver_a16",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quant-root", type=Path, required=True)
    parser.add_argument("--handoff-root", type=Path, required=True)
    parser.add_argument("--segments-manifest", type=Path, required=True)
    parser.add_argument("--gpus", nargs=8, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class EvaluationSupervisor:
    def __init__(self, args: argparse.Namespace) -> None:
        if not args.execute:
            raise PermissionError("pass --execute only for the approved four-candidate evaluation")
        if args.poll_seconds <= 0 or len(set(args.gpus)) != 8:
            raise ValueError("eight unique GPUs and a positive poll interval are required")
        self.args = args
        self.quant = args.quant_root.resolve()
        self.out = self.quant / "evaluation"
        self.logs = self.out / "logs"
        self.state_path = self.out / "evaluation_state.json"
        self.out.mkdir(parents=True, exist_ok=True)
        self.logs.mkdir(exist_ok=True)
        self.active_candidates: list[str] = list(CANDIDATES)

    def frozen_candidates(self) -> list[str]:
        frozen = []
        for candidate in CANDIDATES:
            interface = self.quant / "candidates" / candidate / "interface_manifest.json"
            if interface.is_file() and load_json(interface).get("status") == "FROZEN":
                frozen.append(candidate)
        return frozen

    def state(self, stage: str, **extra: Any) -> None:
        processed = 0
        active = 0
        for candidate in self.active_candidates:
            for shard in range(2):
                path = self.out / candidate / f"shard{shard}.progress.json"
                result = self.out / candidate / f"shard{shard}.result.json"
                if result.is_file():
                    processed += int(load_json(result).get("sample_count", 0))
                elif path.is_file():
                    report = load_json(path)
                    processed += int(report.get("processed_count", 0))
                    if report.get("status") == "RUNNING":
                        active += 1
        atomic_json(self.state_path, {
            "schema": "semantic_sensing.segmented_joint4_evaluation_state.v1",
            "status": stage if stage in {"COMPLETE", "FAILED", "COMPLETE_PARTIAL"} else "RUNNING",
            "stage": stage,
            "quant_root": str(self.quant),
            "active_candidates": list(self.active_candidates),
            "processed_candidate_samples": processed,
            "total_candidate_samples": len(self.active_candidates) * 976,
            "active_shards": active,
            "updated_unix_seconds": time.time(),
            **extra,
        })

    def env(self, gpu: str) -> dict[str, str]:
        value = os.environ.copy()
        value["CUDA_VISIBLE_DEVICES"] = gpu
        value["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
        flags = value.get("XLA_FLAGS", "")
        if "--xla_gpu_autotune_level=" not in flags:
            flags = (flags + " --xla_gpu_autotune_level=0").strip()
        value["XLA_FLAGS"] = flags
        return value

    def wait_for_quantization(self) -> None:
        while True:
            pipeline_state = self.quant / "pipeline_state.json"
            if pipeline_state.is_file():
                report = load_json(pipeline_state)
                status = report.get("status")
                frozen = self.frozen_candidates()
                if status == "COMPLETE":
                    missing = [name for name in CANDIDATES if name not in frozen]
                    if missing:
                        raise ValueError(f"COMPLETE but missing FROZEN interfaces: {missing}")
                    self.active_candidates = list(CANDIDATES)
                    break
                if status == "COMPLETE_PARTIAL":
                    if not frozen:
                        raise ValueError("COMPLETE_PARTIAL without any FROZEN interface")
                    self.active_candidates = frozen
                    break
                if status == "FAILED":
                    # Do not permanently wedge: if any candidate was frozen before
                    # failure, evaluate those; otherwise keep waiting for a resume.
                    if frozen:
                        self.active_candidates = frozen
                        self.state(
                            "QUANT_FAILED_EVALUATING_FROZEN",
                            quantization=report,
                            frozen_candidates=frozen,
                        )
                        break
                    self.state("WAITING_AFTER_QUANT_FAILED", quantization=report)
                else:
                    self.state("WAITING_FOR_FOUR_QUANTIZED_CANDIDATES", quantization=report)
            else:
                self.state("WAITING_FOR_QUANTIZATION_STATE")
            time.sleep(self.args.poll_seconds)

    def evaluate(self) -> None:
        active: dict[str, tuple[subprocess.Popen[Any], Any, float, str]] = {}
        gpu_index = 0
        for candidate in self.active_candidates:
            candidate_out = self.out / candidate
            candidate_out.mkdir(exist_ok=True)
            interface = self.quant / "candidates" / candidate / "interface_manifest.json"
            for shard in range(2):
                label = f"{candidate}_shard{shard}"
                result = candidate_out / f"shard{shard}.result.json"
                if result.is_file():
                    gpu_index += 1
                    continue
                progress = candidate_out / f"shard{shard}.progress.json"
                command = [
                    sys.executable, str(EVALUATOR),
                    "--handoff-root", str(self.args.handoff_root.resolve()),
                    "--segments-manifest", str(self.args.segments_manifest.resolve()),
                    "--interface-manifest", str(interface),
                    "--out", str(result), "--progress-out", str(progress),
                    "--shard-index", str(shard), "--num-shards", "2",
                    "--checkpoint-every", "5",
                ]
                gpu = self.args.gpus[gpu_index % len(self.args.gpus)]
                gpu_index += 1
                log = (self.logs / f"{label}.log").open("ab")
                process = subprocess.Popen(command, env=self.env(gpu), stdout=log, stderr=subprocess.STDOUT)
                active[label] = (process, log, time.time(), gpu)
        finished: dict[str, dict[str, Any]] = {}
        try:
            while active:
                jobs = dict(finished)
                for label, (process, log, started, gpu) in list(active.items()):
                    code = process.poll()
                    entry = {
                        "pid": process.pid, "gpu": gpu, "returncode": code,
                        "elapsed_seconds": time.time() - started,
                    }
                    jobs[label] = entry
                    if code is not None:
                        log.close()
                        del active[label]
                        finished[label] = entry
                self.state("EVALUATING_SHARDS", jobs=jobs)
                if active:
                    time.sleep(self.args.poll_seconds)
        finally:
            for process, log, _started, _gpu in active.values():
                if process.poll() is None:
                    process.terminate()
                log.close()
        failed = [label for label, entry in finished.items() if entry.get("returncode") not in (0, None)]
        if failed:
            detail = ", ".join(f"{label} rc={finished[label]['returncode']}" for label in failed)
            raise RuntimeError(f"{len(failed)} eval shard(s) failed after peers finished: {detail}")

    def merge(self) -> None:
        merged = []
        missing = []
        for candidate in self.active_candidates:
            root = self.out / candidate
            shard0 = root / "shard0.result.json"
            shard1 = root / "shard1.result.json"
            if not (shard0.is_file() and shard1.is_file()):
                missing.append(candidate)
                continue
            command = [
                sys.executable, str(EVALUATOR), "--merge-shards",
                str(shard0), str(shard1),
                "--expected-sample-count", "976", "--out", str(root / "result.json"),
            ]
            subprocess.run(command, check=True)
            merged.append(candidate)
        if missing and not merged:
            raise RuntimeError(f"no candidate shards ready to merge: {missing}")
        if missing:
            self.state(
                "COMPLETE_PARTIAL",
                merged_candidates=merged,
                missing_candidates=missing,
            )
            raise RuntimeError(f"merged {merged}; missing shards for {missing}")
        if len(merged) < 4:
            self.state("COMPLETE_PARTIAL", merged_candidates=merged)
        else:
            self.state("COMPLETE", merged_candidates=merged)

    def run(self) -> None:
        self.wait_for_quantization()
        self.state(
            "QUANTIZATION_READY_STARTING_EVALUATION",
            active_candidates=list(self.active_candidates),
        )
        self.evaluate()
        self.state("MERGING_CANDIDATES", active_candidates=list(self.active_candidates))
        self.merge()


def main() -> int:
    args = parse_args()
    supervisor = EvaluationSupervisor(args)
    try:
        supervisor.run()
    except BaseException as exc:
        current = {}
        if supervisor.state_path.is_file():
            try:
                current = load_json(supervisor.state_path)
            except Exception:
                current = {}
        if current.get("status") != "COMPLETE_PARTIAL":
            supervisor.state("FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
