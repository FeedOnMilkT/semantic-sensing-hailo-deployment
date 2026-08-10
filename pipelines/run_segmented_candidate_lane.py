#!/usr/bin/env python3
"""Resume-safe execution of one formal segmented ConvBN candidate lane.

The CPU rANS boundaries make the receiver graph depend on *quantized* sender
artifacts.  This launcher encodes that dependency explicitly instead of
leaving the receiver/tail commands as a manual follow-up:

  wait for z_hat handoff -> PTQ hyper -> materialize y_hat handoff
  -> PTQ the two receiver/tail precision variants in parallel.

It intentionally owns only one shared upstream lane.  The base lane produces
the W8A8 and validated-g_s-A16 candidates; the QFT lane produces their QFT-4
counterparts.  Every already-PASS artifact is reused, so a detached launcher
can safely be restarted after an interrupted host session.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
MODULE_PTQ = ROOT / "pipelines" / "module_ptq.py"
MATERIALIZE = ROOT / "pipelines" / "materialize_segmented_chain_calib.py"
QFT4_ALLS = ROOT / "configs" / "segmented_candidates" / "qft4_3276.alls"
GS_A16_ALLS = ROOT / "configs" / "segmented_candidates" / "synthesis_tail_validated_gs_a16.alls"


_LANES: dict[str, dict[str, Any]] = {
    "adaround_core": {
        "qft": False,
        "candidates": (
            ("adaround_w8a8_l4", None),
            ("adaround_a16_high_l4", GS_A16_ALLS),
        ),
    },
    "adaround_qft4_core": {
        "qft": True,
        "candidates": (
            ("adaround_qft4_w8a8_l4", None),
            ("adaround_qft4_a16_high_l4", GS_A16_ALLS),
        ),
    },
}


def _args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--lane", choices=sorted(_LANES), required=True)
    parser.add_argument("--student-ckpt", type=Path, required=True)
    parser.add_argument("--n-calib", type=int, required=True)
    parser.add_argument("--hyper-gpu", type=str, required=True)
    parser.add_argument("--tail-gpus", nargs=2, metavar=("GPU_A", "GPU_B"), required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--execute", action="store_true",
                        help="required guard before starting DFC PTQ/materialization")
    return parser.parse_args(argv)


def _pass_result(directory: Path) -> bool:
    result = directory / "result.json"
    if not result.is_file():
        return False
    try:
        return json.loads(result.read_text(encoding="utf-8")).get("status") == "PASS"
    except json.JSONDecodeError:
        return False


def _handoff_ready(directory: Path, n_calib: int) -> bool:
    meta = directory / "meta.json"
    if not meta.is_file():
        return False
    try:
        return int(json.loads(meta.read_text(encoding="utf-8")).get("sample_count", 0)) >= n_calib
    except json.JSONDecodeError:
        return False


class Lane:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.root = args.candidate_root.resolve()
        self.spec = _LANES[args.lane]
        self.lane = self.root / "lanes" / args.lane
        self.manifest = self.root / "onnx" / "segments_manifest.json"
        self.bootstrap = self.root / "bootstrap_calib_3276"
        self.parse = self.root / "parse_native_cpu" / "har"
        self.state_path = self.lane / "receiver_dag_state.json"
        if not self.manifest.is_file() or not self.bootstrap.is_dir() or not self.parse.is_dir():
            raise FileNotFoundError("candidate root lacks frozen ONNX/bootstrap/parse artifacts")
        if not args.student_ckpt.is_file():
            raise FileNotFoundError(args.student_ckpt)
        if args.n_calib <= 0 or args.poll_seconds <= 0:
            raise ValueError("n-calib and poll-seconds must be positive")
        if not args.execute:
            raise PermissionError("pass --execute only for the approved four-candidate run")

    def _state(self, stage: str, **extra: Any) -> None:
        value = {
            "schema": "semantic_sensing.segmented_receiver_dag.v1",
            "lane": self.args.lane,
            "stage": stage,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "n_calib": self.args.n_calib,
            "qft": self.spec["qft"],
            **extra,
        }
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, self.state_path)

    def _env(self, gpu: str) -> dict[str, str]:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        return env

    def _run(self, label: str, command: list[str], gpu: str) -> None:
        self._state(label, command=command, gpu=gpu)
        print(f"[{self.args.lane}] start {label} on GPU {gpu}: {' '.join(command)}", flush=True)
        completed = subprocess.run(command, env=self._env(gpu), check=False)
        if completed.returncode:
            self._state("FAILED", failed_stage=label, returncode=completed.returncode, command=command, gpu=gpu)
            raise RuntimeError(f"{label} failed with return code {completed.returncode}")

    def _wait_for_handoff(self, directory: Path, stage: str) -> None:
        while not _handoff_ready(directory, self.args.n_calib):
            self._state(stage, waiting_for=str(directory))
            print(f"[{self.args.lane}] waiting for {directory}", flush=True)
            time.sleep(self.args.poll_seconds)

    def _ptq_command(self, segment: str, calib_root: Path, out_dir: Path, extra: Path | None) -> list[str]:
        manifest = json.loads(self.manifest.read_text(encoding="utf-8"))
        entries = {entry["id"]: entry for entry in manifest["segments"]}
        command = [
            sys.executable, str(MODULE_PTQ), "calib-root",
            "--module", segment,
            "--onnx", entries[segment]["onnx"],
            "--parse-har", str(self.parse / f"{segment}_hailo_model.har"),
            "--calib-root", str(calib_root),
            "--out-dir", str(out_dir),
            "--n-calib", str(self.args.n_calib),
            "--opt-level", "4",
            "--eval-samples", "4",
            "--context-switch", "allowed",
            "--expect-onnx-sha",
            "--expect-checkpoint-sha", manifest["checkpoint"]["sha256"],
        ]
        if extra is not None:
            command += ["--extra-alls", str(extra)]
        return command

    def run(self) -> None:
        hyper_calib = self.lane / "calib_hyper"
        hyper_ptq = self.lane / "ptq" / "hyper"
        tail_calib = self.lane / "calib_tail"
        analysis_qhar = self.lane / "ptq" / "analysis" / "analysis_l4_quantized.har"

        self._wait_for_handoff(hyper_calib, "WAIT_HYPER_CALIBRATION")
        if not _pass_result(hyper_ptq):
            self._run(
                "PTQ_HYPER",
                self._ptq_command("hyper", hyper_calib, hyper_ptq, QFT4_ALLS if self.spec["qft"] else None),
                self.args.hyper_gpu,
            )
        if not _pass_result(hyper_ptq):
            raise RuntimeError(f"hyper PTQ has no PASS result: {hyper_ptq}")

        if not _handoff_ready(tail_calib, self.args.n_calib):
            command = [
                sys.executable, str(MATERIALIZE),
                "--target", "synthesis_tail",
                "--segments-manifest", str(self.manifest),
                "--bootstrap-calib-root", str(self.bootstrap),
                "--analysis-quantized-har", str(analysis_qhar),
                "--hyper-quantized-har", str(hyper_ptq / "hyper_l4_quantized.har"),
                "--student-ckpt", str(self.args.student_ckpt),
                "--n-calib", str(self.args.n_calib),
                "--out-root", str(tail_calib),
            ]
            self._run("MATERIALIZE_TAIL_CALIBRATION", command, self.args.hyper_gpu)
        if not _handoff_ready(tail_calib, self.args.n_calib):
            raise RuntimeError(f"tail calibration has no valid handoff: {tail_calib}")

        jobs: list[tuple[str, subprocess.Popen[bytes]]] = []
        for (candidate, a16_alls), gpu in zip(self.spec["candidates"], self.args.tail_gpus):
            tail_ptq = self.lane / "candidates" / candidate / "ptq" / "synthesis_tail"
            if _pass_result(tail_ptq):
                print(f"[{self.args.lane}] reusing PASS {candidate}", flush=True)
                continue
            if tail_ptq.exists():
                raise FileExistsError(f"refusing ambiguous non-PASS tail output: {tail_ptq}")
            # The two user-approved policies compose: QFT is applied after the
            # precision override is loaded, exactly as in the static plan.
            command = self._ptq_command(
                "synthesis_tail", tail_calib, tail_ptq,
                QFT4_ALLS if self.spec["qft"] else a16_alls,
            )
            if self.spec["qft"] and a16_alls is not None:
                command += ["--alls", str(a16_alls)]
            self._state("PTQ_SYNTHESIS_TAIL", candidate=candidate, command=command, gpu=gpu)
            print(f"[{self.args.lane}] start {candidate} on GPU {gpu}: {' '.join(command)}", flush=True)
            jobs.append((candidate, subprocess.Popen(command, env=self._env(gpu))))
        failures = [(candidate, job.wait()) for candidate, job in jobs]
        failures = [(candidate, status) for candidate, status in failures if status]
        if failures:
            self._state("FAILED", failed_stage="PTQ_SYNTHESIS_TAIL", failures=failures)
            raise RuntimeError(f"tail PTQ failures: {failures}")
        self._state("TAIL_PTQ_COMPLETE", candidates=[candidate for candidate, _ in self.spec["candidates"]])


def main(argv: list[str] | None = None) -> int:
    Lane(_args(argv)).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
