#!/usr/bin/env python3
"""Resume-safe coordinator for the approved 10-module selective PTQ campaign."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hailo_deploy.io.fullmodule_selection import (
    MODULES,
    atomic_json,
    freeze_candidate,
    load_json,
    select_candidate,
)
from hailo_deploy.io.hash import sha256

MODULE_PTQ = ROOT / "pipelines" / "module_ptq.py"
EVALUATOR = ROOT / "scripts" / "evaluate_fullmodule_hn_chain.py"
SCREEN = ROOT / "scripts" / "evaluate_fullmodule_calibration_screen.py"
DEFAULT_SPEC = ROOT / "configs" / "fullmodule_selective_campaign_20260804.json"
QFT_TOKEN = "post_quantization_optimization(finetune, policy=enabled, dataset_size=1024, epochs=4)"


def parse_gpu_rows() -> list[dict[str, int]]:
    command = [
        "nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.run(command, check=True, text=True, capture_output=True).stdout
    rows = []
    for line in output.splitlines():
        index, memory, utilization = [int(item.strip()) for item in line.split(",")]
        rows.append({"index": index, "memory_used_mib": memory, "utilization_percent": utilization})
    return rows


def gpu_affinities() -> dict[int, str]:
    output = subprocess.run(["nvidia-smi", "topo", "-m"], check=True, text=True, capture_output=True).stdout
    raw: dict[int, str] = {}
    for line in output.splitlines():
        fields = line.replace("\x1b[4m", "").replace("\x1b[0m", "").split()
        if not fields or not fields[0].startswith("GPU") or not fields[0][3:].isdigit():
            continue
        affinity = next((field for field in reversed(fields) if "," in field and "-" in field), None)
        if affinity is not None:
            raw[int(fields[0][3:])] = affinity
    groups: dict[str, list[int]] = {}
    for gpu, affinity in raw.items():
        groups.setdefault(affinity, []).append(gpu)
    result: dict[int, str] = {}
    for affinity, gpus in groups.items():
        cpus: list[int] = []
        for part in affinity.split(","):
            begin, end = [int(value) for value in part.split("-")]
            cpus.extend(range(begin, end + 1))
        gpus.sort()
        chunk = max(1, len(cpus) // len(gpus))
        for position, gpu in enumerate(gpus):
            assigned = cpus[position * chunk:(position + 1) * chunk] if position < len(gpus) - 1 else cpus[position * chunk:]
            result[gpu] = ",".join(str(value) for value in assigned)
    return result


class Campaign:
    def __init__(self, spec_path: Path, approved_sha: str, *, resume: bool = False) -> None:
        self.spec_path = spec_path.resolve()
        self.spec = load_json(self.spec_path)
        self.approved_sha = approved_sha
        self.out = Path(self.spec["out_root"]).resolve()
        self.handoff = Path(self.spec["handoff_root"]).resolve()
        self.base = Path(self.spec["base_root"]).resolve()
        self.resume = resume
        self.state_path = self.out / "campaign_state.json"
        self.affinities: dict[int, str] = {}
        self.job_status: dict[str, Any] = {}

    def validate_spec(self, *, require_out_absent: bool) -> dict[str, Any]:
        if self.spec.get("schema") != "semantic_sensing.fullmodule_selective_campaign.v1":
            raise ValueError("unsupported campaign schema")
        review = self.spec["review_spec"]
        review_path = Path(review["path"])
        actual_review_sha = sha256(review_path)
        if actual_review_sha != review["sha256"] or actual_review_sha != self.approved_sha:
            raise PermissionError("approved review SHA does not match the current review document")
        if tuple(self.spec["modules"]) != MODULES:
            raise ValueError("spec module order differs from the frozen 10-module contract")
        if self.spec["target"]["hw_arch"] != "hailo8" or not self.spec["target"]["dfc_version"].startswith("3."):
            raise ValueError("campaign is not locked to Hailo-8 / DFC 3.x")
        versions = {
            "dfc": importlib.metadata.version("hailo-dataflow-compiler"),
            "model_zoo": importlib.metadata.version("hailo-model-zoo"),
            "hailort": importlib.metadata.version("hailort"),
        }
        expected = self.spec["target"]
        if versions != {"dfc": expected["dfc_version"], "model_zoo": expected["model_zoo_version"],
                        "hailort": expected["hailort_version"]}:
            raise ValueError(f"suite version mismatch: {versions}")
        identity = load_json(self.handoff / "identity.json")
        if identity.get("checkpoint", {}).get("sha256") != self.spec["checkpoint_sha256"]:
            raise ValueError("handoff checkpoint differs from spec")
        qft = Path(self.spec["qft4_alls"])
        if qft.read_text(encoding="utf-8").strip() != QFT_TOKEN:
            raise ValueError("QFT4 script differs from the approved single command")
        if require_out_absent and self.out.exists():
            raise FileExistsError(f"refusing existing out root: {self.out}")
        if not require_out_absent and not self.out.is_dir():
            raise FileNotFoundError(self.out)
        disk = shutil.disk_usage(self.out.parent if self.out.parent.exists() else Path("/local/shared_with_docker"))
        free_gib = disk.free / (1024 ** 3)
        if free_gib < float(self.spec["scheduler"]["minimum_free_disk_gib"]):
            raise OSError(f"insufficient free disk: {free_gib:.1f} GiB")
        gpus = parse_gpu_rows()
        return {"review_sha256": actual_review_sha, "spec_sha256": sha256(self.spec_path),
                "versions": versions, "gpu_snapshot": gpus, "free_disk_gib": free_gib}

    def eligible_gpus(self) -> list[int]:
        scheduler = self.spec["scheduler"]
        return [
            row["index"] for row in parse_gpu_rows()
            if row["memory_used_mib"] < int(scheduler["gpu_memory_used_max_mib"])
            and row["utilization_percent"] < int(scheduler["gpu_utilization_max_percent"])
        ]

    def state(self, stage: str, status: str = "RUNNING", **extra: Any) -> None:
        atomic_json(self.state_path, {
            "schema": "semantic_sensing.fullmodule_selective_campaign_state.v1",
            "status": status, "stage": stage, "updated_unix_seconds": time.time(),
            "out_root": str(self.out), "spec_sha256": sha256(self.spec_path),
            "review_sha256": self.approved_sha, "jobs": self.job_status, **extra,
        })

    def setup(self, preflight: dict[str, Any]) -> None:
        manifest = {
            "schema": "semantic_sensing.fullmodule_selective_campaign_manifest.v1",
            "spec": {"path": str(self.spec_path), "sha256": sha256(self.spec_path)},
            "review_spec": self.spec["review_spec"], "approved_review_sha256": self.approved_sha,
            "checkpoint_sha256": self.spec["checkpoint_sha256"],
            "handoff_root": str(self.handoff), "base_root": str(self.base),
            "out_root": str(self.out), "preflight": preflight,
        }
        if self.resume:
            existing = load_json(self.out / "campaign_manifest.json")
            for key in ("spec", "review_spec", "approved_review_sha256", "checkpoint_sha256",
                        "handoff_root", "base_root", "out_root"):
                if existing.get(key) != manifest.get(key):
                    raise ValueError(f"resume identity mismatch: {key}")
        else:
            self.out.mkdir(parents=True, exist_ok=False)
            atomic_json(self.out / "campaign_manifest.json", manifest)
        self.affinities = gpu_affinities()

    def selection_paths(self, candidate: str) -> tuple[Path, Path]:
        root = self.out / "candidates" / candidate
        return root / "selection_manifest.json", root / "boundary_manifest.json"

    def c1_result_paths(self) -> dict[str, Path]:
        paths = {name: self.base / "ptq" / name / "result.json" for name in MODULES}
        for module, relative in self.spec["c1_replacements"].items():
            paths[module] = self.base / relative
        return paths

    def candidate_result_paths(self, candidate: str) -> dict[str, Path]:
        paths = self.c1_result_paths()
        if candidate in {"C2", "C3"}:
            paths["bn_h_s"] = self.out / "candidates" / candidate / "ptq" / "bn_h_s" / "result.json"
        return paths

    def freeze(self, candidate: str) -> dict[str, Any]:
        selection, boundary = self.selection_paths(candidate)
        return freeze_candidate(
            candidate=candidate, policy=self.spec["candidates"][candidate], handoff_root=self.handoff,
            result_paths=self.candidate_result_paths(candidate), selection_path=selection, boundary_path=boundary,
        )

    def audit(self) -> None:
        output = self.out / "reuse_audit.json"
        if output.is_file():
            if load_json(output).get("status") != "PASS":
                raise ValueError("existing reuse audit is not PASS")
            self.freeze("C1")
            return
        selection = self.freeze("C1")
        calibration = self.handoff / "calibration"
        counts = {}
        for module in MODULES:
            meta = load_json(calibration / module / "meta.json")
            if int(meta.get("sample_count", 0)) != 1024:
                raise ValueError(f"{module}: calibration count is not 1024")
            if meta.get("checkpoint", {}).get("sha256") != self.spec["checkpoint_sha256"]:
                raise ValueError(f"{module}: calibration checkpoint mismatch")
            counts[module] = int(meta["sample_count"])
        evaluation = load_json(self.handoff / "evaluation" / "meta.json")
        if list(evaluation.get("val_indices", [])) != list(range(1024, 2000)):
            raise ValueError("formal split is not exactly 1024..1999")
        atomic_json(output, {
            "schema": "semantic_sensing.fullmodule_reuse_audit.v1", "status": "PASS",
            "checkpoint_sha256": self.spec["checkpoint_sha256"], "calibration_counts": counts,
            "formal_indices": {"start": 1024, "end": 1999, "count": 976},
            "c1_candidate_fingerprint": selection["candidate_fingerprint"],
            "c1_selection_manifest": str(self.selection_paths("C1")[0]),
        })

    def job_env(self, gpu: int, job_dir: Path) -> dict[str, str]:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
        flags = env.get("XLA_FLAGS", "")
        if "--xla_gpu_autotune_level=" not in flags:
            flags = (flags + " --xla_gpu_autotune_level=0").strip()
        env["XLA_FLAGS"] = flags
        env["HAILO_SDK_LOG_DIR"] = str(job_dir / "sdk_logs")
        env["_HAILO_SDK_LOGGED"] = "1"
        env["TMPDIR"] = str(job_dir / "tmp")
        env["OMP_NUM_THREADS"] = "8"
        env["MKL_NUM_THREADS"] = "8"
        (job_dir / "sdk_logs").mkdir(parents=True, exist_ok=True)
        (job_dir / "tmp").mkdir(parents=True, exist_ok=True)
        return env

    @staticmethod
    def output_pass(path: Path) -> bool:
        if not path.is_file():
            return False
        try:
            return load_json(path).get("status") == "PASS"
        except Exception:
            return False

    def run_jobs(self, stage: str, jobs: list[dict[str, Any]], *, allow_failures: bool = False) -> dict[str, bool]:
        pending = [job for job in jobs if not self.output_pass(job["output"])]
        outcomes = {job["label"]: True for job in jobs if self.output_pass(job["output"])}
        while pending:
            gpus = self.eligible_gpus()
            minimum = int(self.spec["scheduler"]["minimum_gpus"])
            if len(gpus) < minimum:
                self.state(stage, waiting_for_gpus=True, eligible_gpus=gpus)
                time.sleep(int(self.spec["scheduler"]["poll_seconds"]))
                continue
            batch, pending = pending[:len(gpus)], pending[len(gpus):]
            active: dict[str, tuple[subprocess.Popen[Any], Any, Path, float, int]] = {}
            for job, gpu in zip(batch, gpus):
                label = job["label"]
                job_dir = self.out / "jobs" / label
                attempt = 1
                while (job_dir / f"attempt{attempt}").exists():
                    attempt += 1
                attempt_dir = job_dir / f"attempt{attempt}"
                attempt_dir.mkdir(parents=True)
                command = list(job["command"])
                affinity = self.affinities.get(gpu)
                if affinity and shutil.which("taskset"):
                    command = ["taskset", "-c", affinity, *command]
                log = (attempt_dir / "stdout.log").open("ab")
                process = subprocess.Popen(command, cwd=attempt_dir, env=self.job_env(gpu, attempt_dir),
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                self.job_status[label] = {"status": "RUNNING", "pid": process.pid, "gpu": gpu,
                                          "attempt": attempt, "command": command, "output": str(job["output"])}
                atomic_json(job_dir / "job.json", self.job_status[label])
                active[label] = (process, log, job_dir, time.time(), gpu)
            while active:
                for label, (process, log, job_dir, started, gpu) in list(active.items()):
                    code = process.poll()
                    if code is None:
                        continue
                    log.close()
                    output = next(job["output"] for job in batch if job["label"] == label)
                    passed = code == 0 and self.output_pass(output)
                    outcomes[label] = passed
                    self.job_status[label].update({"status": "PASS" if passed else "FAIL_FINAL",
                                                   "returncode": code, "elapsed_seconds": time.time() - started})
                    atomic_json(job_dir / "job.json", self.job_status[label])
                    del active[label]
                self.state(stage, eligible_gpus=self.eligible_gpus())
                if active:
                    time.sleep(int(self.spec["scheduler"]["poll_seconds"]))
        failed = [label for label, passed in outcomes.items() if not passed]
        if failed and not allow_failures:
            raise RuntimeError(f"{stage} failed jobs: {failed}")
        return outcomes

    def eval_job(self, candidate: str, label: str, output: Path, *, limit: int | None = None,
                 shard: tuple[int, int] | None = None, progress: Path | None = None) -> dict[str, Any]:
        selection = self.selection_paths(candidate)[0]
        command = [sys.executable, str(EVALUATOR), "--handoff-root", str(self.handoff),
                   "--selection-manifest", str(selection), "--out", str(output)]
        if limit is not None:
            command += ["--limit", str(limit)]
        if shard is not None:
            command += ["--shard-index", str(shard[0]), "--num-shards", str(shard[1])]
        if progress is not None:
            command += ["--progress-out", str(progress), "--checkpoint-every",
                        str(self.spec["scheduler"]["checkpoint_every"])]
        return {"label": label, "command": command, "output": output}

    def screen_job(self, candidate: str, shard: int, count: int) -> dict[str, Any]:
        root = self.out / "candidates" / candidate / "screen64"
        output = root / f"shard{shard}.json"
        command = [sys.executable, str(SCREEN), "--handoff-root", str(self.handoff),
                   "--selection-manifest", str(self.selection_paths(candidate)[0]),
                   "--shard-index", str(shard), "--num-shards", str(count), "--out", str(output)]
        return {"label": f"{candidate}_screen64_shard{shard}", "command": command, "output": output}

    def merge_screen(self, candidate: str) -> Path:
        root = self.out / "candidates" / candidate / "screen64"
        output = root / "result.json"
        if self.output_pass(output):
            return output
        count = int(self.spec["scheduler"]["screen_shards"])
        shards = [root / f"shard{index}.json" for index in range(count)]
        subprocess.run([sys.executable, str(SCREEN), "--merge-shards", *map(str, shards),
                        "--out", str(output)], cwd=ROOT, check=True)
        return output

    def ptq_job(self, candidate: str) -> dict[str, Any]:
        out_dir = self.out / "candidates" / candidate / "ptq" / "bn_h_s"
        command = [sys.executable, str(MODULE_PTQ), "fullmodule", "--handoff-root", str(self.handoff),
                   "--module", "bn_h_s", "--parse-har", str(self.base / "har" / "bn_h_s_hailo_model.har"),
                   "--out-dir", str(out_dir), "--n-calib", "1024", "--opt-level", "4"]
        if candidate == "C3":
            command += ["--extra-alls", self.spec["qft4_alls"]]
        return {"label": f"{candidate}_bn_h_s_ptq", "command": command, "output": out_dir / "result.json"}

    def run_candidate_smokes(self, candidates: list[str]) -> None:
        golden1 = []
        for candidate in candidates:
            output = self.out / "candidates" / candidate / "golden1.json"
            golden1.append(self.eval_job(candidate, f"{candidate}_golden1", output, limit=1))
        self.run_jobs("GOLDEN1", golden1)
        golden8 = []
        for candidate in candidates:
            output = self.out / "candidates" / candidate / "golden8.json"
            golden8.append(self.eval_job(candidate, f"{candidate}_golden8", output, limit=8))
        self.run_jobs("GOLDEN8", golden8)

    def decide(self, candidates: list[str]) -> dict[str, Any]:
        path = self.out / "decision" / "candidate_decision.json"
        if path.is_file():
            return load_json(path)
        reports = {
            candidate: {
                "golden": load_json(self.out / "candidates" / candidate / "golden8.json"),
                "screen": load_json(self.out / "candidates" / candidate / "screen64" / "result.json"),
            }
            for candidate in candidates
        }
        decision = select_candidate(reports, self.spec["selector"])
        atomic_json(path, decision)
        return decision

    def formal(self, candidate: str) -> Path:
        max_wait = int(self.spec["scheduler"]["formal_wait_for_eight_minutes"]) * 60
        started = time.time()
        while True:
            gpus = self.eligible_gpus()
            if len(gpus) >= 8 or time.time() - started >= max_wait:
                break
            self.state("WAITING_FORMAL_GPUS", candidate=candidate, eligible_gpus=gpus,
                       wait_seconds=int(time.time() - started))
            time.sleep(int(self.spec["scheduler"]["poll_seconds"]))
        count = min(len(gpus), int(self.spec["scheduler"]["formal_max_shards"]))
        if count < 2:
            self.state("PAUSED_INSUFFICIENT_FORMAL_GPUS", status="PAUSED", candidate=candidate,
                       eligible_gpus=gpus)
            raise RuntimeError("formal requires at least two eligible GPUs")
        root = self.out / "formal" / candidate
        jobs = [
            self.eval_job(candidate, f"formal_{candidate}_shard{index}", root / f"shard{index}.result.json",
                          shard=(index, count), progress=root / f"shard{index}.progress.json")
            for index in range(count)
        ]
        self.run_jobs("FORMAL976", jobs)
        output = root / "result.json"
        if not self.output_pass(output):
            shards = [root / f"shard{index}.result.json" for index in range(count)]
            subprocess.run([sys.executable, str(EVALUATOR), "--merge-shards", *map(str, shards),
                            "--expected-sample-count", "976", "--out", str(output)], cwd=ROOT, check=True)
        return output

    def freeze_host(self, candidate: str, formal: dict[str, Any]) -> None:
        gates = self.spec["formal_gates"]
        fixed19 = float(formal["semantic"]["sdk_quantized_chain"]["fixed_19_large_object_macro_iou"])
        approved_pass = fixed19 >= float(gates["approved_fixed19_min"])
        project_pass = fixed19 >= float(gates["project_fixed19_min"])
        rans_pass = int(formal.get("real_rans_success_count", 0)) == int(gates["real_rans_required"])
        report = {
            "schema": "semantic_sensing.fullmodule_host_gate.v1",
            "status": "PASS" if approved_pass and rans_pass else "FAIL",
            "candidate": candidate, "fixed19": fixed19,
            "approved_workflow_gate": {"minimum": gates["approved_fixed19_min"], "pass": approved_pass},
            "current_project_gate": {"minimum": gates["project_fixed19_min"], "pass": project_pass},
            "real_rans": {"required": gates["real_rans_required"],
                          "actual": formal.get("real_rans_success_count"), "pass": rans_pass},
            "formal_result": {"path": str(self.out / "formal" / candidate / "result.json"),
                              "sha256": sha256(self.out / "formal" / candidate / "result.json")},
            "selection_manifest": {"path": str(self.selection_paths(candidate)[0]),
                                   "sha256": sha256(self.selection_paths(candidate)[0])},
            "boundary_manifest": {"path": str(self.selection_paths(candidate)[1]),
                                  "sha256": sha256(self.selection_paths(candidate)[1])},
        }
        atomic_json(self.out / "freeze" / "host_candidate_manifest.json", report)
        if report["status"] != "PASS":
            self.state("STOP_FORMAL_FAIL", status="STOP_FORMAL_FAIL", host_gate=report)
            raise RuntimeError("formal host gate failed")
        self.state("COMPLETE_HOST", status="COMPLETE_HOST", host_gate=report)

    def run(self) -> None:
        preflight = self.validate_spec(require_out_absent=not self.resume)
        self.setup(preflight)
        self.state("G1_REUSE_AUDIT")
        self.audit()
        self.run_candidate_smokes(["C1"])

        screen_count = int(self.spec["scheduler"]["screen_shards"])
        concurrent = [self.ptq_job("C2"), self.ptq_job("C3")]
        concurrent += [self.screen_job("C1", index, screen_count) for index in range(screen_count)]
        outcomes = self.run_jobs("C1_SCREEN_AND_C2_C3_PTQ", concurrent, allow_failures=True)
        if not all(outcomes.get(f"C1_screen64_shard{index}", False) for index in range(screen_count)):
            raise RuntimeError("C1 screen64 failed")
        self.merge_screen("C1")

        active = ["C1"]
        for candidate in ("C2", "C3"):
            if outcomes.get(f"{candidate}_bn_h_s_ptq", False):
                self.freeze(candidate)
                active.append(candidate)
        if len(active) > 1:
            self.run_candidate_smokes(active[1:])
            screen_jobs = [self.screen_job(candidate, index, screen_count)
                           for candidate in active[1:] for index in range(screen_count)]
            screen_outcomes = self.run_jobs("C2_C3_SCREEN64", screen_jobs, allow_failures=True)
            active = [candidate for candidate in active if candidate == "C1" or all(
                screen_outcomes.get(f"{candidate}_screen64_shard{index}", False)
                for index in range(screen_count)
            )]
            for candidate in active[1:]:
                self.merge_screen(candidate)

        decision = self.decide(active)
        self.state("CANDIDATE_DECISION", decision=decision)
        if decision["status"] != "UNIQUE_SELECT":
            self.state(decision["status"], status=decision["status"], decision=decision)
            return
        selected = str(decision["selected"])
        formal_path = self.formal(selected)
        self.freeze_host(selected, load_json(formal_path))


def print_plan(spec_path: Path, approved_sha: str) -> None:
    campaign = Campaign(spec_path, approved_sha)
    preflight = campaign.validate_spec(require_out_absent=True)
    plan = {
        "schema": "semantic_sensing.fullmodule_selective_plan.v1", "preflight": preflight,
        "out_root": str(campaign.out), "candidates": list(campaign.spec["candidates"]),
        "dag": ["audit", "C1 golden1/golden8", "C1 screen64 || C2/C3 PTQ",
                "C2/C3 golden/screen", "deterministic select", "one formal976", "freeze host"],
        "eligible_gpus": campaign.eligible_gpus(),
    }
    print(json.dumps(plan, indent=2, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    sub = value.add_subparsers(dest="command", required=True)
    for name in ("validate", "plan", "run"):
        item = sub.add_parser(name)
        item.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
        item.add_argument("--approved-spec-sha", required=True)
        if name == "run":
            item.add_argument("--execute", action="store_true")
    resume = sub.add_parser("resume")
    resume.add_argument("--out-root", type=Path, required=True)
    resume.add_argument("--approved-spec-sha", required=True)
    resume.add_argument("--execute", action="store_true")
    status = sub.add_parser("status")
    status.add_argument("--out-root", type=Path, required=True)
    return value


def main() -> int:
    args = parser().parse_args()
    if args.command == "status":
        print(json.dumps(load_json(args.out_root / "campaign_state.json"), indent=2, sort_keys=True))
        return 0
    if args.command in {"validate", "plan"}:
        print_plan(args.spec, args.approved_spec_sha)
        return 0
    if not args.execute:
        raise PermissionError("execution requires the explicit --execute flag")
    if args.command == "run":
        campaign = Campaign(args.spec, args.approved_spec_sha)
    else:
        manifest = load_json(args.out_root / "campaign_manifest.json")
        campaign = Campaign(Path(manifest["spec"]["path"]), args.approved_spec_sha, resume=True)
        if campaign.out != args.out_root.resolve():
            raise ValueError("--out-root differs from immutable campaign manifest")
    try:
        campaign.run()
    except BaseException as exc:
        if campaign.out.exists():
            current = load_json(campaign.state_path) if campaign.state_path.is_file() else {}
            if current.get("status") not in {
                "PAUSED", "PAUSED_AMBIGUOUS_SELECTION", "STOP_NO_ELIGIBLE_CANDIDATE", "STOP_FORMAL_FAIL",
            }:
                campaign.state("FAILED_INFRASTRUCTURE", status="FAILED_INFRASTRUCTURE",
                               error=f"{type(exc).__name__}: {exc}")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
