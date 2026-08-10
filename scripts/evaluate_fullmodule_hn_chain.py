#!/usr/bin/env python3
"""Evaluate the current split model as an x86 Hailo SDK-quantized module chain.

This is deliberately a *host simulator* gate, not a Raspberry-Pi result.  It
links the independently calibrated Hailo modules with the real CPU rANS codec:

  image -> sender -> align -> g_a -> h_a -> rANS(z) -> h_s -> rANS(y)
        -> g_s -> receiver stage4/stage5 -> DAPPM -> head.

For each held-out sample it records cumulative module-boundary cosine metrics,
the final segmentation logits cosine and mIoU.  The board evaluator is the
separate source of target-side timing and board-vs-host parity.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if __name__ == "__main__":
    from hailo_deploy.logging import reexec_if_unlogged

    reexec_if_unlogged(Path(__file__), sys.argv[1:])

from hailo_deploy.emulate import _match_outputs
from hailo_deploy.io.calib import map_hn_inputs
from hailo_deploy.io.fullmodule_selection import (
    MODULES,
    load_selection,
)
from hailo_deploy.io.hash import sha256
from hailo_deploy.io.metrics import metric, summarize
from hailo_deploy.onnx_io import nchw_to_nhwc


LARGE = [7, 10, 15, 20, 23, 30, 35, 37, 48, 51, 80, 83, 84, 90, 101, 102, 103, 109, 122]
IMAGE_PIXELS = 512 * 512


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Publish a report only after a complete JSON payload is on disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def load_student(checkpoint: Path):
    repo = Path(os.environ.get("SEMANTIC_SENSING_ROOT", "/workspace/semantic-sensing"))
    sys.path.insert(0, str(repo))
    from src import _compat  # noqa: F401
    from src.models import build_model
    from src.models.student import build_student

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = state["cfg"]
    cfg.model.backbone.args.pretrained = None
    teacher = build_model(cfg.model).eval()
    student = build_student(cfg, teacher).eval()
    student.load_state_dict(state["student"], strict=True)
    student.update(force=True)
    return student.eval(), cfg


def code_size(strings: list[bytes], *, fixed_overhead: int = 0) -> int:
    return fixed_overhead + sum(4 + len(item) for item in strings)


def z_roundtrip(codec, z: np.ndarray) -> tuple[np.ndarray, int]:
    value = torch.from_numpy(np.ascontiguousarray(z)).float()
    with torch.inference_mode():
        strings = codec.entropy_bottleneck.compress(value)
        hat = codec.entropy_bottleneck.decompress(strings, value.shape[-2:])
    return hat.numpy().astype(np.float32, copy=False), code_size(strings)


def y_roundtrip(codec, y: np.ndarray, means: np.ndarray, scales: np.ndarray) -> tuple[np.ndarray, int]:
    y_t = torch.from_numpy(np.ascontiguousarray(y)).float()
    means_t = torch.from_numpy(np.ascontiguousarray(means)).float()
    scales_t = torch.from_numpy(np.ascontiguousarray(scales)).float()
    with torch.inference_mode():
        indexes = codec.gaussian_conditional.build_indexes(scales_t)
        strings = codec.gaussian_conditional.compress(y_t, indexes, means=means_t)
        hat = codec.gaussian_conditional.decompress(strings, indexes, means=means_t)
    return hat.numpy().astype(np.float32, copy=False), code_size(strings)


def output_metrics(reference: dict[str, np.ndarray], actual: dict[str, np.ndarray]) -> dict[str, dict[str, float | None]]:
    if set(reference) != set(actual):
        raise ValueError(f"output coverage mismatch: {sorted(reference)} != {sorted(actual)}")
    return {name: metric(reference[name], actual[name]) for name in reference}


def scalar_summary(values: list[float]) -> dict[str, float | int | None]:
    data = np.asarray(values, dtype=np.float64)
    if not data.size:
        return {"count": 0, "mean": None, "min": None, "p50": None, "p95": None, "max": None}
    return {"count": int(data.size), "mean": float(data.mean()), "min": float(data.min()),
            "p50": float(np.percentile(data, 50)), "p95": float(np.percentile(data, 95)),
            "max": float(data.max())}


def update_confusion(confusion: np.ndarray, logits: np.ndarray, target: np.ndarray, ignore: int) -> None:
    classes = confusion.shape[0]
    pred = np.asarray(logits).argmax(axis=1).astype(np.int64, copy=False)
    truth = np.asarray(target, dtype=np.int64)
    valid = truth != ignore
    bins = np.bincount((truth[valid] * classes + pred[valid]).ravel(), minlength=classes * classes)
    confusion += bins.reshape(classes, classes)


def segmentation_metrics(confusion: np.ndarray) -> dict[str, Any]:
    data = confusion.astype(np.float64, copy=False)
    tp = np.diag(data)
    union = data.sum(0) + data.sum(1) - tp
    iou = np.divide(tp, np.maximum(union, 1.0))
    present = union > 0
    return {
        "mIoU": float(iou[present].mean()),
        "pixel_acc": float(tp.sum() / max(data.sum(), 1.0)),
        "fixed_19_large_object_macro_iou": float(iou[LARGE].mean()),
        "per_class_iou": [float(item) for item in iou],
    }


def semantic_delta(fp_metrics: dict[str, Any], quant_metrics: dict[str, Any]) -> dict[str, float]:
    return {key: float(quant_metrics[key] - fp_metrics[key])
            for key in ("mIoU", "pixel_acc", "fixed_19_large_object_macro_iou")}


def bitstream_totals(rows: list[dict[str, float | int]]) -> dict[str, int]:
    sample_count = len(rows)
    return {
        "sample_count": sample_count,
        "fp_z_bytes": sum(int(row["fp_z_bytes"]) for row in rows),
        "fp_y_bytes": sum(int(row["fp_y_bytes"]) for row in rows),
        "quant_z_bytes": sum(int(row["quant_z_bytes"]) for row in rows),
        "quant_y_bytes": sum(int(row["quant_y_bytes"]) for row in rows),
        "container_overhead_bytes": sample_count * 29,
        "fp_bytes": sum(int(row["fp_bytes"]) for row in rows),
        "quant_bytes": sum(int(row["quant_bytes"]) for row in rows),
        "image_pixels": sample_count * IMAGE_PIXELS,
        "container_overhead_bytes_per_image": 29,
    }


def bpp_from_totals(totals: dict[str, int]) -> dict[str, dict[str, float | int | None]]:
    pixels = int(totals["image_pixels"])
    samples = int(totals["sample_count"])

    def summary(byte_key: str) -> dict[str, float | int | None]:
        total_bytes = int(totals[byte_key])
        return {
            "count": samples,
            "mean": (total_bytes * 8.0 / pixels) if pixels else None,
            "total_bytes": total_bytes,
        }

    return {
        "fp": {**summary("fp_bytes"), "z_bytes": int(totals["fp_z_bytes"]),
               "y_bytes": int(totals["fp_y_bytes"])},
        "sdk_quantized": {**summary("quant_bytes"), "z_bytes": int(totals["quant_z_bytes"]),
                          "y_bytes": int(totals["quant_y_bytes"])},
        "container_overhead_bytes": int(totals["container_overhead_bytes"]),
        "container_overhead_bytes_per_image": int(totals["container_overhead_bytes_per_image"]),
    }

class Chain:
    def __init__(self, handoff: Path, result_paths: dict[str, Path], fp_modules: set[str] | None = None):
        from hailo_sdk_client import ClientRunner, InferenceContext

        self.handoff = handoff
        calibration_root = handoff / "calib"
        if not calibration_root.is_dir():
            calibration_root = handoff / "calibration"
        self.meta = {name: load_json(calibration_root / name / "meta.json") for name in MODULES}
        self.sessions: dict[str, ort.InferenceSession] = {}
        self.runners: dict[str, Any] = {}
        self.contexts: dict[str, Any] = {}
        self.context_managers: list[Any] = []
        self.notes: dict[str, list[str]] = defaultdict(list)
        # This is a host-emulator localization aid only.  A selected module
        # runs its ONNX graph on the *quantized-branch inputs*, which removes
        # only that module's numerical quantization while preserving all
        # upstream quantization error.  Such a mixed chain is deliberately
        # not a deployable HEF candidate.
        self.fp_modules = set(fp_modules or ())
        self.ort_provider = ("CUDAExecutionProvider" if "CUDAExecutionProvider" in ort.get_available_providers()
                             else "CPUExecutionProvider")
        providers: list[object] = ([('CUDAExecutionProvider', {'use_tf32': '0'}), 'CPUExecutionProvider']
                                   if self.ort_provider == "CUDAExecutionProvider" else ['CPUExecutionProvider'])
        for name in MODULES:
            result = load_json(result_paths[name])
            if result.get("status") != "PASS":
                raise ValueError(f"{name} does not have a PASS PTQ result")
            har = Path(result["quantized_har"]["path"])
            if sha256(har) != result["quantized_har"]["sha256"]:
                raise ValueError(f"quantized HAR hash mismatch: {name}")
            onnx = handoff / "onnx" / f"{name}.onnx"
            self.sessions[name] = ort.InferenceSession(str(onnx), providers=providers)
            runner = ClientRunner(har=str(har))
            self.runners[name] = runner
            manager = runner.infer_context(InferenceContext.SDK_QUANTIZED)
            self.context_managers.append(manager)
            self.contexts[name] = manager.__enter__()

    def close(self) -> None:
        for manager in reversed(self.context_managers):
            manager.__exit__(None, None, None)

    def fp(self, name: str, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        session = self.sessions[name]
        values = session.run(None, {key: np.ascontiguousarray(value) for key, value in inputs.items()})
        return {item.name: np.asarray(value, dtype=np.float32) for item, value in zip(session.get_outputs(), values)}

    def quant(self, name: str, inputs: dict[str, np.ndarray], expected: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], float]:
        if name in self.fp_modules:
            if not self.notes[name]:
                self.notes[name].append(
                    "FP bypass runs ONNX on quantized-branch inputs for localization; not a deployable result."
                )
            begin = time.perf_counter_ns()
            values = self.fp(name, inputs)
            elapsed_ms = (time.perf_counter_ns() - begin) / 1e6
            return values, elapsed_ms
        runner = self.runners[name]
        feed, _ = map_hn_inputs(runner, {key: nchw_to_nhwc(value) for key, value in inputs.items()})
        begin = time.perf_counter_ns()
        raw = runner.infer(self.contexts[name], feed)
        elapsed_ms = (time.perf_counter_ns() - begin) / 1e6
        pairs = _match_outputs(runner, expected, raw, self.notes[name])
        if set(pairs) != set(expected):
            raise ValueError(f"failed to map all Hailo outputs for {name}: {self.notes[name][-3:]}")
        return {key: np.ascontiguousarray(value[1], dtype=np.float32) for key, value in pairs.items()}, elapsed_ms


def resolve_module_results(
    ptq_root: Path | None, selection_manifest: Path | None
) -> tuple[dict[str, Path], dict[str, Any]]:
    if selection_manifest is not None:
        selection_manifest = selection_manifest.resolve()
        selection = load_selection(selection_manifest, verify_files=False)
        paths = {module: Path(record["result_json"]["path"])
                 for module, record in selection["modules"].items()}
        return paths, {
            "ptq_root": None,
            "selection_manifest": str(selection_manifest),
            "selection_manifest_sha256": sha256(selection_manifest),
            "candidate": selection["candidate"],
            "candidate_fingerprint": selection["candidate_fingerprint"],
        }
    if ptq_root is None:
        raise ValueError("one of --ptq-root or --selection-manifest is required")
    ptq_root = ptq_root.resolve()
    return {name: ptq_root / name / "result.json" for name in MODULES}, {
        "ptq_root": str(ptq_root), "selection_manifest": None,
        "selection_manifest_sha256": None, "candidate": None, "candidate_fingerprint": None,
    }


def validate(
    handoff: Path, result_paths: dict[str, Path]
) -> tuple[dict[str, Any], dict[str, Any], Path, str]:
    identity = load_json(handoff / "identity.json")
    checkpoint_sha = identity.get("checkpoint", {}).get("sha256")
    if not isinstance(checkpoint_sha, str) or len(checkpoint_sha) != 64:
        raise ValueError("handoff has no valid checkpoint SHA-256")
    model = identity.get("model", {})
    if model.get("gfa_mode") != "group_softmax" or model.get("split") != "mid":
        raise ValueError("handoff is not the supported group-softmax mid split")
    if identity.get("splits", {}).get("intersection_empty") is not True:
        raise ValueError("handoff does not prove calibration/evaluation disjointness")
    report = load_json(handoff / "handoff_report.json")
    if report.get("status") != "PASS":
        raise ValueError("training handoff report is not PASS")
    meta_path = handoff / "evaluation" / "meta.json"
    if not meta_path.is_file():
        meta_path = handoff / "evaluation" / "manifest.json"
    meta = load_json(meta_path)
    val_indices = meta.get("val_indices", [])
    calibration = identity["splits"].get("calibration_indices", {})
    if isinstance(calibration, dict) and {"start", "end_inclusive"} <= set(calibration):
        calibration_indices = set(range(int(calibration["start"]), int(calibration["end_inclusive"]) + 1))
    elif isinstance(calibration, list):
        calibration_indices = set(calibration)
    else:
        raise ValueError("handoff does not declare calibration indices")
    if set(val_indices) & calibration_indices:
        raise ValueError("evaluation split overlaps calibration")
    for name in MODULES:
        if not result_paths[name].is_file():
            raise FileNotFoundError(result_paths[name])
    checkpoints = sorted((handoff / "checkpoint").glob("*.pth"))
    if len(checkpoints) != 1:
        raise ValueError(f"expected exactly one final .pth checkpoint, found {checkpoints}")
    if sha256(checkpoints[0]) != checkpoint_sha:
        raise ValueError("local checkpoint hash differs from handoff identity")
    return identity, meta, checkpoints[0], checkpoint_sha


def merge_shards() -> None:
    parser = argparse.ArgumentParser(description="Merge deterministic full-module evaluation shards.")
    parser.add_argument("--merge-shards", nargs="+", type=Path, required=True,
                        help="PASS shard result JSON files from one candidate")
    parser.add_argument("--expected-sample-count", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite: {args.out}")
    if args.expected_sample_count <= 0:
        raise ValueError("--expected-sample-count must be positive")
    if len(set(args.merge_shards)) != len(args.merge_shards):
        raise ValueError("--merge-shards contains a duplicate path")

    reports: list[tuple[Path, dict[str, Any]]] = []
    for path in args.merge_shards:
        if not path.is_file():
            raise FileNotFoundError(path)
        report = load_json(path)
        if report.get("kind") != "x86_sdk_quantized_full_module_chain_semantic_evaluation":
            raise ValueError(f"not a direct evaluator result: {path}")
        if report.get("status") != "PASS":
            raise ValueError(f"shard is not PASS: {path}")
        reports.append((path, report))

    reference: tuple[Any, ...] | None = None
    expected_num_shards: int | None = None
    seen_shards: set[int] = set()
    seen_indices: set[int] = set()
    merged_fp: np.ndarray | None = None
    merged_quant: np.ndarray | None = None
    argmax_equal_pixels = 0
    argmax_pixels = 0
    additive_byte_keys = (
        "fp_z_bytes", "fp_y_bytes", "quant_z_bytes", "quant_y_bytes",
        "container_overhead_bytes", "fp_bytes", "quant_bytes",
    )
    totals = {"sample_count": 0, "image_pixels": 0, "container_overhead_bytes_per_image": 29,
              **{key: 0 for key in additive_byte_keys}}
    sources: list[dict[str, Any]] = []
    for path, report in reports:
        shard = report.get("shard")
        if not isinstance(shard, dict):
            raise ValueError(f"shard provenance missing: {path}")
        shard_index = shard.get("index")
        num_shards = shard.get("num_shards")
        if not isinstance(shard_index, int) or not isinstance(num_shards, int):
            raise ValueError(f"invalid shard metadata: {path}")
        if num_shards <= 0 or not 0 <= shard_index < num_shards:
            raise ValueError(f"invalid shard bounds: {path}")
        if expected_num_shards is None:
            expected_num_shards = num_shards
        elif expected_num_shards != num_shards:
            raise ValueError("all shards must use the same --num-shards")
        if shard_index in seen_shards:
            raise ValueError(f"duplicate shard index {shard_index}")
        seen_shards.add(shard_index)

        full_count = report.get("full_validation_sample_count")
        fingerprint = (
            report.get("handoff_root"), report.get("ptq_root"), report.get("selection_manifest"),
            report.get("selection_manifest_sha256"), report.get("candidate_fingerprint"),
            report.get("checkpoint_sha256"), tuple(report.get("fp_bypass_modules", [])), full_count,
        )
        if reference is None:
            reference = fingerprint
        elif reference != fingerprint:
            raise ValueError("shards do not share one immutable evaluation contract")

        indices = report.get("val_indices")
        if not isinstance(indices, list) or any(not isinstance(index, int) for index in indices):
            raise ValueError(f"invalid val_indices: {path}")
        if int(report.get("sample_count", -1)) != len(indices):
            raise ValueError(f"sample_count mismatch: {path}")
        if int(report.get("real_rans_success_count", -1)) != len(indices):
            raise ValueError(f"real-rANS success count mismatch: {path}")
        local_indices = set(indices)
        if len(local_indices) != len(indices) or seen_indices & local_indices:
            raise ValueError(f"validation indices overlap across shards: {path}")
        seen_indices.update(local_indices)

        raw_confusions = report.get("confusion_matrices")
        if not isinstance(raw_confusions, dict):
            raise ValueError(f"raw confusion matrices missing: {path}")
        local_matrices: list[np.ndarray] = []
        for key in ("fp32_onnx", "sdk_quantized"):
            matrix = np.asarray(raw_confusions.get(key), dtype=np.int64)
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or np.any(matrix < 0):
                raise ValueError(f"invalid {key} confusion matrix: {path}")
            local_matrices.append(matrix)
        if merged_fp is None:
            merged_fp, merged_quant = np.zeros_like(local_matrices[0]), np.zeros_like(local_matrices[1])
        if local_matrices[0].shape != merged_fp.shape or local_matrices[1].shape != merged_quant.shape:
            raise ValueError(f"confusion shape mismatch: {path}")
        merged_fp += local_matrices[0]
        merged_quant += local_matrices[1]

        local_argmax = report.get("final_argmax")
        if not isinstance(local_argmax, dict):
            raise ValueError(f"final argmax totals missing: {path}")
        for key in ("equal_pixels", "pixels"):
            if not isinstance(local_argmax.get(key), int) or local_argmax[key] < 0:
                raise ValueError(f"invalid final argmax {key}: {path}")
        if local_argmax["equal_pixels"] > local_argmax["pixels"]:
            raise ValueError(f"final argmax equal count exceeds pixels: {path}")
        argmax_equal_pixels += int(local_argmax["equal_pixels"])
        argmax_pixels += int(local_argmax["pixels"])

        local_totals = report.get("bitstream_totals")
        if not isinstance(local_totals, dict):
            raise ValueError(f"raw bitstream totals missing: {path}")
        for key in ("sample_count", "image_pixels", "container_overhead_bytes_per_image", *additive_byte_keys):
            value = local_totals.get(key)
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"invalid bitstream total {key}: {path}")
        if local_totals["sample_count"] != len(indices):
            raise ValueError(f"bitstream sample count mismatch: {path}")
        if local_totals["image_pixels"] != len(indices) * IMAGE_PIXELS:
            raise ValueError(f"unexpected image-pixel total: {path}")
        if local_totals["container_overhead_bytes_per_image"] != totals["container_overhead_bytes_per_image"]:
            raise ValueError(f"container overhead mismatch: {path}")
        for key in ("sample_count", "image_pixels", *additive_byte_keys):
            totals[key] += int(local_totals[key])
        sources.append({"path": str(path), "sha256": sha256(path), "shard_index": shard_index,
                        "sample_count": len(indices)})

    if expected_num_shards is None or len(reports) != expected_num_shards:
        raise ValueError("must provide exactly one result for every shard")
    if seen_shards != set(range(expected_num_shards)):
        raise ValueError(f"incomplete shard set: {sorted(seen_shards)}")
    if len(seen_indices) != args.expected_sample_count or totals["sample_count"] != args.expected_sample_count:
        raise ValueError("merged sample count does not match --expected-sample-count")
    if reference is None or merged_fp is None or merged_quant is None:
        raise ValueError("no shard reports to merge")

    fp_metrics, quant_metrics = segmentation_metrics(merged_fp), segmentation_metrics(merged_quant)
    (handoff_root, ptq_root, selection_manifest, selection_manifest_sha, candidate_fingerprint,
     checkpoint_sha, fp_bypass_modules, full_validation_count) = reference
    result = {
        "kind": "x86_sdk_quantized_full_module_chain_semantic_evaluation_shard_merge",
        "status": "PASS", "scope": "Exact global merge of deterministic Hailo SDK_QUANTIZED host shards; not Raspberry-Pi timing",
        "handoff_root": handoff_root, "ptq_root": ptq_root,
        "selection_manifest": selection_manifest,
        "selection_manifest_sha256": selection_manifest_sha,
        "candidate_fingerprint": candidate_fingerprint,
        "checkpoint_sha256": checkpoint_sha,
        "fp_bypass_modules": list(fp_bypass_modules), "sample_count": len(seen_indices),
        "real_rans_success_count": len(seen_indices),
        "val_indices": sorted(seen_indices), "full_validation_sample_count": full_validation_count,
        "shard_merge": {"num_shards": expected_num_shards, "selection": "round_robin",
                        "source_reports": sorted(sources, key=lambda item: item["shard_index"])},
        "confusion_matrices": {"fp32_onnx": merged_fp.tolist(), "sdk_quantized": merged_quant.tolist()},
        "final_argmax": {"equal_pixels": argmax_equal_pixels, "pixels": argmax_pixels,
                         "agreement": argmax_equal_pixels / max(argmax_pixels, 1)},
        "semantic": {"fp32_onnx_chain": fp_metrics, "sdk_quantized_chain": quant_metrics,
                     "quant_minus_fp": semantic_delta(fp_metrics, quant_metrics)},
        "bitstream": bpp_from_totals(totals), "bitstream_totals": totals,
        "merge_notes": ["Confusion matrices and byte totals are summed before metrics are computed.",
                        "Per-shard percentile timing and module metrics are intentionally not averaged."],
    }
    atomic_write_json(args.out, result)
    print(json.dumps({"out": str(args.out), "samples": len(seen_indices),
                      "quant_mIoU": quant_metrics["mIoU"]}, indent=2))



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff-root", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--ptq-root", type=Path,
                        help="legacy root containing <module>/result.json")
    source.add_argument("--selection-manifest", type=Path,
                        help="immutable mixed-root 10-module selection manifest")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None,
                        help="first N samples of this selected shard; omit for the complete shard")
    parser.add_argument("--shard-index", type=int, default=None,
                        help="zero-based deterministic round-robin shard index")
    parser.add_argument("--num-shards", type=int, default=None,
                        help="number of deterministic round-robin shards")
    parser.add_argument("--progress-out", type=Path, default=None,
                        help="atomically refreshed durable progress JSON")
    parser.add_argument("--checkpoint-every", type=int, default=10,
                        help="write --progress-out after this many completed samples")
    parser.add_argument("--fp-modules", nargs="*", default=[], choices=MODULES,
                        help="host-only quantization ablation: bypass these modules with FP ONNX outputs")
    args = parser.parse_args()
    if (args.shard_index is None) != (args.num_shards is None):
        raise ValueError("--shard-index and --num-shards must be supplied together")
    if args.num_shards is not None and (args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards):
        raise ValueError("invalid --shard-index/--num-shards")
    if args.checkpoint_every <= 0:
        raise ValueError("--checkpoint-every must be positive")
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite: {args.out}")
    if args.progress_out is not None:
        if args.progress_out == args.out:
            raise ValueError("--progress-out must differ from --out")

    result_paths, selection_identity = resolve_module_results(args.ptq_root, args.selection_manifest)
    identity, eval_meta, checkpoint, checkpoint_sha = validate(args.handoff_root, result_paths)
    all_indices = list(eval_meta["val_indices"])
    shard_index = args.shard_index if args.shard_index is not None else 0
    num_shards = args.num_shards if args.num_shards is not None else 1
    shard = {"index": shard_index, "num_shards": num_shards, "selection": "round_robin"}
    indices = all_indices[shard_index::num_shards]
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        indices = indices[:args.limit]

    student, cfg = load_student(checkpoint)
    codec = student.bottleneck
    codec.eval()
    chain = Chain(args.handoff_root, result_paths, set(args.fp_modules))
    module_rows: dict[str, dict[str, list[dict[str, float | None]]]] = {
        name: defaultdict(list) for name in MODULES
    }
    timings: dict[str, list[float]] = defaultdict(list)
    fp_confusion = np.zeros((int(cfg.num_classes), int(cfg.num_classes)), dtype=np.int64)
    quant_confusion = np.zeros_like(fp_confusion)
    bitstream_rows: list[dict[str, float | int]] = []
    sample_rows: list[dict[str, Any]] = []
    final_argmax_equal_pixels = 0
    final_argmax_pixels = 0
    image_root = args.handoff_root / "evaluation" / "images"
    target_root = args.handoff_root / "evaluation" / "ground_truth"
    if not target_root.is_dir():
        target_root = args.handoff_root / "evaluation" / "labels"

    def write_progress(completed: int, status: str) -> None:
        if args.progress_out is None:
            return
        partial_fp = segmentation_metrics(fp_confusion)
        partial_quant = segmentation_metrics(quant_confusion)
        payload = {
            "kind": "x86_sdk_quantized_full_module_chain_semantic_evaluation_progress",
            "status": status, "handoff_root": str(args.handoff_root), **selection_identity,
            "checkpoint_sha256": checkpoint_sha, "fp_bypass_modules": sorted(chain.fp_modules),
            "shard": shard, "full_validation_sample_count": len(all_indices),
            "selected_sample_count": len(indices), "processed_count": completed,
            "processed_indices": indices[:completed],
            "confusion_matrices": {"fp32_onnx": fp_confusion.tolist(), "sdk_quantized": quant_confusion.tolist()},
            "semantic": {"fp32_onnx_chain": partial_fp, "sdk_quantized_chain": partial_quant,
                         "quant_minus_fp": semantic_delta(partial_fp, partial_quant)},
            "bitstream_totals": bitstream_totals(bitstream_rows), "updated_unix_seconds": time.time(),
        }
        atomic_write_json(args.progress_out, payload)

    try:
        for position, index in enumerate(indices, start=1):
            image = np.load(image_root / f"{index}.npy", allow_pickle=False).astype(np.float32, copy=False)[None]
            target = np.load(target_root / f"{index}.npy", allow_pickle=False).astype(np.int64, copy=False)[None]
            fp: dict[str, dict[str, np.ndarray]] = {}
            q: dict[str, dict[str, np.ndarray]] = {}

            def run(name: str, fp_inputs: dict[str, np.ndarray], q_inputs: dict[str, np.ndarray]) -> None:
                fp[name] = chain.fp(name, fp_inputs)
                q[name], elapsed = chain.quant(name, q_inputs, fp[name])
                timings[name].append(elapsed)
                for output, values in output_metrics(fp[name], q[name]).items():
                    module_rows[name][output].append(values)

            run("sender_backbone", {"image": image}, {"image": image})
            run("bn_align_h", {"x_h": fp["sender_backbone"]["x_h"]}, {"x_h": q["sender_backbone"]["x_h"]})
            run("bn_g_a", {
                "aligned_x_h": fp["bn_align_h"]["aligned_x_h"], "x_l": fp["sender_backbone"]["x_l"],
            }, {
                "aligned_x_h": q["bn_align_h"]["aligned_x_h"], "x_l": q["sender_backbone"]["x_l"],
            })
            run("bn_h_a", {"latent_y": fp["bn_g_a"]["latent_y"]}, {"latent_y": q["bn_g_a"]["latent_y"]})

            codec_start = time.perf_counter_ns()
            fp_z_hat, fp_z_bytes = z_roundtrip(codec, fp["bn_h_a"]["latent_z"])
            q_z_hat, q_z_bytes = z_roundtrip(codec, q["bn_h_a"]["latent_z"])
            timings["cpu_rans_z"].append((time.perf_counter_ns() - codec_start) / 1e6)
            run("bn_h_s", {"latent_z_hat": fp_z_hat}, {"latent_z_hat": q_z_hat})

            codec_start = time.perf_counter_ns()
            fp_y_hat, fp_y_bytes = y_roundtrip(codec, fp["bn_g_a"]["latent_y"], fp["bn_h_s"]["means"], fp["bn_h_s"]["scales"])
            q_y_hat, q_y_bytes = y_roundtrip(codec, q["bn_g_a"]["latent_y"], q["bn_h_s"]["means"], q["bn_h_s"]["scales"])
            timings["cpu_rans_y"].append((time.perf_counter_ns() - codec_start) / 1e6)
            run("bn_g_s", {"latent_y_hat": fp_y_hat}, {"latent_y_hat": q_y_hat})
            run("receiver_stage4", {
                "x_h": fp["bn_g_s"]["x_h_hat"], "x_l": fp["bn_g_s"]["x_l_hat"],
            }, {
                "x_h": q["bn_g_s"]["x_h_hat"], "x_l": q["bn_g_s"]["x_l_hat"],
            })
            run("receiver_stage5", {
                "stage4_high": fp["receiver_stage4"]["stage4_high"], "stage4_low": fp["receiver_stage4"]["stage4_low"],
            }, {
                "stage4_high": q["receiver_stage4"]["stage4_high"], "stage4_low": q["receiver_stage4"]["stage4_low"],
            })
            run("receiver_dappm", {
                "high_feat": fp["receiver_stage5"]["high_feat"], "low_feat": fp["receiver_stage5"]["low_feat"],
            }, {
                "high_feat": q["receiver_stage5"]["high_feat"], "low_feat": q["receiver_stage5"]["low_feat"],
            })
            run("receiver_head", {"s8": fp["receiver_dappm"]["s8"]}, {"s8": q["receiver_dappm"]["s8"]})

            update_confusion(fp_confusion, fp["receiver_head"]["seg_logits"], target, int(cfg.ignore_index))
            update_confusion(quant_confusion, q["receiver_head"]["seg_logits"], target, int(cfg.ignore_index))
            bitstream_rows.append({
                "val_index": index,
                "fp_z_bytes": fp_z_bytes, "fp_y_bytes": fp_y_bytes,
                "quant_z_bytes": q_z_bytes, "quant_y_bytes": q_y_bytes,
                "fp_bytes": fp_z_bytes + fp_y_bytes + 29,
                "quant_bytes": q_z_bytes + q_y_bytes + 29,
                "fp_bpp": (fp_z_bytes + fp_y_bytes + 29) * 8.0 / IMAGE_PIXELS,
                "quant_bpp": (q_z_bytes + q_y_bytes + 29) * 8.0 / IMAGE_PIXELS,
            })
            fp_argmax = fp["receiver_head"]["seg_logits"].argmax(axis=1)
            quant_argmax = q["receiver_head"]["seg_logits"].argmax(axis=1)
            equal_pixels = int(np.count_nonzero(fp_argmax == quant_argmax))
            pixels = int(fp_argmax.size)
            final_argmax_equal_pixels += equal_pixels
            final_argmax_pixels += pixels
            sample_rows.append({
                "val_index": index,
                "final_logits": module_rows["receiver_head"]["seg_logits"][-1],
                "final_argmax_agreement": equal_pixels / max(pixels, 1),
            })
            if position % args.checkpoint_every == 0 or position == len(indices):
                write_progress(position, "RUNNING")
            if position % 10 == 0 or position == len(indices):
                print(f"evaluated {position}/{len(indices)}", flush=True)
    finally:
        chain.close()

    write_progress(len(indices), "COMPLETE")
    module_summary = {
        module: {output: summarize(rows) for output, rows in outputs.items()}
        for module, outputs in module_rows.items()
    }
    timing_summary = {name: scalar_summary(values) for name, values in timings.items()}
    fp_metrics, quant_metrics = segmentation_metrics(fp_confusion), segmentation_metrics(quant_confusion)
    totals = bitstream_totals(bitstream_rows)
    result = {
        "kind": "x86_sdk_quantized_full_module_chain_semantic_evaluation",
        "status": "PASS", "scope": "Hailo SDK_QUANTIZED host simulation; not Raspberry-Pi timing",
        "handoff_root": str(args.handoff_root), **selection_identity,
        "fp_bypass_modules": sorted(chain.fp_modules),
        "checkpoint_sha256": checkpoint_sha, "sample_count": len(indices), "val_indices": indices,
        "real_rans_success_count": len(indices),
        "shard": shard, "full_validation_sample_count": len(all_indices),
        "calibration_overlap": [], "onnxruntime_providers": ort.get_available_providers(),
        "fp32_onnx_provider": chain.ort_provider,
        "module_boundary_cumulative_metrics": module_summary,
        "final_logits_cumulative_metrics": module_summary["receiver_head"]["seg_logits"],
        "final_argmax": {
            "equal_pixels": final_argmax_equal_pixels,
            "pixels": final_argmax_pixels,
            "agreement": final_argmax_equal_pixels / max(final_argmax_pixels, 1),
        },
        "confusion_matrices": {"fp32_onnx": fp_confusion.tolist(), "sdk_quantized": quant_confusion.tolist()},
        "semantic": {"fp32_onnx_chain": fp_metrics, "sdk_quantized_chain": quant_metrics,
                     "quant_minus_fp": semantic_delta(fp_metrics, quant_metrics)},
        "bitstream": {"fp": scalar_summary([float(row["fp_bpp"]) for row in bitstream_rows]),
                      "sdk_quantized": scalar_summary([float(row["quant_bpp"]) for row in bitstream_rows]),
                      "container_overhead_bytes_per_image": 29},
        "bitstream_totals": totals,
        "timing_ms": {"sdk_quantized_module_wall": timing_summary,
                      "notes": ["SDK simulation wall time is not Hailo-8 target latency.",
                                "h_s is required on both endpoints for y entropy coding and decoding.",
                                "rANS is CPU-only and reported separately."]},
        "mapping_notes": dict(chain.notes), "per_sample": sample_rows,
    }
    atomic_write_json(args.out, result)
    print(json.dumps({"out": str(args.out), "samples": len(indices),
                      "quant_mIoU": quant_metrics["mIoU"]}, indent=2))


if __name__ == "__main__":
    if "--merge-shards" in sys.argv[1:]:
        merge_shards()
    else:
        main()
