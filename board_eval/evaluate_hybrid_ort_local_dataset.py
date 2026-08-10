#!/usr/bin/env python3
"""Run one mixed Hailo/ORT arm against a Pi-local immutable eval set.

This avoids sending a 3 MiB FP32 image through SSH for every sample.  The
process is serial, sleeps between samples, records temperature/frequency, and
writes an atomic progress report so a detached host orchestrator is auditable.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from hybrid_ort_runtime_runner import HybridRuntime


LARGE = [7, 10, 15, 20, 23, 30, 35, 37, 48, 51, 80, 83, 84, 90, 101, 102, 103, 109, 122]
IMAGE_PIXELS = 512 * 512


def atomic_write(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "min": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def update_confusion(confusion: np.ndarray, prediction: np.ndarray, target: np.ndarray) -> None:
    classes = confusion.shape[0]
    prediction = np.asarray(prediction, dtype=np.int64)
    target = np.asarray(target, dtype=np.int64)
    if prediction.shape != target.shape:
        raise ValueError(f"prediction/target mismatch: {prediction.shape} != {target.shape}")
    valid = (target >= 0) & (target < classes) & (prediction >= 0) & (prediction < classes)
    encoded = target[valid] * classes + prediction[valid]
    confusion += np.bincount(encoded, minlength=classes * classes).reshape(classes, classes)


def semantic_metrics(confusion: np.ndarray) -> dict[str, object]:
    data = confusion.astype(np.float64, copy=False)
    true = np.diag(data)
    union = data.sum(axis=1) + data.sum(axis=0) - true
    iou = np.divide(true, np.maximum(union, 1.0))
    present = union > 0
    return {
        "mIoU": float(iou[present].mean()) if np.any(present) else 0.0,
        "pixel_accuracy": float(true.sum() / max(data.sum(), 1.0)),
        "fixed_19_large_object_macro_iou": float(iou[LARGE].mean()),
        "classes_with_union": int(present.sum()),
        "per_class_iou": [float(item) for item in iou],
    }


def report_payload(
    *, arm: str, mode: str, checkpoint_sha256: str, hello: dict,
    indices: list[int], rows: list[dict], confusion: np.ndarray,
    warmup: int, interval_seconds: float, status: str,
) -> dict:
    numeric_keys = sorted({
        key for row in rows for key, value in row.items()
        if isinstance(value, (int, float)) and key != "val_index"
    })
    total_bytes = sum(int(row["payload_bytes"]) for row in rows)
    return {
        "kind": "pi_local_dataset_hybrid_ort_hailort_evaluation",
        "status": status,
        "mode": mode,
        "arm": arm,
        "checkpoint_sha256": checkpoint_sha256,
        "sample_count": len(rows),
        "val_indices": indices[:len(rows)],
        "board_hello": hello,
        "policy": {
            "warmup": warmup,
            "interval_seconds": interval_seconds,
            "serial_pipeline": True,
            "stress_test": False,
            "dataset_transport": "pi_local_immutable_npy",
        },
        "semantic": semantic_metrics(confusion),
        "bitstream": {
            "total_bytes": total_bytes,
            "mean_bpp": total_bytes * 8.0 / max(len(rows) * IMAGE_PIXELS, 1),
            "z_bytes": sum(int(row["z_bytes"]) for row in rows),
            "y_bytes": sum(int(row["y_bytes"]) for row in rows),
        },
        "timing": {
            key: stats([float(row[key]) for row in rows if row.get(key) is not None])
            for key in numeric_keys
        },
        "per_sample": rows,
        "scope": [
            "Pi executes the complete image-to-segmentation chain with real z/y rANS roundtrips.",
            "board_total_pipeline_ms excludes cooldown and the intentional idle interval.",
            "Inputs and labels are loaded from an immutable Pi-local copy of the 976-sample handoff.",
            "Samples are serialized; this is an accuracy/latency experiment, not a throughput stress test.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("preflight", "latency", "accuracy"), required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--interval-seconds", type=float, default=1.0)
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.out}")
    if args.limit <= 0 or args.warmup < 0 or args.interval_seconds < 0 or args.progress_every <= 0:
        raise ValueError("invalid limit/warmup/interval/progress policy")

    dataset_manifest = json.loads(
        (args.dataset_root / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    runtime_manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    checkpoint = str(runtime_manifest["checkpoint_sha256"])
    if dataset_manifest.get("checkpoint_sha256") != checkpoint:
        raise ValueError("runtime/dataset checkpoint mismatch")
    all_indices = [int(item) for item in dataset_manifest["val_indices"]]
    if args.limit > len(all_indices):
        raise ValueError(f"requested {args.limit} samples but dataset has {len(all_indices)}")
    indices = all_indices[:args.limit]
    image_root = args.dataset_root / "images"
    target_root = args.dataset_root / "ground_truth"

    confusion = np.zeros((150, 150), dtype=np.int64)
    rows: list[dict] = []
    progress_path = args.out.with_suffix(".progress.json")
    runtime = HybridRuntime(args.manifest)
    hello = runtime.hello()
    try:
        warm_image = np.load(image_root / f"{indices[0]}.npy", allow_pickle=False)
        for _ in range(args.warmup):
            runtime.infer(warm_image.transpose(1, 2, 0))

        for position, index in enumerate(indices, start=1):
            image = np.load(image_root / f"{index}.npy", allow_pickle=False)
            target = np.load(target_root / f"{index}.npy", allow_pickle=False)
            outputs, timing = runtime.infer(image.transpose(1, 2, 0))
            prediction = np.asarray(outputs["argmax"], dtype=np.int64)
            update_confusion(confusion, prediction, target)
            container = timing["stages"]["rtff_container"]
            thermal = timing.get("thermal", {})
            row: dict[str, object] = {
                "val_index": index,
                "board_total_pipeline_ms": float(timing["total_pipeline_ms"]),
                "thermal_initial_c": thermal.get("initial_c"),
                "thermal_start_c": thermal.get("start_c"),
                "thermal_end_c": thermal.get("end_c"),
                "cooldown_wait_ms": thermal.get("cooldown_wait_ms", 0.0),
                "cpu_frequency_khz": thermal.get("cpu_frequency_khz", {}),
                "payload_bytes": int(container["bytes"]),
                "z_bytes": int(container["z_bytes"]),
                "y_bytes": int(container["y_bytes"]),
                "bpp": float(container["bytes"] * 8.0 / IMAGE_PIXELS),
            }
            for stage, stage_timing in timing["stages"].items():
                for key, value in stage_timing.items():
                    if isinstance(value, (int, float)):
                        row[f"{stage}.{key}"] = value
            rows.append(row)
            if position == 1 or position % args.progress_every == 0 or position == len(indices):
                partial = report_payload(
                    arm=runtime.arm, mode=args.mode, checkpoint_sha256=checkpoint,
                    hello=hello, indices=indices, rows=rows, confusion=confusion,
                    warmup=args.warmup, interval_seconds=args.interval_seconds,
                    status="RUNNING" if position < len(indices) else "PASS",
                )
                atomic_write(progress_path, partial)
                print(json.dumps({
                    "arm": runtime.arm,
                    "mode": args.mode,
                    "progress": f"{position}/{len(indices)}",
                    "fixed19": partial["semantic"]["fixed_19_large_object_macro_iou"],
                    "temperature_c": thermal.get("end_c"),
                    "pipeline_ms": timing["total_pipeline_ms"],
                }), flush=True)
            if position != len(indices) and args.interval_seconds:
                time.sleep(args.interval_seconds)
    finally:
        runtime.close()

    report = report_payload(
        arm=str(runtime_manifest["arm"]), mode=args.mode, checkpoint_sha256=checkpoint,
        hello=hello, indices=indices, rows=rows, confusion=confusion,
        warmup=args.warmup, interval_seconds=args.interval_seconds, status="PASS",
    )
    atomic_write(args.out, report)
    print(json.dumps({
        "status": "PASS", "out": str(args.out), "arm": runtime_manifest["arm"],
        "samples": len(rows), "semantic": report["semantic"], "bitstream": report["bitstream"],
    }), flush=True)


if __name__ == "__main__":
    main()
