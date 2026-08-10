#!/usr/bin/env python3
"""Evaluate causal high/low ``bn_g_s`` branch splices on a Pi-local set."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from bn_g_s_branch_attribution_runtime import BRANCHES, BranchAttributionRuntime
from evaluate_hybrid_ort_local_dataset import semantic_metrics, stats, update_confusion


def atomic_write(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def flatten_numeric(prefix: str, value: object, output: dict[str, float]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            flatten_numeric(f"{prefix}.{key}" if prefix else str(key), child, output)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        output[prefix] = float(value)


def payload(*, status: str, manifest: dict, hello: dict, indices: list[int], rows: list[dict],
            confusions: dict[str, np.ndarray], interval: float) -> dict:
    numeric_keys = sorted({key for row in rows for key in row.get("numeric", {})})
    return {
        "kind": "bn_g_s_same_y_hat_high_low_causal_splice_evaluation",
        "status": status,
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "sample_count": len(rows),
        "val_indices": indices[:len(rows)],
        "board_hello": hello,
        "policy": {
            "serial_pipeline": True,
            "stress_test": False,
            "interval_seconds": interval,
            "same_decoded_y_hat_for_hailo_and_fp32_bn_g_s": True,
            "per_sample_receiver_tail_invocations": 4,
        },
        "semantic_by_splice": {
            arm: semantic_metrics(confusions[arm]) for arm in BRANCHES
        },
        "numeric_timing_and_drift": {
            key: stats([row["numeric"][key] for row in rows if key in row.get("numeric", {})])
            for key in numeric_keys
        },
        "per_sample": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--interval-seconds", type=float, default=1.0)
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.out}")
    if args.limit <= 0 or args.interval_seconds < 0 or args.progress_every <= 0:
        raise ValueError("invalid limit/interval/progress policy")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    dataset_manifest = json.loads(
        (args.dataset_root / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    if dataset_manifest.get("checkpoint_sha256") != manifest.get("checkpoint_sha256"):
        raise ValueError("runtime/dataset checkpoint mismatch")
    all_indices = [int(item) for item in dataset_manifest["val_indices"]]
    if args.limit > len(all_indices):
        raise ValueError("requested more samples than the immutable dataset contains")
    indices = all_indices[:args.limit]
    image_root = args.dataset_root / "images"
    target_root = args.dataset_root / "ground_truth"
    confusions = {arm: np.zeros((150, 150), dtype=np.int64) for arm in BRANCHES}
    rows: list[dict] = []
    progress_path = args.out.with_suffix(".progress.json")

    runtime = BranchAttributionRuntime(args.manifest)
    hello = runtime.hello()
    try:
        for position, index in enumerate(indices, start=1):
            image = np.load(image_root / f"{index}.npy", allow_pickle=False)
            target = np.load(target_root / f"{index}.npy", allow_pickle=False)
            predictions, timing = runtime.infer_branches(image.transpose(1, 2, 0))
            for arm, prediction in predictions.items():
                update_confusion(confusions[arm], prediction, target)
            numeric: dict[str, float] = {}
            flatten_numeric("", timing, numeric)
            row = {
                "val_index": index,
                "latent_y_hat_sha256": timing["exact_input_gate"]["latent_y_hat_sha256"],
                "numeric": numeric,
            }
            rows.append(row)
            if position == 1 or position % args.progress_every == 0 or position == len(indices):
                partial = payload(
                    status="RUNNING" if position < len(indices) else "PASS",
                    manifest=manifest, hello=hello, indices=indices, rows=rows,
                    confusions=confusions, interval=args.interval_seconds,
                )
                atomic_write(progress_path, partial)
                print(json.dumps({
                    "progress": f"{position}/{len(indices)}",
                    "mIoU": {
                        arm: partial["semantic_by_splice"][arm]["mIoU"] for arm in BRANCHES
                    },
                    "temperature_c": timing["thermal"]["end_c"],
                    "experiment_ms": timing["total_experiment_ms"],
                }), flush=True)
            if position != len(indices) and args.interval_seconds:
                time.sleep(args.interval_seconds)
    finally:
        runtime.close()

    report = payload(
        status="PASS", manifest=manifest, hello=hello, indices=indices, rows=rows,
        confusions=confusions, interval=args.interval_seconds,
    )
    atomic_write(args.out, report)
    print(json.dumps({
        "status": "PASS", "out": str(args.out), "samples": len(rows),
        "semantic_by_splice": report["semantic_by_splice"],
    }), flush=True)


if __name__ == "__main__":
    main()
