#!/usr/bin/env python3
"""Capture exact current-topology decoded ``y_hat`` tensors on the Pi.

This is a bounded correctness artifact generator.  It preserves the permanent
hybrid Hailo/ORT boundary through the entropy-decoder output and never runs a
semantic metric.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from hybrid_ort_runtime_runner import HybridRuntime, cpu_temperature_c


CPU_MODULES = ("bn_align_h", "bn_g_a", "bn_h_a", "bn_h_s")
HAILO_MODULES = (
    "sender_backbone", "bn_g_s", "receiver_stage4", "receiver_stage5",
    "receiver_dappm", "receiver_head",
)


def atomic_write(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


class CaptureRuntime(HybridRuntime):
    def __init__(self, manifest_path: Path):
        super().__init__(manifest_path)
        if tuple(self.manifest.get("cpu_modules", ())) != CPU_MODULES:
            raise ValueError("permanent ORT CPU boundary mismatch")
        if tuple(self.manifest.get("hailo_modules", ())) != HAILO_MODULES:
            raise ValueError("permanent Hailo boundary mismatch")

    def capture(self, image: np.ndarray) -> tuple[np.ndarray, dict]:
        start = time.perf_counter_ns()
        guard = self._thermal_guard()
        timings: dict[str, dict] = {}
        sender = self._module("sender_backbone", {"image": image}, timings)
        aligned = self._module("bn_align_h", {"x_h": sender["x_h"]}, timings)
        y = self._module(
            "bn_g_a", {"aligned_x_h": aligned["aligned_x_h"], "x_l": sender["x_l"]},
            timings,
        )["latent_y"]
        z = self._module("bn_h_a", {"latent_y": y}, timings)["latent_z"]
        z_hat, _z_stream, z_timing = self.codec.z_roundtrip(z)
        timings["rans_z"] = z_timing
        hyper = self._module("bn_h_s", {"latent_z_hat": z_hat}, timings)
        y_hat, _y_stream, y_timing = self.codec.y_roundtrip(
            y, hyper["means"], hyper["scales"]
        )
        timings["rans_y"] = y_timing
        expected = (16, 16, 192)
        if y_hat.shape != expected or y_hat.dtype != np.float32 or not np.isfinite(y_hat).all():
            raise ValueError(f"invalid decoded y_hat: {y_hat.shape}/{y_hat.dtype}")
        self.calls += 1
        contiguous = np.ascontiguousarray(y_hat)
        return contiguous, {
            "timings": timings,
            "thermal": {**guard, "end_c": cpu_temperature_c()},
            "total_ms": (time.perf_counter_ns() - start) / 1e6,
            "sha256": hashlib.sha256(contiguous.view(np.uint8)).hexdigest(),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=32)
    args = parser.parse_args()
    if args.out_root.exists():
        raise FileExistsError(f"refusing to overwrite: {args.out_root}")
    if args.limit != 32:
        raise ValueError("this diagnostic bundle is frozen to exactly 32 samples")
    args.out_root.mkdir(parents=True)
    tensor_root = args.out_root / "latent_y_hat"
    tensor_root.mkdir()
    dataset = json.loads((args.dataset_root / "dataset_manifest.json").read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if dataset.get("checkpoint_sha256") != manifest.get("checkpoint_sha256"):
        raise ValueError("dataset/runtime checkpoint mismatch")
    indices = [int(x) for x in dataset["val_indices"][: args.limit]]
    rows: list[dict] = []
    progress = args.out_root / "progress.json"
    runtime = CaptureRuntime(args.manifest)
    hello = runtime.hello()
    try:
        for position, index in enumerate(indices, start=1):
            image = np.load(args.dataset_root / "images" / f"{index}.npy", allow_pickle=False)
            y_hat, metadata = runtime.capture(image.transpose(1, 2, 0))
            target = tensor_root / f"{index}.npy"
            np.save(target, y_hat, allow_pickle=False)
            rows.append({"val_index": index, **metadata})
            atomic_write(progress, {
                "status": "RUNNING" if position < len(indices) else "PASS",
                "sample_count": position,
                "indices": indices[:position],
                "last_temperature_c": metadata["thermal"]["end_c"],
            })
            print(json.dumps({
                "progress": f"{position}/{len(indices)}", "val_index": index,
                "temperature_c": metadata["thermal"]["end_c"],
            }), flush=True)
    finally:
        runtime.close()
    atomic_write(args.out_root / "result.json", {
        "kind": "current_topology_bn_g_s_yhat_correctness_bundle",
        "status": "PASS", "sample_count": len(rows), "val_indices": indices,
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "runtime_hello": hello, "rows": rows,
        "policy": {"diagnostic_only": True, "semantic_ranking": False},
    })


if __name__ == "__main__":
    main()
