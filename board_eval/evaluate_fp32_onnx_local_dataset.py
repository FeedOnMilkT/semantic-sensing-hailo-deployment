#!/usr/bin/env python3
"""Evaluate the immutable held-out package through the all-FP32 ONNX chain.

This is a host-only semantic reference.  Neural modules run through ONNX
Runtime CPU FP32, while rounding, CDF/index construction, rANS, and the RTFF
container use the same frozen implementation and codec tables as the Pi
runtime.  The result intentionally omits ADE20K-150 mIoU.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
DEMO_ROOT = HERE.parent
SEMANTIC_ROOT = Path("/workspace/semantic-sensing")
for path in (str(HERE), str(DEMO_ROOT), str(SEMANTIC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from evaluate_hybrid_ort_local_dataset import atomic_write, update_confusion  # noqa: E402
from fullmodule_runtime_runner import FullRuntime, RuntimeCodec, SEQUENCE  # noqa: E402
from hailo_deploy.io.hash import sha256  # noqa: E402
from hybrid_ort_runtime_runner import OrtModuleEngine  # noqa: E402
from src.codec import rans_fast  # noqa: E402


FIXED_19 = [7, 10, 15, 20, 23, 30, 35, 37, 48, 51, 80, 83, 84, 90, 101, 102, 103, 109, 122]
HIGH_SUPPORT_MIN_IMAGES = 20
EXPECTED_HIGH_SUPPORT = [7, 10, 15, 20, 23, 30, 37, 83, 102]
IMAGE_PIXELS = 512 * 512


class HostFp32Runtime(FullRuntime):
    """FullRuntime using only validated FP32 ONNX module engines."""

    def __init__(self, handoff: Path, codec_tables: Path, threads: int):
        self.root = handoff
        manifest_path = handoff / "onnx" / "manifest.json"
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        by_name = {str(item["name"]): item for item in self.manifest["modules"]}
        if tuple(by_name) != SEQUENCE:
            raise ValueError(f"ONNX manifest sequence differs from runtime sequence: {tuple(by_name)}")

        if not rans_fast.available():
            raise ImportError("the x86 rans_fast extension is unavailable")
        # RuntimeCodec imports the extension by its board-side top-level name.
        # Reuse the already validated x86 build without changing codec logic.
        sys.modules["_rans_fast"] = rans_fast._ext
        self.codec = RuntimeCodec(codec_tables)
        self.vdevice = None
        self.modules: dict[str, OrtModuleEngine] = {}
        try:
            for name in SEQUENCE:
                source = by_name[name]
                onnx_path = handoff / "onnx" / f"{name}.onnx"
                if not onnx_path.is_file():
                    raise FileNotFoundError(onnx_path)
                expected_sha = str(source["sha256"])
                actual_sha = sha256(onnx_path)
                if actual_sha != expected_sha:
                    raise ValueError(f"{name}: ONNX SHA mismatch: {actual_sha} != {expected_sha}")
                spec = {
                    "name": name,
                    "engine": "ort_cpu_fp32",
                    "onnx_filename": str(onnx_path),
                    "input_names": list(source["input_names"]),
                    "output_names": list(source["output_names"]),
                    "in_shapes_nchw": list(source["in_shapes_nchw"]),
                    "out_shapes_nchw": list(source["out_shapes_nchw"]),
                }
                self.modules[name] = OrtModuleEngine(Path("/"), spec, threads)
        except BaseException:
            self.close()
            raise
        self.calls = 0


def semantic_result(confusion: np.ndarray, occurrences: np.ndarray) -> dict[str, object]:
    data = confusion.astype(np.float64, copy=False)
    true_positive = np.diag(data)
    union = data.sum(axis=1) + data.sum(axis=0) - true_positive
    per_class = np.divide(true_positive, np.maximum(union, 1.0))
    high_support = [index for index in FIXED_19 if int(occurrences[index]) >= HIGH_SUPPORT_MIN_IMAGES]
    if high_support != EXPECTED_HIGH_SUPPORT:
        raise ValueError(
            f"high-support class set changed: {high_support} != {EXPECTED_HIGH_SUPPORT}"
        )
    return {
        "fixed_19_large_object_macro_iou": float(per_class[FIXED_19].mean()),
        "high_support_9_macro_iou": float(per_class[high_support].mean()),
        "high_support_rule": {
            "source_set": "fixed_19",
            "minimum_positive_images": HIGH_SUPPORT_MIN_IMAGES,
            "class_indices": high_support,
            "positive_image_counts": {str(index): int(occurrences[index]) for index in high_support},
        },
        "classes_with_union": int((union > 0).sum()),
        "per_class_iou": [float(value) for value in per_class],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--codec-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--formal", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    if args.threads <= 0 or (args.limit is not None and args.limit <= 0):
        raise ValueError("threads and limit must be positive")

    handoff = args.handoff.resolve()
    codec_dir = args.codec_dir.resolve()
    meta = json.loads((handoff / "evaluation" / "meta.json").read_text(encoding="utf-8"))
    indices = [int(value) for value in meta["val_indices"]]
    if args.limit is not None:
        indices = indices[: args.limit]
    if args.formal and indices != list(range(1024, 2000)):
        raise ValueError("formal evaluation requires exactly ADE20K val 1024..1999")

    checkpoint_path = handoff / "checkpoint" / "midf_b0.01_convbn_s2_gfa-gs-v1.pth"
    checkpoint_sha = sha256(checkpoint_path)
    declared_checkpoint = str(meta["checkpoint"]["sha256"])
    if checkpoint_sha != declared_checkpoint:
        raise ValueError(f"checkpoint SHA mismatch: {checkpoint_sha} != {declared_checkpoint}")

    codec_metadata_path = codec_dir / "codec_runtime.json"
    codec_tables_path = codec_dir / "codec_tables.npz"
    codec_metadata = json.loads(codec_metadata_path.read_text(encoding="utf-8"))
    expected_tables_sha = str(codec_metadata["tables"]["sha256"])
    actual_tables_sha = sha256(codec_tables_path)
    if actual_tables_sha != expected_tables_sha:
        raise ValueError(f"codec tables SHA mismatch: {actual_tables_sha} != {expected_tables_sha}")

    image_root = handoff / "evaluation" / "images"
    target_root = handoff / "evaluation" / "ground_truth"
    golden_root = handoff / "evaluation" / "golden"
    confusion = np.zeros((150, 150), dtype=np.int64)
    occurrences = np.zeros(150, dtype=np.int64)
    payload_bytes = 0
    pipeline_ms: list[float] = []
    golden_checks: list[dict[str, object]] = []
    started = time.time()

    runtime = HostFp32Runtime(handoff, codec_tables_path, args.threads)
    try:
        for position, index in enumerate(indices, start=1):
            image = np.load(image_root / f"{index}.npy", allow_pickle=False)
            target = np.load(target_root / f"{index}.npy", allow_pickle=False)
            outputs, timing = runtime.infer(image.transpose(1, 2, 0))
            prediction = np.asarray(outputs["argmax"], dtype=np.int64)
            update_confusion(confusion, prediction, target)
            present = np.unique(target[(target >= 0) & (target < 150)])
            occurrences[present.astype(np.int64)] += 1
            container = timing["stages"]["rtff_container"]
            payload_bytes += int(container["bytes"])
            pipeline_ms.append(float(timing["total_pipeline_ms"]))

            golden_path = golden_root / str(index) / "seg_logits.npy"
            if len(golden_checks) < 1 and golden_path.is_file():
                golden = np.load(golden_path, allow_pickle=False)[0].transpose(1, 2, 0)
                golden_mask = np.argmax(golden, axis=-1)
                golden_checks.append({
                    "val_index": index,
                    "max_abs_logit_diff": float(np.max(np.abs(outputs["seg_logits"] - golden))),
                    "mask_pixel_agreement": float(np.mean(prediction == golden_mask)),
                })

            if position % 25 == 0 or position == len(indices):
                elapsed = time.time() - started
                print(json.dumps({
                    "progress": f"{position}/{len(indices)}",
                    "elapsed_seconds": elapsed,
                    "mean_pipeline_ms": float(np.mean(pipeline_ms)),
                    "eta_seconds": elapsed / position * (len(indices) - position),
                }), flush=True)
    finally:
        runtime.close()

    semantic = semantic_result(confusion, occurrences) if args.formal else None
    elapsed = time.time() - started
    report = {
        "schema": "semantic_sensing.host_fp32_onnx_full976.v1",
        "status": "PASS",
        "scope": "formal_full976" if args.formal else "diagnostic",
        "runtime": "host ONNX Runtime CPU FP32 + frozen real rANS/RTFF codec",
        "sample_count": len(indices),
        "val_indices": indices,
        "checkpoint_sha256": checkpoint_sha,
        "onnx_manifest": {
            "path": str(handoff / "onnx" / "manifest.json"),
            "sha256": sha256(handoff / "onnx" / "manifest.json"),
        },
        "codec": {
            "metadata_path": str(codec_metadata_path),
            "metadata_sha256": sha256(codec_metadata_path),
            "tables_path": str(codec_tables_path),
            "tables_sha256": actual_tables_sha,
        },
        "execution": {
            "provider": "CPUExecutionProvider",
            "threads": args.threads,
            "elapsed_seconds": elapsed,
            "mean_pipeline_ms": float(np.mean(pipeline_ms)),
            "p95_pipeline_ms": float(np.percentile(pipeline_ms, 95)),
            "cpu_reason": "installed ONNX Runtime exposes no CUDAExecutionProvider",
        },
        "golden_check": golden_checks,
        "semantic": semantic,
        "bitstream": {
            "total_bytes": payload_bytes,
            "mean_bpp": payload_bytes * 8.0 / (len(indices) * IMAGE_PIXELS),
        },
        "policy": {
            "ade20k_150_class_miou": "NOT_COMPUTED_OR_REPORTED",
            "high_support_rule_depends_only_on_ground_truth_occurrence": True,
        },
    }
    atomic_write(args.out, report)
    print(json.dumps({
        "status": "PASS",
        "out": str(args.out),
        "semantic": semantic,
        "mean_bpp": report["bitstream"]["mean_bpp"],
        "elapsed_seconds": elapsed,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
