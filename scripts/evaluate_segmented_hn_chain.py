#!/usr/bin/env python3
"""Evaluate one frozen three-HEF candidate with real CPU rANS boundaries."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

if __name__ == "__main__":
    from hailo_deploy.logging import reexec_if_unlogged

    reexec_if_unlogged(Path(__file__), sys.argv[1:])

from evaluate_fullmodule_hn_chain import (
    IMAGE_PIXELS,
    atomic_write_json,
    bitstream_totals,
    bpp_from_totals,
    load_json,
    load_student,
    output_metrics,
    segmentation_metrics,
    semantic_delta,
    update_confusion,
    y_roundtrip,
    z_roundtrip,
)
from hailo_deploy.io.hash import sha256
from hailo_deploy.io.metrics import summarize
from pipelines.materialize_segmented_chain_calib import QuantizedSegment


SEGMENTS = ("analysis", "hyper", "synthesis_tail")


def artifact(record: dict[str, Any], key: str, segment: str) -> Path:
    item = record.get(key)
    if not isinstance(item, dict):
        raise ValueError(f"{segment}: missing {key}")
    path = Path(item.get("path", ""))
    if not path.is_file() or sha256(path) != item.get("sha256"):
        raise ValueError(f"{segment}: invalid {key} identity: {path}")
    return path.resolve()


def validate(
    handoff: Path, segments_manifest_path: Path, interface_path: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]], Path, list[int]]:
    identity = load_json(handoff / "identity.json")
    if identity.get("model", {}).get("split") != "mid" or identity.get("model", {}).get("gfa_mode") != "group_softmax":
        raise ValueError("formal handoff is not the group-softmax mid split")
    if identity.get("splits", {}).get("intersection_empty") is not True:
        raise ValueError("formal handoff does not prove calibration/evaluation disjointness")
    checkpoint_sha = identity.get("checkpoint", {}).get("sha256")
    interface = load_json(interface_path)
    if interface.get("status") != "FROZEN" or interface.get("checkpoint", {}).get("sha256") != checkpoint_sha:
        raise ValueError("candidate interface is not frozen to the formal checkpoint")
    if len(interface.get("codec_boundaries", [])) != 2:
        raise ValueError("candidate lacks both CPU codec boundaries")
    for boundary in interface["codec_boundaries"]:
        if boundary.get("raw_integer_boundary") is not False:
            raise ValueError("raw integer HEF handoff is forbidden")
        if boundary.get("requires_equal_qp_across_hefs") is not False:
            raise ValueError("invalid direct-QP-equality boundary contract")

    manifest = load_json(segments_manifest_path)
    if manifest.get("checkpoint", {}).get("sha256") != checkpoint_sha:
        raise ValueError("segments manifest checkpoint mismatch")
    entries = {entry["id"]: entry for entry in manifest.get("segments", [])}
    if set(entries) != set(SEGMENTS):
        raise ValueError("segments manifest is not the frozen three-graph DAG")
    for name, entry in entries.items():
        onnx = Path(entry["onnx"])
        if not onnx.is_file() or sha256(onnx) != entry.get("sha256"):
            raise ValueError(f"{name}: ONNX identity mismatch")
        record = interface.get(name, {})
        artifact(record, "quantized_har", name)
        artifact(record, "hef", name)
        script = artifact(record, "model_script", name).read_text(encoding="utf-8")
        if "model_optimization_flavor(optimization_level=4" not in script:
            raise ValueError(f"{name}: candidate is not Level-4 optimized")
        io = interface.get("hef_io_quantization", {}).get(name, {})
        if io.get("host_vstream_format") != "FLOAT32":
            raise ValueError(f"{name}: interface is not FLOAT32 VStreams")
        if io.get("quantized_har", {}).get("sha256") != record["quantized_har"]["sha256"]:
            raise ValueError(f"{name}: IO contract qHAR mismatch")

    sender_sha = interface["analysis"]["quantized_har"]["sha256"]
    hyper_sha = interface["hyper"]["quantized_har"]["sha256"]
    candidate = interface.get("candidate")
    if candidate not in {
        "A_hyper_adaround_receiver_w8", "B_hyper_qft4_receiver_w8",
        "C_hyper_adaround_receiver_a16", "D_hyper_qft4_receiver_a16",
    }:
        raise ValueError(f"unknown candidate policy: {candidate}")
    scripts = {
        name: artifact(interface[name], "model_script", name).read_text(encoding="utf-8")
        for name in SEGMENTS
    }
    qft_token = "post_quantization_optimization(finetune, policy=enabled, dataset_size=3276, epochs=4)"
    expects_qft = candidate.startswith(("B_", "D_"))
    if (qft_token in scripts["hyper"]) != expects_qft:
        raise ValueError(f"{candidate}: Hyper QFT4 policy mismatch")
    if qft_token in scripts["analysis"] or qft_token in scripts["synthesis_tail"]:
        raise ValueError(f"{candidate}: QFT escaped the physical Hyper graph")
    # Validated fused A16 is head_l only (synthesis_tail/conv3). Explicit A16 on
    # post-resize conv2 poisons resize1 and is not an approved policy.
    a16_token = "quantization_param(synthesis_tail/conv3, precision_mode=a16_w16)"
    poison_token = "quantization_param(synthesis_tail/conv2, precision_mode=a16_w16)"
    expects_a16 = candidate.startswith(("C_", "D_"))
    has_a16 = a16_token in scripts["synthesis_tail"]
    if has_a16 != expects_a16:
        raise ValueError(f"{candidate}: Receiver A16 high-path policy mismatch")
    if poison_token in scripts["synthesis_tail"]:
        raise ValueError(f"{candidate}: A16 on synthesis_tail/conv2 poisons resize1")
    if not expects_a16 and "precision_mode=a16_w16" in scripts["synthesis_tail"]:
        raise ValueError(f"{candidate}: unexpected Receiver A16 override")
    hyper_meta = interface["hyper"].get("calibration", {}).get("metadata", {})
    tail_meta = interface["synthesis_tail"].get("calibration", {}).get("metadata", {})
    if hyper_meta.get("calibration_mode") != "quantized_analysis_z_rans":
        raise ValueError("Hyper was not calibrated from quantized Sender + real rANS")
    if hyper_meta.get("quantized_upstream", {}).get("analysis_quantized_har", {}).get("sha256") != sender_sha:
        raise ValueError("Hyper calibration does not bind the frozen Sender")
    if tail_meta.get("calibration_mode") != "quantized_analysis_hyper_y_rans":
        raise ValueError("Receiver was not calibrated from the full quantized upstream chain")
    upstream = tail_meta.get("quantized_upstream", {})
    if upstream.get("analysis_quantized_har", {}).get("sha256") != sender_sha:
        raise ValueError("Receiver calibration Sender mismatch")
    if upstream.get("hyper_quantized_har", {}).get("sha256") != hyper_sha:
        raise ValueError("Receiver calibration Hyper mismatch")

    checkpoints = sorted((handoff / "checkpoint").glob("*.pth"))
    if len(checkpoints) != 1 or sha256(checkpoints[0]) != checkpoint_sha:
        raise ValueError("formal checkpoint copy mismatch")
    meta_path = handoff / "evaluation" / "meta.json"
    if not meta_path.is_file():
        meta_path = handoff / "evaluation" / "manifest.json"
    indices = list(load_json(meta_path).get("val_indices", []))
    if len(indices) != 976 or len(set(indices)) != 976:
        raise ValueError("formal evaluation split must contain 976 unique indices")
    return identity, interface, entries, checkpoints[0], indices


class Chain:
    def __init__(self, interface: dict[str, Any], entries: dict[str, dict[str, Any]]) -> None:
        providers: list[Any]
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers = [("CUDAExecutionProvider", {"use_tf32": "0"}), "CPUExecutionProvider"]
            self.fp_provider = "CUDAExecutionProvider"
        else:
            providers = ["CPUExecutionProvider"]
            self.fp_provider = "CPUExecutionProvider"
        self.sessions = {
            name: ort.InferenceSession(entries[name]["onnx"], providers=providers)
            for name in SEGMENTS
        }
        self.quantized = {
            name: QuantizedSegment(artifact(interface[name], "quantized_har", name), entries[name])
            for name in SEGMENTS
        }

    def __enter__(self) -> "Chain":
        for name in SEGMENTS:
            self.quantized[name].__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for name in reversed(SEGMENTS):
            self.quantized[name].__exit__(exc_type, exc, tb)

    def fp(self, name: str, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        session = self.sessions[name]
        values = session.run(None, {key: np.ascontiguousarray(value) for key, value in inputs.items()})
        return {output.name: np.asarray(value, dtype=np.float32) for output, value in zip(session.get_outputs(), values)}

    def quant(self, name: str, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return self.quantized[name].run(inputs)


def merge_shards() -> None:
    parser = argparse.ArgumentParser(description="Merge deterministic segmented-chain shards")
    parser.add_argument("--merge-shards", nargs="+", type=Path, required=True)
    parser.add_argument("--expected-sample-count", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    reports = [load_json(path) for path in args.merge_shards]
    if not reports or any(report.get("kind") != "segmented_three_hef_real_rans_evaluation" or report.get("status") != "PASS" for report in reports):
        raise ValueError("all shards must be PASS segmented evaluations")
    expected_shards = reports[0]["shard"]["num_shards"]
    fingerprints = {
        (report["interface_manifest_sha256"], report["checkpoint_sha256"], report["full_validation_sample_count"])
        for report in reports
    }
    if len(fingerprints) != 1 or len(reports) != expected_shards:
        raise ValueError("incomplete or mixed shard set")
    shard_ids = {report["shard"]["index"] for report in reports}
    if shard_ids != set(range(expected_shards)):
        raise ValueError("duplicate or missing shard index")
    seen: set[int] = set()
    fp_confusion: np.ndarray | None = None
    q_confusion: np.ndarray | None = None
    totals = {"sample_count": 0, "fp_bytes": 0, "quant_bytes": 0, "image_pixels": 0,
              "container_overhead_bytes_per_image": 29}
    sources = []
    for path, report in zip(args.merge_shards, reports):
        indices = report["val_indices"]
        if len(indices) != report["sample_count"] or seen.intersection(indices):
            raise ValueError(f"overlapping or invalid indices: {path}")
        seen.update(indices)
        matrices = report["confusion_matrices"]
        local_fp = np.asarray(matrices["fp32_onnx"], dtype=np.int64)
        local_q = np.asarray(matrices["sdk_quantized"], dtype=np.int64)
        if fp_confusion is None:
            fp_confusion = np.zeros_like(local_fp)
            q_confusion = np.zeros_like(local_q)
        fp_confusion += local_fp
        q_confusion += local_q
        local_totals = report["bitstream_totals"]
        for key in ("sample_count", "fp_bytes", "quant_bytes", "image_pixels"):
            totals[key] += int(local_totals[key])
        sources.append({"path": str(path), "sha256": sha256(path), "shard": report["shard"]})
    if len(seen) != args.expected_sample_count or totals["sample_count"] != args.expected_sample_count:
        raise ValueError("merged sample count mismatch")
    assert fp_confusion is not None and q_confusion is not None
    fp_metrics = segmentation_metrics(fp_confusion)
    q_metrics = segmentation_metrics(q_confusion)
    first = reports[0]
    result = {
        "kind": "segmented_three_hef_real_rans_evaluation_shard_merge",
        "status": "PASS",
        "candidate": first["candidate"],
        "checkpoint_sha256": first["checkpoint_sha256"],
        "interface_manifest_sha256": first["interface_manifest_sha256"],
        "sample_count": len(seen),
        "val_indices": sorted(seen),
        "shard_merge": {"num_shards": expected_shards, "selection": "round_robin", "sources": sources},
        "confusion_matrices": {"fp32_onnx": fp_confusion.tolist(), "sdk_quantized": q_confusion.tolist()},
        "semantic": {"fp32_onnx_chain": fp_metrics, "sdk_quantized_chain": q_metrics,
                     "quant_minus_fp": semantic_delta(fp_metrics, q_metrics)},
        "bitstream": bpp_from_totals(totals),
        "bitstream_totals": totals,
    }
    atomic_write_json(args.out, result)
    print(json.dumps({"status": "PASS", "out": str(args.out), "candidate": result["candidate"],
                      "samples": len(seen), "fixed19": q_metrics["fixed_19_large_object_macro_iou"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff-root", type=Path, required=True)
    parser.add_argument("--segments-manifest", type=Path, required=True)
    parser.add_argument("--interface-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--progress-out", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    args = parser.parse_args()
    if args.out.exists() or args.progress_out.exists():
        raise FileExistsError("refusing to overwrite evaluation output/progress")
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard")
    if args.checkpoint_every <= 0:
        raise ValueError("--checkpoint-every must be positive")

    identity, interface, entries, checkpoint, all_indices = validate(
        args.handoff_root, args.segments_manifest, args.interface_manifest
    )
    indices = all_indices[args.shard_index::args.num_shards]
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        indices = indices[:args.limit]
    student, cfg = load_student(checkpoint)
    codec = student.bottleneck.eval()
    fp_confusion = np.zeros((int(cfg.num_classes), int(cfg.num_classes)), dtype=np.int64)
    q_confusion = np.zeros_like(fp_confusion)
    metric_rows: dict[str, dict[str, list[dict[str, float | None]]]] = {
        name: defaultdict(list) for name in SEGMENTS
    }
    bit_rows: list[dict[str, int]] = []
    image_root = args.handoff_root / "evaluation" / "images"
    target_root = args.handoff_root / "evaluation" / "ground_truth"
    if not target_root.is_dir():
        target_root = args.handoff_root / "evaluation" / "labels"
    shard = {"index": args.shard_index, "num_shards": args.num_shards, "selection": "round_robin"}

    def progress(completed: int, status: str) -> None:
        fp_metrics = segmentation_metrics(fp_confusion)
        q_metrics = segmentation_metrics(q_confusion)
        atomic_write_json(args.progress_out, {
            "kind": "segmented_three_hef_real_rans_evaluation_progress",
            "status": status,
            "candidate": interface["candidate"],
            "shard": shard,
            "selected_sample_count": len(indices),
            "processed_count": completed,
            "processed_indices": indices[:completed],
            "semantic": {"fp32_onnx_chain": fp_metrics, "sdk_quantized_chain": q_metrics,
                         "quant_minus_fp": semantic_delta(fp_metrics, q_metrics)},
            "bitstream_totals": bitstream_totals(bit_rows),
            "updated_unix_seconds": time.time(),
        })

    with Chain(interface, entries) as chain:
        for position, index in enumerate(indices, start=1):
            image = np.load(image_root / f"{index}.npy", allow_pickle=False).astype(np.float32, copy=False)[None]
            target = np.load(target_root / f"{index}.npy", allow_pickle=False).astype(np.int64, copy=False)[None]
            fp_analysis = chain.fp("analysis", {"image": image})
            q_analysis = chain.quant("analysis", {"image": image})
            fp_z_hat, fp_z_bytes = z_roundtrip(codec, fp_analysis["latent_z"])
            q_z_hat, q_z_bytes = z_roundtrip(codec, q_analysis["latent_z"])
            fp_hyper = chain.fp("hyper", {"latent_z_hat": fp_z_hat})
            q_hyper = chain.quant("hyper", {"latent_z_hat": q_z_hat})
            fp_y_hat, fp_y_bytes = y_roundtrip(codec, fp_analysis["latent_y"], fp_hyper["means"], fp_hyper["scales"])
            q_y_hat, q_y_bytes = y_roundtrip(codec, q_analysis["latent_y"], q_hyper["means"], q_hyper["scales"])
            fp_tail = chain.fp("synthesis_tail", {"latent_y_hat": fp_y_hat})
            q_tail = chain.quant("synthesis_tail", {"latent_y_hat": q_y_hat})
            for name, fp_values, q_values in (
                ("analysis", fp_analysis, q_analysis),
                ("hyper", fp_hyper, q_hyper),
                ("synthesis_tail", fp_tail, q_tail),
            ):
                for output, values in output_metrics(fp_values, q_values).items():
                    metric_rows[name][output].append(values)
            update_confusion(fp_confusion, fp_tail["seg_logits"], target, int(cfg.ignore_index))
            update_confusion(q_confusion, q_tail["seg_logits"], target, int(cfg.ignore_index))
            bit_rows.append({"fp_bytes": fp_z_bytes + fp_y_bytes + 29,
                             "quant_bytes": q_z_bytes + q_y_bytes + 29})
            if position % args.checkpoint_every == 0 or position == len(indices):
                progress(position, "RUNNING")
                print(f"evaluated {position}/{len(indices)}", flush=True)
    progress(len(indices), "COMPLETE")
    fp_metrics = segmentation_metrics(fp_confusion)
    q_metrics = segmentation_metrics(q_confusion)
    totals = bitstream_totals(bit_rows)
    result = {
        "kind": "segmented_three_hef_real_rans_evaluation",
        "status": "PASS",
        "scope": "three Hailo SDK_QUANTIZED graphs with real CPU rANS; not board timing",
        "candidate": interface["candidate"],
        "handoff_root": str(args.handoff_root),
        "checkpoint_sha256": identity["checkpoint"]["sha256"],
        "interface_manifest": str(args.interface_manifest),
        "interface_manifest_sha256": sha256(args.interface_manifest),
        "sample_count": len(indices),
        "val_indices": indices,
        "shard": shard,
        "full_validation_sample_count": len(all_indices),
        "fp32_onnx_provider": chain.fp_provider,
        "segment_boundary_cumulative_metrics": {
            name: {output: summarize(rows) for output, rows in outputs.items()}
            for name, outputs in metric_rows.items()
        },
        "confusion_matrices": {"fp32_onnx": fp_confusion.tolist(), "sdk_quantized": q_confusion.tolist()},
        "semantic": {"fp32_onnx_chain": fp_metrics, "sdk_quantized_chain": q_metrics,
                     "quant_minus_fp": semantic_delta(fp_metrics, q_metrics)},
        "bitstream": bpp_from_totals(totals),
        "bitstream_totals": totals,
    }
    atomic_write_json(args.out, result)
    print(json.dumps({"status": "PASS", "out": str(args.out), "candidate": interface["candidate"],
                      "samples": len(indices), "fixed19": q_metrics["fixed_19_large_object_macro_iou"]}, indent=2))


if __name__ == "__main__":
    if "--merge-shards" in sys.argv[1:]:
        merge_shards()
    else:
        main()
