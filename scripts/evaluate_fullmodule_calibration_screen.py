#!/usr/bin/env python3
"""Calibration-overlap screen for one immutable 10-module Hailo selection.

This is a rate/drift engineering screen, not held-out semantic accuracy.  It
uses the already materialized FP32 module-boundary dumps for indices 0..63,
propagates one real SDK_QUANTIZED chain, and executes both real rANS codecs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    from hailo_deploy.logging import reexec_if_unlogged

    reexec_if_unlogged(Path(__file__), sys.argv[1:])

from hailo_deploy.io.hash import sha256
from hailo_deploy.io.metrics import metric, summarize
from scripts.evaluate_fullmodule_hn_chain import (
    IMAGE_PIXELS,
    MODULES,
    Chain,
    atomic_write_json,
    bitstream_totals,
    bpp_from_totals,
    load_student,
    resolve_module_results,
    scalar_summary,
    validate,
    y_roundtrip,
    z_roundtrip,
)


def load_dump(calibration: Path, module: str, semantic: str, index: int) -> np.ndarray:
    path = calibration / module / semantic / f"{index:05d}.npy"
    value = np.load(path, allow_pickle=False).astype(np.float32, copy=False)
    if value.ndim == 3:
        value = np.transpose(value, (2, 0, 1))[None]
    elif value.ndim == 4:
        value = np.transpose(value, (0, 3, 1, 2))
    else:
        raise ValueError(f"unexpected dump rank at {path}: {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"nonfinite FP32 dump: {path}")
    return np.ascontiguousarray(value)


def output_metrics(reference: dict[str, np.ndarray], actual: dict[str, np.ndarray]) -> dict[str, Any]:
    if set(reference) != set(actual):
        raise ValueError(f"output coverage mismatch: {sorted(reference)} != {sorted(actual)}")
    return {name: metric(reference[name], actual[name]) for name in reference}


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    modules: dict[str, dict[str, list[dict[str, float | None]]]] = {
        name: defaultdict(list) for name in MODULES
    }
    codec: dict[str, list[dict[str, float | None]]] = defaultdict(list)
    final_argmax: list[float] = []
    for row in rows:
        for module, outputs in row["module_metrics"].items():
            for output, values in outputs.items():
                modules[module][output].append(values)
        for name, values in row["codec_metrics"].items():
            codec[name].append(values)
        final_argmax.append(float(row["final_argmax_agreement"]))
    return {
        "module_boundary_cumulative_metrics": {
            module: {output: summarize(values) for output, values in outputs.items()}
            for module, outputs in modules.items()
        },
        "codec_boundary_metrics": {name: summarize(values) for name, values in codec.items()},
        "final_logits_cumulative_metrics": summarize(
            [row["module_metrics"]["receiver_head"]["seg_logits"] for row in rows]
        ),
        "final_argmax_agreement": scalar_summary(final_argmax),
    }


def evaluate(args: argparse.Namespace) -> None:
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite: {args.out}")
    result_paths, selection_identity = resolve_module_results(None, args.selection_manifest)
    _, _, checkpoint, checkpoint_sha = validate(args.handoff_root, result_paths)
    student, _ = load_student(checkpoint)
    codec = student.bottleneck.eval()
    chain = Chain(args.handoff_root, result_paths)
    calibration = args.handoff_root / "calibration"
    if not calibration.is_dir():
        calibration = args.handoff_root / "calib"
    indices = list(range(64))[args.shard_index::args.num_shards]
    rows: list[dict[str, Any]] = []
    try:
        for position, index in enumerate(indices, start=1):
            truth: dict[str, dict[str, np.ndarray]] = {}
            q: dict[str, dict[str, np.ndarray]] = {}
            module_metrics: dict[str, Any] = {}

            image = load_dump(calibration, "sender_backbone", "image", index)
            truth["sender_backbone"] = {
                "x_h": load_dump(calibration, "bn_align_h", "x_h", index),
                "x_l": load_dump(calibration, "bn_g_a", "x_l", index),
            }
            q["sender_backbone"], _ = chain.quant("sender_backbone", {"image": image}, truth["sender_backbone"])
            module_metrics["sender_backbone"] = output_metrics(truth["sender_backbone"], q["sender_backbone"])

            truth["bn_align_h"] = {"aligned_x_h": load_dump(calibration, "bn_g_a", "aligned_x_h", index)}
            q["bn_align_h"], _ = chain.quant("bn_align_h", {"x_h": q["sender_backbone"]["x_h"]}, truth["bn_align_h"])
            module_metrics["bn_align_h"] = output_metrics(truth["bn_align_h"], q["bn_align_h"])

            truth["bn_g_a"] = {"latent_y": load_dump(calibration, "bn_h_a", "latent_y", index)}
            q["bn_g_a"], _ = chain.quant("bn_g_a", {
                "aligned_x_h": q["bn_align_h"]["aligned_x_h"], "x_l": q["sender_backbone"]["x_l"],
            }, truth["bn_g_a"])
            module_metrics["bn_g_a"] = output_metrics(truth["bn_g_a"], q["bn_g_a"])

            truth["bn_h_a"] = chain.fp("bn_h_a", {"latent_y": truth["bn_g_a"]["latent_y"]})
            q["bn_h_a"], _ = chain.quant("bn_h_a", {"latent_y": q["bn_g_a"]["latent_y"]}, truth["bn_h_a"])
            module_metrics["bn_h_a"] = output_metrics(truth["bn_h_a"], q["bn_h_a"])

            fp_z_hat, fp_z_bytes = z_roundtrip(codec, truth["bn_h_a"]["latent_z"])
            q_z_hat, q_z_bytes = z_roundtrip(codec, q["bn_h_a"]["latent_z"])
            dumped_z_hat = load_dump(calibration, "bn_h_s", "latent_z_hat", index)
            truth["bn_h_s"] = chain.fp("bn_h_s", {"latent_z_hat": dumped_z_hat})
            q["bn_h_s"], _ = chain.quant("bn_h_s", {"latent_z_hat": q_z_hat}, truth["bn_h_s"])
            module_metrics["bn_h_s"] = output_metrics(truth["bn_h_s"], q["bn_h_s"])

            fp_y_hat, fp_y_bytes = y_roundtrip(
                codec, truth["bn_g_a"]["latent_y"], truth["bn_h_s"]["means"], truth["bn_h_s"]["scales"]
            )
            q_y_hat, q_y_bytes = y_roundtrip(
                codec, q["bn_g_a"]["latent_y"], q["bn_h_s"]["means"], q["bn_h_s"]["scales"]
            )
            dumped_y_hat = load_dump(calibration, "bn_g_s", "latent_y_hat", index)
            truth["bn_g_s"] = {
                "x_h_hat": load_dump(calibration, "receiver_stage4", "x_h_hat", index),
                "x_l_hat": load_dump(calibration, "receiver_stage4", "x_l_hat", index),
            }
            q["bn_g_s"], _ = chain.quant("bn_g_s", {"latent_y_hat": q_y_hat}, truth["bn_g_s"])
            module_metrics["bn_g_s"] = output_metrics(truth["bn_g_s"], q["bn_g_s"])

            truth["receiver_stage4"] = {
                "stage4_high": load_dump(calibration, "receiver_stage5", "stage4_high", index),
                "stage4_low": load_dump(calibration, "receiver_stage5", "stage4_low", index),
            }
            q["receiver_stage4"], _ = chain.quant("receiver_stage4", {
                "x_h": q["bn_g_s"]["x_h_hat"], "x_l": q["bn_g_s"]["x_l_hat"],
            }, truth["receiver_stage4"])
            module_metrics["receiver_stage4"] = output_metrics(truth["receiver_stage4"], q["receiver_stage4"])

            truth["receiver_stage5"] = {
                "high_feat": load_dump(calibration, "receiver_dappm", "high_feat", index),
                "low_feat": load_dump(calibration, "receiver_dappm", "low_feat", index),
            }
            q["receiver_stage5"], _ = chain.quant("receiver_stage5", {
                "stage4_high": q["receiver_stage4"]["stage4_high"],
                "stage4_low": q["receiver_stage4"]["stage4_low"],
            }, truth["receiver_stage5"])
            module_metrics["receiver_stage5"] = output_metrics(truth["receiver_stage5"], q["receiver_stage5"])

            truth["receiver_dappm"] = {"s8": load_dump(calibration, "receiver_head", "s8", index)}
            q["receiver_dappm"], _ = chain.quant("receiver_dappm", {
                "high_feat": q["receiver_stage5"]["high_feat"],
                "low_feat": q["receiver_stage5"]["low_feat"],
            }, truth["receiver_dappm"])
            module_metrics["receiver_dappm"] = output_metrics(truth["receiver_dappm"], q["receiver_dappm"])

            truth["receiver_head"] = chain.fp("receiver_head", {"s8": truth["receiver_dappm"]["s8"]})
            q["receiver_head"], _ = chain.quant("receiver_head", {"s8": q["receiver_dappm"]["s8"]}, truth["receiver_head"])
            module_metrics["receiver_head"] = output_metrics(truth["receiver_head"], q["receiver_head"])
            fp_argmax = truth["receiver_head"]["seg_logits"].argmax(axis=1)
            q_argmax = q["receiver_head"]["seg_logits"].argmax(axis=1)

            rows.append({
                "val_index": index,
                "module_metrics": module_metrics,
                "codec_metrics": {
                    "z_hat_vs_dump": metric(dumped_z_hat, q_z_hat),
                    "y_hat_vs_dump": metric(dumped_y_hat, q_y_hat),
                    "fp_z_roundtrip_vs_dump": metric(dumped_z_hat, fp_z_hat),
                    "fp_y_roundtrip_vs_dump": metric(dumped_y_hat, fp_y_hat),
                },
                "final_argmax_agreement": float(np.mean(fp_argmax == q_argmax)),
                "fp_z_bytes": fp_z_bytes, "fp_y_bytes": fp_y_bytes,
                "quant_z_bytes": q_z_bytes, "quant_y_bytes": q_y_bytes,
                "fp_bytes": fp_z_bytes + fp_y_bytes + 29,
                "quant_bytes": q_z_bytes + q_y_bytes + 29,
            })
            if position % 4 == 0 or position == len(indices):
                print(f"screen shard {args.shard_index}: {position}/{len(indices)}", flush=True)
    finally:
        chain.close()

    totals = bitstream_totals(rows)
    result = {
        "kind": "x86_sdk_quantized_fullmodule_calibration_screen",
        "status": "PASS",
        "scope": "Calibration-overlap rate/drift diagnostic; not held-out or final semantic accuracy",
        "handoff_root": str(args.handoff_root.resolve()), **selection_identity,
        "checkpoint_sha256": checkpoint_sha,
        "shard": {"index": args.shard_index, "num_shards": args.num_shards, "selection": "round_robin"},
        "sample_count": len(rows), "val_indices": indices, "real_rans_success_count": len(rows),
        "reference": "Existing immutable FP32 module-boundary dumps; no quantized-upstream rematerialization",
        **summarize_rows(rows),
        "bitstream": bpp_from_totals(totals), "bitstream_totals": totals,
        "per_sample": rows,
    }
    atomic_write_json(args.out, result)


def merge(args: argparse.Namespace) -> None:
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite: {args.out}")
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.merge_shards]
    if not reports:
        raise ValueError("no shards")
    num_shards = reports[0].get("shard", {}).get("num_shards")
    if len(reports) != num_shards:
        raise ValueError("must provide exactly one result for every shard")
    identity = (
        reports[0].get("handoff_root"), reports[0].get("selection_manifest_sha256"),
        reports[0].get("candidate_fingerprint"), reports[0].get("checkpoint_sha256"),
    )
    seen_shards: set[int] = set()
    rows: list[dict[str, Any]] = []
    sources = []
    for path, report in zip(args.merge_shards, reports):
        if report.get("kind") != "x86_sdk_quantized_fullmodule_calibration_screen" or report.get("status") != "PASS":
            raise ValueError(f"not a PASS screen shard: {path}")
        if (report.get("handoff_root"), report.get("selection_manifest_sha256"),
            report.get("candidate_fingerprint"), report.get("checkpoint_sha256")) != identity:
            raise ValueError("screen shard identity mismatch")
        shard = report["shard"]
        if shard.get("num_shards") != num_shards or shard.get("index") in seen_shards:
            raise ValueError("screen shard topology mismatch")
        seen_shards.add(int(shard["index"]))
        rows.extend(report.get("per_sample", []))
        sources.append({"path": str(path), "sha256": sha256(path), "shard_index": shard["index"]})
    indices = [int(row["val_index"]) for row in rows]
    if seen_shards != set(range(num_shards)) or sorted(indices) != list(range(64)):
        raise ValueError("screen shards do not cover calibration indices 0..63 exactly once")
    rows.sort(key=lambda row: row["val_index"])
    totals = bitstream_totals(rows)
    first = reports[0]
    result = {
        "kind": "x86_sdk_quantized_fullmodule_calibration_screen_merge",
        "status": "PASS", "scope": first["scope"],
        "handoff_root": first["handoff_root"], "selection_manifest": first["selection_manifest"],
        "selection_manifest_sha256": first["selection_manifest_sha256"],
        "candidate": first.get("candidate"), "candidate_fingerprint": first["candidate_fingerprint"],
        "checkpoint_sha256": first["checkpoint_sha256"],
        "sample_count": len(rows), "val_indices": indices, "real_rans_success_count": len(rows),
        "shard_merge": {"num_shards": num_shards, "source_reports": sorted(sources, key=lambda x: x["shard_index"])},
        **summarize_rows(rows),
        "bitstream": bpp_from_totals(totals), "bitstream_totals": totals,
        "per_sample": rows,
    }
    atomic_write_json(args.out, result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--merge-shards", nargs="+", type=Path)
    parser.add_argument("--handoff-root", type=Path)
    parser.add_argument("--selection-manifest", type=Path)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    if args.merge_shards:
        return args
    if args.handoff_root is None or args.selection_manifest is None:
        parser.error("evaluation requires --handoff-root and --selection-manifest")
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        parser.error("invalid shard topology")
    return args


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.merge_shards:
        merge(parsed)
    else:
        evaluate(parsed)
