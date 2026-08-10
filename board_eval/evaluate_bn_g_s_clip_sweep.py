#!/usr/bin/env python3
"""Evaluate a bn_g_s clipping sweep on one frozen 32-y_hat board bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from bn_g_s_branch_attribution_runtime import tensor_drift
from fullmodule_runtime_runner import ModuleEngine
from hybrid_ort_runtime_runner import OrtModuleEngine, cpu_temperature_c


BRANCH_WEIGHTS = {"x_h_hat": 0.37, "x_l_hat": 0.63}


def atomic_write(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def summarize(rows: list[dict], key: str) -> dict:
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    return {
        "count": int(values.size), "mean": float(values.mean()),
        "min": float(values.min()), "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)), "max": float(values.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite: {args.out}")
    spec = json.loads(args.manifest.read_text(encoding="utf-8"))
    bundle = json.loads((args.bundle_root / "result.json").read_text(encoding="utf-8"))
    if bundle.get("status") != "PASS" or bundle.get("sample_count") != 32:
        raise ValueError("frozen y_hat bundle is incomplete")
    if bundle.get("checkpoint_sha256") != spec.get("checkpoint_sha256"):
        raise ValueError("bundle/candidate checkpoint mismatch")

    from hailo_platform import VDevice

    root = args.manifest.parent
    fp = OrtModuleEngine(root, spec["fp32"], int(spec.get("ort_threads", 4)))
    frozen_samples = []
    try:
        for position, index in enumerate(bundle["val_indices"], start=1):
            y_hat = np.ascontiguousarray(
                np.load(args.bundle_root / "latent_y_hat" / f"{index}.npy", allow_pickle=False),
                dtype=np.float32,
            )
            digest = hashlib.sha256(y_hat.view(np.uint8)).hexdigest()
            if digest != bundle["rows"][position - 1]["sha256"]:
                raise ValueError(f"y_hat hash mismatch for {index}")
            reference, fp_timing = fp.infer({"latent_y_hat": y_hat})
            frozen_samples.append({
                "val_index": index, "y_hat": y_hat, "digest": digest,
                "reference": reference, "fp32_ort_timing": fp_timing,
            })
    finally:
        fp.close()

    per_arm = {
        arm["name"]: {
            "x_h_hat": [], "x_l_hat": [], "latency": [], "saturation": [],
            "sse": {branch: 0.0 for branch in BRANCH_WEIGHTS},
            "reference_energy": {branch: 0.0 for branch in BRANCH_WEIGHTS},
            "temperature": {},
        }
        for arm in spec["arms"]
    }
    sample_rows = [{
        "val_index": sample["val_index"], "latent_y_hat_sha256": sample["digest"],
        "fp32_ort_timing": sample["fp32_ort_timing"], "arms": {},
    } for sample in frozen_samples]
    progress = args.out.with_suffix(".progress.json")
    for arm_index, arm in enumerate(spec["arms"], start=1):
        name = arm["name"]
        vdevice = VDevice()
        engine = None
        per_arm[name]["temperature"]["start_c"] = cpu_temperature_c()
        try:
            engine = ModuleEngine(vdevice, root, arm["module_spec"])
            for position, sample in enumerate(frozen_samples):
                y_hat, reference = sample["y_hat"], sample["reference"]
                got, timing = engine.infer({"latent_y_hat": y_hat})
                arm_row = {"timing": timing, "outputs": {}}
                for output in BRANCH_WEIGHTS:
                    metrics = tensor_drift(got[output], reference[output])
                    candidate64 = np.asarray(got[output], dtype=np.float64)
                    reference64 = np.asarray(reference[output], dtype=np.float64)
                    delta = candidate64 - reference64
                    sse = float(np.sum(delta * delta))
                    energy = float(np.sum(reference64 * reference64))
                    metrics.update({"sse": sse, "reference_energy": energy})
                    arm_row["outputs"][output] = metrics
                    per_arm[name][output].append(metrics)
                    per_arm[name]["sse"][output] += sse
                    per_arm[name]["reference_energy"][output] += energy
                bounds = arm["input_quantization"]
                below = int(np.count_nonzero(y_hat < float(bounds["limvals_min"])))
                above = int(np.count_nonzero(y_hat > float(bounds["limvals_max"])))
                saturation = {
                    "below_count": below, "above_count": above,
                    "fraction": float((below + above) / y_hat.size),
                }
                arm_row["input_saturation"] = saturation
                per_arm[name]["saturation"].append(saturation)
                per_arm[name]["latency"].append(timing)
                sample_rows[position]["arms"][name] = arm_row
        finally:
            if engine is not None:
                try:
                    engine.close()
                except Exception:
                    pass
            vdevice.release()
        temperature = cpu_temperature_c()
        per_arm[name]["temperature"]["end_c"] = temperature
        atomic_write(progress, {
            "status": "RUNNING" if arm_index < len(spec["arms"]) else "PASS",
            "completed_arms": arm_index, "total_arms": len(spec["arms"]),
            "last_arm": name, "temperature_c": temperature,
        })
        print(json.dumps({"progress": f"{arm_index}/{len(spec['arms'])} arms", "arm": name,
                          "temperature_c": temperature}), flush=True)
    for row in sample_rows:
        row["temperature_end_c"] = cpu_temperature_c()

    summary = {}
    for name, values in per_arm.items():
        nmse = {
            output: float(values["sse"][output] / values["reference_energy"][output])
            for output in BRANCH_WEIGHTS
        }
        summary[name] = {
            "outputs": {
                output: {
                    **{metric: summarize(rows, metric) for metric in ("cosine", "mae", "rmse", "max_abs")},
                    "aggregate_nmse": nmse[output],
                    "aggregate_sse": values["sse"][output],
                    "aggregate_reference_energy": values["reference_energy"][output],
                }
                for output, rows in (("x_h_hat", values["x_h_hat"]), ("x_l_hat", values["x_l_hat"]))
            },
            "weighted_branch_nmse": float(sum(BRANCH_WEIGHTS[x] * nmse[x] for x in BRANCH_WEIGHTS)),
            "input_saturation_fraction": summarize(values["saturation"], "fraction"),
            "activation_plus_infer_ms": summarize(values["latency"], "activation_plus_infer_ms"),
            "infer_wall_ms": summarize(values["latency"], "infer_wall_ms"),
            "temperature_c": values["temperature"],
        }
    atomic_write(args.out, {
        "kind": "bn_g_s_clip_sweep_same_yhat_board_correctness",
        "status": "PASS", "sample_count": 32, "val_indices": bundle["val_indices"],
        "checkpoint_sha256": spec["checkpoint_sha256"],
        "branch_weights": BRANCH_WEIGHTS,
        "ranking_metric": "0.37*NMSE(x_h_hat)+0.63*NMSE(x_l_hat)",
        "policy": {
            "diagnostic_only": True, "semantic_ranking": False,
            "not_a_full976_fixed19_result": True,
        },
        "arms": summary, "samples": sample_rows,
    })


if __name__ == "__main__":
    main()
