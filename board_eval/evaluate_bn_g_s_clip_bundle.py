#!/usr/bin/env python3
"""Compare two ``bn_g_s`` HEFs with FP32 ORT on one frozen y_hat bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from bn_g_s_branch_attribution_runtime import tensor_drift
from fullmodule_runtime_runner import ModuleEngine
from hybrid_ort_runtime_runner import OrtModuleEngine, cpu_temperature_c


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
    vdevice = VDevice()
    fp = OrtModuleEngine(root, spec["fp32"], int(spec.get("ort_threads", 4)))
    engines = {
        arm["name"]: ModuleEngine(vdevice, root, arm["module_spec"])
        for arm in spec["arms"]
    }
    per_arm: dict[str, dict[str, list[dict]]] = {
        name: {"x_h_hat": [], "x_l_hat": [], "latency": [], "saturation": []}
        for name in engines
    }
    sample_rows: list[dict] = []
    progress = args.out.with_suffix(".progress.json")
    try:
        for position, index in enumerate(bundle["val_indices"], start=1):
            y_hat = np.load(args.bundle_root / "latent_y_hat" / f"{index}.npy", allow_pickle=False)
            y_hat = np.ascontiguousarray(y_hat, dtype=np.float32)
            digest = hashlib.sha256(y_hat.view(np.uint8)).hexdigest()
            expected_digest = bundle["rows"][position - 1]["sha256"]
            if digest != expected_digest:
                raise ValueError(f"y_hat hash mismatch for {index}")
            reference, fp_timing = fp.infer({"latent_y_hat": y_hat})
            row = {"val_index": index, "latent_y_hat_sha256": digest,
                   "fp32_ort_timing": fp_timing, "temperature_start_c": cpu_temperature_c(),
                   "arms": {}}
            for arm in spec["arms"]:
                name = arm["name"]
                got, timing = engines[name].infer({"latent_y_hat": y_hat})
                arm_row = {"timing": timing, "outputs": {}}
                for output in ("x_h_hat", "x_l_hat"):
                    metrics = tensor_drift(got[output], reference[output])
                    arm_row["outputs"][output] = metrics
                    per_arm[name][output].append(metrics)
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
                row["arms"][name] = arm_row
            row["temperature_end_c"] = cpu_temperature_c()
            sample_rows.append(row)
            atomic_write(progress, {
                "status": "RUNNING" if position < 32 else "PASS",
                "sample_count": position, "last_val_index": index,
                "temperature_c": row["temperature_end_c"],
            })
            print(json.dumps({"progress": f"{position}/32", "temperature_c": row["temperature_end_c"]}), flush=True)
    finally:
        fp.close()
        for engine in engines.values():
            try:
                engine.close()
            except Exception:
                pass
        vdevice.release()

    summary: dict[str, dict] = {}
    for name, values in per_arm.items():
        summary[name] = {
            "outputs": {
                output: {metric: summarize(rows, metric) for metric in ("cosine", "mae", "rmse", "max_abs")}
                for output, rows in (("x_h_hat", values["x_h_hat"]), ("x_l_hat", values["x_l_hat"]))
            },
            "input_saturation_fraction": summarize(values["saturation"], "fraction"),
            "activation_plus_infer_ms": summarize(values["latency"], "activation_plus_infer_ms"),
            "infer_wall_ms": summarize(values["latency"], "infer_wall_ms"),
        }
    atomic_write(args.out, {
        "kind": "bn_g_s_clip_same_yhat_board_correctness",
        "status": "PASS", "sample_count": 32, "val_indices": bundle["val_indices"],
        "checkpoint_sha256": spec["checkpoint_sha256"],
        "policy": {"diagnostic_only": True, "semantic_ranking": False},
        "arms": summary, "samples": sample_rows,
    })


if __name__ == "__main__":
    main()
