#!/usr/bin/env python3
"""Resumable bn_g_s-only input clipping range sweep (no full976 semantics)."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from board_eval.deploy import stream_upload  # noqa: E402
from hailo_deploy.emulate import _build_native_feed, _match_outputs  # noqa: E402
from hailo_deploy.io.hash import sha256  # noqa: E402
from hailo_deploy.io.metrics import metric, summarize  # noqa: E402
from hailo_deploy.onnx_io import run_onnxruntime, signature  # noqa: E402
from pipelines.run_bn_g_s_clip_feasibility import (  # noqa: E402
    BASE_MANIFEST, CALIB, CHECKPOINT_SHA, IDENTITY, KNOWN_HOSTS, ONNX, ONNX_SHA,
    PROXY, REMOTE_PYTHON, REMOTE_SEED, SSH_TARGET, atomic_write, compile_arm,
    hardlink, hef_static, launch_remote, optimize_arm, poll_remote, ssh,
    ssh_options, ssh_readonly, update_state,
)


SOURCE = Path(
    "/local/shared_with_docker/deplyment-demo/artifacts/fused/experiments/"
    "bn-gs-input-force-range-continuation-retry-20260805T161225Z"
)
SOURCE_EXPECTED = {
    "precompile_report.json": "f16f87ff25d64900a3266da9d497303b4ccc6a042aa2a478ab4393d772b1e308",
    "parse/har/bn_g_s_hailo_model.har": "9d6b2fc89deb9a37e296c2b4cbd579e981f3c4873663c597ef475451e2dabaeb",
    "ptq/F0_minmax/bn_g_s_l2_quantized.har": "7f85ec35b7416e27afe6df1b5c22254359e198e28ab8c3488756e39a0721dd08",
    "ptq/F0_minmax/bn_g_s_l2.hef": "c7c5e1462a6f2ecca77d116d749be6c5392c7b7223a33b546b1a96cb721914fc",
    "ptq/F1_q0p1_q99p9/bn_g_s_l2_quantized.har": "e5465d3247d8be8d7207de3146c909d614e1c83a482a38610e5d694c774febda",
    "ptq/F1_q0p1_q99p9/bn_g_s_l2.hef": "d4d36a473b3497e9db76efa854dab56f5f18e60ca1dcdf2d2aa5c2bb040df740",
    "board/clip_yhat_bundle/result.json": "f9c14c3e3430774055bf0df0bc43d62d0af37f9378fd4392a32523fd4a484c7f",
}
FROZEN_ARMS = ("F0_minmax", "F1_q0p1_q99p9")
NEW_ARM_ORDER = (
    "C1_fixed_m127_p128", "C2_q0p01_q99p99", "C3_dfc_hist_equivalent",
    "C4_constrained_input_mse", "C5_branch_weighted_output_nmse",
)
WEIGHTS = {"x_h_hat": 0.37, "x_l_hat": 0.63}
SEED = 20260806
DFC_CLIP_SOURCE = Path(
    "/local/workspace/hailo_virtualenv/lib/python3.10/site-packages/"
    "hailo_sdk_client/quantization/tools/activation_clipping_tool.py"
)
DFC_ATOMIC_SOURCE = Path(
    "/local/workspace/hailo_virtualenv/lib/python3.10/site-packages/"
    "hailo_model_optimization/acceleras/atomic_ops/base_atomic_op.py"
)


def affine_uint8(low: float, high: float) -> dict:
    if not np.isfinite(low) or not np.isfinite(high) or not low < 0 < high:
        raise ValueError(f"range must straddle zero: {[low, high]}")
    zp = int(np.clip(np.rint(-255.0 * low / (high - low)), 0, 255))
    scale = max(-low / zp if zp else 0.0, high / (255 - zp) if zp < 255 else 0.0)
    return {
        "requested_min": float(low), "requested_max": float(high),
        "qp_zp": zp, "qp_scale": float(scale),
        "limvals_min": float(-zp * scale), "limvals_max": float((255 - zp) * scale),
    }


def fake_quantize(values: np.ndarray, quant: dict) -> np.ndarray:
    scale, zp = float(quant["qp_scale"]), int(quant["qp_zp"])
    return ((np.clip(np.rint(values / scale + zp), 0, 255) - zp) * scale).astype(np.float32)


def error_stats(values: np.ndarray, quant: dict) -> dict:
    reconstructed = fake_quantize(values, quant)
    delta = reconstructed - values
    energy = float(np.sum(values * values, dtype=np.float64))
    sse = float(np.sum(delta * delta, dtype=np.float64))
    lo, hi = float(quant["limvals_min"]), float(quant["limvals_max"])
    clipped = int(np.count_nonzero((values < lo) | (values > hi)))
    return {
        "element_count": int(values.size), "sse": sse,
        "mse": float(sse / values.size), "nmse": float(sse / energy),
        "clipped_count": clipped, "clipped_fraction": float(clipped / values.size),
    }


def dfc_percentile(hist: np.ndarray, bins: np.ndarray, percentile: float) -> float:
    cumulative = np.cumsum(hist)
    index_in_values = percentile / 100.0 * (np.sum(hist) - 1)
    bin_index = int(np.where(cumulative - 1 >= np.floor(index_in_values))[0][0])
    if cumulative[bin_index] - 1 > index_in_values:
        return float(bins[bin_index])
    next_nonempty = bin_index + 1
    while next_nonempty < len(bins) and hist[next_nonempty] == 0:
        next_nonempty += 1
    if next_nonempty == len(bins):
        return float(bins[bin_index])
    return float(bins[bin_index] + (index_in_values % 1) * (bins[next_nonempty] - bins[bin_index]))


def load_values(files: list[Path], indices: np.ndarray | None = None) -> np.ndarray:
    selected = files if indices is None else [files[int(i)] for i in indices]
    return np.concatenate([
        np.load(path, allow_pickle=False).reshape(-1) for path in selected
    ]).astype(np.float32, copy=False)


def output_nmse(indices: np.ndarray, candidates: dict[str, dict]) -> dict:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(ONNX), sess_options=options, providers=["CPUExecutionProvider"])
    output_names = [output.name for output in session.get_outputs()]
    if set(output_names) != set(WEIGHTS):
        raise ValueError(f"unexpected ONNX outputs: {output_names}")
    accum = {
        name: {"sse": {x: 0.0 for x in WEIGHTS}, "energy": {x: 0.0 for x in WEIGHTS}}
        for name in candidates
    }
    files = sorted((CALIB / "latent_y_hat").glob("*.npy"))
    for index in indices:
        y = np.load(files[int(index)], allow_pickle=False).astype(np.float32, copy=False)
        nchw = y.transpose(2, 0, 1)[None]
        refs = dict(zip(output_names, session.run(output_names, {"latent_y_hat": nchw})))
        reference_energy = {
            name: float(np.sum(np.asarray(value, dtype=np.float64) ** 2))
            for name, value in refs.items()
        }
        for name, quant in candidates.items():
            quantized = fake_quantize(y, quant).transpose(2, 0, 1)[None]
            outputs = dict(zip(output_names, session.run(output_names, {"latent_y_hat": quantized})))
            for branch in WEIGHTS:
                delta = np.asarray(outputs[branch], dtype=np.float64) - np.asarray(refs[branch], dtype=np.float64)
                accum[name]["sse"][branch] += float(np.sum(delta * delta))
                accum[name]["energy"][branch] += reference_energy[branch]
    report = {}
    for name, values in accum.items():
        nmse = {branch: values["sse"][branch] / values["energy"][branch] for branch in WEIGHTS}
        report[name] = {
            "branch_nmse": nmse,
            "weighted_branch_nmse": float(sum(WEIGHTS[x] * nmse[x] for x in WEIGHTS)),
            "sse": values["sse"], "reference_energy": values["energy"],
        }
    return report


def design_ranges(root: Path) -> dict:
    output = root / "range_design.json"
    if output.is_file():
        return json.loads(output.read_text(encoding="utf-8"))
    files = sorted((CALIB / "latent_y_hat").glob("*.npy"))
    if len(files) != 4096:
        raise ValueError(f"expected 4096 calibration tensors, got {len(files)}")
    rng = np.random.default_rng(SEED)
    design_indices = np.arange(0, 3072, dtype=np.int64)
    audit_indices = np.arange(3072, 4096, dtype=np.int64)
    output_design = np.sort(rng.choice(design_indices, size=64, replace=False))
    output_audit = np.sort(rng.choice(audit_indices, size=64, replace=False))

    all_values = load_values(files)
    minimum, maximum = float(all_values.min()), float(all_values.max())
    exact = np.quantile(all_values, [0.0001, 0.001, 0.999, 0.9999], method="linear")
    hist = np.histogram(all_values, bins=1000, range=(minimum, maximum))[0].astype(np.int64)
    hist_bins = np.linspace(minimum, maximum, 1000, dtype=np.float32)
    ranges = {
        "F0_minmax": affine_uint8(minimum, maximum),
        "F1_q0p1_q99p9": affine_uint8(float(exact[1]), float(exact[2])),
        "C1_fixed_m127_p128": affine_uint8(-127.0, 128.0),
        "C2_q0p01_q99p99": affine_uint8(float(exact[0]), float(exact[3])),
        "C3_dfc_hist_equivalent": affine_uint8(
            dfc_percentile(hist, hist_bins, 0.1), dfc_percentile(hist, hist_bins, 99.9)
        ),
    }
    full4096_input_error = {name: error_stats(all_values, quant) for name, quant in ranges.items()}
    del all_values

    design_values, audit_values = load_values(files, design_indices), load_values(files, audit_indices)
    totals = (0.0001, 0.0002, 0.0004, 0.0008, 0.0012, 0.0016, 0.0020)
    allocations = tuple(i / 8 for i in range(9))
    quantile_levels = sorted({
        level for total in totals for ratio in allocations
        for level in (total * ratio, 1.0 - total * (1.0 - ratio))
    })
    quantiles = dict(zip(quantile_levels, np.quantile(design_values, quantile_levels, method="linear")))
    grid = []
    rejected_grid = []
    for total in totals:
        for ratio in allocations:
            lower_tail, upper_tail = total * ratio, total * (1.0 - ratio)
            quant = affine_uint8(float(quantiles[lower_tail]), float(quantiles[1.0 - upper_tail]))
            design_error = error_stats(design_values, quant)
            row = {
                "total_tail_budget": total, "lower_tail_budget": lower_tail,
                "upper_tail_budget": upper_tail, "quantization": quant,
                "design_input_error": design_error,
            }
            if design_error["clipped_fraction"] <= 0.002 + 1e-12:
                grid.append(row)
            else:
                rejected_grid.append({
                    **row,
                    "rejection_reason": "actual Hailo-affine clipping fraction exceeds 0.002",
                })
    if not grid:
        raise ValueError("C4 constrained grid has no eligible candidates")
    grid.sort(key=lambda row: row["design_input_error"]["mse"])
    c4 = dict(grid[0]["quantization"])
    ranges["C4_constrained_input_mse"] = c4
    top = grid[:8]
    c5_candidates = {f"grid_rank_{rank:02d}": row["quantization"] for rank, row in enumerate(grid, 1)}
    c5_candidates.update({f"anchor_{name}": quant for name, quant in ranges.items()})
    # Exact-range de-duplication keeps ORT search small and deterministic.
    unique = {}
    for name, quant in c5_candidates.items():
        key = (quant["requested_min"], quant["requested_max"])
        unique.setdefault(key, (name, quant))
    c5_candidates = {name: quant for name, quant in unique.values()}
    design_output = output_nmse(output_design, c5_candidates)
    selected_name = min(design_output, key=lambda name: design_output[name]["weighted_branch_nmse"])
    c5 = dict(c5_candidates[selected_name])
    ranges["C5_branch_weighted_output_nmse"] = c5
    audit_output = output_nmse(output_audit, {"selected_C5": c5})
    aliases = {}
    c5_key = (c5["requested_min"], c5["requested_max"])
    for prior in (*FROZEN_ARMS, *NEW_ARM_ORDER[:-1]):
        prior_quant = ranges[prior]
        if c5_key == (prior_quant["requested_min"], prior_quant["requested_max"]):
            aliases["C5_branch_weighted_output_nmse"] = prior
            break
    payload = {
        "kind": "bn_g_s_input_clip_range_design", "status": "PASS",
        "seed": SEED, "calibration_sample_count": 4096,
        "identity": {"checkpoint_sha256": CHECKPOINT_SHA, "onnx_sha256": sha256(ONNX)},
        "split": {
            "input_design_3072": design_indices.tolist(), "input_audit_1024": audit_indices.tolist(),
            "output_design_64": output_design.tolist(), "output_audit_64": output_audit.tolist(),
            "source": "ADE20K train bn_g_s calibration only; no held-out labels or semantics",
        },
        "fixed_full4096_statistics": {
            "minimum": minimum, "maximum": maximum,
            "q0p01": float(exact[0]), "q0p1": float(exact[1]),
            "q99p9": float(exact[2]), "q99p99": float(exact[3]),
        },
        "fixed_full4096_input_fake_quant_error": full4096_input_error,
        "dfc_hist_equivalent": {
            "nbins": 1000, "hist_range": [minimum, maximum],
            "percentiles": [0.1, 99.9],
            "method": "external exact reproduction of DFC output-percentile histogram selector on input data",
            "not_builtin_input_activation_clipping": True,
            "reason": "DFC 3.33 does not collect input histograms; conv1 output clipping is forbidden here",
            "source_files": {str(path): sha256(path) for path in (DFC_CLIP_SOURCE, DFC_ATOMIC_SOURCE)},
        },
        "c4_search": {
            "constraint": "actual affine-range clipped_fraction <= 0.002 on design split",
            "objective": "input fake-quantization MSE", "totals": totals,
            "asymmetric_lower_allocations": allocations,
            "proposed_candidate_count": len(totals) * len(allocations),
            "eligible_candidate_count": len(grid),
            "rejected_candidate_count": len(rejected_grid),
            "rejected_candidates": rejected_grid,
            "winner": grid[0], "winner_audit_input_error": error_stats(audit_values, c4),
            "top8": top,
        },
        "c5_search": {
            "pool": "all eligible C4 grid candidates plus F0/F1/C1/C2/C3-hist/C4 anchors, exact-range deduplicated",
            "objective": "0.37*NMSE(x_h_hat)+0.63*NMSE(x_l_hat)",
            "reference": "FP32 ORT bn_g_s output from original calibration y_hat",
            "candidate": "FP32 ORT bn_g_s output from Hailo-affine fake-quantized y_hat",
            "design_scores": design_output, "winner_name": selected_name,
            "winner_audit": audit_output["selected_C5"],
        },
        "ranges": ranges, "aliases": aliases,
        "policy": {"bn_g_s_only": True, "no_heldout_labels": True, "full976_not_run": True},
    }
    atomic_write(output, payload)
    return payload


def link_tree(source: Path, destination: Path) -> list[dict]:
    links = []
    for path in sorted(source.rglob("*")):
        if path.is_file():
            links.append(hardlink(path, destination / path.relative_to(source)))
    return links


def prepare(root: Path) -> None:
    lineage = root / "reuse_lineage.json"
    if lineage.is_file():
        return
    for relative, expected in SOURCE_EXPECTED.items():
        if sha256(SOURCE / relative) != expected:
            raise ValueError(f"frozen source SHA mismatch: {relative}")
    graph_check = json.loads((SOURCE / "precompile_report.json").read_text(encoding="utf-8"))[
        "range_control_substitution"
    ]["static_graph_check"]
    if graph_check != {
        "input_layer": "bn_g_s/input_layer1", "input_shape_nhwc": [-1, 16, 16, 192],
        "exact_first_and_only_consumer": "bn_g_s/conv1", "consumer_input_index": 0,
    }:
        raise ValueError(f"frozen first-consumer proof mismatch: {graph_check}")
    links = []
    for relative in (
        "parse/har/bn_g_s_hailo_model.har", "parse/results/pipeline.json",
        "parse/results/native_cpu_fp32.json", "parse/reports/pipeline.md",
    ):
        links.append(hardlink(SOURCE / relative, root / relative))
    links.append(hardlink(SOURCE / "precompile_report.json", root / "references/source_precompile_report.json"))
    for arm in FROZEN_ARMS:
        links.extend(link_tree(SOURCE / "ptq" / arm, root / "ptq" / arm))
    links.extend(link_tree(SOURCE / "board/clip_yhat_bundle", root / "board/clip_yhat_bundle"))
    atomic_write(lineage, {
        "kind": "bn_g_s_clip_sweep_reuse", "source_root": str(SOURCE),
        "source_policy": "immutable SHA-checked hardlinks; no source mutation",
        "links": links,
    })


def write_scripts(root: Path, design: dict) -> dict[str, Path]:
    directory = root / "model_scripts"
    directory.mkdir(exist_ok=True)
    common = (
        "# Freeze actual x_l low-spatial output at A16; x_h remains UINT8.\n"
        "quantization_param(bn_g_s/conv3, precision_mode=a16_w16)\n"
        "quantization_param(bn_g_s/output_layer1, precision_mode=a16_w16)\n"
    )
    paths = {}
    for arm in NEW_ARM_ORDER:
        if arm in design["aliases"]:
            continue
        quant = design["ranges"][arm]
        control = (
            "# Input-layer activation_clipping is unsupported in DFC 3.33.\n"
            "# bn_g_s/conv1 is the verified first/only input consumer, slot 0.\n"
            "quantization_param([bn_g_s/conv1], "
            f"force_range_in=[{quant['requested_min']:.17g}, {quant['requested_max']:.17g}], "
            "force_range_index=0)\n"
        )
        path = directory / f"{arm}.alls"
        intended = common + control
        if path.exists() and path.read_text(encoding="utf-8") != intended:
            raise ValueError(f"existing model script differs: {path}")
        if not path.exists():
            path.write_text(intended, encoding="utf-8")
        paths[arm] = path
    return paths


def sdk32(root: Path, arms: list[str]) -> dict:
    output = root / "sdk32.json"
    if output.is_file():
        return json.loads(output.read_text(encoding="utf-8"))
    bundle = root / "board/clip_yhat_bundle"
    bundle_result = json.loads((bundle / "result.json").read_text(encoding="utf-8"))
    from hailo_sdk_client import ClientRunner, InferenceContext

    sig = signature(ONNX)
    report = {
        "kind": "bn_g_s_clip_sweep_sdk_quantized_same_yhat", "status": "PASS",
        "sample_count": 32, "branch_weights": WEIGHTS, "arms": {},
        "policy": {"diagnostic_only": True, "semantic_ranking": False},
    }
    for arm in arms:
        runner = ClientRunner(har=str(root / "ptq" / arm / "bn_g_s_l2_quantized.har"))
        rows = {x: [] for x in WEIGHTS}
        sse = {x: 0.0 for x in WEIGHTS}
        energy = {x: 0.0 for x in WEIGHTS}
        notes = []
        with runner.infer_context(InferenceContext.SDK_QUANTIZED) as context:
            for index in bundle_result["val_indices"]:
                y = np.load(bundle / "latent_y_hat" / f"{index}.npy", allow_pickle=False)
                sample = {"latent_y_hat": y.transpose(2, 0, 1)[None].astype(np.float32)}
                reference = run_onnxruntime(ONNX, sample)
                pairs = _match_outputs(runner, reference, runner.infer(context, _build_native_feed(runner, sig, sample, notes)), notes)
                for name, (ref, got) in pairs.items():
                    rows[name].append(metric(ref, got))
                    delta = np.asarray(got, dtype=np.float64) - np.asarray(ref, dtype=np.float64)
                    sse[name] += float(np.sum(delta * delta))
                    energy[name] += float(np.sum(np.asarray(ref, dtype=np.float64) ** 2))
        nmse = {name: sse[name] / energy[name] for name in WEIGHTS}
        report["arms"][arm] = {
            "outputs": {name: {**summarize(values), "aggregate_nmse": nmse[name]} for name, values in rows.items()},
            "weighted_branch_nmse": float(sum(WEIGHTS[x] * nmse[x] for x in WEIGHTS)),
            "notes": sorted(set(notes)),
        }
    atomic_write(output, report)
    return report


def static_check(root: Path, design: dict, arms: list[str]) -> dict:
    output = root / "hef_static_report.json"
    if output.is_file():
        return json.loads(output.read_text(encoding="utf-8"))["arms"]
    rows = {arm: hef_static(root / "ptq" / arm / "bn_g_s_l2.hef") for arm in arms}
    for arm, row in rows.items():
        if len(row["network_groups"]) != 1 or len(row["inputs"]) != 1 or len(row["outputs"]) != 2:
            raise ValueError(f"{arm}: HEF topology mismatch")
        inp = row["inputs"][0]
        if inp["shape_hwc"] != [16, 16, 192] or inp["format"] != "UINT8":
            raise ValueError(f"{arm}: input contract mismatch")
        formats = {(tuple(x["shape_hwc"]), x["format"]) for x in row["outputs"]}
        if formats != {((64, 64, 128), "UINT8"), ((32, 32, 256), "UINT16")}:
            raise ValueError(f"{arm}: output contract mismatch: {formats}")
        if arm in design["ranges"]:
            expected = design["ranges"][arm]
            actual = inp["quantization"]
            tolerance = max(2.0 * float(actual["qp_scale"]), 1e-4)
            if abs(actual["limvals_min"] - expected["limvals_min"]) > tolerance or abs(actual["limvals_max"] - expected["limvals_max"]) > tolerance:
                raise ValueError(f"{arm}: requested input range did not reach HEF")
    atomic_write(output, {"status": "PASS", "arms": rows})
    return rows


def board32(root: Path, state: dict, static: dict, arms: list[str]) -> dict:
    leftovers = ssh_readonly(
        "pgrep -af '[e]valuate_bn_g_s_clip_sweep.py|[e]valuate_bn_g_s_clip_bundle.py|"
        "[r]un_board_hybrid_ort.py|[r]un_board_formal_eval.py' || true"
    ).strip()
    update_state(root, state, board_prelaunch_processes=leftovers)
    if leftovers:
        raise RuntimeError(f"board experiment process exists; refusing overlap: {leftovers}")
    remote_root = f".local/share/semantic-sensing-board/hybrid-ort/{root.name}"
    existing = ssh(f"test ! -e {shlex.quote(remote_root)} && echo ABSENT || echo PRESENT").strip()
    if existing == "ABSENT":
        ssh(
            f"test -d {shlex.quote(REMOTE_SEED)}; mkdir -p {shlex.quote(remote_root)}; "
            f"cp -al {shlex.quote(REMOTE_SEED + '/.')} {shlex.quote(remote_root + '/')}"
        )
    opts = ssh_options()
    for source in (
        ROOT / "board_eval/evaluate_bn_g_s_clip_sweep.py",
        ROOT / "board_eval/bn_g_s_branch_attribution_runtime.py",
        ROOT / "board_eval/fullmodule_runtime_runner.py",
        ROOT / "board_eval/hybrid_ort_runtime_runner.py",
    ):
        stream_upload(source, f"{remote_root}/{source.name}", opts, SSH_TARGET)
    candidates = f"{remote_root}/clip_sweep_candidates"
    ssh(f"mkdir -p {shlex.quote(candidates)}")
    manifest_arms = []
    for arm in arms:
        hef = root / "ptq" / arm / "bn_g_s_l2.hef"
        destination = f"{candidates}/{arm}.hef"
        stream_upload(hef, destination, opts, SSH_TARGET)
        manifest_arms.append({
            "name": arm, "input_quantization": static[arm]["inputs"][0]["quantization"],
            "module_spec": {
                "engine": "hailort_float32_vstream", "hef_filename": f"clip_sweep_candidates/{arm}.hef",
                "input_names": ["latent_y_hat"], "name": "bn_g_s",
                "in_shapes_nchw": [[1, 192, 16, 16]], "output_names": ["x_h_hat", "x_l_hat"],
                "out_shapes_nchw": [[1, 128, 64, 64], [1, 256, 32, 32]],
            },
        })
    manifest = {
        "kind": "bn_g_s_clip_sweep_board_manifest", "checkpoint_sha256": CHECKPOINT_SHA,
        "ort_threads": 4,
        "fp32": {
            "engine": "ort_cpu_fp32", "name": "bn_g_s", "onnx_filename": "onnx/bn_g_s.onnx",
            "input_names": ["latent_y_hat"], "in_shapes_nchw": [[1, 192, 16, 16]],
            "output_names": ["x_h_hat", "x_l_hat"],
            "out_shapes_nchw": [[1, 128, 64, 64], [1, 256, 32, 32]],
        },
        "arms": manifest_arms,
    }
    local_manifest = root / "board/board_clip_sweep_manifest.json"
    atomic_write(local_manifest, manifest)
    remote_manifest = f"{remote_root}/board_clip_sweep_manifest.json"
    stream_upload(local_manifest, remote_manifest, opts, SSH_TARGET)
    result_remote = f"{remote_root}/board_clip_sweep32/result.json"
    if ssh(f"test -f {shlex.quote(result_remote)} && echo PASS || true").strip() != "PASS":
        ssh(f"mkdir -p {shlex.quote(result_remote.rsplit('/', 1)[0])}")
        job = launch_remote(remote_root, f"board_clip_sweep32_attempt{state.get('resume_count', 0)}", [
            REMOTE_PYTHON, f"{remote_root}/evaluate_bn_g_s_clip_sweep.py",
            "--manifest", remote_manifest,
            "--bundle-root", ".local/share/semantic-sensing-board/hybrid-ort/"
            "bn-gs-input-force-range-continuation-retry-20260805T161225Z/clip_yhat_bundle",
            "--out", result_remote,
        ], result_override=result_remote, progress_override=f"{remote_root}/board_clip_sweep32/result.progress.json")
        poll_remote(root, state, job, timeout=2400)
    output = root / "board/board_clip_sweep32.json"
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        completed = subprocess.run(["ssh", *opts, SSH_TARGET, "cat", result_remote], stdout=handle)
    if completed.returncode:
        raise RuntimeError("failed to download board result")
    os.replace(temporary, output)
    return json.loads(output.read_text(encoding="utf-8"))


def summarize_campaign(root: Path, design: dict, sdk: dict, board: dict, static: dict, arms: list[str], started: float) -> dict:
    ranking = sorted(arms, key=lambda arm: board["arms"][arm]["weighted_branch_nmse"])
    pareto = []
    for arm in arms:
        point = [board["arms"][arm]["outputs"][x]["aggregate_nmse"] for x in WEIGHTS]
        if not any(
            other != arm and all(
                board["arms"][other]["outputs"][x]["aggregate_nmse"] <= point[i]
                for i, x in enumerate(WEIGHTS)
            ) and any(
                board["arms"][other]["outputs"][x]["aggregate_nmse"] < point[i]
                for i, x in enumerate(WEIGHTS)
            ) for other in arms
        ):
            pareto.append(arm)
    discrepancy = {}
    for arm in arms:
        discrepancy[arm] = {
            branch: {
                "sdk_nmse": sdk["arms"][arm]["outputs"][branch]["aggregate_nmse"],
                "board_nmse": board["arms"][arm]["outputs"][branch]["aggregate_nmse"],
                "board_over_sdk_ratio": (
                    board["arms"][arm]["outputs"][branch]["aggregate_nmse"] /
                    sdk["arms"][arm]["outputs"][branch]["aggregate_nmse"]
                ),
            } for branch in WEIGHTS
        }
    return {
        "kind": "bn_g_s_clip_range_sweep_summary", "status": "PASS",
        "elapsed_seconds": time.monotonic() - started, "range_design": design,
        "hef_static": static, "sdk32": sdk, "board32": board,
        "diagnostic_board32_order": ranking, "diagnostic_board32_pareto": pareto,
        "sdk_quantized_vs_board_hef": discrepancy,
        "interpretation": (
            "Ordering is tensor-level correctness/localization only. It cannot rank deployment semantics; "
            "only an explicitly authorized full976 fixed-19 run may select a candidate."
        ),
        "policy": {
            "bn_g_s_only": True, "runtime_boundary_unchanged": True,
            "full976_not_run": True, "semantic_ranking": False,
            "ade150_miou_not_computed_or_reported": True, "canonical_promotion": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--cuda-visible-devices", default="2")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        source_sha_ok = SOURCE.is_dir() and all(
            (SOURCE / relative).is_file() and sha256(SOURCE / relative) == expected
            for relative, expected in SOURCE_EXPECTED.items()
        )
        meta = json.loads((CALIB / "meta.json").read_text(encoding="utf-8"))
        payload = {
            "status": "PASS", "heavy_work_started": False,
            "source_exists": SOURCE.is_dir(), "onnx_sha256": sha256(ONNX),
            "source_sha_contract_pass": source_sha_ok,
            "calibration_checkpoint_sha256": meta.get("checkpoint", {}).get("sha256"),
            "calibration_onnx_sha256": meta.get("onnx", {}).get("sha256"),
            "calibration_files": len(list((CALIB / "latent_y_hat").glob("*.npy"))),
            "planned_arms": [*FROZEN_ARMS, *NEW_ARM_ORDER],
            "default_cuda_visible_devices": args.cuda_visible_devices,
        }
        if (
            payload["onnx_sha256"] != ONNX_SHA or payload["calibration_files"] != 4096
            or not source_sha_ok or payload["calibration_checkpoint_sha256"] != CHECKPOINT_SHA
            or payload["calibration_onnx_sha256"] != ONNX_SHA
        ):
            raise ValueError(payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    root = args.out_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    state_path = root / "campaign_state.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["resume_count"] = int(state.get("resume_count", 0)) + 1
    else:
        state = {"kind": "bn_g_s_clip_range_sweep_state", "status": "RUNNING", "resume_count": 0,
                 "started_unix_seconds": time.time()}
    update_state(root, state, status="RUNNING", stage="preflight", error=None)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    try:
        if sha256(ONNX) != ONNX_SHA:
            raise ValueError("frozen ONNX SHA mismatch")
        meta = json.loads((CALIB / "meta.json").read_text(encoding="utf-8"))
        if meta["checkpoint"]["sha256"] != CHECKPOINT_SHA or meta["onnx"]["sha256"] != ONNX_SHA:
            raise ValueError("calibration identity mismatch")
        prepare(root)
        update_state(root, state, stage="range_design")
        design = design_ranges(root)
        scripts = write_scripts(root, design)
        aliases = design["aliases"]
        physical_new = [arm for arm in NEW_ARM_ORDER if arm not in aliases]
        physical_arms = [*FROZEN_ARMS, *physical_new]
        har = root / "parse/har/bn_g_s_hailo_model.har"
        ptq = {
            arm: json.loads((root / "ptq" / arm / "result.json").read_text(encoding="utf-8"))
            for arm in FROZEN_ARMS
        }
        for arm in physical_new:
            update_state(root, state, stage=f"optimize_{arm}")
            ptq[arm] = optimize_arm(root, arm, scripts[arm], har, env)
        update_state(root, state, stage="sdk32_precompile")
        sdk = sdk32(root, physical_arms)
        precompile_path = root / "precompile_report.json"
        first_report = not precompile_path.exists()
        if first_report:
            atomic_write(precompile_path, {
                "kind": "bn_g_s_clip_range_sweep_precompile", "status": "CONTINUE_TO_COMPILE",
                "created_unix_seconds": time.time(),
                "identity": {
                    "checkpoint_sha256": CHECKPOINT_SHA, "onnx_sha256": ONNX_SHA,
                    "parse_har_sha256": sha256(har), "calibration_sample_count": 4096,
                    "optimization_level": 2, "compression_level": 0, "hw_arch": "hailo8",
                    "dfc": "3.33.0", "hailort_host": "4.23.0",
                    "x_h_format": "UINT8", "x_l_format": "UINT16",
                },
                "arms": {arm: {
                    "range": design["ranges"].get(arm),
                    "quantized_har": ptq[arm]["quantized_har"],
                    "sdk_quantized_calibration_tail8": ptq[arm]["quantized_check"],
                    "sdk_quantized_same_yhat32": sdk["arms"][arm],
                } for arm in physical_arms},
                "aliases": aliases,
                "policy": {"bn_g_s_only": True, "diagnostic_only": True, "full976_not_run": True},
            })
        update_state(root, state, stage="precompile_report_written", precompile_report_sha256=sha256(precompile_path))
        print(json.dumps({"PRECOMPILE_REPORT": str(precompile_path), "sha256": sha256(precompile_path)}), flush=True)
        if first_report:
            time.sleep(90)
        for arm in physical_new:
            update_state(root, state, stage=f"compile_{arm}")
            compile_arm(root, arm)
        static = static_check(root, design, physical_arms)
        update_state(root, state, stage="board32")
        board = board32(root, state, static, physical_arms)
        summary = summarize_campaign(root, design, sdk, board, static, physical_arms, started)
        if aliases:
            summary["aliases"] = aliases
        atomic_write(root / "summary.json", summary)
        update_state(root, state, status="PASS", stage="complete", elapsed_seconds=summary["elapsed_seconds"])
        return 0
    except Exception as exc:
        import traceback
        error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        atomic_write(root / "failure.json", {"status": "FAIL", "error": error})
        update_state(root, state, status="FAIL", stage="failed", error=error)
        print(error, file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
