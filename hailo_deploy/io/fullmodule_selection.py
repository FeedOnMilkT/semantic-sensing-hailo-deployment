"""Immutable mixed-root selection and Hailo interface contracts for 10-module chains."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from hailo_deploy.io.hash import sha256
from hailo_deploy.io.interface_contract import extract_hef_io_contract


MODULES = (
    "sender_backbone", "bn_align_h", "bn_g_a", "bn_h_a", "bn_h_s", "bn_g_s",
    "receiver_stage4", "receiver_stage5", "receiver_dappm", "receiver_head",
)
REQUIRED_ARTIFACTS = (
    "parse_har", "model_script", "quantized_har", "hef", "compiled_har", "auto_alls",
)


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def canonical_fingerprint(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _artifact(item: Any, label: str) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise ValueError(f"missing artifact record: {label}")
    path = Path(str(item.get("path", ""))).resolve()
    expected = item.get("sha256")
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"SHA mismatch for {label}: {path}")
    return {"path": str(path), "sha256": actual, "bytes": path.stat().st_size}


def freeze_result(result_path: Path, module: str, onnx_entry: Mapping[str, Any]) -> dict[str, Any]:
    result_path = result_path.resolve()
    result = load_json(result_path)
    if result.get("schema") != "ptq.v1" or result.get("status") != "PASS":
        raise ValueError(f"{module}: result is not ptq.v1 PASS: {result_path}")
    if result.get("module") != module or int(result.get("n_calib", 0)) != 1024:
        raise ValueError(f"{module}: result identity/count mismatch: {result_path}")
    onnx = _artifact(result.get("onnx"), f"{module}.onnx")
    if onnx["sha256"] != onnx_entry.get("sha256"):
        raise ValueError(f"{module}: ONNX differs from handoff manifest")
    artifacts = {key: _artifact(result.get(key), f"{module}.{key}") for key in REQUIRED_ARTIFACTS}
    return {
        "module": module,
        "result_json": {
            "path": str(result_path), "sha256": sha256(result_path), "bytes": result_path.stat().st_size,
        },
        "n_calib": int(result["n_calib"]),
        "optimization_level": int(result["optimization_level"]),
        "num_contexts": int(result["num_contexts"]),
        "onnx": onnx,
        **artifacts,
    }


def load_selection(path: Path, *, verify_files: bool = True) -> dict[str, Any]:
    value = load_json(path)
    if value.get("schema") != "semantic_sensing.fullmodule_selection.v1":
        raise ValueError(f"unsupported selection manifest: {path}")
    contract = {key: item for key, item in value.items() if key != "candidate_fingerprint"}
    if canonical_fingerprint(contract) != value.get("candidate_fingerprint"):
        raise ValueError(f"selection fingerprint mismatch: {path}")
    modules = value.get("modules")
    if not isinstance(modules, dict) or set(modules) != set(MODULES) or value.get("module_order") != list(MODULES):
        raise ValueError(f"selection must contain ordered 10 modules: {path}")
    if verify_files:
        for module, record in modules.items():
            for key in ("result_json", "onnx", *REQUIRED_ARTIFACTS):
                _artifact(record.get(key), f"{module}.{key}")
    return value


def result_paths_from_selection(path: Path) -> dict[str, Path]:
    value = load_selection(path)
    return {module: Path(record["result_json"]["path"]) for module, record in value["modules"].items()}


def freeze_candidate(
    *, candidate: str, policy: Mapping[str, Any], handoff_root: Path,
    result_paths: Mapping[str, Path], selection_path: Path, boundary_path: Path,
) -> dict[str, Any]:
    if selection_path.exists() or boundary_path.exists():
        existing = load_selection(selection_path)
        if existing.get("candidate") != candidate:
            raise ValueError(f"existing selection candidate mismatch: {selection_path}")
        if not boundary_path.is_file():
            raise FileNotFoundError(boundary_path)
        boundary = load_json(boundary_path)
        if boundary.get("candidate_fingerprint") != existing["candidate_fingerprint"]:
            raise ValueError(f"boundary fingerprint mismatch: {boundary_path}")
        return existing
    if set(result_paths) != set(MODULES):
        raise ValueError("result paths must contain the frozen 10 modules")
    handoff_root = handoff_root.resolve()
    identity_path = handoff_root / "identity.json"
    identity = load_json(identity_path)
    onnx_manifest_path = handoff_root / "onnx" / "manifest.json"
    onnx_manifest = load_json(onnx_manifest_path)
    entries = {entry["name"]: entry for entry in onnx_manifest.get("modules", [])}
    if set(entries) != set(MODULES):
        raise ValueError("handoff ONNX manifest does not contain the exact 10 modules")
    checkpoint_sha = identity.get("checkpoint", {}).get("sha256")
    modules: dict[str, Any] = {}
    for module in MODULES:
        modules[module] = freeze_result(Path(result_paths[module]), module, entries[module])
        result = load_json(Path(result_paths[module]))
        result_checkpoint = result.get("identity", {}).get("checkpoint", {}).get("sha256")
        if result_checkpoint not in (None, checkpoint_sha):
            raise ValueError(f"{module}: checkpoint identity mismatch")
    contract: dict[str, Any] = {
        "schema": "semantic_sensing.fullmodule_selection.v1",
        "candidate": candidate,
        "policy": dict(policy),
        "handoff_root": str(handoff_root),
        "handoff_identity": {"path": str(identity_path), "sha256": sha256(identity_path)},
        "onnx_manifest": {"path": str(onnx_manifest_path), "sha256": sha256(onnx_manifest_path)},
        "checkpoint_sha256": checkpoint_sha,
        "host_vstream_format": "FLOAT32",
        "raw_integer_handoff": False,
        "module_order": list(MODULES),
        "modules": modules,
    }
    contract["candidate_fingerprint"] = canonical_fingerprint(contract)

    interfaces: dict[str, Any] = {}
    for module in MODULES:
        entry = {**entries[module], "id": module}
        interfaces[module] = extract_hef_io_contract(
            Path(modules[module]["quantized_har"]["path"]),
            Path(modules[module]["hef"]["path"]), entry,
        )
    boundaries = {
        "schema": "semantic_sensing.fullmodule_boundary.v1",
        "candidate": candidate,
        "candidate_fingerprint": contract["candidate_fingerprint"],
        "host_vstream_format": "FLOAT32",
        "raw_integer_handoff": False,
        "requires_equal_qp_across_hefs": False,
        "interfaces": interfaces,
        "codec_boundaries": {
            "z": {
                "producer": interfaces["bn_h_a"]["outputs"]["latent_z"],
                "consumer": interfaces["bn_h_s"]["inputs"]["latent_z_hat"],
                "operation": "entropy_bottleneck real rANS roundtrip",
            },
            "y": {
                "producer": interfaces["bn_g_a"]["outputs"]["latent_y"],
                "parameters": {
                    "means": interfaces["bn_h_s"]["outputs"]["means"],
                    "scales": interfaces["bn_h_s"]["outputs"]["scales"],
                },
                "consumer": interfaces["bn_g_s"]["inputs"]["latent_y_hat"],
                "operation": "gaussian_conditional real rANS roundtrip",
            },
        },
    }
    atomic_json(selection_path, contract)
    atomic_json(boundary_path, boundaries)
    return contract


def select_candidate(
    reports: Mapping[str, Mapping[str, Any]], thresholds: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply the approved screen/golden selector without reading formal results."""
    rows: dict[str, dict[str, Any]] = {}
    for candidate, bundle in reports.items():
        golden = bundle.get("golden")
        screen = bundle.get("screen")
        reasons: list[str] = []
        if not isinstance(golden, dict) or golden.get("status") != "PASS":
            reasons.append("golden8_not_pass")
        if not isinstance(screen, dict) or screen.get("status") != "PASS":
            reasons.append("screen64_not_pass")
        row: dict[str, Any] = {"candidate": candidate, "reasons": reasons}
        if not reasons:
            try:
                row.update({
                    "bpp": float(screen["bitstream"]["sdk_quantized"]["mean"]),
                    "logits_cosine": float(golden["final_logits_cumulative_metrics"]["cosine"]["mean"]),
                    "argmax_agreement": float(golden["final_argmax"]["agreement"]),
                    "fixed19": float(golden["semantic"]["sdk_quantized_chain"]["fixed_19_large_object_macro_iou"]),
                    "real_rans": int(golden["real_rans_success_count"]) + int(screen["real_rans_success_count"]),
                })
            except (KeyError, TypeError, ValueError) as exc:
                reasons.append(f"invalid_report_schema:{exc}")
        if not reasons:
            if row["real_rans"] != 72:
                reasons.append("real_rans_not_72")
            if row["bpp"] > float(thresholds["max_total_bpp"]):
                reasons.append("bpp_above_limit")
            if row["logits_cosine"] < float(thresholds["min_final_logits_cosine"]):
                reasons.append("logits_cosine_below_limit")
        rows[candidate] = row

    base = [row for row in rows.values() if not row["reasons"]]
    if not base:
        return {"schema": "semantic_sensing.fullmodule_candidate_decision.v1",
                "status": "STOP_NO_ELIGIBLE_CANDIDATE", "selected": None, "candidates": rows}
    best_argmax = max(row["argmax_agreement"] for row in base)
    best_fixed19 = max(row["fixed19"] for row in base)
    argmax_gap = float(thresholds["max_argmax_gap_pp"]) / 100.0
    fixed19_gap = float(thresholds["max_fixed19_gap_pp"]) / 100.0
    for row in base:
        if best_argmax - row["argmax_agreement"] > argmax_gap:
            row["reasons"].append("argmax_quality_veto")
        if best_fixed19 - row["fixed19"] > fixed19_gap:
            row["reasons"].append("fixed19_quality_veto")

    rate_winner = min(base, key=lambda row: row["bpp"])
    quality_winner = max(base, key=lambda row: (row["fixed19"], row["argmax_agreement"]))
    if rate_winner is not quality_winner and any(reason.endswith("quality_veto") for reason in rate_winner["reasons"]):
        return {
            "schema": "semantic_sensing.fullmodule_candidate_decision.v1",
            "status": "PAUSED_AMBIGUOUS_SELECTION", "selected": None,
            "ambiguity": {"rate_winner": rate_winner["candidate"],
                          "quality_winner": quality_winner["candidate"]},
            "candidates": rows,
        }

    eligible = [row for row in base if not row["reasons"]]
    if not eligible:
        return {"schema": "semantic_sensing.fullmodule_candidate_decision.v1",
                "status": "STOP_NO_ELIGIBLE_CANDIDATE", "selected": None, "candidates": rows}

    c3 = rows.get("C3")
    non_qft = [row for row in eligible if row["candidate"] != "C3"]
    if c3 in eligible and non_qft:
        best_non_qft = min(non_qft, key=lambda row: row["bpp"])
        absolute = best_non_qft["bpp"] - c3["bpp"]
        relative = absolute / max(best_non_qft["bpp"], 1e-12)
        c3["improvement_vs_best_non_qft"] = {"absolute_bpp": absolute, "relative": relative}
        if absolute < float(thresholds["rate_absolute_bpp"]) or relative < float(thresholds["rate_relative_fraction"]):
            c3["reasons"].append("qft_improvement_below_both_required_thresholds")
            eligible = [row for row in eligible if row is not c3]

    ranked = sorted(eligible, key=lambda row: row["bpp"])
    selected = ranked[0]
    simplicity = list(thresholds["simplicity_order"])
    for row in eligible:
        if simplicity.index(row["candidate"]) >= simplicity.index(selected["candidate"]):
            continue
        improvement = row["bpp"] - selected["bpp"]
        relative = improvement / max(row["bpp"], 1e-12)
        if improvement < float(thresholds["rate_absolute_bpp"]) or relative < float(thresholds["rate_relative_fraction"]):
            selected = row
    return {
        "schema": "semantic_sensing.fullmodule_candidate_decision.v1",
        "status": "UNIQUE_SELECT", "selected": selected["candidate"],
        "selection_basis": "screen64 rate + golden8 quality proxy; formal unseen",
        "candidates": rows,
    }
