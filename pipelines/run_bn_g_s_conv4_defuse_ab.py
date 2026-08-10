#!/usr/bin/env python3
"""Gated compile-only A/B for the ``bn_g_s/conv4`` spatial defuse count.

The immutable formal spatial-8 HEF is the baseline.  The only scheduled new
compile is spatial-4.  Other DFC 3.33 commands are parser-tested fallbacks and
are never compiled by this entry point.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from board_eval.deploy import stream_upload  # noqa: E402
from hailo_deploy.artifacts.layout import (  # noqa: E402
    commit_or_fail,
    refuse_existing,
    stage_dir,
)
from hailo_deploy.gates.compile_real import compile_quantized  # noqa: E402
from hailo_deploy.gates.compiler_probe import (  # noqa: E402
    hn_sha256,
    load_compile_only_script,
    optimized_params_sha256,
    parse_allocation_script,
)
from hailo_deploy.io.hash import sha256  # noqa: E402
from pipelines.run_bn_g_s_clip_feasibility import (  # noqa: E402
    CHECKPOINT_SHA,
    ONNX,
    ONNX_SHA,
    REMOTE_PYTHON,
    SSH_TARGET,
    atomic_write,
    launch_remote,
    poll_remote,
    ssh,
    ssh_options,
    ssh_readonly,
)
from pipelines.run_bn_g_s_compiler_probe import (  # noqa: E402
    FORMAL_AUTO_ALLS,
    FORMAL_AUTO_ALLS_SHA,
    FORMAL_HEF,
    FORMAL_HEF_SHA,
    FORMAL_QHAR,
    FORMAL_QHAR_SHA,
    GOLDEN,
    GOLDEN_RESULT_SHA,
    _map_static_outputs,
    _same_quantization,
    _validate_golden,
    hef_static,
)
import pipelines.run_bn_g_s_vstream_parity as parity  # noqa: E402


EXPECTED_HN_SHA = "3454dda68e200b164f7fcf565694731a05b16b8d5c0a341ed46a5c5bc69de0c9"
EXPECTED_PARAMS_SHA = "fc708e800b9dbabc6cf2d964b569e2d28c6c0643cdf7fb8c0b0c18c6ad2f41c5"
SCHEDULED_ARM = "conv4_spatial4"
REMOTE_GOLDEN = parity.REMOTE_GOLDEN
SAMPLE_INDICES = parity.SAMPLE_INDICES
CPU_MODULES = parity.CPU_MODULES


def spatial_script(count: int) -> str:
    outputs = [f"ab{count}_sd{i}" for i in range(count)] + [f"ab{count}_sdc"]
    return (
        f"{', '.join(outputs)} = defuse(bn_g_s/conv4, {count}, "
        "defuse_type=SPATIAL)\n"
    )


@dataclass(frozen=True)
class Candidate:
    script: str
    disposition: str
    reason: str


CANDIDATES = {
    SCHEDULED_ARM: Candidate(
        spatial_script(4),
        "SCHEDULED",
        "smallest targeted reduction from the formal spatial-8 mapping",
    ),
    "conv4_spatial2": Candidate(
        spatial_script(2),
        "PARSER_ONLY_FALLBACK",
        "more aggressive split reduction; compile only after spatial-4 evidence",
    ),
    "conv4_spatial1_explicit": Candidate(
        spatial_script(1),
        "PARSER_ONLY_NOT_UNDEFUSE",
        "DFC has no per-layer undefuse; spatial-1 is not proven equivalent to off",
    ),
    "conv4_spatial16": Candidate(
        spatial_script(16),
        "PARSER_ONLY_DOSE_FALLBACK",
        "stronger than formal spatial-8 and not the first causal contrast",
    ),
    "conv4_normal4": Candidate(
        "abn4_d0, abn4_d1, abn4_d2, abn4_d3, abn4_dc = "
        "defuse(bn_g_s/conv4, 4)\n",
        "PARSER_ONLY_AXIS_CHANGE",
        "changes defuse type as well as count and weakens attribution",
    ),
    "global_auto_defuse_off": Candidate(
        "allocator_param(width_splitter_defuse=disabled, max_auto_defuse=0, "
        "defuse_by_validator=False, enable_defuse_with_slack=False)\n",
        "PARSER_ONLY_GLOBAL",
        "global allocator policy can remap conv3/resize2/conv5 and explicit defuse may override it",
    ),
    "conv4_logical_hint": Candidate(
        "compilation_param(bn_g_s/conv4, defuse_num_layers=1, "
        "defuse_spatial_w=False)\n",
        "PARSER_ONLY_WEAK_HINT",
        "logical compilation hint does not prove the realized physical mapping",
    ),
}

# Wrapper entry points may narrow this to a more tightly locked topology.  The
# compile gate compares each named layer with the generated auto.alls exactly.
EXPECTED_REALIZED_DEFUSES = {
    "conv4": {"count": 4, "type": "SPATIAL"},
}


def allocation_facts(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"defuse\((?:bn_g_s/)?(?P<layer>[^,]+),\s*(?P<count>\d+)"
        r"(?:,\s*defuse_type=(?P<kind>[A-Z_]+))?\)"
    )
    rows = [
        {
            "layer": match.group("layer"),
            "count": int(match.group("count")),
            "type": match.group("kind") or "NORMAL",
            "text": match.group(0),
        }
        for match in pattern.finditer(text)
    ]
    return {
        "path": str(path),
        "sha256": sha256(path),
        "defuse_commands": rows,
        "conv4": [row for row in rows if row["layer"] == "conv4"],
        "context_command_count": text.count(" = context(["),
    }


def validate_realized_topology(realized: dict[str, Any]) -> dict[str, Any]:
    rows = realized["defuse_commands"]
    checked = {}
    for layer, expected in EXPECTED_REALIZED_DEFUSES.items():
        actual = [
            {"count": row["count"], "type": row["type"]}
            for row in rows
            if row["layer"] == layer
        ]
        wanted = [dict(expected)]
        status = "PASS" if actual == wanted else "MISMATCH"
        checked[layer] = {"status": status, "expected": wanted, "actual": actual}
        if status != "PASS":
            raise ValueError(
                f"compiler did not realize exact {layer} topology: "
                f"{actual} != {wanted}"
            )
    return {"status": "PASS", "layers": checked}


def parser_matrix() -> dict[str, Any]:
    from hailo_sdk_client import ClientRunner

    rows: dict[str, Any] = {}
    for name, candidate in CANDIDATES.items():
        runner = ClientRunner(har=str(FORMAL_QHAR))
        parsed = [str(command) for command in parse_allocation_script(
            runner, candidate.script
        )]
        guard = load_compile_only_script(runner, candidate.script)
        if guard["hn_sha256_before"] != EXPECTED_HN_SHA:
            raise ValueError(f"{name}: unexpected source HN hash")
        if guard["optimized_params_sha256_before"] != EXPECTED_PARAMS_SHA:
            raise ValueError(f"{name}: unexpected optimized parameter hash")
        rows[name] = {
            "status": "PASS",
            "disposition": candidate.disposition,
            "reason": candidate.reason,
            "script": candidate.script,
            "parsed_commands": parsed,
            "guard": guard,
        }
        del runner
    return {
        "status": "PASS",
        "mode": "DFC_3_33_ALLOCATION_ONLY",
        "meaning": "syntax and immutable numerical-state check only; not compile feasibility",
        "rows": rows,
    }


def validate_sources() -> dict[str, Any]:
    expected = {
        FORMAL_QHAR: FORMAL_QHAR_SHA,
        FORMAL_HEF: FORMAL_HEF_SHA,
        FORMAL_AUTO_ALLS: FORMAL_AUTO_ALLS_SHA,
        ONNX: ONNX_SHA,
        GOLDEN / "result.json": GOLDEN_RESULT_SHA,
    }
    for path, digest in expected.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if actual != digest:
            raise ValueError(f"immutable source mismatch: {path}: {actual} != {digest}")

    from hailo_sdk_client import ClientRunner

    runner = ClientRunner(har=str(FORMAL_QHAR))
    hn = runner.get_hn_dict()
    conv4 = hn.get("layers", {}).get("bn_g_s/conv4")
    if conv4 is None:
        raise ValueError("qHAR has no bn_g_s/conv4")
    originals = list(conv4.get("original_names") or [])
    conv_endpoint = "/head_h/head_h.1/head_h.1.0/Conv"
    relu_endpoint = "/head_h/head_h.1/head_h.1.2/Relu"
    if conv4.get("params", {}).get("activation") != "relu":
        raise ValueError("conv4 is not a fused-ReLU HN layer")
    if conv_endpoint not in originals or relu_endpoint not in originals:
        raise ValueError("conv4 original_names no longer prove Conv+ReLU fusion")
    if conv4.get("output") != ["bn_g_s/conv5"]:
        raise ValueError(f"conv4 consumer changed: {conv4.get('output')}")
    source_hn_sha = hn_sha256(runner)
    source_params_sha = optimized_params_sha256(runner)
    del runner
    if source_hn_sha != EXPECTED_HN_SHA or source_params_sha != EXPECTED_PARAMS_SHA:
        raise ValueError("immutable qHAR numerical identity changed")

    formal_allocation = allocation_facts(FORMAL_AUTO_ALLS)
    if formal_allocation["conv4"] != [{
        "layer": "conv4",
        "count": 8,
        "type": "SPATIAL",
        "text": "defuse(conv4, 8, defuse_type=SPATIAL)",
    }]:
        raise ValueError(f"formal conv4 allocation changed: {formal_allocation['conv4']}")

    static = hef_static(FORMAL_HEF)
    mapped = _map_static_outputs(static["outputs"], include_probe=False)
    if mapped["x_h_hat"]["format_type"] != "UINT8":
        raise ValueError("formal x_h is not UINT8")
    if mapped["x_l_hat"]["format_type"] != "UINT16":
        raise ValueError("formal x_l is not UINT16")

    return {
        "status": "PASS",
        "policy": {
            "diagnostic_only": True,
            "full976_not_run": True,
            "semantic_ranking": False,
            "source_qhar_immutable": True,
            "parse_called": False,
            "optimize_called": False,
            "compile_called": False,
            "board_called": False,
        },
        "source": {
            "checkpoint_sha256": CHECKPOINT_SHA,
            "quantized_har": {"path": str(FORMAL_QHAR), "sha256": FORMAL_QHAR_SHA},
            "formal_hef": {"path": str(FORMAL_HEF), "sha256": FORMAL_HEF_SHA},
            "formal_auto_alls": formal_allocation,
            "onnx": {"path": str(ONNX), "sha256": ONNX_SHA},
        },
        "conv4_tap_semantics": {
            "status": "PASS",
            "conclusion": "output_layer(conv4) is after the fused ReLU",
            "hn_type": conv4.get("type"),
            "hn_activation": conv4["params"]["activation"],
            "original_names": originals,
            "consumer": conv4["output"],
        },
        "quantized_model": {
            "state": "quantized_model",
            "hn_sha256": source_hn_sha,
            "optimized_params_sha256": source_params_sha,
        },
        "formal_hef_static": static,
        "golden": _validate_golden(),
        "parser_matrix": parser_matrix(),
        "compile_plan": {
            "baseline": "immutable formal spatial-8 HEF; no compile",
            "scheduled_new_arm": SCHEDULED_ARM,
            "scheduled_script": CANDIDATES[SCHEDULED_ARM].script,
            "new_compile_count": 1,
            "board_hef_count": 2,
        },
    }


def prepare(root: Path, cuda_visible_devices: str) -> dict[str, Any]:
    refuse_existing(root)
    root.mkdir(parents=True)
    (root / "logs").mkdir()
    (root / "model_scripts").mkdir()
    validation = validate_sources()
    atomic_write(root / "source_validation.json", validation)
    script_path = root / "model_scripts" / f"{SCHEDULED_ARM}.alls"
    script_path.write_text(CANDIDATES[SCHEDULED_ARM].script, encoding="utf-8")
    report = {
        "kind": "bn_g_s_conv4_defuse_ab_precompile",
        "status": "WAITING_FOR_COMPILE_APPROVAL",
        "created_unix_seconds": time.time(),
        "out_root": str(root),
        "source_validation": {
            "path": str(root / "source_validation.json"),
            "sha256": sha256(root / "source_validation.json"),
        },
        "scheduled_arm": {
            "name": SCHEDULED_ARM,
            "script": CANDIDATES[SCHEDULED_ARM].script,
            "script_path": str(script_path),
            "script_sha256": sha256(script_path),
        },
        "commands_after_approval": {
            "compile": shlex.join([
                sys.executable, str(Path(__file__).relative_to(ROOT)),
                "--phase", "compile", "--out-root", str(root),
                "--cuda-visible-devices", cuda_visible_devices, "--allow-compile",
            ]),
            "board": shlex.join([
                sys.executable, str(Path(__file__).relative_to(ROOT)),
                "--phase", "board", "--out-root", str(root), "--allow-board",
            ]),
        },
        "risk_register": [
            "ALLOCATION_ONLY parser PASS does not prove allocator or hardware feasibility",
            "a targeted conv4 constraint can still cause global placement/context changes",
            "generated auto.alls must prove realized conv4 spatial-4 after compile",
            "final output dtype, shape and quantization must match the formal HEF",
            "eight samples are diagnostic only; no semantic ranking or acceptance",
        ],
        "policy": {
            "compile_started": False,
            "board_started": False,
            "parse_called": False,
            "optimize_called": False,
            "new_compile_count": 1,
        },
    }
    atomic_write(root / "precompile_report.json", report)
    atomic_write(root / "campaign_state.json", {
        "kind": "bn_g_s_conv4_defuse_ab_state",
        "status": "WAITING_FOR_COMPILE_APPROVAL",
        "stage": "prepared",
        "scheduled_arm": SCHEDULED_ARM,
    })
    return report


def _relocate_paths(value: Any, source: Path, destination: Path) -> Any:
    """Rewrite staged absolute paths before the directory is atomically renamed."""

    if isinstance(value, dict):
        return {
            key: _relocate_paths(item, source, destination)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_relocate_paths(item, source, destination) for item in value]
    if isinstance(value, str):
        return value.replace(str(source), str(destination))
    return value


def compile_arm(root: Path, cuda_visible_devices: str, allowed: bool) -> dict[str, Any]:
    if not allowed:
        raise PermissionError("compile phase requires explicit --allow-compile")
    pre_path = root / "precompile_report.json"
    if not pre_path.is_file():
        raise FileNotFoundError("prepare phase has not completed")
    pre = json.loads(pre_path.read_text(encoding="utf-8"))
    if pre.get("status") != "WAITING_FOR_COMPILE_APPROVAL":
        raise ValueError(f"unexpected precompile state: {pre.get('status')}")
    script_path = root / "model_scripts" / f"{SCHEDULED_ARM}.alls"
    script = script_path.read_text(encoding="utf-8")
    if script != CANDIDATES[SCHEDULED_ARM].script:
        raise ValueError("scheduled compiler script changed after prepare")
    validation = validate_sources()

    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    os.environ["HAILO_SDK_LOG_DIR"] = str(root / "logs")
    destination = root / "compile" / SCHEDULED_ARM
    refuse_existing(destination)
    temporary = stage_dir(destination)
    from hailo_sdk_client import ClientRunner

    try:
        runner = ClientRunner(har=str(FORMAL_QHAR))
        guard = load_compile_only_script(runner, script)
        (temporary / "compiler_override.alls").write_text(script, encoding="utf-8")
        qhar_before = sha256(FORMAL_QHAR)
        compiled = compile_quantized(
            runner,
            hef_path=temporary / "bn_g_s.hef",
            compiled_har=temporary / "bn_g_s_compiled.har",
            auto_alls=temporary / "bn_g_s.auto.alls",
        )
        del runner
        qhar_after = sha256(FORMAL_QHAR)
        if qhar_before != FORMAL_QHAR_SHA or qhar_after != FORMAL_QHAR_SHA:
            raise ValueError("immutable source qHAR changed during compile")
        auto_path = temporary / "bn_g_s.auto.alls"
        if not auto_path.is_file():
            raise ValueError(f"compiler did not emit auto.alls: {compiled.get('auto_alls')}")
        realized = allocation_facts(auto_path)
        realized_gate = validate_realized_topology(realized)

        candidate_static = hef_static(temporary / "bn_g_s.hef")
        candidate_outputs = _map_static_outputs(
            candidate_static["outputs"], include_probe=False
        )
        formal_outputs = _map_static_outputs(
            validation["formal_hef_static"]["outputs"], include_probe=False
        )
        comparisons = {}
        for logical in ("x_h_hat", "x_l_hat"):
            same = _same_quantization(formal_outputs[logical], candidate_outputs[logical])
            comparisons[logical] = {
                "status": "PASS" if same else "MISMATCH",
                "formal": formal_outputs[logical],
                "candidate": candidate_outputs[logical],
            }
        if any(row["status"] != "PASS" for row in comparisons.values()):
            raise ValueError("spatial-4 changed a final output quantization contract")

        reopened = ClientRunner(har=str(FORMAL_QHAR))
        source_after = {
            "hn_sha256": hn_sha256(reopened),
            "optimized_params_sha256": optimized_params_sha256(reopened),
        }
        del reopened
        if source_after != {
            "hn_sha256": EXPECTED_HN_SHA,
            "optimized_params_sha256": EXPECTED_PARAMS_SHA,
        }:
            raise ValueError("source qHAR numerical state failed postcompile reopen guard")
        report = {
            "kind": "bn_g_s_conv4_defuse_ab_compile",
            "status": "PASS",
            "arm": SCHEDULED_ARM,
            "source_qhar_sha256_before": qhar_before,
            "source_qhar_sha256_after": qhar_after,
            "compile_only_numerical_guard": guard,
            "source_reopen_guard": source_after,
            "compile": compiled,
            "formal_allocation": allocation_facts(FORMAL_AUTO_ALLS),
            "candidate_allocation": realized,
            "realized_topology_gate": realized_gate,
            "hef_static": candidate_static,
            "output_contract_comparison": comparisons,
            "policy": {
                "parse_called": False,
                "optimize_called": False,
                "source_qhar_immutable": True,
            },
        }
        atomic_write(
            temporary / "compile_result.json",
            _relocate_paths(report, temporary, destination),
        )
        committed = commit_or_fail(temporary, destination, True)
    except Exception:
        commit_or_fail(temporary, destination, False)
        raise

    candidate_hef = committed / "bn_g_s.hef"
    post = {
        "kind": "bn_g_s_conv4_defuse_ab_postcompile",
        "status": "WAITING_FOR_BOARD_APPROVAL",
        "arm": SCHEDULED_ARM,
        "candidate_hef": {"path": str(candidate_hef), "sha256": sha256(candidate_hef)},
        "compile_result": str(committed / "compile_result.json"),
        "next_command": shlex.join([
            sys.executable, str(Path(__file__).relative_to(ROOT)),
            "--phase", "board", "--out-root", str(root), "--allow-board",
        ]),
    }
    atomic_write(root / "postcompile_report.json", post)
    atomic_write(root / "campaign_state.json", {
        "kind": "bn_g_s_conv4_defuse_ab_state",
        "status": "WAITING_FOR_BOARD_APPROVAL",
        "stage": "compiled",
        "scheduled_arm": SCHEDULED_ARM,
    })
    return post


def board(root: Path, allowed: bool) -> dict[str, Any]:
    if not allowed:
        raise PermissionError("board phase requires explicit --allow-board")
    post_path = root / "postcompile_report.json"
    if not post_path.is_file():
        raise FileNotFoundError("compile phase has not completed")
    post = json.loads(post_path.read_text(encoding="utf-8"))
    if post.get("status") != "WAITING_FOR_BOARD_APPROVAL":
        raise ValueError(f"unexpected postcompile state: {post.get('status')}")
    candidate_hef = root / "compile" / SCHEDULED_ARM / "bn_g_s.hef"
    if sha256(candidate_hef) != post["candidate_hef"]["sha256"]:
        raise ValueError("candidate HEF identity changed")
    if sha256(FORMAL_HEF) != FORMAL_HEF_SHA or sha256(FORMAL_QHAR) != FORMAL_QHAR_SHA:
        raise ValueError("immutable formal source identity changed")

    arms = {
        "formal_spatial8": {
            "hef": FORMAL_HEF, "hef_sha256": FORMAL_HEF_SHA,
            "har": FORMAL_QHAR, "har_sha256": FORMAL_QHAR_SHA,
            "role": "immutable formal spatial-8 baseline",
        },
        SCHEDULED_ARM: {
            "hef": candidate_hef, "hef_sha256": sha256(candidate_hef),
            "har": FORMAL_QHAR, "har_sha256": FORMAL_QHAR_SHA,
            "role": "compile-only targeted conv4 spatial-4",
        },
    }
    parity.ARMS = arms
    atomic_write(root / "campaign_state.json", {
        "kind": "bn_g_s_conv4_defuse_ab_state",
        "status": "RUNNING", "stage": "host_sdk8",
    })
    references_path, host_report = parity.make_host_references(root)

    prelaunch = ssh_readonly(
        "pgrep -af '[e]valuate_bn_g_s|[r]un_bn_g_s' || true; "
        "vcgencmd measure_temp 2>/dev/null || true"
    )
    if "evaluate_bn_g_s" in prelaunch or "run_bn_g_s" in prelaunch:
        raise RuntimeError(f"board has an active bn_g_s campaign:\n{prelaunch}")
    remote_root = f".local/share/semantic-sensing-board/hybrid-ort/{root.name}"
    if ssh_readonly(
        f"test ! -e {shlex.quote(remote_root)} && echo ABSENT || echo PRESENT"
    ).strip() != "ABSENT":
        raise FileExistsError(f"remote result root exists: {remote_root}")
    ssh(f"mkdir -p {shlex.quote(remote_root + '/hef')} {shlex.quote(remote_root + '/onnx')}")
    options = ssh_options()
    evaluator = ROOT / "board_eval/evaluate_bn_g_s_vstream_parity.py"
    dependencies = (
        ROOT / "board_eval/bn_g_s_branch_attribution_runtime.py",
        ROOT / "board_eval/hybrid_ort_runtime_runner.py",
        ROOT / "board_eval/fullmodule_runtime_runner.py",
    )
    for source in (evaluator, *dependencies):
        stream_upload(source, f"{remote_root}/{source.name}", options, SSH_TARGET)
    stream_upload(ONNX, f"{remote_root}/onnx/bn_g_s.onnx", options, SSH_TARGET)
    stream_upload(references_path, f"{remote_root}/{references_path.name}", options, SSH_TARGET)
    manifest_arms = []
    for name, spec in arms.items():
        remote_hef = f"hef/{name}.hef"
        stream_upload(spec["hef"], f"{remote_root}/{remote_hef}", options, SSH_TARGET)
        manifest_arms.append({
            "name": name, "role": spec["role"], "hef_filename": remote_hef,
            "hef_sha256": spec["hef_sha256"], "har_sha256": FORMAL_QHAR_SHA,
        })
    manifest = {
        "kind": "bn_g_s_conv4_defuse_ab_manifest",
        "checkpoint_sha256": CHECKPOINT_SHA,
        "sample_indices": list(SAMPLE_INDICES),
        "ort_threads": 4,
        "onnx": {"filename": "onnx/bn_g_s.onnx", "sha256": ONNX_SHA},
        "golden_bundle": {"result_sha256": GOLDEN_RESULT_SHA},
        "references": {
            "filename": references_path.name,
            "sha256": host_report["bundle"]["sha256"],
            "arrays": host_report["bundle"]["arrays"],
        },
        "arms": manifest_arms,
        "policy": {
            "diagnostic_only": True, "full976_not_run": True,
            "semantic_ranking": False, "cpu_modules": CPU_MODULES,
            "source_quantized_har_sha256": FORMAL_QHAR_SHA,
        },
    }
    manifest_path = root / "board_manifest.json"
    atomic_write(manifest_path, manifest)
    stream_upload(manifest_path, f"{remote_root}/board_manifest.json", options, SSH_TARGET)
    remote_golden_sha = ssh_readonly(
        f"sha256sum {shlex.quote(REMOTE_GOLDEN + '/result.json')} | cut -d' ' -f1"
    ).strip()
    if remote_golden_sha != GOLDEN_RESULT_SHA:
        raise ValueError("remote golden bundle identity mismatch")

    result_remote = f"{remote_root}/result.json"
    raw_remote = f"{remote_root}/raw_outputs.npz"
    job = launch_remote(
        remote_root,
        "board_conv4_defuse_ab_attempt0",
        [
            REMOTE_PYTHON, f"{remote_root}/{evaluator.name}",
            "--manifest", f"{remote_root}/board_manifest.json",
            "--bundle-root", REMOTE_GOLDEN,
            "--out", result_remote, "--raw-out", raw_remote,
        ],
        result_override=result_remote,
        progress_override=f"{remote_root}/result.progress.json",
    )
    state = {
        "kind": "bn_g_s_conv4_defuse_ab_state", "status": "RUNNING",
        "stage": "board8", "remote_job": job, "board_prelaunch": prelaunch,
    }
    atomic_write(root / "campaign_state.json", state)
    poll_remote(root, state, job, timeout=1800)
    result_path = root / "board_result.json"
    raw_path = root / "raw_outputs.npz"
    parity.remote_download(result_remote, result_path)
    parity.remote_download(raw_remote, raw_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "PASS" or result.get("sample_count") != len(SAMPLE_INDICES):
        raise RuntimeError("board A/B result is incomplete")
    if sha256(raw_path) != result["raw_outputs"]["sha256"]:
        raise ValueError("downloaded raw output archive hash mismatch")
    per_arm = {
        name: {
            branch: arm["comparisons"]["board_float_vs_sdk"][branch]["aggregate_nmse"]
            for branch in ("x_h_hat", "x_l_hat")
        }
        for name, arm in result["arms"].items()
    }
    baseline = per_arm["formal_spatial8"]
    ratio = {
        branch: per_arm[SCHEDULED_ARM][branch] / baseline[branch]
        for branch in ("x_h_hat", "x_l_hat")
    }
    summary = {
        "kind": "bn_g_s_conv4_defuse_ab_summary",
        "status": "PASS",
        "board_vs_sdk_aggregate_nmse": per_arm,
        "spatial4_to_spatial8_nmse_ratio": ratio,
        "interpretation": (
            "SPATIAL_DEFUSE_COUNT_CAUSAL"
            if ratio["x_h_hat"] < 0.7
            else "SPATIAL4_DID_NOT_MATERIALLY_REDUCE_XH_GAP"
        ),
        "artifacts": {
            "board_result": {"path": str(result_path), "sha256": sha256(result_path)},
            "raw_outputs": {"path": str(raw_path), "sha256": sha256(raw_path)},
        },
        "policy": {
            "diagnostic_only": True, "full976_not_run": True,
            "semantic_ranking": False, "bn_g_s_only": True,
        },
    }
    atomic_write(root / "summary.json", summary)
    atomic_write(root / "campaign_state.json", {
        "kind": "bn_g_s_conv4_defuse_ab_state", "status": "PASS",
        "stage": "complete", "interpretation": summary["interpretation"],
    })
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "compile", "board"), default="prepare")
    parser.add_argument("--out-root", type=Path)
    parser.add_argument("--cuda-visible-devices", default="2")
    parser.add_argument("--allow-compile", action="store_true")
    parser.add_argument("--allow-board", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        print(json.dumps(validate_sources(), indent=2, sort_keys=True))
        return 0
    if args.out_root is None:
        parser.error("--out-root is required unless --validate-only is used")
    root = args.out_root.resolve()
    if args.phase == "prepare":
        report = prepare(root, args.cuda_visible_devices)
    elif args.phase == "compile":
        report = compile_arm(root, args.cuda_visible_devices, args.allow_compile)
    else:
        report = board(root, args.allow_board)
    print(json.dumps({"phase": args.phase, "status": report["status"], "out_root": str(root)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
