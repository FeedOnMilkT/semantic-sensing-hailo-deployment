#!/usr/bin/env python3
"""Create the immutable Phase-2 OC4 hybrid-runtime freeze bundle.

This is an archival operation only: it copies already-produced artifacts, validates
their identities, writes an inventory, and refuses to overwrite an existing bundle.
It does not parse, optimize, compile, run inference, or contact the board.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/local/shared_with_docker/deplyment-demo/artifacts/fused")
HANDOFF = Path(
    "/local/shared_with_docker/hailo_full_candidate/"
    "convbn-gfa-gs-3a3ea630af6d2d547808137be678bd909f9789f5594b7bdc76c36b90"
)
GRAPH = ROOT / "experiments/bn-gs-conv4-graph-equivalence-20260806T070529Z"
FORMAL = ROOT / "experiments/bn-gs-equivalent-rewrites-formal-20260806T090000Z"
CONT = ROOT / "experiments/bn-gs-equivalent-rewrites-formal-continuation-20260806T093058Z"
THERMAL = ROOT / "experiments/bn-gs-equivalent-rewrites-thermal-continuation-20260806T103815Z"
BASELINE = ROOT / "board-hybrid-ort-formal-20260805T030725Z"
HOST_FP32 = ROOT / "experiments/host-fp32-onnx-high-support-20260806T124957Z"
CLIP = ROOT / "experiments/bn-gs-input-clip-range-sweep-retry-20260806T025328Z"
CONV4 = ROOT / "experiments/bn-gs-compiler-conv4-bisection-20260806T055105Z"
BRANCH = ROOT / "experiments/bn-gs-current-topology-20260805T075923Z"
WORKSPACE = Path("/workspace")

CHECKPOINT_SHA = "13dc6c4b3a3ea630af6d2d547808137be678bd909f9789f5594b7bdc76c36b90"
OC4_ONNX_SHA = "4ddbf7d4a3aa599ad8e1fe297db36f280265476c52b60131ad513f9ad949b48e"
OC4_PARSE_HAR_SHA = "18765b7eeaa1b094887cf46285680c753e6f55a13b596840507a44bfa2a28995"
OC4_QHAR_SHA = "a5fae52e28ab74d7b17888796dba81bd78dd357cd483f3ffc4cdd4e4611f16ad"
OC4_HEF_SHA = "aa800804a2f5821648bfb15aac54d406a29232503ec5e9b19f86609c726eff9a"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        type=Path,
        default=ROOT / "current-ort-oc4-mainline-20260806",
    )
    parser.add_argument(
        "--board-stage",
        type=Path,
        default=ROOT / ".oc4-freeze-board-stage-20260806",
        help="Read-only download of the five unchanged base HEFs from the evaluated Pi bundle.",
    )
    args = parser.parse_args()
    target = args.target.resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite frozen bundle: {target}")
    stage = target.with_name(f".{target.name}.building-{os.getpid()}")
    stage.mkdir(parents=True, exist_ok=False)

    sources: list[tuple[Path, str, str | None, str]] = [
        (
            HANDOFF / "checkpoint/midf_b0.01_convbn_s2_gfa-gs-v1.pth",
            "model/checkpoint.pth",
            CHECKPOINT_SHA,
            "trained Phase-2 checkpoint",
        ),
        (HANDOFF / "configs/phase2_mid_fused_stage1_convbn.yaml", "model/phase2_stage1.yaml", None, "resolved training config"),
        (HANDOFF / "configs/phase2_mid_fused_stage2_convbn_gfa_groupsoftmax.yaml", "model/phase2_stage2.yaml", None, "resolved training config"),
        (HANDOFF / "onnx/manifest.json", "model/source_onnx_manifest.json", None, "source handoff manifest"),
        (GRAPH / "onnx/output_channel_split_4/bn_g_s.onnx", "oc4/bn_g_s_oc4.onnx", OC4_ONNX_SHA, "exact OC4 FP32 graph"),
        (GRAPH / "har/output_channel_split_4.har", "oc4/bn_g_s_oc4_parsed.har", OC4_PARSE_HAR_SHA, "parsed OC4 HAR"),
        (GRAPH / "ptq/output_channel_split_4_l2/bn_g_s_l2_quantized.har", "oc4/bn_g_s_oc4_quantized.har", OC4_QHAR_SHA, "L2 PTQ OC4 HAR"),
        (GRAPH / "ptq/output_channel_split_4_l2/bn_g_s_l2.model_script.alls", "oc4/ptq_model_script.alls", None, "PTQ model script"),
        (GRAPH / "ptq/output_channel_split_4_l2/result.json", "oc4/ptq_result.json", None, "PTQ report"),
        (FORMAL / "compile/output_channel_split_4/bn_g_s_compiled.har", "oc4/bn_g_s_oc4_compiled.har", None, "compiled OC4 HAR"),
        (FORMAL / "compile/output_channel_split_4/bn_g_s.auto.alls", "oc4/compile_model_script.alls", None, "compiler-generated model script"),
        (FORMAL / "compile/output_channel_split_4/bn_g_s.hef", "runtime/hef/bn_g_s.hef", OC4_HEF_SHA, "selected OC4 HEF"),
        (FORMAL / "compile_results.json", "oc4/compile_results.json", None, "same-policy compile comparison"),
        (THERMAL / "runtime_manifests/runtime_manifest_output_channel_split_4.json", "runtime/runtime_manifest_oc4.json", None, "evaluated runtime manifest"),
        (BASELINE / "runtime_manifests/runtime_manifest_gs_only_hailo_fp32.json", "runtime/runtime_manifest_pre_oc4.json", None, "pre-OC4 runtime lineage"),
        (HANDOFF / "onnx/bn_align_h.onnx", "runtime/onnx/bn_align_h.onnx", None, "ORT FP32 module"),
        (HANDOFF / "onnx/bn_g_a.onnx", "runtime/onnx/bn_g_a.onnx", None, "ORT FP32 module"),
        (HANDOFF / "onnx/bn_h_a.onnx", "runtime/onnx/bn_h_a.onnx", None, "ORT FP32 module"),
        (HANDOFF / "onnx/bn_h_s.onnx", "runtime/onnx/bn_h_s.onnx", None, "ORT FP32 module"),
        (HANDOFF / "diagnostics/codec_runtime_20260723T1209Z/codec_runtime.json", "runtime/codec_runtime.json", None, "CPU entropy metadata"),
        (HANDOFF / "diagnostics/codec_runtime_20260723T1209Z/codec_tables.npz", "runtime/codec_tables.npz", None, "CPU entropy tables"),
        (args.board_stage / "hef/sender_backbone.hef", "runtime/hef/sender_backbone.hef", None, "unchanged Hailo module recovered from evaluated board bundle"),
        (args.board_stage / "hef/receiver_stage4.hef", "runtime/hef/receiver_stage4.hef", None, "unchanged Hailo module recovered from evaluated board bundle"),
        (args.board_stage / "hef/receiver_stage5.hef", "runtime/hef/receiver_stage5.hef", None, "unchanged Hailo module recovered from evaluated board bundle"),
        (args.board_stage / "hef/receiver_dappm.hef", "runtime/hef/receiver_dappm.hef", None, "unchanged Hailo module recovered from evaluated board bundle"),
        (args.board_stage / "hef/receiver_head.hef", "runtime/hef/receiver_head.hef", None, "unchanged Hailo module recovered from evaluated board bundle"),
        (THERMAL / "formal_results/output_channel_split_4/accuracy976.json", "evaluation/oc4_accuracy976.json", None, "formal OC4 full976 result"),
        (THERMAL / "formal_results/output_channel_split_4/preflight1.json", "evaluation/oc4_preflight1.json", None, "board preflight"),
        (THERMAL / "formal_results/spatial_height_split_2/accuracy976.json", "evaluation/ht2_accuracy976.json", None, "formal alternative full976 result"),
        (THERMAL / "summary.json", "evaluation/formal_comparison_summary.json", None, "formal comparison summary"),
        (BASELINE / "gs_only_hailo_fp32/accuracy976.json", "evaluation/pre_oc4_accuracy976.json", None, "formal pre-OC4 baseline"),
        (HOST_FP32 / "accuracy976.json", "evaluation/fp32_onnx_accuracy976.json", None, "host FP32 ONNX plus real-rANS reference"),
        (CONT / "golden8_summary.json", "diagnostics/golden8_equivalent_rewrites.json", None, "diagnostic only"),
        (GRAPH / "report.json", "diagnostics/oc4_graph_equivalence.json", None, "ONNX/ORT/NATIVE/parse equivalence evidence"),
        (CLIP / "summary.json", "diagnostics/input_clip_sweep32.json", None, "diagnostic only"),
        (CONV4 / "conv4_same_input_causal_report.json", "diagnostics/conv4_same_input8.json", None, "diagnostic only"),
        (BRANCH / "results/branch_attribution.json", "diagnostics/branch_attribution256.json", None, "diagnostic only"),
        (WORKSPACE / "semantic-sensing/notes/phase2_hailo_ort_runtime_policy_2026-08-05.md", "policy/runtime_policy_20260805.md", None, "runtime policy"),
        (WORKSPACE / "demo/deplyment-demo/hailo_deploy/semantic_acceptance_large_object.json", "policy/semantic_acceptance.json", None, "semantic evaluation policy"),
        (ROOT / "current-ort-mainline-20260805/manifest.json", "policy/prior_mainline_manifest.json", None, "pre-OC4 canonical manifest"),
        (WORKSPACE / "demo/deplyment-demo/pipelines/run_bn_g_s_conv4_graph_equivalence.py", "source/run_bn_g_s_conv4_graph_equivalence.py", None, "OC4 graph rewrite implementation"),
        (WORKSPACE / "demo/deplyment-demo/pipelines/run_bn_g_s_equivalent_rewrites_formal.py", "source/run_bn_g_s_equivalent_rewrites_formal.py", None, "formal compile/evaluation orchestrator"),
        (WORKSPACE / "demo/deplyment-demo/pipelines/continue_bn_g_s_equivalent_rewrites_thermal.py", "source/continue_bn_g_s_equivalent_rewrites_thermal.py", None, "thermal continuation orchestrator"),
        (WORKSPACE / "demo/deplyment-demo/board_eval/hybrid_ort_runtime_runner.py", "source/hybrid_ort_runtime_runner.py", None, "hybrid runtime implementation"),
        (WORKSPACE / "semantic-sensing/src/models/bottlenecks/rtformer_fused.py", "source/rtformer_fused.py", None, "bottleneck implementation snapshot"),
        (WORKSPACE / "semantic-sensing/src/models/student.py", "source/student.py", None, "split topology implementation snapshot"),
        (WORKSPACE / "semantic-sensing/tools/export_modules_fused.py", "source/export_modules_fused.py", None, "module export implementation snapshot"),
    ]

    runtime_manifest = load_json(THERMAL / "runtime_manifests/runtime_manifest_output_channel_split_4.json")
    runtime_expected = {}
    for module in runtime_manifest["modules"]:
        if "hef_sha256" in module:
            runtime_expected[f"runtime/hef/{module['name']}.hef"] = module["hef_sha256"]
        if "onnx_sha256" in module:
            runtime_expected[f"runtime/onnx/{module['name']}.onnx"] = module["onnx_sha256"]
    runtime_expected["runtime/codec_runtime.json"] = runtime_manifest["codec"]["metadata_sha256"]
    runtime_expected["runtime/codec_tables.npz"] = runtime_manifest["codec"]["tables_sha256"]

    inventory = {}
    for source, relative, expected, role in sources:
        if not source.is_file():
            raise FileNotFoundError(source)
        actual = sha256(source)
        required = expected or runtime_expected.get(relative)
        if required is not None and actual != required:
            raise ValueError(f"SHA mismatch for {source}: {actual} != {required}")
        destination = stage / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied_sha = sha256(destination)
        if copied_sha != actual:
            raise IOError(f"copy verification failed: {destination}")
        inventory[relative] = {
            "bytes": destination.stat().st_size,
            "role": role,
            "sha256": copied_sha,
            "source": str(source),
        }

    oc4_result = load_json(stage / "evaluation/oc4_accuracy976.json")
    fp32_result = load_json(stage / "evaluation/fp32_onnx_accuracy976.json")
    baseline_result = load_json(stage / "evaluation/pre_oc4_accuracy976.json")
    fixed19 = oc4_result["semantic"]["fixed_19_large_object_macro_iou"]
    high9_indices = [7, 10, 15, 20, 23, 30, 37, 83, 102]
    oc4_high9 = sum(oc4_result["semantic"]["per_class_iou"][i] for i in high9_indices) / len(high9_indices)
    baseline_high9 = sum(baseline_result["semantic"]["per_class_iou"][i] for i in high9_indices) / len(high9_indices)
    manifest = {
        "schema": "semantic_sensing.current_ort_oc4_mainline.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FROZEN",
        "freeze_semantics": {
            "weights_and_artifacts": "immutable Phase-2 reference",
            "oc4_recipe": "carry forward to new checkpoints; re-export/PTQ/compile required",
            "old_hef_reuse_for_retrained_weights": "PROHIBITED",
        },
        "checkpoint_sha256": CHECKPOINT_SHA,
        "selected_bn_g_s": {
            "name": "OC4",
            "definition": "exact Conv4 output-channel split: Conv(384->128,3x3,s1,p1) becomes four Conv(384->32,3x3,s1,p1) plus channel Concat before the original ReLU",
            "mathematical_function_change": False,
            "ptq": "optimization_level_2",
            "calibration_samples": 1024,
            "hef_sha256": OC4_HEF_SHA,
            "outputs": {"x_h": "UINT8 128x64x64", "x_l": "UINT16 256x32x32"},
            "num_contexts": 1,
        },
        "runtime_boundary": {
            "hailo": ["sender_backbone", "bn_g_s", "receiver_stage4", "receiver_stage5", "receiver_dappm", "receiver_head"],
            "ort_cpu_fp32_default": ["bn_align_h", "bn_g_a", "bn_h_a", "bn_h_s"],
            "cpu_codec": ["rounding", "CDF/index construction", "rANS", "RTFF byte container"],
        },
        "formal_full976": {
            "sample_count": 976,
            "val_indices": [1024, 1999],
            "oc4_fixed_19_large_object_macro_iou": fixed19,
            "oc4_high_support_9_macro_iou": oc4_high9,
            "oc4_mean_bpp": oc4_result["bitstream"]["mean_bpp"],
            "pre_oc4_fixed19": baseline_result["semantic"]["fixed_19_large_object_macro_iou"],
            "pre_oc4_high9": baseline_high9,
            "fp32_fixed19_reproduction": fp32_result["semantic"]["fixed_19_large_object_macro_iou"],
            "fp32_high9": fp32_result["semantic"]["high_support_9_macro_iou"],
            "selection_metric": "high_support_9_macro_iou",
            "ade20k_150_class_miou": "NOT_USED",
        },
        "diagnostic_policy": "Results with fewer than 976 samples are localization/correctness evidence only.",
        "inventory": inventory,
    }
    manifest_path = stage / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    checksums = []
    for path in sorted(p for p in stage.rglob("*") if p.is_file() and p.name != "SHA256SUMS"):
        checksums.append(f"{sha256(path)}  {path.relative_to(stage)}")
    (stage / "SHA256SUMS").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    readme = f"""# Phase-2 OC4 frozen mainline

Status: **FROZEN**. This bundle is a byte-verified archive of the selected Phase-2
hybrid ORT/Hailo-8 runtime and its evidence. It was assembled without training,
quantization, compilation, or inference.

- Selected `bn_g_s`: OC4, exact output-channel split of Conv4 into four 384-to-32
  convolutions followed by `Concat`; FP32 graph behavior is unchanged.
- Formal held-out evaluation: 976 ADE20K validation samples, indices 1024..1999.
- OC4 results: fixed-19 `{fixed19:.16f}`, high-support-9 `{oc4_high9:.16f}`, mean bpp
  `{oc4_result['bitstream']['mean_bpp']:.16f}`.
- Runtime: six Hailo HEFs, four ORT FP32 modules, and CPU entropy/byte-container logic.

`manifest.json` records provenance and all file hashes. `SHA256SUMS` verifies the
physical bundle. The old HEF encodes old weights: future reduced-class or ImageNet-
strengthened checkpoints must repeat export, PTQ, compile, and full-976 validation,
while carrying forward the OC4 graph rewrite and hybrid runtime boundary.
"""
    (stage / "README.md").write_text(readme, encoding="utf-8")
    os.rename(stage, target)
    print(json.dumps({"status": "FROZEN", "target": str(target), "manifest_sha256": sha256(target / "manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
