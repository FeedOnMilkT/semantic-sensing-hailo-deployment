#!/usr/bin/env python3
"""Export a non-equivalent GFA-mode receiver_stage45 feasibility candidate.

This is deliberately deployment-only: it loads only ``teacher.*`` tensors from
the frozen Stage-2 checkpoint, so it does not need CompressAI or a training
environment.  The candidate must still be trained before it is evaluated as a
semantic model.  Its only purpose here is the G0/G1/G2 hardware gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

SEMANTIC_ROOT = Path("/workspace/semantic-sensing")
sys.path.insert(0, str(SEMANTIC_ROOT))

from src import _compat  # noqa: F401,E402
from src.models import build_model  # noqa: E402


class ReceiverStage45(nn.Module):
    """The runtime-correct stage4 -> stage5 receiver boundary."""

    def __init__(self, stage4: nn.Module, stage5: nn.Module):
        super().__init__()
        self.stage4 = stage4
        self.stage5 = stage5

    def forward(self, x_h: torch.Tensor, x_l: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        high, low = self.stage4(x_h, x_l)
        return self.stage5(high, low)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ops(model) -> dict[str, int]:
    values = [node.op_type for node in model.graph.node]
    return {op: values.count(op) for op in sorted(set(values))}


def _matmul_audit(model) -> list[dict]:
    initializers = {value.name for value in model.graph.initializer}
    report = []
    for node in model.graph.node:
        if node.op_type not in {"MatMul", "Gemm"}:
            continue
        report.append({
            "name": node.name,
            "op": node.op_type,
            "inputs": list(node.input),
            "rhs_is_initializer": len(node.input) > 1 and node.input[1] in initializers,
        })
    return report


def _max_abs(reference: tuple[torch.Tensor, ...], actual: list[np.ndarray]) -> float:
    return max(float(np.max(np.abs(expected.detach().cpu().numpy() - received)))
               for expected, received in zip(reference, actual))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "/local/shared_with_docker/ckpt/midf_b0.01_s2.pth"))
    parser.add_argument("--gfa-mode", choices=["group_softmax", "memory_softmax"],
                        required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--opset", type=int, default=13)
    parser.add_argument("--adapted", action="store_true",
                        help="mark this as a receiver-adapted checkpoint, not a zero-training probe")
    parser.add_argument("--acceptance-note", default=None,
                        help="record a human acceptance/waiver without rewriting automated gates")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}")
    if args.out_dir.exists():
        parser.error(f"refusing to overwrite candidate directory: {args.out_dir}")

    torch.manual_seed(0)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint["cfg"]
    cfg.model.backbone.args.pretrained = None
    cfg.model.backbone.args.gfa_mode = args.gfa_mode
    teacher = build_model(cfg.model).eval()
    teacher_state = {
        key.removeprefix("teacher."): value
        for key, value in checkpoint["student"].items()
        if key.startswith("teacher.")
    }
    teacher.load_state_dict(teacher_state, strict=True)

    module = ReceiverStage45(teacher.backbone.stage4, teacher.backbone.stage5).eval()
    x_h = torch.randn(1, 128, 64, 64)
    x_l = torch.randn(1, 256, 32, 32)
    with torch.no_grad():
        reference = module(x_h, x_l)

    args.out_dir.mkdir(parents=True, exist_ok=False)
    onnx_path = args.out_dir / "receiver_stage45.onnx"
    torch.onnx.export(
        module, (x_h, x_l), str(onnx_path),
        input_names=["x_h", "x_l"], output_names=["high_feat", "low_feat"],
        opset_version=args.opset, do_constant_folding=True, dynamic_axes=None,
    )

    import onnx
    import onnxruntime as ort

    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    simplified = False
    try:
        from onnxsim import simplify

        candidate, ok = simplify(model)
        if ok:
            onnx.save(candidate, onnx_path)
            model = candidate
            simplified = True
    except ImportError:
        pass

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_outputs = session.run(None, {"x_h": x_h.numpy(), "x_l": x_l.numpy()})
    ort_max = _max_abs(reference, ort_outputs)
    inputs = [[item.name, [dim.dim_value for dim in item.type.tensor_type.shape.dim]]
              for item in model.graph.input]
    outputs = [[item.name, [dim.dim_value for dim in item.type.tensor_type.shape.dim]]
               for item in model.graph.output]
    manifest = {
        "kind": (
            "trained_gfa_receiver_candidate"
            if args.adapted else "non_equivalent_gfa_hardware_feasibility"
        ),
        "receiver_adapted": args.adapted,
        "acceptance_note": args.acceptance_note,
        "gfa_mode": args.gfa_mode,
        "formula": (
            "reshape(Q@Kg.T,[B,N,8,16]) then softmax over the fixed 16-slot group"
            if args.gfa_mode == "group_softmax"
            else "softmax(Q@Kg.T) over the fixed 128-slot memory axis"
        ),
        "checkpoint": {"path": str(args.checkpoint), "sha256": _sha256(args.checkpoint)},
        "state_load": {"strict": True, "tensor_source": "checkpoint.student teacher.*"},
        "onnx": {
            "path": str(onnx_path), "sha256": _sha256(onnx_path), "opset": args.opset,
            "inputs": inputs, "outputs": outputs, "nodes": len(model.graph.node),
            "ops": _ops(model), "simplified": simplified, "matmul_audit": _matmul_audit(model),
        },
        "validation": {"pytorch_to_ort_max_abs": ort_max, "pass": ort_max < 1e-3},
        "warning": (
            "Human acceptance does not rewrite the automated G3 result. Re-run Parse, "
            "NATIVE, PTQ, full ADE20K QUANTIZED evaluation, Level-2 calibration, and HEF."
            if args.adapted else
            "This checkpoint has not been adapted to the new GFA normalization; "
            "passing this export gate is hardware feasibility only, not task accuracy."
        ),
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if not manifest["validation"]["pass"]:
        raise SystemExit("PyTorch/ORT parity failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
