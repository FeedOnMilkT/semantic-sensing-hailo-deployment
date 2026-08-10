#!/usr/bin/env python3
"""Export receiver stage4/stage5 separately and verify their cascade."""
from __future__ import annotations

import argparse
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


class ReceiverStage(nn.Module):
    def __init__(self, stage: nn.Module):
        super().__init__()
        self.stage = stage

    def forward(self, high, low):
        return self.stage(high, low)


def _export(module, inputs, path, input_names, output_names, opset):
    with torch.no_grad():
        reference = module(*inputs)
    torch.onnx.export(
        module, inputs, str(path), input_names=input_names,
        output_names=output_names, opset_version=opset,
        do_constant_folding=True, dynamic_axes=None,
    )
    import onnx
    model = onnx.load(path)
    onnx.checker.check_model(model)
    simplified = False
    try:
        from onnxsim import simplify
        model, ok = simplify(model)
        if ok:
            onnx.save(model, path)
            simplified = True
    except ImportError:
        pass
    return reference, model, simplified


def _ort(path, feed):
    import onnxruntime as ort
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return session.run(None, feed)


def _entry(name, path, model, inputs, outputs, risk_note, simplified, max_abs):
    op_types = [node.op_type for node in model.graph.node]
    risky_names = {"MatMul", "Softmax", "Resize"}
    return {
        "name": name,
        "onnx": str(path),
        "nodes": len(model.graph.node),
        "size_kb": round(path.stat().st_size / 1024, 1),
        "in_shapes": [list(x.shape) for x in inputs],
        "out_shapes": [list(x.shape) for x in outputs],
        "ops": sorted(set(op_types)),
        "risky_ops": {op: op_types.count(op) for op in sorted(set(op_types) & risky_names)},
        "gdn_ops": {},
        "hailo_risk": "medium",
        "risk_note": risk_note,
        "ort_max_abs": max_abs,
        "ort_ok": max_abs < 1e-3,
        "simplified": simplified,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "/local/shared_with_docker/ckpt/midf_b0.01_s2.pth"))
    parser.add_argument("--out-dir", type=Path, default=Path(
        "/local/shared_with_docker/onnx/modules"))
    parser.add_argument("--opset", type=int, default=13)
    args = parser.parse_args()

    torch.manual_seed(0)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint["cfg"]
    cfg.model.backbone.args.pretrained = None
    teacher = build_model(cfg.model).eval()
    teacher_state = {
        key.removeprefix("teacher."): value
        for key, value in checkpoint["student"].items()
        if key.startswith("teacher.")
    }
    teacher.load_state_dict(teacher_state, strict=True)
    backbone = teacher.backbone

    args.out_dir.mkdir(parents=True, exist_ok=True)
    high0 = torch.randn(1, 128, 64, 64)
    low0 = torch.randn(1, 256, 32, 32)
    stage4 = ReceiverStage(backbone.stage4).eval()
    stage5 = ReceiverStage(backbone.stage5).eval()
    path4 = args.out_dir / "receiver_stage4.onnx"
    path5 = args.out_dir / "receiver_stage5.onnx"

    ref4, model4, sim4 = _export(
        stage4, (high0, low0), path4, ["x_h", "x_l"],
        ["stage4_high", "stage4_low"], args.opset)
    ref5, model5, sim5 = _export(
        stage5, ref4, path5, ["stage4_high", "stage4_low"],
        ["high_feat", "low_feat"], args.opset)

    ort4 = _ort(path4, {"x_h": high0.numpy(), "x_l": low0.numpy()})
    ort5 = _ort(path5, {"stage4_high": ort4[0], "stage4_low": ort4[1]})
    stage4_max = max(float(np.max(np.abs(a.detach().numpy() - b)))
                     for a, b in zip(ref4, ort4))
    cascade_max = max(float(np.max(np.abs(a.detach().numpy() - b)))
                      for a, b in zip(ref5, ort5))

    manifest_path = args.out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    modules = [item for item in manifest["modules"]
               if item["name"] not in {"receiver_stage45", "receiver_stage4", "receiver_stage5"}]
    modules.extend([
        _entry("receiver_stage4", path4, model4, (high0, low0), ref4,
               "x_h/x_l -> stage4 pair; feeds receiver_stage5", sim4, stage4_max),
        _entry("receiver_stage5", path5, model5, ref4, ref5,
               "stage4 pair -> pre-DAPPM pair", sim5, cascade_max),
    ])
    order = ["sender_backbone", "bn_align_h", "bn_g_a", "bn_h_a", "bn_h_s",
             "bn_g_s", "receiver_stage4", "receiver_stage5", "receiver_dappm",
             "receiver_head"]
    manifest["modules"] = sorted(modules, key=lambda item: order.index(item["name"]))
    manifest["ckpt"] = str(args.checkpoint)
    manifest["receiver_cascade"] = {
        "order": ["receiver_stage4", "receiver_stage5", "receiver_dappm"],
        "stage4_ort_max_abs": stage4_max,
        "stage45_cascade_ort_max_abs": cascade_max,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(json.dumps(manifest["receiver_cascade"], indent=2))
    if stage4_max >= 1e-3 or cascade_max >= 1e-3:
        raise SystemExit("cascade parity failed")


if __name__ == "__main__":
    main()
