#!/usr/bin/env python3
"""Export topology-preserving low/high receiver probes for Stage4/5.

The two ONNX models are a diagnostic cut only:

  low_path:  x_l -> low4 -> low5
  high_path: x_h, low4, low5 -> high5

By default they use the checkpoint's original RTFormer modules verbatim.  The
optional ``--gfa-mode`` flag is a non-equivalent deployment candidate: parameters
remain unchanged, but the GFA normalization is selected by the shared RTFormer
implementation.  The exporter proves the cut against the matching complete
stage4 -> stage5 reference before emitting its manifest.
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


class ReceiverLowPath(nn.Module):
    """Both stepped low branches; it has no dependency on the high branch."""

    def __init__(self, stage4: nn.Module, stage5: nn.Module):
        super().__init__()
        self.stage4 = stage4
        self.stage5 = stage5

    @staticmethod
    def _low(stage: nn.Module, low: torch.Tensor) -> torch.Tensor:
        return stage.ffn_low(stage.gfa(stage.low_trans(low)))

    def forward(self, x_l: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        low4 = self._low(self.stage4, x_l)
        low5 = self._low(self.stage5, low4)
        return low4, low5


class ReceiverHighPath(nn.Module):
    """Both stepped high branches, consuming the low-path boundary tensors."""

    def __init__(self, stage4: nn.Module, stage5: nn.Module):
        super().__init__()
        self.stage4 = stage4
        self.stage5 = stage5

    @staticmethod
    def _high(stage: nn.Module, high: torch.Tensor, low: torch.Tensor) -> torch.Tensor:
        return stage.ffn_high(stage.ca(high, low))

    def forward(self, x_h: torch.Tensor, low4: torch.Tensor,
                low5: torch.Tensor) -> torch.Tensor:
        high4 = self._high(self.stage4, x_h, low4)
        return self._high(self.stage5, high4, low5)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _export(module: nn.Module, inputs: tuple[torch.Tensor, ...], path: Path,
            input_names: list[str], output_names: list[str], opset: int) -> tuple:
    with torch.no_grad():
        reference = module(*inputs)
    torch.onnx.export(
        module, inputs, str(path), input_names=input_names, output_names=output_names,
        opset_version=opset, do_constant_folding=True, dynamic_axes=None,
    )
    import onnx

    model = onnx.load(path)
    onnx.checker.check_model(model)
    simplified = False
    try:
        from onnxsim import simplify

        simplified_model, ok = simplify(model)
        if ok:
            onnx.save(simplified_model, path)
            model = simplified_model
            simplified = True
    except ImportError:
        pass
    return reference if isinstance(reference, tuple) else (reference,), model, simplified


def _run_ort(path: Path, feed: dict[str, torch.Tensor]) -> list[np.ndarray]:
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return session.run(None, {name: value.detach().cpu().numpy() for name, value in feed.items()})


def _max_abs(reference: tuple[torch.Tensor, ...], actual: list[np.ndarray]) -> float:
    return max(float(np.max(np.abs(expected.detach().cpu().numpy() - received)))
               for expected, received in zip(reference, actual))


def _ops(model) -> dict[str, int]:
    types = [node.op_type for node in model.graph.node]
    return {item: types.count(item) for item in sorted(set(types))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "/local/shared_with_docker/ckpt/midf_b0.01_s2.pth"))
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="new candidate directory; it must not be a baseline ONNX directory")
    parser.add_argument("--gfa-mode", choices=["original", "group_softmax", "memory_softmax"],
                        default="original")
    parser.add_argument("--opset", type=int, default=13)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error(f"refusing to overwrite non-empty candidate directory: {args.out_dir}")

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
    backbone = teacher.backbone

    x_h = torch.randn(1, 128, 64, 64)
    x_l = torch.randn(1, 256, 32, 32)
    stage4, stage5 = backbone.stage4.eval(), backbone.stage5.eval()
    low_path = ReceiverLowPath(stage4, stage5).eval()
    high_path = ReceiverHighPath(stage4, stage5).eval()

    with torch.no_grad():
        ref4_high, ref4_low = stage4(x_h, x_l)
        ref5_high, ref5_low = stage5(ref4_high, ref4_low)

    args.out_dir.mkdir(parents=True, exist_ok=False)
    low_onnx = args.out_dir / "receiver_low_path.onnx"
    high_onnx = args.out_dir / "receiver_high_path.onnx"
    low_ref, low_model, low_simplified = _export(
        low_path, (x_l,), low_onnx, ["x_l"], ["low4", "low5"], args.opset)
    high_ref, high_model, high_simplified = _export(
        high_path, (x_h, ref4_low, ref5_low), high_onnx,
        ["x_h", "low4", "low5"], ["high5"], args.opset)

    split_reference_max = max(
        _max_abs(low_ref, [ref4_low.detach().cpu().numpy(), ref5_low.detach().cpu().numpy()]),
        _max_abs(high_ref, [ref5_high.detach().cpu().numpy()]),
    )
    low_ort = _run_ort(low_onnx, {"x_l": x_l})
    high_ort = _run_ort(high_onnx, {
        "x_h": x_h,
        "low4": torch.from_numpy(low_ort[0]),
        "low5": torch.from_numpy(low_ort[1]),
    })
    ort_max = max(
        _max_abs(low_ref, low_ort),
        _max_abs((ref5_high,), high_ort),
    )
    manifest = {
        "kind": "diagnostic_low_high_cut",
        "gfa_mode": args.gfa_mode,
        "mathematical_change": (
            "none; original stage4/stage5 submodules are reused verbatim"
            if args.gfa_mode == "original"
            else "non-equivalent GFA normalization candidate; all learned tensors are reused"
        ),
        "checkpoint": {"path": str(args.checkpoint), "sha256": _sha256(args.checkpoint)},
        "opset": args.opset,
        "interfaces": {
            "low_path": {"inputs": {"x_l": [1, 256, 32, 32]},
                         "outputs": {"low4": [1, 512, 16, 16], "low5": [1, 512, 16, 16]}},
            "high_path": {"inputs": {"x_h": [1, 128, 64, 64], "low4": [1, 512, 16, 16],
                                        "low5": [1, 512, 16, 16]},
                          "outputs": {"high5": [1, 128, 64, 64]}},
        },
        "artifacts": {
            "low_path": {"onnx": str(low_onnx), "sha256": _sha256(low_onnx),
                         "nodes": len(low_model.graph.node), "ops": _ops(low_model),
                         "simplified": low_simplified},
            "high_path": {"onnx": str(high_onnx), "sha256": _sha256(high_onnx),
                          "nodes": len(high_model.graph.node), "ops": _ops(high_model),
                          "simplified": high_simplified},
        },
        "validation": {"split_reference_max_abs": split_reference_max,
                       "pytorch_to_ort_cascade_max_abs": ort_max,
                       "pass": split_reference_max == 0.0 and ort_max < 1e-3},
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if not manifest["validation"]["pass"]:
        raise SystemExit("diagnostic split parity failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
