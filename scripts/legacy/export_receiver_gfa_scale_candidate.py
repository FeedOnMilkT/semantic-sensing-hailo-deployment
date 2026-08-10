#!/usr/bin/env python3
"""Export one strictly-equivalent power-of-two GFA grouped-L1 candidate.

For q=2**k, the candidate evaluates

    (q * S) / (sum_group(q * S) + q * eps)

instead of S / (sum_group(S) + eps).  The checkpoint, GFA weights, softmax and
epsilon are unchanged; this is an FP32 reparameterization intended only to
alter the DFC intermediate representation.  It never changes rtformer.py.
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


class ScaledGFA(nn.Module):
    """Deployment-only GFA view that shares all original GFA modules/parameters."""

    def __init__(self, original: nn.Module, scale_exponent: int, normalizer: str):
        super().__init__()
        if scale_exponent < 0:
            raise ValueError("scale_exponent must be non-negative")
        if normalizer not in {"sum", "l1"}:
            raise ValueError(f"unsupported normalizer: {normalizer}")
        self.original = original
        self.scale_exponent = scale_exponent
        self.normalizer = normalizer
        self.register_buffer("normalization_scale", torch.tensor(float(2 ** scale_exponent)))
        self.eps = 1e-6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        gfa = self.original
        attn = gfa.q(x).flatten(2).transpose(1, 2)
        attn = attn @ gfa.kg.t()
        attn = attn.softmax(dim=1)
        attn = attn.view(B, H * W, gfa.groups, gfa.per)
        # Multiplication by q is omitted for q=1 so the L1 candidate changes only
        # the reduction operator. Both forms retain the same epsilon semantics.
        scaled = attn if self.scale_exponent == 0 else attn * self.normalization_scale
        if self.normalizer == "sum":
            norm = scaled.sum(dim=3, keepdim=True)
        else:
            # Softmax is non-negative, so L1 exactly equals ReduceSum in real FP32.
            norm = torch.linalg.vector_norm(scaled, ord=1, dim=3, keepdim=True)
        denominator = norm + self.eps * self.normalization_scale
        attn = scaled / denominator
        attn = attn.view(B, H * W, gfa.mem)
        out = (attn @ gfa.vg).transpose(1, 2).reshape(B, C, H, W)
        return torch.relu(x + gfa.proj(out))


class ReceiverLowPathOriginal(nn.Module):
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


class ReceiverLowPathScaled(nn.Module):
    def __init__(self, stage4: nn.Module, stage5: nn.Module, scale_exponent: int,
                 normalizer: str):
        super().__init__()
        self.stage4 = stage4
        self.stage5 = stage5
        self.stage4_gfa = ScaledGFA(stage4.gfa, scale_exponent, normalizer)
        self.stage5_gfa = ScaledGFA(stage5.gfa, scale_exponent, normalizer)
        self.scale_exponent = scale_exponent
        self.normalizer = normalizer

    def _low4(self, low: torch.Tensor) -> torch.Tensor:
        return self.stage4.ffn_low(self.stage4_gfa(self.stage4.low_trans(low)))

    def _low5(self, low: torch.Tensor) -> torch.Tensor:
        return self.stage5.ffn_low(self.stage5_gfa(self.stage5.low_trans(low)))

    def forward(self, x_l: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        low4 = self._low4(x_l)
        return low4, self._low5(low4)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _max_abs(reference: tuple[torch.Tensor, ...], actual: tuple[torch.Tensor, ...] | list[np.ndarray]) -> float:
    values = []
    for expected, received in zip(reference, actual):
        array = received.detach().cpu().numpy() if isinstance(received, torch.Tensor) else received
        values.append(float(np.max(np.abs(expected.detach().cpu().numpy() - array))))
    return max(values)


def _ops(model) -> dict[str, int]:
    types = [node.op_type for node in model.graph.node]
    return {item: types.count(item) for item in sorted(set(types))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "/local/shared_with_docker/ckpt/midf_b0.01_s2.pth"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--scale-exp", type=int, required=True)
    parser.add_argument("--normalizer", choices=["sum", "l1"], default="sum")
    parser.add_argument("--opset", type=int, default=13)
    args = parser.parse_args()
    if args.scale_exp not in range(17):
        parser.error("--scale-exp must be in 0..16")
    if args.scale_exp == 0 and args.normalizer == "sum":
        parser.error("k=0 + sum is the preserved baseline, not a candidate")
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}")
    if args.out_dir.exists():
        parser.error(f"refusing to overwrite candidate directory: {args.out_dir}")

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
    original = ReceiverLowPathOriginal(backbone.stage4.eval(), backbone.stage5.eval()).eval()
    candidate = ReceiverLowPathScaled(backbone.stage4.eval(), backbone.stage5.eval(),
                                      args.scale_exp, args.normalizer).eval()
    x_l = torch.randn(1, 256, 32, 32)
    with torch.no_grad():
        reference = original(x_l)
        scaled_reference = candidate(x_l)
    torch_equivalence = _max_abs(reference, scaled_reference)

    args.out_dir.mkdir(parents=True, exist_ok=False)
    onnx_path = args.out_dir / "receiver_low_path.onnx"
    torch.onnx.export(
        candidate, (x_l,), str(onnx_path), input_names=["x_l"],
        output_names=["low4", "low5"], opset_version=args.opset,
        do_constant_folding=True, dynamic_axes=None,
    )
    import onnx
    import onnxruntime as ort

    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    simplified = False
    try:
        from onnxsim import simplify

        simplified_model, ok = simplify(model)
        if ok:
            onnx.save(simplified_model, onnx_path)
            model = simplified_model
            simplified = True
    except ImportError:
        pass
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_outputs = session.run(None, {"x_l": x_l.detach().cpu().numpy()})
    ort_equivalence = _max_abs(scaled_reference, ort_outputs)
    manifest = {
        "kind": "strict_equivalent_gfa_normalizer_rewrite",
        "formula": ("(q*S) / (sum_group(q*S) + q*eps)" if args.normalizer == "sum"
                    else "S / (L1_group(S) + eps), with softmax(S)>=0"),
        "normalizer": args.normalizer,
        "scale_exponent": args.scale_exp,
        "scale": 2 ** args.scale_exp,
        "epsilon": 1e-6,
        "checkpoint": {"path": str(args.checkpoint), "sha256": _sha256(args.checkpoint)},
        "onnx": {"path": str(onnx_path), "sha256": _sha256(onnx_path),
                 "nodes": len(model.graph.node), "ops": _ops(model), "simplified": simplified},
        "validation": {
            "original_vs_scaled_pytorch_max_abs": torch_equivalence,
            "scaled_pytorch_vs_ort_max_abs": ort_equivalence,
            "pass": torch_equivalence <= 1e-5 and ort_equivalence < 1e-3,
        },
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if not manifest["validation"]["pass"]:
        raise SystemExit("strict-equivalence gate failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
