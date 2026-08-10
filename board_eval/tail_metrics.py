#!/usr/bin/env python3
"""Evaluate the frozen receiver tail on held-out Stage45 tensor outputs.

This intentionally accepts only a frozen board-evaluation directory produced by
``evaluate_receiver_stage45.py``.  The HEF contains Stage45 only, so this is a
hybrid quality gate: Stage45 runs on the board and the unchanged decoder/head
runs on x86 CUDA when available.  It must never be called a full-board mIoU.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hailo_deploy.io.hash import sha256

REPO = Path(os.environ.get("SEMANTIC_SENSING_ROOT", "/workspace/semantic-sensing"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
from src import _compat  # noqa: E402,F401
from src.models import build_model  # noqa: E402
from src.models.student import build_student  # noqa: E402


LARGE = [7, 10, 15, 20, 23, 30, 35, 37, 48, 51, 80, 83, 84, 90, 101, 102, 103, 109, 122]


def load_student(checkpoint: Path, device: torch.device):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = state["cfg"]
    cfg.model.backbone.args.pretrained = None
    teacher = build_model(cfg.model).eval()
    student = build_student(cfg, teacher).eval()
    student.load_state_dict(state["student"], strict=True)
    student.update(force=True)
    for parameter in student.parameters():
        parameter.requires_grad_(False)
    return student.to(device).eval(), cfg


def load_hwc(root: Path, indices: list[int], shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    arrays = [np.load(root / f"{index:05d}.npy", allow_pickle=False) for index in indices]
    value = np.ascontiguousarray(np.stack(arrays).transpose(0, 3, 1, 2)).astype(np.float32, copy=False)
    if tuple(value.shape[1:]) != (shape[2], shape[0], shape[1]) or not np.isfinite(value).all():
        raise ValueError(f"invalid tensor batch from {root}: {value.shape}")
    return torch.from_numpy(value).to(device, non_blocking=True)


def load_labels(root: Path, indices: list[int], device: torch.device) -> torch.Tensor:
    arrays = [np.load(root / f"{index:05d}.npy", allow_pickle=False) for index in indices]
    value = np.ascontiguousarray(np.stack(arrays)).astype(np.int64, copy=False)
    if value.ndim != 3 or not np.logical_or(value < 150, value == 255).all():
        raise ValueError(f"invalid label batch from {root}: {value.shape}")
    return torch.from_numpy(value).to(device, non_blocking=True)


def update_confusion(confusion: torch.Tensor, logits: torch.Tensor, target: torch.Tensor, ignore_index: int) -> None:
    classes = confusion.shape[0]
    prediction = logits.argmax(dim=1)
    valid = target != ignore_index
    bins = torch.bincount((target[valid] * classes + prediction[valid]).reshape(-1), minlength=classes * classes)
    confusion += bins.reshape(classes, classes).to(confusion.device)


def metrics(confusion: torch.Tensor) -> dict:
    data = confusion.cpu().double()
    true_positive = data.diag()
    union = data.sum(0) + data.sum(1) - true_positive
    iou = true_positive / union.clamp(min=1)
    present = union > 0
    return {
        "mIoU": float(iou[present].mean()),
        "pixel_acc": float(true_positive.sum() / data.sum().clamp(min=1)),
        "fixed_19_large_object_macro_iou": float(iou[LARGE].mean()),
        "per_class_iou": [float(item) for item in iou],
        "confusion_matrix": [[int(item) for item in row] for row in data.tolist()],
    }


def run_tail(student, high: torch.Tensor, low: torch.Tensor) -> torch.Tensor:
    teacher = student.teacher
    features = teacher.backbone.forward_receiver_late(high, low)
    decoded = teacher.decoder(teacher.neck(features))
    return teacher.heads["seg"](decoded)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tensor-eval", type=Path, required=True,
                        help="output directory created by evaluate_receiver_stage45.py")
    parser.add_argument("--student-ckpt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite: {args.out}")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    tensor_report = json.loads((args.tensor_eval / "report.json").read_text(encoding="utf-8"))
    if tensor_report.get("status") != "PASS":
        raise ValueError("tensor evaluation is not PASS")
    if tensor_report.get("manifest_sha256") != sha256(args.manifest):
        raise ValueError("tensor evaluation does not belong to this frozen manifest")
    indices = list(tensor_report["val_indices"])
    overlap = set(indices) & set(manifest["calibration"]["val_indices"])
    if overlap or len(indices) != int(tensor_report["sample_count"]):
        raise ValueError("invalid held-out index contract")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu explicitly")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    student, cfg = load_student(args.student_ckpt, device)
    classes, ignore = int(cfg.num_classes), int(cfg.ignore_index)
    input_root = Path(manifest["evaluation"]["input_root"])
    output_roots = {
        "fp_onnx": Path(tensor_report["outputs"]["fp_onnx"]),
        "board_stage45": Path(tensor_report["outputs"]["board"]),
    }
    result: dict[str, dict] = {}
    with torch.inference_mode():
        for label, root in output_roots.items():
            confusion = torch.zeros((classes, classes), dtype=torch.int64, device=device)
            elapsed = 0.0
            for start in range(0, len(indices), args.batch_size):
                part = indices[start:start + args.batch_size]
                high = load_hwc(root / "high_feat", part, (64, 64, 128), device)
                low = load_hwc(root / "low_feat", part, (16, 16, 512), device)
                target = load_labels(input_root / "ground_truth", part, device)
                begin = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
                end = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
                if begin is not None:
                    begin.record()
                else:
                    import time
                    host_start = time.perf_counter()
                logits = run_tail(student, high, low)
                if end is not None:
                    end.record()
                    torch.cuda.synchronize(device)
                    elapsed += begin.elapsed_time(end)
                else:
                    elapsed += (time.perf_counter() - host_start) * 1000.0
                update_confusion(confusion, logits, target, ignore)
            result[label] = {**metrics(confusion), "receiver_tail_compute_ms": elapsed,
                             "receiver_tail_compute_ms_per_sample": elapsed / len(indices)}
    fp = result["fp_onnx"]
    board = result["board_stage45"]
    report = {
        "kind": "receiver_stage45_hybrid_board_semantic_evaluation",
        "status": "PASS",
        "mode": "hybrid: board receiver_stage45 + x86 frozen receiver tail",
        "manifest_sha256": sha256(args.manifest),
        "tensor_eval": str(args.tensor_eval.resolve()),
        "student_checkpoint": {"path": str(args.student_ckpt.resolve()), "sha256": sha256(args.student_ckpt)},
        "sample_count": len(indices), "val_indices": indices,
        "calibration_overlap": [], "device": str(device),
        "tf32_disabled": bool(device.type == "cuda"), "metrics": result,
        "board_minus_fp": {
            "mIoU": board["mIoU"] - fp["mIoU"],
            "fixed_19_large_object_macro_iou": board["fixed_19_large_object_macro_iou"] - fp["fixed_19_large_object_macro_iou"],
            "pixel_acc": board["pixel_acc"] - fp["pixel_acc"],
        },
        "limitations": ["This is not full-board mIoU: DAPPM/head execute on x86."]
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "sample_count": len(indices),
                      "board_fixed19": board["fixed_19_large_object_macro_iou"]}, indent=2))


if __name__ == "__main__":
    main()
