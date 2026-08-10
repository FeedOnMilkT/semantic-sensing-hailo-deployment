#!/usr/bin/env python3
"""Materialize formal downstream calibration from quantized upstream HARs.

Run this only *after* the relevant upstream graph has been PTQ'd and compiled
to a quantized HAR.  It executes the Hailo SDK QUANTIZED emulator, applies the
real CPU rANS decode boundary, and writes a normal ``load_calib.py`` handoff:

  hyper           = quantized analysis(latent_z) -> rANS(z) -> latent_z_hat
  synthesis_tail  = quantized analysis(latent_y/z) -> rANS(z) -> quantized
                    hyper(means/scales) -> rANS(y) -> latent_y_hat

The script intentionally cannot materialize a calibration set from the
bootstrap FP tensors.  That is the guard which keeps independent HEF scales
from being evaluated against mismatched upstream activations.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    from hailo_deploy.logging import reexec_if_unlogged
    reexec_if_unlogged(Path(__file__), sys.argv[1:])

from hailo_deploy.emulate import _match_outputs
from hailo_deploy.io.calib import load_calib_handoff, map_hn_inputs
from hailo_deploy.io.hash import sha256
from hailo_deploy.onnx_io import nchw_to_nhwc, nhwc_to_nchw


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=["hyper", "synthesis_tail"], required=True)
    parser.add_argument("--segments-manifest", type=Path, required=True)
    parser.add_argument("--bootstrap-calib-root", type=Path, required=True,
                        help="root emitted by prepare_hailo_segmented_calib.py")
    parser.add_argument("--analysis-quantized-har", type=Path, required=True)
    parser.add_argument("--hyper-quantized-har", type=Path,
                        help="required only for --target synthesis_tail")
    parser.add_argument("--semantic-root", type=Path, default=Path("/workspace/semantic-sensing"))
    parser.add_argument("--student-ckpt", type=Path, required=True)
    parser.add_argument("--n-calib", type=int, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    return parser.parse_args(argv)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest = _load_json(path)
    if manifest.get("schema") != "semantic_sensing.segmented_onnx.v1":
        raise ValueError("unsupported segments manifest")
    entries = {entry["id"]: entry for entry in manifest.get("segments", [])}
    required = {"analysis", "hyper", "synthesis_tail"}
    if set(entries) != required:
        raise ValueError(f"expected {sorted(required)}, got {sorted(entries)}")
    for name, entry in entries.items():
        onnx = Path(entry["onnx"])
        if not onnx.is_file() or sha256(onnx) != entry["sha256"]:
            raise ValueError(f"{name}: ONNX identity mismatch")
    return manifest, entries


def _expected(entry: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        name: np.empty(shape, dtype=np.float32)
        for name, shape in zip(entry["output_names"], entry["out_shapes_nchw"])
    }



def _exact_quantized_outputs(
    runner, expected: dict[str, np.ndarray], raw
) -> dict[str, np.ndarray] | None:
    """Map raw Hailo outputs by HN ``original_names`` before any shape fallback."""
    layers = runner.get_hn_dict().get("layers", {})
    output_layers = [
        (name, list(layer.get("original_names") or []))
        for name, layer in layers.items()
        if "output" in str(layer.get("type", "")).lower()
    ]
    if isinstance(raw, dict):
        values = [(name, value) for name, value in raw.items()]
    elif isinstance(raw, (list, tuple)):
        if len(raw) != len(output_layers):
            return None
        values = [(name, value) for (name, _originals), value in zip(output_layers, raw)]
    else:
        if len(output_layers) != 1:
            return None
        values = [(output_layers[0][0], raw)]
    original_by_hn = dict(output_layers)
    mapped: dict[str, np.ndarray] = {}
    for hn_name, value in values:
        hits = [name for name in original_by_hn.get(hn_name, []) if name in expected]
        if len(hits) != 1 or hits[0] in mapped:
            return None
        mapped[hits[0]] = np.ascontiguousarray(
            nhwc_to_nchw(np.asarray(value)), dtype=np.float32
        )
    return mapped if set(mapped) == set(expected) else None


class QuantizedSegment:
    """One parsed/optimized HAR, with output matching through HN original names."""

    def __init__(self, har: Path, entry: dict[str, Any]) -> None:
        from hailo_sdk_client import ClientRunner

        if not har.is_file():
            raise FileNotFoundError(har)
        self.runner = ClientRunner(har=str(har))
        self.expected = _expected(entry)
        self._context_manager = None
        self._context = None

    def __enter__(self):
        from hailo_sdk_client import InferenceContext

        self._context_manager = self.runner.infer_context(InferenceContext.SDK_QUANTIZED)
        self._context = self._context_manager.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        assert self._context_manager is not None
        return self._context_manager.__exit__(exc_type, exc, tb)

    def run(self, inputs_nchw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        arrays = {name: np.ascontiguousarray(nchw_to_nhwc(value), dtype=np.float32)
                  for name, value in inputs_nchw.items()}
        feed, _mapping = map_hn_inputs(self.runner, arrays)
        raw = self.runner.infer(self._context, feed)
        exact = _exact_quantized_outputs(self.runner, self.expected, raw)
        if exact is not None:
            return exact
        # The existing project matcher is retained only for unambiguous output
        # shapes. Equal-shaped means/scales must map through original_names.
        shapes = [tuple(value.shape) for value in self.expected.values()]
        if len(shapes) != len(set(shapes)):
            raise ValueError("ambiguous equal-shaped Hailo outputs lack original_names mapping")
        notes: list[str] = []
        pairs = _match_outputs(self.runner, self.expected, raw, notes)
        if set(pairs) != set(self.expected):
            raise ValueError(
                f"Hailo outputs do not map exactly: got={sorted(pairs)} "
                f"expected={sorted(self.expected)} notes={notes}"
            )
        return {name: np.ascontiguousarray(actual, dtype=np.float32)
                for name, (_ref, actual) in pairs.items()}


def _source_image(meta: dict[str, Any], root: Path, index: int) -> np.ndarray:
    mapping = meta.get("tensor_mapping", {}).get("image")
    if not mapping or mapping.get("onnx_name") != "image":
        raise ValueError("bootstrap analysis handoff does not expose image")
    path = root / mapping["directory"] / f"{index:05d}.npy"
    array = np.load(path, allow_pickle=False)
    if array.dtype != np.float32 or array.ndim != 3 or not np.isfinite(array).all():
        raise ValueError(f"invalid bootstrap image {path}")
    return np.transpose(array[None], (0, 3, 1, 2)).copy()


def _load_codec_helpers(semantic_root: Path):
    if not semantic_root.is_dir():
        raise FileNotFoundError(semantic_root)
    tools = semantic_root / "tools"
    for path in (semantic_root, tools):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from prepare_hailo_full_module_calib import StreamingStats, _generated_loader, _load_student
    return StreamingStats, _generated_loader, _load_student


def _z_roundtrip(bn: torch.nn.Module, latent_z: np.ndarray) -> torch.Tensor:
    z = torch.from_numpy(np.ascontiguousarray(latent_z)).float()
    strings = bn.entropy_bottleneck.compress(z)
    return bn.entropy_bottleneck.decompress(strings, z.shape[-2:])


def _y_roundtrip(
    bn: torch.nn.Module, latent_y: np.ndarray, means: np.ndarray, scales: np.ndarray
) -> torch.Tensor:
    y = torch.from_numpy(np.ascontiguousarray(latent_y)).float()
    means_t = torch.from_numpy(np.ascontiguousarray(means)).float()
    scales_t = torch.from_numpy(np.ascontiguousarray(scales)).float()
    indexes = bn.gaussian_conditional.build_indexes(scales_t)
    strings = bn.gaussian_conditional.compress(y, indexes, means=means_t)
    return bn.gaussian_conditional.decompress(strings, indexes, means=means_t)


def _mapping(name: str, tensor: torch.Tensor) -> dict[str, dict[str, Any]]:
    shape = [int(value) for value in tensor.shape]
    if len(shape) != 4:
        raise ValueError(f"{name}: expected NCHW tensor, got {shape}")
    return {name: {
        "onnx_name": name,
        "directory": name,
        "shape_nchw": shape,
        "shape_nhwc": [shape[0], shape[2], shape[3], shape[1]],
        "dtype": "float",
    }}


def _save(root: Path, name: str, index: int, tensor: torch.Tensor, mapping: dict, stats) -> None:
    value = tensor.detach().cpu().float().numpy()
    expected = mapping[name]["shape_nchw"]
    if list(value.shape) != expected:
        raise ValueError(f"{name}: got {list(value.shape)}, expected {expected}")
    nhwc = np.ascontiguousarray(np.transpose(value, (0, 2, 3, 1))[0], dtype=np.float32)
    if not np.isfinite(nhwc).all():
        raise ValueError(f"{name}: non-finite output")
    np.save(root / name / f"{index:05d}.npy", nhwc)
    stats.update(nhwc)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.n_calib <= 0:
        raise ValueError("--n-calib must be positive")
    if args.out_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.out_root}")
    manifest_path = args.segments_manifest.resolve()
    manifest, entries = _manifest(manifest_path)
    if sha256(args.student_ckpt) != manifest["checkpoint"]["sha256"]:
        raise ValueError("checkpoint/ONNX manifest identity mismatch")
    if args.target == "synthesis_tail" and args.hyper_quantized_har is None:
        raise ValueError("--hyper-quantized-har is required for synthesis_tail")

    base_root = args.bootstrap_calib_root.resolve() / "analysis"
    _first, base_meta = load_calib_handoff(base_root, min(1, args.n_calib))
    if base_meta.get("calibration_mode") != "bootstrap_fp_codec":
        raise ValueError("base input must be the validated bootstrap analysis calibration root")
    if int(base_meta.get("sample_count", 0)) < args.n_calib:
        raise ValueError("bootstrap handoff does not have enough samples")
    if base_meta.get("onnx", {}).get("sha256") != entries["analysis"]["sha256"]:
        raise ValueError("bootstrap analysis ONNX identity mismatch")

    StreamingStats, generated_loader, load_student = _load_codec_helpers(args.semantic_root.resolve())
    student, cfg = load_student(args.student_ckpt.resolve(), torch.device("cpu"), "group_softmax")
    if cfg.bottleneck.args.get("act") != "convbn":
        raise ValueError("only ConvBN checkpoint is supported")
    bn = student.bottleneck.cpu().eval()
    bn.update(force=True)

    output_name = "latent_z_hat" if args.target == "hyper" else "latent_y_hat"
    mapping: dict[str, dict[str, Any]] | None = None
    stats = None
    temporary = args.out_root.parent / f".{args.out_root.name}.tmp-{os.getpid()}"
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.mkdir(parents=True)
    try:
        with ExitStack() as stack:
            analysis = stack.enter_context(QuantizedSegment(args.analysis_quantized_har, entries["analysis"]))
            hyper = None
            if args.target == "synthesis_tail":
                hyper = stack.enter_context(QuantizedSegment(args.hyper_quantized_har, entries["hyper"]))
            for index in range(args.n_calib):
                image = _source_image(base_meta, base_root, index)
                analysis_out = analysis.run({"image": image})
                z_hat = _z_roundtrip(bn, analysis_out["latent_z"])
                if args.target == "hyper":
                    value = z_hat
                else:
                    assert hyper is not None
                    hyper_out = hyper.run({"latent_z_hat": z_hat.numpy()})
                    value = _y_roundtrip(
                        bn, analysis_out["latent_y"], hyper_out["means"], hyper_out["scales"]
                    )
                if mapping is None:
                    mapping = _mapping(output_name, value)
                    (temporary / output_name).mkdir()
                    stats = StreamingStats(seed=20260730 + sum(map(ord, args.target)))
                _save(temporary, output_name, index, value, mapping, stats)
                if (index + 1) % 50 == 0 or index + 1 == args.n_calib:
                    print(f"materialized {index + 1}/{args.n_calib}", flush=True)
        assert mapping is not None and stats is not None
        mode = "quantized_analysis_z_rans" if args.target == "hyper" else "quantized_analysis_hyper_y_rans"
        parent = {
            "analysis_quantized_har": {"path": str(args.analysis_quantized_har.resolve()), "sha256": sha256(args.analysis_quantized_har)},
        }
        if args.hyper_quantized_har is not None:
            parent["hyper_quantized_har"] = {"path": str(args.hyper_quantized_har.resolve()), "sha256": sha256(args.hyper_quantized_har)}
        metadata = {
            "schema": "semantic_sensing.segmented_calibration.v1",
            "segment": args.target,
            "calibration_mode": mode,
            "checkpoint": manifest["checkpoint"],
            "model": manifest["model"],
            "onnx": {key: entries[args.target].get(key) for key in (
                "onnx", "sha256", "opset", "input_names", "output_names", "in_shapes_nchw", "out_shapes_nchw")},
            "sample_count": args.n_calib,
            "source_bootstrap": {"root": str(args.bootstrap_calib_root.resolve()), "meta_sha256": sha256(base_root / "meta.json")},
            "quantized_upstream": parent,
            "codec": {
                "path": "roundtrip",
                "definition": "Hailo QUANTIZED upstream tensor(s) followed by CPU real rANS compress/decompress",
            },
            "tensor_mapping": mapping,
            "storage": {"dtype": "float32", "layout": "NHWC", "per_sample_npy": True},
            "statistics": {output_name: stats.result()},
        }
        (temporary / "meta.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (temporary / "load_calib.py").write_text(generated_loader(), encoding="utf-8")
        os.replace(temporary, args.out_root)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    print(json.dumps({"status": "PASS", "target": args.target, "mode": mode,
                      "out_root": str(args.out_root)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
