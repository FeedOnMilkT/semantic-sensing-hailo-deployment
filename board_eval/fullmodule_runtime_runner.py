#!/usr/bin/env python3
"""Pi-side runtime for the complete ConvBN fused-mid semantic pipeline.

This process intentionally owns *all* per-module HEFs and executes the actual
order used by the split model:

sender -> align_h -> g_a -> h_a -> rANS(z) -> h_s -> rANS(y) -> g_s
       -> stage4 -> stage5 -> DAPPM -> segmentation head.

The entropy models are not neural Hailo graphs. Their immutable CDF state is
loaded from ``codec_tables.npz`` and rANS is the project's C++ extension. No
checkpoint, PyTorch or CompressAI is required on the Pi. Requests and results
are framed compressed NPZ messages on stdin/stdout; logs stay on stderr.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np


FRAME = struct.Struct(">Q")
MAX_FRAME_BYTES = 256 * 1024 * 1024
RTFF_HEADER = struct.Struct(">4sHHHHHHHHB")
RTFF_MAGIC = b"RTFF"
SEQUENCE = (
    "sender_backbone", "bn_align_h", "bn_g_a", "bn_h_a", "bn_h_s", "bn_g_s",
    "receiver_stage4", "receiver_stage5", "receiver_dappm", "receiver_head",
)


def read_exact(handle, size: int) -> bytes:
    parts: list[bytes] = []
    remaining = size
    while remaining:
        piece = handle.read(remaining)
        if not piece:
            raise EOFError("unexpected EOF in framed request")
        parts.append(piece)
        remaining -= len(piece)
    return b"".join(parts)


def read_frame(handle) -> bytes:
    header = handle.read(FRAME.size)
    if not header:
        raise EOFError
    if len(header) != FRAME.size:
        raise EOFError("truncated frame header")
    (size,) = FRAME.unpack(header)
    if not 0 < size <= MAX_FRAME_BYTES:
        raise ValueError(f"invalid frame size: {size}")
    return read_exact(handle, size)


def write_frame(handle, payload: bytes) -> None:
    handle.write(FRAME.pack(len(payload)))
    handle.write(payload)
    handle.flush()


def pack(op: str, *, metadata: dict | None = None, arrays: dict[str, np.ndarray] | None = None) -> bytes:
    out = io.BytesIO()
    values: dict[str, object] = {"op": np.asarray(op), "metadata": np.asarray(json.dumps(metadata or {}))}
    values.update(arrays or {})
    np.savez_compressed(out, **values)
    payload = out.getvalue()
    if len(payload) > MAX_FRAME_BYTES:
        raise ValueError(f"response exceeds {MAX_FRAME_BYTES} byte framed transport limit")
    return payload


def unpack(payload: bytes) -> tuple[str, dict, dict[str, np.ndarray]]:
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        op = str(archive["op"].item())
        metadata = json.loads(str(archive["metadata"].item()))
        arrays = {name: np.asarray(archive[name]) for name in archive.files if name not in {"op", "metadata"}}
    return op, metadata, arrays


def as_hwc(value: np.ndarray, nchw: list[int], name: str) -> np.ndarray:
    if len(nchw) != 4 or nchw[0] != 1:
        raise ValueError(f"only static batch-one NCHW contracts are supported: {name}={nchw}")
    expected = (nchw[2], nchw[3], nchw[1])
    value = np.asarray(value, dtype=np.float32)
    if value.shape == (1, *expected):
        value = value[0]
    if value.shape != expected or not np.isfinite(value).all():
        raise ValueError(f"invalid {name}: got={value.shape} expected={expected} dtype={value.dtype}")
    return np.ascontiguousarray(value)


class ModuleEngine:
    """One configured network group, with activation measured per invocation."""

    def __init__(self, vdevice, root: Path, spec: dict):
        from hailo_platform import (ConfigureParams, FormatType, HailoStreamInterface,
                                    HEF, InferVStreams, InputVStreamParams,
                                    OutputVStreamParams)

        self.name = str(spec["name"])
        self.input_names = list(spec["input_names"])
        self.output_names = list(spec["output_names"])
        self.input_shapes = [list(shape) for shape in spec["in_shapes_nchw"]]
        self.output_shapes = [list(shape) for shape in spec["out_shapes_nchw"]]
        if self.name not in SEQUENCE:
            raise ValueError(f"unknown full-runtime module {self.name!r}")
        self.hef_path = root / spec["hef_filename"]
        if not self.hef_path.is_file():
            raise FileNotFoundError(self.hef_path)
        self.hef = HEF(str(self.hef_path))
        raw_inputs = list(self.hef.get_input_vstream_infos())
        raw_outputs = list(self.hef.get_output_vstream_infos())
        if len(raw_inputs) != len(self.input_names) or len(raw_outputs) != len(self.output_names):
            raise ValueError(f"{self.name} HEF IO count differs from frozen ONNX contract")
        self.input_infos = self._map_infos(raw_inputs, self.input_names, self.input_shapes, "input")
        self.output_infos = self._map_infos(raw_outputs, self.output_names, self.output_shapes, "output")
        configure = ConfigureParams.create_from_hef(self.hef, interface=HailoStreamInterface.PCIe)
        groups = vdevice.configure(self.hef, configure)
        if len(groups) != 1:
            raise ValueError(f"{self.name}: expected one network group, got {len(groups)}")
        self.network_group = groups[0]
        in_params = InputVStreamParams.make_from_network_group(
            self.network_group, quantized=False, format_type=FormatType.FLOAT32)
        out_params = OutputVStreamParams.make_from_network_group(
            self.network_group, quantized=False, format_type=FormatType.FLOAT32)
        self.pipeline = InferVStreams(self.network_group, in_params, out_params)
        self.pipeline.__enter__()

    def _map_infos(self, infos: list, names: list[str], shapes: list[list[int]], direction: str) -> list:
        """Map logical ONNX tensors to HEF vstreams by static HWC contract.

        DFC may reorder independent inputs/outputs. Most contracts have unique
        shapes, so indexing by returned vstream position is unsafe. Identical
        shapes (currently h_s means/scales) are preserved in HEF order and are
        separately golden-checked before any semantic evaluation.
        """
        remaining = list(infos)
        mapped: list[object | None] = [None] * len(names)
        ambiguous: list[int] = []
        for position, (name, nchw) in enumerate(zip(names, shapes)):
            expected = (nchw[2], nchw[3], nchw[1])
            candidates = [info for info in remaining if tuple(info.shape) == expected]
            if len(candidates) == 1:
                mapped[position] = candidates[0]
                remaining.remove(candidates[0])
            elif len(candidates) == 0:
                actual = [(info.name, tuple(info.shape)) for info in infos]
                raise ValueError(f"{self.name}:{name} {direction} HEF shapes={actual} expected={expected}")
            else:
                ambiguous.append(position)
        if ambiguous:
            # Only accept an ambiguity where every unresolved logical tensor and
            # every unresolved HEF stream truly share one contract. The runtime
            # preflight records the order; semantic gate compares it to golden.
            expected_set = {(shapes[pos][2], shapes[pos][3], shapes[pos][1]) for pos in ambiguous}
            actual_set = {tuple(info.shape) for info in remaining}
            if len(expected_set) != 1 or expected_set != actual_set or len(remaining) != len(ambiguous):
                raise ValueError(f"{self.name}: ambiguous {direction} HEF mapping is not shape-equivalent")
            for position, info in zip(ambiguous, remaining):
                mapped[position] = info
        if any(info is None for info in mapped):
            raise AssertionError(f"{self.name}: incomplete {direction} vstream mapping")
        return list(mapped)

    def infer(self, arrays: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        feed: dict[str, np.ndarray] = {}
        for logical, shape, info in zip(self.input_names, self.input_shapes, self.input_infos):
            value = as_hwc(arrays.get(logical), shape, f"{self.name}:{logical}")
            feed[info.name] = value[None, ...]
        total_start = time.perf_counter_ns()
        activation_start = total_start
        with self.network_group.activate():
            infer_start = time.perf_counter_ns()
            raw = self.pipeline.infer(feed)
            infer_ms = (time.perf_counter_ns() - infer_start) / 1e6
        total_ms = (time.perf_counter_ns() - total_start) / 1e6
        result: dict[str, np.ndarray] = {}
        for logical, shape, info in zip(self.output_names, self.output_shapes, self.output_infos):
            result[logical] = as_hwc(raw[info.name], shape, f"{self.name}:{logical}")
        return result, {
            "activation_plus_infer_ms": total_ms,
            "infer_wall_ms": infer_ms,
            "activation_overhead_ms": max(0.0, total_ms - infer_ms),
        }

    def close(self) -> None:
        self.pipeline.__exit__(None, None, None)


class RuntimeCodec:
    """Numpy implementation of the trained CompressAI quantize/CDF contract."""

    def __init__(self, tables_path: Path):
        extension_dir = Path(__file__).resolve().parent / "rans_fast"
        sys.path.insert(0, str(extension_dir))
        try:
            import _rans_fast as rans
        except ImportError as exc:
            raise ImportError(f"Pi rANS extension missing in {extension_dir}") from exc
        self.rans = rans
        with np.load(tables_path, allow_pickle=False) as data:
            self.z_cdf = np.ascontiguousarray(data["z_cdf"], dtype=np.int32)
            self.z_lengths = np.ascontiguousarray(data["z_cdf_lengths"], dtype=np.int32)
            self.z_offsets = np.ascontiguousarray(data["z_offsets"], dtype=np.int32)
            self.z_medians = np.ascontiguousarray(data["z_medians_nchw"][0, :, 0, 0], dtype=np.float32)
            self.y_cdf = np.ascontiguousarray(data["y_cdf"], dtype=np.int32)
            self.y_lengths = np.ascontiguousarray(data["y_cdf_lengths"], dtype=np.int32)
            self.y_offsets = np.ascontiguousarray(data["y_offsets"], dtype=np.int32)
            self.scale_table = np.ascontiguousarray(data["y_scale_table"], dtype=np.float32)
            self.scale_bound = float(np.asarray(data["y_scale_bound"]).reshape(-1)[0])
        if self.z_cdf.shape[0] != self.z_medians.size or self.y_cdf.shape[0] != self.scale_table.size:
            raise ValueError("codec CDF table/channel contract mismatch")

    @staticmethod
    def _symbols(value: np.ndarray, means: np.ndarray) -> np.ndarray:
        if value.shape != means.shape or not np.isfinite(value).all() or not np.isfinite(means).all():
            raise ValueError("invalid entropy-model value or means")
        # numpy.rint follows IEEE-754 ties-to-even, matching torch.round for finite values.
        return np.ascontiguousarray(np.rint(value - means), dtype=np.int32)

    @staticmethod
    def _channel_indexes(shape: tuple[int, int, int], channels: int) -> np.ndarray:
        if shape[-1] != channels:
            raise ValueError(f"channel mismatch: {shape[-1]} != {channels}")
        return np.ascontiguousarray(np.broadcast_to(np.arange(channels, dtype=np.int32), shape))

    def _gaussian_indexes(self, scales: np.ndarray) -> np.ndarray:
        if scales.ndim != 3 or scales.shape[-1] <= 0 or not np.isfinite(scales).all():
            raise ValueError("invalid Gaussian scales")
        clamped = np.maximum(scales, self.scale_bound)
        indexes = np.full(scales.shape, self.scale_table.size - 1, dtype=np.int32)
        for scale in self.scale_table[:-1]:
            indexes -= (clamped <= scale).astype(np.int32)
        return np.ascontiguousarray(indexes)

    def _roundtrip(self, symbols: np.ndarray, indexes: np.ndarray,
                   cdf: np.ndarray, lengths: np.ndarray, offsets: np.ndarray) -> tuple[np.ndarray, bytes, dict]:
        flat_symbols = np.ascontiguousarray(symbols.reshape(-1), dtype=np.int32)
        flat_indexes = np.ascontiguousarray(indexes.reshape(-1), dtype=np.int32)
        encode_start = time.perf_counter_ns()
        stream = self.rans.encode_with_indexes(flat_symbols, flat_indexes, cdf, lengths, offsets)
        encode_ms = (time.perf_counter_ns() - encode_start) / 1e6
        decode_start = time.perf_counter_ns()
        decoded = self.rans.decode_with_indexes(stream, flat_indexes, cdf, lengths, offsets)
        decode_ms = (time.perf_counter_ns() - decode_start) / 1e6
        decoded = np.ascontiguousarray(decoded.reshape(symbols.shape), dtype=np.int32)
        if not np.array_equal(decoded, symbols):
            raise RuntimeError("rANS symbol roundtrip mismatch")
        return decoded, stream, {"encode_ms": encode_ms, "decode_ms": decode_ms, "bytes": len(stream)}

    def z_roundtrip(self, z: np.ndarray) -> tuple[np.ndarray, bytes, dict]:
        medians = np.broadcast_to(self.z_medians.reshape(1, 1, -1), z.shape)
        symbols = self._symbols(z, medians)
        indexes = self._channel_indexes(z.shape, self.z_medians.size)
        decoded, stream, timing = self._roundtrip(symbols, indexes, self.z_cdf, self.z_lengths, self.z_offsets)
        return np.ascontiguousarray(decoded.astype(np.float32) + medians), stream, timing

    def y_roundtrip(self, y: np.ndarray, means: np.ndarray, scales: np.ndarray) -> tuple[np.ndarray, bytes, dict]:
        symbols = self._symbols(y, means)
        indexes = self._gaussian_indexes(scales)
        decoded, stream, timing = self._roundtrip(symbols, indexes, self.y_cdf, self.y_lengths, self.y_offsets)
        return np.ascontiguousarray(decoded.astype(np.float32) + means), stream, timing

    @staticmethod
    def container(z_stream: bytes, y_stream: bytes, z: np.ndarray, y: np.ndarray,
                  x_h: np.ndarray, x_l: np.ndarray) -> tuple[bytes, dict]:
        start = time.perf_counter_ns()
        header = RTFF_HEADER.pack(RTFF_MAGIC, z.shape[0], z.shape[1], y.shape[0], y.shape[1],
                                  x_h.shape[0], x_h.shape[1], x_l.shape[0], x_l.shape[1], 1)
        payload = header + struct.pack(">I", len(z_stream)) + z_stream + struct.pack(">I", len(y_stream)) + y_stream
        if payload[:4] != RTFF_MAGIC:
            raise RuntimeError("RTFF container magic failure")
        offset = RTFF_HEADER.size
        z_len, = struct.unpack_from(">I", payload, offset); offset += 4
        parsed_z = payload[offset:offset + z_len]; offset += z_len
        y_len, = struct.unpack_from(">I", payload, offset); offset += 4
        parsed_y = payload[offset:offset + y_len]; offset += y_len
        if offset != len(payload) or parsed_z != z_stream or parsed_y != y_stream:
            raise RuntimeError("RTFF container parse mismatch")
        return payload, {"pack_and_parse_ms": (time.perf_counter_ns() - start) / 1e6,
                         "bytes": len(payload), "z_bytes": len(z_stream), "y_bytes": len(y_stream)}


class FullRuntime:
    def __init__(self, manifest_path: Path):
        from hailo_platform import VDevice

        self.root = manifest_path.parent
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if tuple(self.manifest.get("sequence", ())) != SEQUENCE:
            raise ValueError("runtime manifest sequence is not the frozen full-module order")
        codec_spec = self.manifest["codec"]
        self.codec = RuntimeCodec(self.root / codec_spec["tables_filename"])
        self.vdevice = VDevice()
        self.modules: dict[str, ModuleEngine] = {}
        try:
            for spec in self.manifest["modules"]:
                engine = ModuleEngine(self.vdevice, self.root, spec)
                if engine.name in self.modules:
                    raise ValueError(f"duplicate module {engine.name}")
                self.modules[engine.name] = engine
            if tuple(self.modules) != SEQUENCE:
                raise ValueError(f"runtime modules differ from required sequence: {tuple(self.modules)}")
        except BaseException:
            self.close()
            raise
        self.calls = 0

    def hello(self) -> dict:
        return {
            "pid": os.getpid(), "sequence": list(SEQUENCE), "calls": self.calls,
            "hefs": {name: str(self.modules[name].hef_path) for name in SEQUENCE},
            "codec": {"z_cdf_rows": int(self.codec.z_cdf.shape[0]), "y_cdf_rows": int(self.codec.y_cdf.shape[0]),
                      "scale_table_entries": int(self.codec.scale_table.size)},
        }

    def _module(self, name: str, arrays: dict[str, np.ndarray], timings: dict) -> dict[str, np.ndarray]:
        outputs, timing = self.modules[name].infer(arrays)
        timings[name] = timing
        return outputs

    def infer(self, image: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
        total_start = time.perf_counter_ns()
        timings: dict[str, object] = {}
        sender = self._module("sender_backbone", {"image": image}, timings)
        aligned = self._module("bn_align_h", {"x_h": sender["x_h"]}, timings)
        y = self._module("bn_g_a", {"aligned_x_h": aligned["aligned_x_h"], "x_l": sender["x_l"]}, timings)["latent_y"]
        z = self._module("bn_h_a", {"latent_y": y}, timings)["latent_z"]
        z_hat, z_stream, z_timing = self.codec.z_roundtrip(z)
        timings["rans_z"] = z_timing
        hyper = self._module("bn_h_s", {"latent_z_hat": z_hat}, timings)
        y_hat, y_stream, y_timing = self.codec.y_roundtrip(y, hyper["means"], hyper["scales"])
        timings["rans_y"] = y_timing
        _container, container_timing = self.codec.container(z_stream, y_stream, z, y, sender["x_h"], sender["x_l"])
        timings["rtff_container"] = container_timing
        recon = self._module("bn_g_s", {"latent_y_hat": y_hat}, timings)
        stage4 = self._module("receiver_stage4", {"x_h": recon["x_h_hat"], "x_l": recon["x_l_hat"]}, timings)
        stage5 = self._module("receiver_stage5", {"stage4_high": stage4["stage4_high"], "stage4_low": stage4["stage4_low"]}, timings)
        s8 = self._module("receiver_dappm", {"high_feat": stage5["high_feat"], "low_feat": stage5["low_feat"]}, timings)
        logits = self._module("receiver_head", {"s8": s8["s8"]}, timings)["seg_logits"]
        self.calls += 1
        return {"seg_logits": logits, "argmax": np.asarray(np.argmax(logits, axis=-1), dtype=np.uint8)}, {
            "call": self.calls, "stages": timings,
            "total_pipeline_ms": (time.perf_counter_ns() - total_start) / 1e6,
        }

    def close(self) -> None:
        for engine in reversed(list(self.modules.values())):
            try:
                engine.close()
            except Exception:
                pass
        self.modules = {}
        if getattr(self, "vdevice", None) is not None:
            try:
                self.vdevice.release()
            except Exception:
                pass


def serve(manifest: Path) -> int:
    runtime = FullRuntime(manifest)
    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    try:
        while True:
            try:
                op, metadata, arrays = unpack(read_frame(stdin))
            except EOFError:
                return 0
            try:
                if op == "hello":
                    response = pack("hello", metadata=runtime.hello())
                elif op == "infer":
                    image = as_hwc(arrays.get("image"), [1, 3, 512, 512], "image")
                    outputs, timing = runtime.infer(image)
                    returned = {"argmax": outputs["argmax"]}
                    sample_indices = arrays.get("cosine_indices")
                    if sample_indices is not None:
                        sample_indices = np.asarray(sample_indices, dtype=np.int64)
                        if sample_indices.ndim != 1 or not 0 < sample_indices.size <= 1_000_000:
                            raise ValueError("cosine_indices must be a non-empty 1-D vector of at most 1,000,000 entries")
                        flat = outputs["seg_logits"].reshape(-1)
                        if sample_indices.min() < 0 or sample_indices.max() >= flat.size:
                            raise ValueError("cosine_indices out of logit range")
                        returned["logit_samples"] = np.ascontiguousarray(flat[sample_indices], dtype=np.float32)
                        timing["cosine_sample_count"] = int(sample_indices.size)
                    if bool(metadata.get("return_logits", False)):
                        returned["seg_logits"] = outputs["seg_logits"]
                    response = pack("result", metadata=timing, arrays=returned)
                elif op == "shutdown":
                    write_frame(stdout, pack("shutdown", metadata={"calls": runtime.calls}))
                    return 0
                else:
                    raise ValueError(f"unknown op {op!r}")
            except Exception as exc:
                response = pack("error", metadata={"error": f"{type(exc).__name__}: {exc}"})
            write_frame(stdout, response)
    finally:
        runtime.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if not args.serve:
        parser.error("only --serve is supported")
    raise SystemExit(serve(args.manifest))


if __name__ == "__main__":
    main()
