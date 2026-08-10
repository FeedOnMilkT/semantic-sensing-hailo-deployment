#!/usr/bin/env python3
"""Memory-only HailoRT runner used by the x86 board evaluator.

The process owns one configured HEF and speaks length-prefixed NPZ frames on
stdin/stdout.  Logs always go to stderr, so no image activation is persisted on
the Raspberry Pi.
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


def read_exact(handle, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        part = handle.read(remaining)
        if not part:
            raise EOFError("unexpected EOF in framed request")
        chunks.append(part)
        remaining -= len(part)
    return b"".join(chunks)


def read_frame(handle) -> bytes:
    header = handle.read(FRAME.size)
    if not header:
        raise EOFError
    if len(header) != FRAME.size:
        raise EOFError("truncated frame header")
    (size,) = FRAME.unpack(header)
    if not 0 < size <= 64 * 1024 * 1024:
        raise ValueError(f"invalid frame size: {size}")
    return read_exact(handle, size)


def write_frame(handle, payload: bytes) -> None:
    handle.write(FRAME.pack(len(payload)))
    handle.write(payload)
    handle.flush()


def pack(op: str, *, metadata: dict | None = None, arrays: dict[str, np.ndarray] | None = None) -> bytes:
    buffer = io.BytesIO()
    values: dict[str, object] = {"op": np.asarray(op), "metadata": np.asarray(json.dumps(metadata or {}))}
    values.update(arrays or {})
    np.savez_compressed(buffer, **values)
    return buffer.getvalue()


def unpack(payload: bytes) -> tuple[str, dict, dict[str, np.ndarray]]:
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        op = str(archive["op"].item())
        metadata = json.loads(str(archive["metadata"].item()))
        arrays = {name: np.asarray(archive[name]) for name in archive.files if name not in {"op", "metadata"}}
    return op, metadata, arrays


class Engine:
    def __init__(self, manifest_path: Path):
        self.manifest_path = manifest_path
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        contract = self.manifest["contract"]["tensors"]
        self.input_specs = {name: tuple(contract[name]["shape"]) for name in ("x_h", "x_l")}
        self.output_specs = {name: tuple(contract[name]["shape"]) for name in ("high_feat", "low_feat")}
        self.vstream = {name: contract[name]["vstream"] for name in contract}
        hef_name = self.manifest["board"]["hef_filename"]
        self.hef_path = manifest_path.parent / hef_name
        if not self.hef_path.is_file():
            raise FileNotFoundError(f"HEF absent beside manifest: {self.hef_path}")
        start = time.perf_counter_ns()
        from hailo_platform import (ConfigureParams, FormatType, HailoStreamInterface,
                                    HEF, InferVStreams, InputVStreamParams,
                                    OutputVStreamParams, VDevice)
        self.hef = HEF(str(self.hef_path))
        actual_inputs = {info.name: tuple(info.shape) for info in self.hef.get_input_vstream_infos()}
        actual_outputs = {info.name: tuple(info.shape) for info in self.hef.get_output_vstream_infos()}
        for name, shape in self.input_specs.items():
            if actual_inputs.get(self.vstream[name]) != shape:
                raise ValueError(f"HEF input contract mismatch for {name}: {actual_inputs}")
        for name, shape in self.output_specs.items():
            if actual_outputs.get(self.vstream[name]) != shape:
                raise ValueError(f"HEF output contract mismatch for {name}: {actual_outputs}")
        self.vdevice = VDevice()
        configure = ConfigureParams.create_from_hef(self.hef, interface=HailoStreamInterface.PCIe)
        groups = self.vdevice.configure(self.hef, configure)
        if len(groups) != 1:
            raise ValueError(f"expected one network group, got {len(groups)}")
        self.network_group = groups[0]
        in_params = InputVStreamParams.make_from_network_group(
            self.network_group, quantized=False, format_type=FormatType.FLOAT32)
        out_params = OutputVStreamParams.make_from_network_group(
            self.network_group, quantized=False, format_type=FormatType.FLOAT32)
        self.pipeline = InferVStreams(self.network_group, in_params, out_params)
        self.pipeline.__enter__()
        # HailoRT 4.23 requires an active network-group context even when the
        # vstream pipeline itself stays alive for a persistent request stream.
        # Keep this context open for the runner lifetime so each framed request
        # is measured without a per-image configure/activate cost.
        self.activation = self.network_group.activate()
        self.activation.__enter__()
        self.init_ms = (time.perf_counter_ns() - start) / 1e6
        self.calls = 0

    def hello(self) -> dict:
        return {
            "pid": os.getpid(), "hef": str(self.hef_path), "init_ms": self.init_ms,
            "inputs": {name: list(shape) for name, shape in self.input_specs.items()},
            "outputs": {name: list(shape) for name, shape in self.output_specs.items()},
            "vstreams": self.vstream,
        }

    def infer(self, arrays: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict]:
        feed: dict[str, np.ndarray] = {}
        for name, shape in self.input_specs.items():
            value = np.asarray(arrays.get(name))
            if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError(f"invalid {name}: shape={value.shape}, dtype={value.dtype}")
            feed[self.vstream[name]] = np.ascontiguousarray(value[None, ...])
        start = time.perf_counter_ns()
        raw = self.pipeline.infer(feed)
        infer_ms = (time.perf_counter_ns() - start) / 1e6
        outputs: dict[str, np.ndarray] = {}
        for name, shape in self.output_specs.items():
            value = np.asarray(raw[self.vstream[name]], dtype=np.float32)
            if value.shape == (1, *shape):
                value = value[0]
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"invalid Hailo output {name}: {value.shape}")
            outputs[name] = np.ascontiguousarray(value)
        self.calls += 1
        extra = {}
        for attr in ("get_hw_time", "get_total_time"):
            try:
                extra[attr] = float(getattr(self.pipeline, attr)())
            except Exception:
                pass
        return outputs, {"infer_wall_ms": infer_ms, "call": self.calls, **extra}

    def close(self) -> None:
        try:
            try:
                self.activation.__exit__(None, None, None)
            finally:
                self.pipeline.__exit__(None, None, None)
        finally:
            self.vdevice.release()


def serve(manifest: Path) -> int:
    engine = Engine(manifest)
    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    try:
        while True:
            try:
                op, _metadata, arrays = unpack(read_frame(stdin))
            except EOFError:
                return 0
            try:
                if op == "hello":
                    response = pack("hello", metadata=engine.hello())
                elif op == "infer":
                    outputs, timing = engine.infer(arrays)
                    response = pack("result", metadata=timing, arrays=outputs)
                elif op == "shutdown":
                    write_frame(stdout, pack("shutdown", metadata={"calls": engine.calls}))
                    return 0
                else:
                    raise ValueError(f"unknown op: {op}")
            except Exception as exc:
                response = pack("error", metadata={"error": f"{type(exc).__name__}: {exc}"})
            write_frame(stdout, response)
    finally:
        engine.close()


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
