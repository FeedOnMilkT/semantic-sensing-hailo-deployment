#!/usr/bin/env python3
"""Benchmark one resident HEF on a Hailo-8 board without host transport.

This intentionally uses zero FP32 tensors with the HEF's own vstream shapes:
the result is a target-side HailoRT/PCIe wall-time measurement, not a semantic
evaluation and not a sender-to-receiver latency.  It is suitable for comparing
the deployable modules before the complete serial runtime exists.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def summarize(samples: list[float]) -> dict[str, float | int]:
    values = np.asarray(samples, dtype=np.float64)
    if values.size == 0:
        raise ValueError("no timed iterations")
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "min": float(values.min()),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }


def benchmark(hef_path: Path, warmup: int, iterations: int) -> dict[str, object]:
    from hailo_platform import (ConfigureParams, FormatType, HailoStreamInterface,
                                HEF, InferVStreams, InputVStreamParams,
                                OutputVStreamParams, VDevice)

    if not hef_path.is_file():
        raise FileNotFoundError(hef_path)
    start = time.perf_counter_ns()
    hef = HEF(str(hef_path))
    inputs = {item.name: tuple(item.shape) for item in hef.get_input_vstream_infos()}
    outputs = {item.name: tuple(item.shape) for item in hef.get_output_vstream_infos()}
    if not inputs or not outputs:
        raise ValueError(f"HEF has empty IO contract: inputs={inputs}, outputs={outputs}")
    vdevice = VDevice()
    pipeline = None
    activation = None
    try:
        configure = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        groups = vdevice.configure(hef, configure)
        if len(groups) != 1:
            raise ValueError(f"expected one network group, got {len(groups)}")
        network_group = groups[0]
        in_params = InputVStreamParams.make_from_network_group(
            network_group, quantized=False, format_type=FormatType.FLOAT32)
        out_params = OutputVStreamParams.make_from_network_group(
            network_group, quantized=False, format_type=FormatType.FLOAT32)
        pipeline = InferVStreams(network_group, in_params, out_params)
        pipeline.__enter__()
        activation = network_group.activate()
        activation.__enter__()
        init_ms = (time.perf_counter_ns() - start) / 1e6
        feed = {name: np.zeros((1, *shape), dtype=np.float32) for name, shape in inputs.items()}
        for _ in range(warmup):
            pipeline.infer(feed)
        samples: list[float] = []
        for _ in range(iterations):
            begin = time.perf_counter_ns()
            pipeline.infer(feed)
            samples.append((time.perf_counter_ns() - begin) / 1e6)
        return {
            "status": "PASS",
            "scope": "Pi HailoRT InferVStreams wall time; resident HEF; excludes SSH and model load",
            "hef": str(hef_path),
            "init_ms": init_ms,
            "warmup": warmup,
            "inputs": {name: list(shape) for name, shape in inputs.items()},
            "outputs": {name: list(shape) for name, shape in outputs.items()},
            "infer_wall_ms": summarize(samples),
        }
    finally:
        if activation is not None:
            activation.__exit__(None, None, None)
        if pipeline is not None:
            pipeline.__exit__(None, None, None)
        vdevice.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hef", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be >=0 and iterations must be >0")
    print(json.dumps(benchmark(args.hef, args.warmup, args.iterations), sort_keys=True))


if __name__ == "__main__":
    main()
