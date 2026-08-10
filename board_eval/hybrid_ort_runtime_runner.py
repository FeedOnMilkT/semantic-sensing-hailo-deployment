#!/usr/bin/env python3
"""Pi-side mixed HailoRT/ORT runtime for bounded bottleneck fallback tests.

The manifest selects, per module, either the frozen C1 HEF or the matching
FP32 ONNX graph.  Entropy coding remains the same C++ rANS implementation used
by the all-Hailo full-module runtime.  The process is intentionally serial and
records board temperature around every request; it is a correctness/latency
probe, not a throughput runner.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

# The ORT virtualenv is isolated, while the board HailoRT bindings are supplied
# by the OS image.  Append (rather than prepend) the system dist-packages so the
# pinned ORT/numpy wheels remain authoritative.
SYSTEM_DIST_PACKAGES = Path("/usr/lib/python3/dist-packages")
if SYSTEM_DIST_PACKAGES.is_dir() and str(SYSTEM_DIST_PACKAGES) not in sys.path:
    sys.path.append(str(SYSTEM_DIST_PACKAGES))

from fullmodule_runtime_runner import (  # noqa: E402
    SEQUENCE,
    FullRuntime,
    ModuleEngine,
    RuntimeCodec,
    as_hwc,
    pack,
    read_frame,
    unpack,
    write_frame,
)


def cpu_temperature_c() -> float | None:
    path = Path("/sys/class/thermal/thermal_zone0/temp")
    try:
        return float(path.read_text(encoding="utf-8").strip()) / 1000.0
    except (FileNotFoundError, ValueError):
        return None


def cpu_frequency_khz() -> dict[str, int]:
    result: dict[str, int] = {}
    for path in sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_cur_freq")):
        try:
            result[path.parts[-3]] = int(path.read_text(encoding="utf-8").strip())
        except (FileNotFoundError, ValueError):
            continue
    return result


class OrtModuleEngine:
    """One serial FP32/FP16 ORT module with explicit NCHW/HWC conversion."""

    def __init__(self, root: Path, spec: dict, threads: int):
        self.name = str(spec["name"])
        self.input_names = list(spec["input_names"])
        self.output_names = list(spec["output_names"])
        self.input_shapes = [list(shape) for shape in spec["in_shapes_nchw"]]
        self.output_shapes = [list(shape) for shape in spec["out_shapes_nchw"]]
        self.onnx_path = root / spec["onnx_filename"]
        self.engine_kind = str(spec["engine"])
        if self.engine_kind not in {"ort_cpu_fp32", "ort_cpu_fp16"}:
            raise ValueError(f"{self.name}: invalid ORT engine {self.engine_kind!r}")
        if not self.onnx_path.is_file():
            raise FileNotFoundError(self.onnx_path)
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3
        start = time.perf_counter_ns()
        self.session = ort.InferenceSession(
            str(self.onnx_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.init_ms = (time.perf_counter_ns() - start) / 1e6
        if [item.name for item in self.session.get_inputs()] != self.input_names:
            raise ValueError(f"{self.name}: ONNX input names differ from frozen manifest")
        if [item.name for item in self.session.get_outputs()] != self.output_names:
            raise ValueError(f"{self.name}: ONNX output names differ from frozen manifest")
        dtype_by_ort_type = {"tensor(float)": np.float32, "tensor(float16)": np.float16}
        try:
            self.input_dtypes = [dtype_by_ort_type[item.type] for item in self.session.get_inputs()]
            self.output_dtypes = [dtype_by_ort_type[item.type] for item in self.session.get_outputs()]
        except KeyError as exc:
            raise ValueError(f"{self.name}: unsupported ORT tensor type {exc.args[0]!r}") from exc
        expected_dtype = np.float16 if self.engine_kind == "ort_cpu_fp16" else np.float32
        if any(dtype != expected_dtype for dtype in (*self.input_dtypes, *self.output_dtypes)):
            raise ValueError(
                f"{self.name}: {self.engine_kind} graph did not preserve its declared I/O precision"
            )

    def infer(self, arrays: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, float | str]]:
        feed: dict[str, np.ndarray] = {}
        for logical, shape, dtype in zip(
            self.input_names, self.input_shapes, self.input_dtypes, strict=True
        ):
            value_hwc = as_hwc(arrays.get(logical), shape, f"{self.name}:{logical}")
            feed[logical] = np.ascontiguousarray(value_hwc.transpose(2, 0, 1)[None], dtype=dtype)
        start = time.perf_counter_ns()
        raw = self.session.run(self.output_names, feed)
        elapsed_ms = (time.perf_counter_ns() - start) / 1e6
        result: dict[str, np.ndarray] = {}
        for logical, shape, value in zip(self.output_names, self.output_shapes, raw, strict=True):
            value = np.asarray(value, dtype=np.float32)
            if value.shape != tuple(shape) or not np.isfinite(value).all():
                raise ValueError(
                    f"{self.name}:{logical} invalid ORT output shape/dtype: "
                    f"got={value.shape}, expected={tuple(shape)}"
                )
            result[logical] = np.ascontiguousarray(value[0].transpose(1, 2, 0))
        return result, {"engine": self.engine_kind, "infer_wall_ms": elapsed_ms}

    def close(self) -> None:
        self.session = None


class HybridRuntime(FullRuntime):
    """Full semantic chain with a manifest-selected engine for each module."""

    def __init__(self, manifest_path: Path):
        from hailo_platform import VDevice

        self.root = manifest_path.parent
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("kind") != "semantic_sensing_pi_hybrid_ort_runtime":
            raise ValueError("not a hybrid ORT runtime manifest")
        if tuple(self.manifest.get("sequence", ())) != SEQUENCE:
            raise ValueError("runtime manifest sequence is not the frozen full-module order")
        self.arm = str(self.manifest["arm"])
        self.ort_threads = int(self.manifest["ort_threads"])
        if self.ort_threads <= 0:
            raise ValueError("ort_threads must be positive")
        thermal = self.manifest.get("thermal_guard", {})
        self.max_start_temp_c = float(thermal.get("max_start_temp_c", 70.0))
        self.resume_temp_c = float(thermal.get("resume_temp_c", 60.0))
        self.poll_seconds = float(thermal.get("poll_seconds", 2.0))
        self.max_wait_seconds = float(thermal.get("max_wait_seconds", 600.0))
        if not self.resume_temp_c < self.max_start_temp_c:
            raise ValueError("thermal resume threshold must be lower than start threshold")

        codec_spec = self.manifest["codec"]
        self.codec = RuntimeCodec(self.root / codec_spec["tables_filename"])
        self.vdevice = VDevice()
        self.modules: dict[str, ModuleEngine | OrtModuleEngine] = {}
        self.engine_kinds: dict[str, str] = {}
        try:
            for spec in self.manifest["modules"]:
                engine_kind = str(spec["engine"])
                if engine_kind in {"ort_cpu_fp32", "ort_cpu_fp16"}:
                    engine = OrtModuleEngine(self.root, spec, self.ort_threads)
                elif engine_kind == "hailort_float32_vstream":
                    engine = ModuleEngine(self.vdevice, self.root, spec)
                else:
                    raise ValueError(f"{spec.get('name')}: unknown engine {engine_kind!r}")
                if engine.name in self.modules:
                    raise ValueError(f"duplicate module {engine.name}")
                self.modules[engine.name] = engine
                self.engine_kinds[engine.name] = engine_kind
            if tuple(self.modules) != SEQUENCE:
                raise ValueError(f"runtime modules differ from required sequence: {tuple(self.modules)}")
        except BaseException:
            self.close()
            raise
        self.calls = 0

    def _thermal_guard(self) -> dict[str, float | None]:
        initial = cpu_temperature_c()
        waited = 0.0
        if initial is not None and initial > self.max_start_temp_c:
            start = time.monotonic()
            while True:
                current = cpu_temperature_c()
                if current is None or current <= self.resume_temp_c:
                    break
                waited = time.monotonic() - start
                if waited >= self.max_wait_seconds:
                    raise TimeoutError(
                        f"thermal guard timed out: current={current:.2f}C "
                        f"resume={self.resume_temp_c:.2f}C"
                    )
                time.sleep(self.poll_seconds)
        return {
            "initial_c": initial,
            "start_c": cpu_temperature_c(),
            "cooldown_wait_ms": waited * 1000.0,
        }

    def infer(self, image: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
        guard = self._thermal_guard()
        outputs, timing = super().infer(image)
        timing["thermal"] = {
            **guard,
            "end_c": cpu_temperature_c(),
            "cpu_frequency_khz": cpu_frequency_khz(),
        }
        timing["arm"] = self.arm
        return outputs, timing

    def hello(self) -> dict:
        return {
            "pid": os.getpid(),
            "arm": self.arm,
            "sequence": list(SEQUENCE),
            "calls": self.calls,
            "engines": self.engine_kinds,
            "ort": {
                "version": ort.__version__,
                "threads": self.ort_threads,
                "providers": ort.get_available_providers(),
                "session_init_ms": {
                    name: engine.init_ms
                    for name, engine in self.modules.items()
                    if isinstance(engine, OrtModuleEngine)
                },
                "session_io_types": {
                    name: {
                        "inputs": [item.type for item in engine.session.get_inputs()],
                        "outputs": [item.type for item in engine.session.get_outputs()],
                    }
                    for name, engine in self.modules.items()
                    if isinstance(engine, OrtModuleEngine)
                },
            },
            "thermal_guard": {
                "max_start_temp_c": self.max_start_temp_c,
                "resume_temp_c": self.resume_temp_c,
                "max_wait_seconds": self.max_wait_seconds,
            },
            "artifacts": {
                name: str(engine.onnx_path if isinstance(engine, OrtModuleEngine) else engine.hef_path)
                for name, engine in self.modules.items()
            },
            "codec": {
                "z_cdf_rows": int(self.codec.z_cdf.shape[0]),
                "y_cdf_rows": int(self.codec.y_cdf.shape[0]),
                "scale_table_entries": int(self.codec.scale_table.size),
            },
        }


def serve(manifest: Path) -> int:
    runtime = HybridRuntime(manifest)
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
                            raise ValueError("cosine_indices must be a non-empty 1-D vector")
                        flat = outputs["seg_logits"].reshape(-1)
                        if sample_indices.min() < 0 or sample_indices.max() >= flat.size:
                            raise ValueError("cosine_indices out of logit range")
                        returned["logit_samples"] = np.ascontiguousarray(
                            flat[sample_indices], dtype=np.float32
                        )
                        timing["cosine_sample_count"] = int(sample_indices.size)
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
