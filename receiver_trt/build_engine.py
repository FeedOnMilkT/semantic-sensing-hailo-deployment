#!/usr/bin/env python3
"""Build the TensorRT engine for the receiver-side ONNX graph.

The engine is bound to the TensorRT version + GPU architecture it was built on,
so it MUST be built on the machine that will run it ("build in place on the
receiver box").  This script therefore records the TRT version, the CUDA
runtime version and the GPU name in a sidecar JSON next to the engine.

Precision
---------
``--precision fp32``  : everything FP32.
``--precision fp16``  : FP16 where allowed, but the *numerically sensitive*
                        layers are pinned to FP32.  The sensitive set is:
                          * the IGDN synthesis blocks (sqrt of a conv sum -> the
                            divisive norm amplifies activation errors), and
                          * the two cross-attention softmax blocks (they feed the
                            K/V memory, so an fp16 error there propagates into
                            every high-branch pixel).
                        ``--fp32-layers`` allows overriding/adding to that set
                        (substring match against the layer names).
                        ``--fp16-all`` forces full FP16 (for comparison).

TensorRT version
----------------
This script targets **TensorRT 10.x** (tested with 10.16.1.11, cu12).
TensorRT 11.x removed ``BuilderFlag.FP16``/``INT8`` (strongly-typed networks
only), so the weak-typed FP16-flag path used here does not exist there.

Timing cache
------------
A serialised timing cache is written next to the engine and re-used on the next
build of the same ONNX (it only stores the *profiling* results of layers, not
the weights), which cuts the build time noticeably.

Usage (TRT venv):
    /root/autodl-tmp/venvs/trt/bin/python receiver_trt/build_engine.py \
        --onnx /root/autodl-tmp/deploy_artifacts/package/receiver/receiver.onnx \
        --precision fp16 --workspace 4096 \
        --out-dir /root/autodl-tmp/deploy_artifacts/package/receiver

Equivalent trtexec command (printed by this script):

    trtexec --onnx=receiver.onnx --saveEngine=receiver_fp16.engine \
            --fp16 --workspace=4096 --buildOnly --verbose
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time

# --------------------------------------------------------------------------- #
# layer-name substrings pinned to FP32 in the fp16 mode (numerically sensitive)
#
# The ONNX export names the nodes with the torch module path, e.g.
#   /synth/g_s_trunk/g_s_trunk.0/g_s_trunk.0.1/Sqrt
#   /synth/head_h/head_h.1/head_h.1.1/Sqrt
#   /layer4/attn_h/Softmax
#   /layer5/attn_l/Softmax
# so the patterns below are matched as plain substrings.
# --------------------------------------------------------------------------- #
DEFAULT_FP32_PATTERNS = (
    "/synth/g_s_trunk",     # IGDN synthesis (divisive normalisation)
    "/synth/head_h",        # IGDN + up-sampled mixing
    "attn_h",               # cross-attention softmax over the 12x12 memory
    "attn_l",               # external attention (double normalisation)
)

# layers that must never be re-typed: constants / shape tensors / casts
_SKIP_PRECISION_TYPES = {
    "LayerType.CONSTANT",
    "LayerType.SHAPE",
    "LayerType.IDENTITY",
}


def trt_import():
    try:
        import tensorrt as trt
    except ImportError as e:                                   # noqa: BLE001
        raise SystemExit(
            "TensorRT is not importable in this interpreter.  Install it in a "
            "dedicated venv, e.g.\n"
            "  python3 -m venv /root/autodl-tmp/venvs/trt\n"
            "  /root/autodl-tmp/venvs/trt/bin/pip install "
            "--cache-dir /root/autodl-tmp/pipcache tensorrt-cu12==10.16.1.11 "
            "cuda-python==13.0.0 numpy==1.26.4"
        ) from e
    return trt


# --------------------------------------------------------------------------- #
# environment / GPU introspection
# --------------------------------------------------------------------------- #
def _nvidia_smi(query):
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20)
        if out.returncode == 0:
            return out.stdout.strip().splitlines()[0].strip()
    except Exception:                                          # noqa: BLE001
        pass
    return None


def cuda_runtime_version():
    """CUDA runtime version string, or None."""
    try:
        from cuda.bindings import runtime as cudart
        err, ver = cudart.cudaRuntimeGetVersion()
        if int(err) == 0:
            return f"{int(ver) // 1000}.{(int(ver) % 1000) // 10}"
    except Exception:                                          # noqa: BLE001
        pass
    return None


def gpu_info(trt):
    """GPU name + compute capability + driver version."""
    info = {}
    try:
        from cuda.bindings import driver as cuda_driver
        err, _ = cuda_driver.cuInit(0)
        if int(err) == 0:
            err, dev = cuda_driver.cuDeviceGet(0)
            if int(err) == 0:
                err, name = cuda_driver.cuDeviceGetName(128, dev)
                if int(err) == 0:
                    info["gpu_name"] = name.decode() if isinstance(name, bytes) else str(name)
                try:
                    major, minor = cuda_driver.cuDeviceComputeCapability(dev)
                    if int(major) == 0:
                        info["compute_capability"] = f"{int(major)}.{int(minor)}"
                except Exception:                              # noqa: BLE001
                    pass
                try:
                    err, mem = cuda_driver.cuDeviceTotalMem_v2(dev)
                    if int(err) == 0:
                        info["total_memory_bytes"] = int(mem)
                except Exception:                              # noqa: BLE001
                    pass
    except Exception:                                          # noqa: BLE001
        pass
    if "gpu_name" not in info:
        smi = _nvidia_smi("name")
        if smi:
            info["gpu_name"] = smi
    if "compute_capability" not in info:
        smi = _nvidia_smi("compute_cap")
        if smi:
            info["compute_capability"] = smi.replace(".", ".", 1)
    drv = _nvidia_smi("driver_version")
    if drv:
        info["driver_version"] = drv
    try:
        info["platform_has_fast_fp16"] = bool(trt.Builder(trt.Logger(trt.Logger.ERROR))
                                               .platform_has_fast_fp16)
    except Exception:                                          # noqa: BLE001
        pass
    return info


# --------------------------------------------------------------------------- #
# engine build
# --------------------------------------------------------------------------- #
def _is_static(network):
    for i in range(network.num_inputs):
        shape = tuple(network.get_input(i).shape)
        if any(d < 0 for d in shape):
            return False
    return True


def _make_logger(trt, verbose, sink):
    """Build a TensorRT logger that also appends every message to ``sink``.

    TensorRT's ``ILogger`` is a pybind11 abstract base, so it can only be
    sub-classed once the module has been imported -- hence this factory.
    """

    class TeeLogger(trt.ILogger):
        def __init__(self):
            trt.ILogger.__init__(self)
            # INFO is always collected so the build log is useful; the
            # VERBOSE spam is filtered out in log().
            self.min_severity = trt.Logger.INFO

        def log(self, severity, msg):
            text = msg.decode("utf-8", "replace") if isinstance(msg, bytes) else str(msg)
            if verbose:
                print(f"[TRT][{severity}] {text}", flush=True)
            # The build log keeps everything except the VERBOSE chatter
            # (which is ~18k lines and drowns out the useful messages).
            if "VERBOSE" in str(severity) or "INTERNAL_ERROR" in str(severity):
                return
            sink.append(f"[TRT][{severity}] {text}")

    return TeeLogger()


def build_engine(trt, onnx_path, engine_path, precision, workspace_mib,
                 fp32_patterns, timing_cache_path, verbose=False, log_sink=None):
    log_sink = log_sink if log_sink is not None else []
    logger = _make_logger(trt, verbose, log_sink)
    builder = trt.Builder(logger)
    # TensorRT 10: an empty flag set == explicit batch (EXPLICIT_BATCH is the
    # default and its enum value is 0).  Passing STRONGLY_TYPED would disable
    # the FP16 builder flag, which is exactly what we need here.
    network = builder.create_network()
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise RuntimeError("ONNX parse failed:\n" + "\n".join(errs))

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_mib) << 20)

    # ---- timing cache ------------------------------------------------------
    # TRT 10 only exposes a cache through ITimingCache objects, so we ALWAYS
    # create one (an empty buffer = fresh cache), attach it and serialise it
    # after the build.  A previously saved cache is loaded into it first.
    timing_cache = config.create_timing_cache(b"")
    if timing_cache_path and os.path.isfile(timing_cache_path):
        with open(timing_cache_path, "rb") as f:
            raw = f.read()
        if raw:
            try:
                timing_cache = config.create_timing_cache(raw)
                ok = config.set_timing_cache(timing_cache, ignore_mismatch=False)
                print(f"[build] timing cache loaded ({len(raw)} bytes, accepted={bool(ok)})",
                      flush=True)
                if not ok:
                    timing_cache = config.create_timing_cache(b"")
                    config.set_timing_cache(timing_cache, ignore_mismatch=True)
            except Exception as e:                            # noqa: BLE001
                print(f"[build] timing cache rejected ({e}); continuing", flush=True)
    else:
        config.set_timing_cache(timing_cache, ignore_mismatch=True)

    # ---- precision ---------------------------------------------------------
    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    else:
        config.clear_flag(trt.BuilderFlag.FP16)

    # ---- per-layer precision pinning --------------------------------------
    # NOTE: only *activation* layers can be re-typed.  CONSTANT layers carry
    # integer weights (Int64 shape tensors) and would fail validation with
    # "cannot use precision Float with weights of type Int64", so they are
    # skipped.  The same holds for shape-only layers.
    pinned = []
    n_pinned = 0
    if precision == "fp16" and fp32_patterns:
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            name = layer.name
            if str(layer.type) in _SKIP_PRECISION_TYPES:
                continue
            if not any(p in name for p in fp32_patterns):
                continue
            try:
                layer.reset_precision()
                layer.precision = trt.DataType.FLOAT
                n_pinned += 1
                pinned.append(name)
            except Exception as e:                        # noqa: BLE001
                print(f"[build] could not pin {name}: {e}", flush=True)
    elif precision == "fp16":
        print("[build] fp32-pattern list empty -> full FP16", flush=True)

    # ---- optimisation profile (only needed for dynamic shapes) --------------
    static = _is_static(network)
    if not static:
        profile = builder.create_optimization_profile()
        profile.set_shape("y_hat", (1, 192, 16, 16), (1, 192, 16, 16))
        profile.set_shape("logits", (1, 150, 128, 128), (1, 150, 128, 128))
        config.add_optimization_profile(profile)

    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    build_s = time.time() - t0
    if serialized is None:
        raise RuntimeError("TensorRT build failed (builder returned None)")

    payload = bytes(serialized)
    with open(engine_path, "wb") as f:
        f.write(payload)

    # ---- save the timing cache for the next build --------------------------
    cache_bytes = 0
    if timing_cache_path:
        try:
            tc = config.get_timing_cache()
            raw = b""
            if tc is not None:
                hm = tc.serialize()          # -> IHostMemory
                if hm is not None:
                    raw = bytes(memoryview(hm))
            if raw:
                os.makedirs(os.path.dirname(os.path.abspath(timing_cache_path)), exist_ok=True)
                with open(timing_cache_path, "wb") as f:
                    f.write(raw)
                cache_bytes = len(raw)
                print(f"[build] timing cache -> {timing_cache_path} ({cache_bytes} bytes)",
                      flush=True)
            else:
                print("[build] timing cache is empty (nothing to save)", flush=True)
        except Exception as e:                                # noqa: BLE001
            print(f"[build] could not save the timing cache: {e}", flush=True)

    return {"build_seconds": build_s, "pinned_fp32_layers": n_pinned,
            "pinned_layer_names": pinned[:64], "engine_bytes": len(payload),
            "timing_cache_bytes": cache_bytes, "static_shapes": static,
            "num_layers": network.num_layers}


def trtexec_example(onnx_path, engine_path, precision, workspace_mib,
                    timing_cache_path=None):
    flags = ""
    if precision == "fp16":
        flags = " --fp16"
    cache = f" --timingCacheFile={timing_cache_path}" if timing_cache_path else ""
    return (f"trtexec --onnx={onnx_path} --saveEngine={engine_path}{flags} "
            f"--workspace={workspace_mib} --buildOnly --verbose{cache}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", default="/root/autodl-tmp/deploy_artifacts/package/receiver/receiver.onnx")
    ap.add_argument("--out-dir", default="/root/autodl-tmp/deploy_artifacts/package/receiver")
    ap.add_argument("--precision", choices=("fp32", "fp16"), default="fp16")
    ap.add_argument("--workspace", type=int, default=4096, help="TRT workspace MiB")
    ap.add_argument("--fp32-layers", default=",".join(DEFAULT_FP32_PATTERNS),
                    help="comma-separated substrings pinned to FP32 in fp16 mode "
                         "(empty string = pin nothing)")
    ap.add_argument("--fp16-all", action="store_true",
                    help="ignore --fp32-layers and build a full-FP16 engine")
    ap.add_argument("--no-timing-cache", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--name", default=None, help="engine base name (default: receiver_<prec>)")
    a = ap.parse_args(argv)

    trt = trt_import()
    os.makedirs(a.out_dir, exist_ok=True)
    name = a.name or f"receiver_{a.precision}"
    engine_path = os.path.join(a.out_dir, f"{name}.engine")
    log_path = os.path.join(a.out_dir, f"{name}.build.log")
    cache_path = None if a.no_timing_cache else os.path.join(a.out_dir, f"{name}.timing.cache")
    patterns = [] if a.fp16_all else [p for p in (s.strip() for s in a.fp32_layers.split(",")) if p]

    info = {
        "tensorrt_version": trt.__version__,
        "trt_module": trt.__file__,
        "python": platform.python_version(),
        "precision": a.precision,
        "fp16_all": bool(a.fp16_all),
        "workspace_mib": a.workspace,
        "fp32_patterns": patterns,
        "onnx": os.path.abspath(a.onnx),
        "engine": engine_path,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    info.update(gpu_info(trt))
    info["cuda_runtime_version"] = cuda_runtime_version()
    try:
        import cuda.bindings
        info["cuda_bindings"] = "present"
    except Exception:                                          # noqa: BLE001
        info["cuda_bindings"] = "absent"

    print(f"[build] TRT {trt.__version__} | precision={a.precision} | "
          f"workspace={a.workspace} MiB", flush=True)
    print(f"[build] gpu: {info.get('gpu_name')} cc={info.get('compute_capability')} "
          f"driver={info.get('driver_version')} cuda={info.get('cuda_runtime_version')}",
          flush=True)
    print(f"[build] trtexec equivalent:\n    "
          f"{trtexec_example(a.onnx, engine_path, a.precision, a.workspace, cache_path)}",
          flush=True)

    log_sink = []
    stats = build_engine(trt, a.onnx, engine_path, a.precision, a.workspace,
                         patterns, cache_path, verbose=a.verbose, log_sink=log_sink)
    info.update(stats)
    info["trtexec_equivalent"] = trtexec_example(a.onnx, engine_path, a.precision,
                                                 a.workspace, cache_path)

    meta_path = os.path.join(a.out_dir, f"{name}.json")
    with open(meta_path, "w") as f:
        json.dump(info, f, indent=1)
    # The build log contains the metadata JSON followed by every message the
    # TensorRT logger emitted during the build.
    with open(log_path, "w") as f:
        f.write(json.dumps(info, indent=1) + "\n")
        if log_sink:
            f.write("\n".join(log_sink) + "\n")
    print(f"[build] engine  -> {engine_path} ({stats['engine_bytes'] / 1e6:.1f} MB) "
          f"in {stats['build_seconds']:.1f}s", flush=True)
    print(f"[build] pinned FP32 layers: {stats['pinned_fp32_layers']}", flush=True)
    print(f"[build] metadata -> {meta_path}", flush=True)
    print(f"[build] build log -> {log_path} ({len(log_sink)} TRT messages)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
