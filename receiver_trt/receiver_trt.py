#!/usr/bin/env python3
"""Receiver-side runtime: RTFF bitstream -> TensorRT logits -> segmentation.

The receiver is the mirror image of the C++ sender:

    *.rtff (RTFF container)
      -> libsscodec.so :: ss_decode()          [ctypes]  -> y_hat [1,192,16,16]
      -> TensorRT engine (receiver_fp32/fp16.engine)     -> logits [1,150,128,128]
      -> bilinear upsample to 512x512 (align_corners=False)
      -> argmax -> predicted label map [512,512]

Why the decode happens in C++ and not in the graph
--------------------------------------------------
The rANS payload is coded against per-element CDF indexes that the sender
derived from the deterministic h_s hyper-synthesis.  Recomputing h_s in
PyTorch/ONNX would NOT reproduce the sender's indexes (the Hailo int8 h_s and
the FP32 h_s disagree on ~17% of the elements) and the rANS decoder would
desynchronise.  So the receiver MUST use the same C++ h_s -- which is exactly
what ``ss_decode()`` does.  The TensorRT graph therefore starts at ``y_hat``.

The decoded ``y_hat`` is ``symbols + means`` (already dequantised), i.e. the
same tensor the sender's own receiver path reconstructs with
``torch.round(y - means) + means``.

Upsampling / argmax convention
------------------------------
The evaluation code in ``/root/Semantic-sensing/src/utils/metrics.py``
(``SegMeter.update``) does:

    * bilinear ``F.interpolate(logits, size=gt.shape[-2:], mode="bilinear",
      align_corners=False)`` when the logit resolution differs from the GT,
    * ``pred = logits.argmax(1)``,
    * the confusion matrix is built over the pixels whose GT label is not
      ``ignore_index`` (255).

The exported ONNX already ends in a ``Resize`` to 128x128, so this module
resizes 128x128 -> 512x512 with the same bilinear / align_corners=False rule
before the argmax.  (A 128->512 upscale followed by a 512->128 upscale is
*not* the same as a direct 256->512, but it is the deployed graph, and the
metric code resizes whatever resolution the head emits to the GT size.)

Usage
-----
    # single bitstream -> logits + label map
    python receiver_trt/receiver_trt.py \
        --engine /root/autodl-tmp/deploy_artifacts/package/receiver/receiver_fp16.engine \
        --bitstream /root/autodl-tmp/deploy_artifacts/package/sender_out/bitstreams/ADE_val_00000001.rtff \
        --out-npy /root/autodl-tmp/tmp/pred01.npy

    # as a library
    from receiver_trt import ReceiverPipeline
    pipe = ReceiverPipeline(engine_path)
    out = pipe.infer_bitstream("....rtff")
    out.logits        # (1,150,128,128) float32
    out.label         # (512,512) int64
    out.decode_seconds, out.infer_seconds
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from postprocess import make_postprocess, numpy_argmax, numpy_upsample  # noqa: E402

# --------------------------------------------------------------------------- #
# defaults
# --------------------------------------------------------------------------- #
DEFAULT_PACKAGE = "/root/autodl-tmp/deploy_artifacts/package"
DEFAULT_SENDER = os.path.join(DEFAULT_PACKAGE, "sender_out")
DEFAULT_RECEIVER = os.path.join(DEFAULT_PACKAGE, "receiver")
DEFAULT_TABLES = os.path.join(DEFAULT_SENDER, "codec_tables.bin")
DEFAULT_LIB = os.path.join(DEFAULT_SENDER, "libsscodec.so")
DEFAULT_ENGINE = os.path.join(DEFAULT_RECEIVER, "receiver_fp16.engine")

NUM_CLASSES = 150
IGNORE_INDEX = 255
GT_SIZE = 512
LOGIT_SIZE = 128


# --------------------------------------------------------------------------- #
# ctypes binding for libsscodec.so
# --------------------------------------------------------------------------- #
class SsCodecError(RuntimeError):
    """Raised when the C++ codec returns a negative status code."""


class SsCodec:
    """ctypes wrapper around the shared codec (``cpp/include/sscodec.h``).

    Only the entry points the receiver needs are bound:

    ``ss_open``      -- open ``codec_tables.bin``
    ``ss_decode``    -- RTFF container -> ``y_hat`` (symbols + means)
    ``ss_close``     -- release the handle
    ``ss_version``   -- library version string
    """

    def __init__(self, lib_path=DEFAULT_LIB, tables_path=DEFAULT_TABLES):
        self.lib_path = os.path.abspath(lib_path)
        self.tables_path = os.path.abspath(tables_path)
        if not os.path.isfile(self.lib_path):
            raise FileNotFoundError(f"libsscodec.so not found: {self.lib_path}")
        if not os.path.isfile(self.tables_path):
            raise FileNotFoundError(f"codec_tables.bin not found: {self.tables_path}")

        self._lib = ctypes.CDLL(self.lib_path)
        self._lib.ss_open.argtypes = [ctypes.c_char_p]
        self._lib.ss_open.restype = ctypes.c_void_p
        self._lib.ss_close.argtypes = [ctypes.c_void_p]
        self._lib.ss_close.restype = None
        self._lib.ss_decode.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
        ]
        self._lib.ss_decode.restype = ctypes.c_int
        self._lib.ss_version.argtypes = []
        self._lib.ss_version.restype = ctypes.c_char_p
        # z-stream only decode + the deterministic h_s (means / scales / indexes)
        self._lib.ss_decode_z.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
        ]
        self._lib.ss_decode_z.restype = ctypes.c_int
        self._lib.ss_hyper_synthesis.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ]
        self._lib.ss_hyper_synthesis.restype = ctypes.c_int

        self._handle = self._lib.ss_open(self.tables_path.encode())
        if not self._handle:
            raise SsCodecError(
                f"ss_open('{self.tables_path}') failed -- bad tables file?")

    # -- lifecycle ----------------------------------------------------------
    def close(self):
        h = getattr(self, "_handle", None)
        if h:
            self._lib.ss_close(h)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:                                      # noqa: BLE001
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @property
    def version(self):
        v = self._lib.ss_version()
        return v.decode() if v else "unknown"

    # -- the actual decode --------------------------------------------------
    def decode(self, payload: bytes, cap_floats: int = 192 * 16 * 16):
        """Decode one RTFF container.

        Returns ``(y_hat, dims)`` where ``y_hat`` is a float32 array of shape
        ``(1, 192, y_h, y_w)`` and ``dims`` is
        ``[z_h, z_w, y_h, y_w, fh_h, fh_w, fl_h, fl_w]``.
        """
        y_hat = np.empty(cap_floats, dtype=np.float32)
        dims = (ctypes.c_int * 8)()
        src = ctypes.create_string_buffer(payload, len(payload))
        rc = self._lib.ss_decode(self._handle,
                                 ctypes.cast(src, ctypes.c_void_p), len(payload),
                                 y_hat.ctypes.data_as(ctypes.c_void_p),
                                 cap_floats, dims)
        if rc != 0:
            raise SsCodecError(f"ss_decode failed (rc={rc})")
        zh, zw, yh, yw = int(dims[0]), int(dims[1]), int(dims[2]), int(dims[3])
        n = 192 * yh * yw
        return y_hat[:n].reshape(1, 192, yh, yw), [int(d) for d in dims]

    def decode_file(self, path, cap_floats: int = 192 * 16 * 16):
        with open(path, "rb") as f:
            return self.decode(f.read(), cap_floats=cap_floats)

    # -- the internals (used by the self-test) -------------------------------
    def hyper_synthesis(self, z_hat: np.ndarray):
        """The deterministic C++ h_s: ``z_hat -> (means, scales, indexes)``.

        ``z_hat`` is float32 NCHW ``(1,128,8,8)``.  This is the SAME function
        the sender used to compute the CDF indexes the y-stream was coded
        against.
        """
        z = np.ascontiguousarray(np.asarray(z_hat, np.float32))
        if z.ndim != 4 or z.shape[0] != 1:
            raise ValueError(f"expected (1,C,H,W), got {z.shape}")
        _, ch, zh, zw = z.shape
        n = 192 * zh * 2 * zw * 2
        means = np.empty(n, dtype=np.float32)
        scales = np.empty(n, dtype=np.float32)
        indexes = np.empty(n, dtype=np.int32)
        rc = self._lib.ss_hyper_synthesis(
            self._handle, z.ctypes.data_as(ctypes.c_void_p), zh, zw,
            means.ctypes.data_as(ctypes.c_void_p),
            scales.ctypes.data_as(ctypes.c_void_p),
            indexes.ctypes.data_as(ctypes.c_void_p))
        if rc != 0:
            raise SsCodecError(f"ss_hyper_synthesis failed (rc={rc})")
        yh, yw = zh * 2, zw * 2
        return (means.reshape(1, 192, yh, yw),
                scales.reshape(1, 192, yh, yw),
                indexes.reshape(1, 192, yh, yw))

    def decode_z_and_hs(self, payload: bytes):
        """Decode the z-stream and recompute the deterministic h_s.

        Returns ``(z_hat, means, scales, indexes)``.  This is the codec's own
        view of the latent: the z-stream gives ``z_hat``, the C++ h_s turns it
        into the per-element ``means`` / ``scales`` / CDF ``indexes`` that the
        y-stream was coded against.
        """
        z_hat = np.empty(128 * 8 * 8, dtype=np.float32)
        dims = (ctypes.c_int * 8)()
        src = ctypes.create_string_buffer(payload, len(payload))
        rc = self._lib.ss_decode_z(self._handle,
                                   ctypes.cast(src, ctypes.c_void_p), len(payload),
                                   z_hat.ctypes.data_as(ctypes.c_void_p),
                                   z_hat.size, dims)
        if rc != 0:
            raise SsCodecError(f"ss_decode_z failed (rc={rc})")
        zh, zw = int(dims[0]), int(dims[1])
        z = z_hat[:128 * zh * zw].reshape(1, 128, zh, zw)

        n = 192 * zh * 2 * zw * 2
        means = np.empty(n, dtype=np.float32)
        scales = np.empty(n, dtype=np.float32)
        indexes = np.empty(n, dtype=np.int32)
        rc = self._lib.ss_hyper_synthesis(
            self._handle,
            np.ascontiguousarray(z).ctypes.data_as(ctypes.c_void_p), zh, zw,
            means.ctypes.data_as(ctypes.c_void_p),
            scales.ctypes.data_as(ctypes.c_void_p),
            indexes.ctypes.data_as(ctypes.c_void_p))
        if rc != 0:
            raise SsCodecError(f"ss_hyper_synthesis failed (rc={rc})")
        yh, yw = zh * 2, zw * 2
        return (z,
                means.reshape(1, 192, yh, yw),
                scales.reshape(1, 192, yh, yw),
                indexes.reshape(1, 192, yh, yw))


# --------------------------------------------------------------------------- #
# TensorRT engine wrapper
# --------------------------------------------------------------------------- #
def _trt():
    try:
        import tensorrt as trt
    except ImportError as e:                                   # noqa: BLE001
        raise SystemExit(
            "TensorRT is not importable -- run this script with the TRT venv:\n"
            "  /root/autodl-tmp/venvs/trt/bin/python receiver_trt/receiver_trt.py ..."
        ) from e
    return trt


class TrtEngine:
    """Thin wrapper around a serialised TensorRT engine (name-based I/O).

    Uses the TensorRT 10 name-based API (``set_tensor_address`` /
    ``execute_async_v3``) and ``cuda-python`` for the device buffers.
    """

    def __init__(self, engine_path, device=0):
        self.engine_path = os.path.abspath(engine_path)
        self._closed = False
        if not os.path.isfile(self.engine_path):
            raise FileNotFoundError(f"engine not found: {self.engine_path}")

        self.trt = _trt()
        from cuda.bindings import runtime as cudart

        self.cudart = cudart
        self.logger = self.trt.Logger(self.trt.Logger.WARNING)
        self.runtime = self.trt.Runtime(self.logger)
        with open(self.engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"could not deserialize engine: {self.engine_path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("could not create the execution context")

        # ---- tensor inventory ----------------------------------------------
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(
            self.engine.num_io_tensors - 1)
        self.in_shape = tuple(self.engine.get_tensor_shape(self.input_name))
        self.out_shape = tuple(self.engine.get_tensor_shape(self.output_name))
        self.in_dtype = self.engine.get_tensor_dtype(self.input_name)
        self.out_dtype = self.engine.get_tensor_dtype(self.output_name)

        # ---- device buffers -------------------------------------------------
        err, = cudart.cudaSetDevice(device)
        if int(err) != 0:
            raise RuntimeError(f"cudaSetDevice({device}) failed: {err}")
        err, self.stream = cudart.cudaStreamCreate()
        if int(err) != 0:
            raise RuntimeError(f"cudaStreamCreate failed: {err}")

        self.in_bytes = int(np.prod(self.in_shape)) * self._itemsize(self.in_dtype)
        self.out_bytes = int(np.prod(self.out_shape)) * self._itemsize(self.out_dtype)
        err, self.d_in = cudart.cudaMalloc(self.in_bytes)
        if int(err) != 0:
            raise RuntimeError(f"cudaMalloc(input) failed: {err}")
        err, self.d_out = cudart.cudaMalloc(self.out_bytes)
        if int(err) != 0:
            raise RuntimeError(f"cudaMalloc(output) failed: {err}")
        self.context.set_tensor_address(self.input_name, int(self.d_in))
        self.context.set_tensor_address(self.output_name, int(self.d_out))
        self._stream_handle = int(self.stream)

    @staticmethod
    def _itemsize(dtype):
        # TRT DataType -> numpy dtype
        return 4 if str(dtype) in ("DataType.FLOAT", "DataType.HALF",
                                   "DataType.INT32", "DataType.INT8",
                                   "DataType.UINT8", "DataType.BOOL") else 8

    def _to_numpy_dtype(self):
        s = str(self.out_dtype)
        if "HALF" in s:
            return np.float16
        if "INT32" in s:
            return np.int32
        return np.float32

    # -- inference ----------------------------------------------------------
    def infer(self, y_hat: np.ndarray):
        """Run the engine on a ``y_hat`` array, return the raw logits array."""
        y_hat = np.ascontiguousarray(y_hat, dtype=np.float32)
        if tuple(y_hat.shape) != self.in_shape:
            raise ValueError(f"expected input {self.in_shape}, got {y_hat.shape}")

        self.cudart.cudaMemcpyAsync(int(self.d_in), y_hat.ctypes.data,
                                    y_hat.nbytes,
                                    self.cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                                    self.stream)
        ok = self.context.execute_async_v3(self._stream_handle)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 failed")

        out = np.empty(self.out_shape, dtype=self._to_numpy_dtype())
        self.cudart.cudaMemcpyAsync(out.ctypes.data, int(self.d_out), out.nbytes,
                                    self.cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                                    self.stream)
        self.cudart.cudaStreamSynchronize(self.stream)
        return out.astype(np.float32) if out.dtype != np.float32 else out

    def close(self):
        if getattr(self, "_closed", False):
            return
        self._closed = True
        for attr in ("d_in", "d_out"):
            d = getattr(self, attr, None)
            if d:
                try:
                    self.cudart.cudaFree(d)
                except Exception:                              # noqa: BLE001
                    pass
                setattr(self, attr, None)
        stream = getattr(self, "stream", None)
        if stream is not None:
            try:
                self.cudart.cudaStreamDestroy(stream)
            except Exception:                                  # noqa: BLE001
                pass
            self.stream = None

    def __del__(self):
        try:
            self.close()
        except Exception:                                      # noqa: BLE001
            pass


# --------------------------------------------------------------------------- #
# post-processing (matches /root/Semantic-sensing/src/utils/metrics.py)
#
# The canonical implementation lives in ``postprocess.py``:
#   * ``postprocess.numpy_upsample``  -- bilinear align_corners=False resize
#   * ``postprocess.numpy_argmax``    -- class argmax
#   * ``postprocess.CudaPostprocess`` -- the same thing fused into one CUDA
#     kernel (NVRTC-compiled at runtime), used by default because the numpy
#     version costs ~200 ms per image for a 150-class 128->512 map.
# The thin aliases below are kept for backwards compatibility.
# --------------------------------------------------------------------------- #
def bilinear_upsample(logits: np.ndarray, size: int = GT_SIZE) -> np.ndarray:
    """Bilinear resize to ``size x size`` with ``align_corners=False``."""
    return numpy_upsample(logits, size)


def argmax_labels(logits: np.ndarray) -> np.ndarray:
    """``argmax`` over the class axis -> int64 label map."""
    return numpy_argmax(logits)


# --------------------------------------------------------------------------- #
# pipeline
# --------------------------------------------------------------------------- #
@dataclass
class InferResult:
    name: str
    logits: np.ndarray                 # (1,150,128,128) float32
    label: np.ndarray                  # (512,512) int64
    y_hat: np.ndarray = field(repr=False, default=None)   # (1,192,16,16)
    decode_seconds: float = 0.0
    infer_seconds: float = 0.0
    postprocess_seconds: float = 0.0


class ReceiverPipeline:
    """bitstream -> (logits, label map) with per-stage timing."""

    def __init__(self, engine_path=DEFAULT_ENGINE,
                 lib_path=DEFAULT_LIB, tables_path=DEFAULT_TABLES,
                 gt_size=GT_SIZE, warmup: int = 3,
                 postprocess: str = "auto"):
        self.codec = SsCodec(lib_path, tables_path)
        self.engine = TrtEngine(engine_path)
        self.gt_size = int(gt_size)
        self.codec_version = self.codec.version
        self.postprocess = make_postprocess(postprocess)
        # warm-up: the first inference pays the CUDA context / kernel load cost
        dummy = np.zeros(self.engine.in_shape, dtype=np.float32)
        for _ in range(max(0, warmup)):
            self.engine.infer(dummy)

    # -- helpers -------------------------------------------------------------
    def decode(self, payload: bytes):
        return self.codec.decode(payload)

    def decode_file(self, path):
        with open(path, "rb") as f:
            return self.codec.decode(f.read())

    # -- main path -----------------------------------------------------------
    def infer_payload(self, payload: bytes, name: str = "") -> InferResult:
        t0 = time.perf_counter()
        y_hat, dims = self.codec.decode(payload)
        t1 = time.perf_counter()

        logits = self.engine.infer(y_hat)
        t2 = time.perf_counter()

        label = self.postprocess.label_map(logits, self.gt_size)
        t3 = time.perf_counter()
        return InferResult(name=name, logits=logits, label=label, y_hat=y_hat,
                            decode_seconds=t1 - t0, infer_seconds=t2 - t1,
                            postprocess_seconds=t3 - t2)

    def infer_bitstream(self, path: str) -> InferResult:
        with open(path, "rb") as f:
            payload = f.read()
        return self.infer_payload(payload, name=os.path.basename(path))

    def close(self):
        try:
            self.engine.close()
        except Exception:                                      # noqa: BLE001
            pass
        try:
            self.codec.close()
        except Exception:                                      # noqa: BLE001
            pass
        try:
            self.postprocess.close()
        except Exception:                                      # noqa: BLE001
            pass


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", default=DEFAULT_ENGINE)
    ap.add_argument("--lib", default=DEFAULT_LIB)
    ap.add_argument("--tables", default=DEFAULT_TABLES)
    ap.add_argument("--bitstream", required=True, help="path to one .rtff file")
    ap.add_argument("--out-npy", default=None, help="save the label map as .npy")
    ap.add_argument("--out-logits", default=None, help="save the logits as .npy")
    ap.add_argument("--out-yhat", default=None, help="save the decoded y_hat as .npy")
    ap.add_argument("--gt-size", type=int, default=GT_SIZE)
    ap.add_argument("--json", default=None, help="write a small JSON summary here")
    a = ap.parse_args(argv)

    pipe = ReceiverPipeline(a.engine, a.lib, a.tables, gt_size=a.gt_size)
    try:
        res = pipe.infer_bitstream(a.bitstream)
    finally:
        pipe.close()

    print(f"[receiver] codec: {pipe.codec_version}")
    print(f"[receiver] engine: {a.engine}")
    print(f"[receiver] {a.bitstream}")
    print(f"[receiver] logits {res.logits.shape}  label {res.label.shape}")
    print(f"[receiver] decode {res.decode_seconds * 1e3:.2f} ms | "
          f"infer {res.infer_seconds * 1e3:.2f} ms | "
          f"post {res.postprocess_seconds * 1e3:.2f} ms")

    if a.out_npy:
        np.save(a.out_npy, res.label)
        print(f"[receiver] label -> {a.out_npy}")
    if a.out_logits:
        np.save(a.out_logits, res.logits)
        print(f"[receiver] logits -> {a.out_logits}")
    if a.out_yhat:
        np.save(a.out_yhat, res.y_hat)
        print(f"[receiver] y_hat -> {a.out_yhat}")
    if a.json:
        summary = {
            "engine": os.path.abspath(a.engine),
            "codec_version": pipe.codec_version,
            "bitstream": os.path.abspath(a.bitstream),
            "logits_shape": list(res.logits.shape),
            "label_shape": list(res.label.shape),
            "decode_ms": res.decode_seconds * 1e3,
            "infer_ms": res.infer_seconds * 1e3,
            "postprocess_ms": res.postprocess_seconds * 1e3,
        }
        with open(a.json, "w") as f:
            json.dump(summary, f, indent=1)
        print(f"[receiver] summary -> {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
