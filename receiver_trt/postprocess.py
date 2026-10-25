#!/usr/bin/env python3
"""Post-processing of the receiver logits: bilinear upsample + argmax.

The deployed graph emits ``logits [1,150,128,128]``; the segmentation metric
compares against the 512x512 ground truth, so the logits are resized to
512x512 and the class argmax is taken.

Convention (matches ``/root/Semantic-sensing/src/utils/metrics.py``):

    logits = F.interpolate(logits, size=gt_size, mode="bilinear",
                           align_corners=False)
    pred   = logits.argmax(1)

Two implementations are provided:

``NumpyPostprocess``
    A pure-numpy separable bilinear resize (the weight matrices are the exact
    ``align_corners=False`` weights) followed by ``np.argmax``.  Portable but
    it costs ~200 ms per image for a 150-class 128->512 map.

``CudaPostprocess``
    A single fused CUDA kernel (compiled at runtime with NVRTC) that computes
    the bilinear interpolation and the argmax in one pass, writing only the
    512x512 label map back to the host (~1 ms per image).  Falls back to the
    numpy path automatically if NVRTC / the driver API is unavailable.

Both paths produce the *same* labels (verified in ``selftest_receiver.py``).
"""
from __future__ import annotations

import ctypes

import numpy as np

# --------------------------------------------------------------------------- #
# numpy reference
# --------------------------------------------------------------------------- #
def _axis_weights(in_size: int, out_size: int):
    """The ``align_corners=False`` bilinear weights of one axis.

    Returns ``(i0, i1, w0, w1)`` with

        src = (q + 0.5) * in_size/out_size - 0.5     (clamped to [0, in-1])
        i0  = floor(src),  i1 = min(i0 + 1, in-1),  lam = src - i0
        out[q] = in[i0] * (1 - lam) + in[i1] * lam
    """
    scale = in_size / float(out_size)
    src = (np.arange(out_size, dtype=np.float64) + 0.5) * scale - 0.5
    src = np.clip(src, 0.0, in_size - 1.0)
    i0 = np.floor(src).astype(np.int64)
    i1 = np.minimum(i0 + 1, in_size - 1)
    lam = (src - i0).astype(np.float32)
    return i0, i1, (1.0 - lam).astype(np.float32), lam.astype(np.float32)


def bilinear_weights(in_size: int, out_size: int) -> np.ndarray:
    """The ``align_corners=False`` bilinear weight matrix ``(out_size, in_size)``."""
    i0, i1, w0, w1 = _axis_weights(in_size, out_size)
    W = np.zeros((out_size, in_size), np.float32)
    rows = np.arange(out_size)
    np.add.at(W, (rows, i0), w0)
    np.add.at(W, (rows, i1), w1)
    return W


def numpy_upsample(logits: np.ndarray, out_size: int = 512) -> np.ndarray:
    """Separable bilinear resize ``(1,C,H,W) -> (1,C,out,out)``.

    The resize is done as two separable passes (x first, then y), which is the
    same order PyTorch's CPU kernel uses.
    """
    x = np.asarray(logits, np.float32)
    squeeze = (x.ndim == 4 and x.shape[0] == 1)
    if squeeze:
        x = x[0]
    c, h, w = x.shape
    Wy = bilinear_weights(h, out_size)
    Wx = bilinear_weights(w, out_size)
    u = np.tensordot(x, Wx, axes=([2], [1]))       # (C, H, out)
    o = np.tensordot(Wy, u, axes=([1], [1]))       # (out, C, out)
    o = np.ascontiguousarray(o.transpose(1, 0, 2))  # -> (C, out, out)
    if squeeze:
        o = o[None]
    return o


def numpy_argmax(logits: np.ndarray) -> np.ndarray:
    """argmax over the class axis of a ``(1,C,H,W)`` (or ``(C,H,W)``) array."""
    x = np.asarray(logits, np.float32)
    if x.ndim == 4:
        return np.argmax(x, axis=1).astype(np.int64)
    return np.argmax(x, axis=0).astype(np.int64)


def numpy_label_map(logits: np.ndarray, out_size: int = 512) -> np.ndarray:
    """Bilinear resize + argmax in one call -> ``(out, out)`` label map."""
    up = numpy_upsample(logits, out_size)
    if up.ndim == 4:
        up = up[0]
    return np.argmax(up, axis=0).astype(np.int64)


# --------------------------------------------------------------------------- #
# fused CUDA kernel
# --------------------------------------------------------------------------- #
_CUDA_SRC = r'''
extern "C" __global__ void upsample_argmax_kernel(
    const float* __restrict__ src,     // (C, H, W) NCHW float32
    const float* __restrict__ ws,      // packed axis weights, see layout below
    int* __restrict__ out,             // (OH, OW) int32 labels
    const int C, const int H, const int W, const int OW)
{
    // ws layout (n = OW), all float32:
    //   [0*n .. 1*n)  x0[j]
    //   [1*n .. 2*n)  x1[j]
    //   [2*n .. 3*n)  wx0[j]   (1 - lam)
    //   [3*n .. 4*n)  wx1[j]   (lam)
    //   [4*n .. 5*n)  y0[i]
    //   [5*n .. 6*n)  y1[i]
    //   [6*n .. 7*n)  wy0[i]
    //   [7*n .. 8*n)  wy1[i]
    const int n = OW;
    const int i = blockIdx.y;
    const int y0 = (int)ws[4*n + i];
    const int y1 = (int)ws[5*n + i];
    const float wy0 = ws[6*n + i];
    const float wy1 = ws[7*n + i];

    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        const int x0 = (int)ws[j];
        const int x1 = (int)ws[n + j];
        const float wx0 = ws[2*n + j];
        const float wx1 = ws[3*n + j];
        float best = -3.0e38f;
        int bi = 0;
        for (int c = 0; c < C; ++c) {
            const float* s = src + (size_t)c * H * W;
            const float a = s[(size_t)y0 * W + x0] * wx0 + s[(size_t)y0 * W + x1] * wx1;
            const float b = s[(size_t)y1 * W + x0] * wx0 + s[(size_t)y1 * W + x1] * wx1;
            const float v = wy0 * a + wy1 * b;
            if (v > best) { best = v; bi = c; }   // first max wins (numpy/torch rule)
        }
        out[(size_t)i * n + j] = bi;
    }
}
'''


class CudaPostprocess:
    """Fused bilinear-upsample + argmax on the GPU (NVRTC-compiled)."""

    def __init__(self, device=0):
        from cuda.bindings import driver as cuda_driver
        from cuda.bindings import nvrtc
        from cuda.bindings import runtime as cudart

        self.driver = cuda_driver
        self.cudart = cudart

        err, = cuda_driver.cuInit(0)
        if int(err) != 0:
            raise RuntimeError(f"cuInit failed: {err}")
        err, self.dev = cuda_driver.cuDeviceGet(device)
        if int(err) != 0:
            raise RuntimeError(f"cuDeviceGet failed: {err}")

        # ---- a CUDA context is required for the driver API --------------------
        err, self.ctx = cuda_driver.cuDevicePrimaryCtxRetain(self.dev)
        if int(err) != 0:
            raise RuntimeError(f"cuDevicePrimaryCtxRetain failed: {err}")
        err, = cuda_driver.cuCtxSetCurrent(self.ctx)
        if int(err) != 0:
            raise RuntimeError(f"cuCtxSetCurrent failed: {err}")

        # ---- compile the kernel ---------------------------------------------
        err, prog = nvrtc.nvrtcCreateProgram(_CUDA_SRC.encode(), b"upsample_argmax.cu",
                                             0, [], [])
        if int(err) != 0:
            raise RuntimeError(f"nvrtcCreateProgram failed: {err}")
        arch = self._compute_capability()
        opts = [f"--gpu-architecture=compute_{arch}".encode()]
        (err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        if int(err) != 0:
            _, logsz = nvrtc.nvrtcGetProgramLogSize(prog)
            log = b" " * logsz
            nvrtc.nvrtcGetProgramLog(prog, log)
            raise RuntimeError(f"nvrtcCompileProgram failed: {log.decode()[:500]}")
        _, ptxsz = nvrtc.nvrtcGetPTXSize(prog)
        ptx = b" " * ptxsz
        nvrtc.nvrtcGetPTX(prog, ptx)

        # ---- load the module -------------------------------------------------
        err, self.module = cuda_driver.cuModuleLoadData(ptx)
        if int(err) != 0:
            raise RuntimeError(f"cuModuleLoadData failed: {err}")
        err, self.func = cuda_driver.cuModuleGetFunction(self.module,
                                                         b"upsample_argmax_kernel")
        if int(err) != 0:
            raise RuntimeError(f"cuModuleGetFunction failed: {err}")

        # ---- stream (driver API, so it shares the context) --------------------
        err, self.stream = cuda_driver.cuStreamCreate(0)
        if int(err) != 0:
            raise RuntimeError(f"cuStreamCreate failed: {err}")

        self._ws_cache = {}
        self._src_buf = None
        self._out_buf = None
        self._src_bytes = 0
        self._out_bytes = 0

    def _compute_capability(self):
        _, major, minor = self.driver.cuDeviceComputeCapability(self.dev)
        return f"{int(major)}{int(minor)}"

    # -- weights -------------------------------------------------------------
    def _weights(self, in_size: int, out_size: int) -> np.ndarray:
        key = (in_size, out_size)
        if key not in self._ws_cache:
            xi0, xi1, wx0, wx1 = _axis_weights(in_size, out_size)
            yi0, yi1, wy0, wy1 = _axis_weights(in_size, out_size)
            ws = np.concatenate([
                xi0.astype(np.float32), xi1.astype(np.float32), wx0, wx1,
                yi0.astype(np.float32), yi1.astype(np.float32), wy0, wy1,
            ]).astype(np.float32)
            self._ws_cache[key] = ws
        return self._ws_cache[key]

    # -- inference -----------------------------------------------------------
    def label_map(self, logits: np.ndarray, out_size: int = 512) -> np.ndarray:
        x = np.ascontiguousarray(np.asarray(logits, np.float32))
        if x.ndim == 4:
            assert x.shape[0] == 1
            x = x[0]
        c, h, w = x.shape
        src_bytes = x.nbytes
        out_bytes = out_size * out_size * 4

        if self._src_buf is None or self._src_bytes < src_bytes:
            if self._src_buf is not None:
                self.driver.cuMemFree(self._src_buf)
            err, self._src_buf = self.driver.cuMemAlloc(src_bytes)
            if int(err) != 0:
                raise RuntimeError(f"cuMemAlloc(src) failed: {err}")
            self._src_bytes = src_bytes
        if self._out_buf is None or self._out_bytes < out_bytes:
            if self._out_buf is not None:
                self.driver.cuMemFree(self._out_buf)
            err, self._out_buf = self.driver.cuMemAlloc(out_bytes)
            if int(err) != 0:
                raise RuntimeError(f"cuMemAlloc(out) failed: {err}")
            self._out_bytes = out_bytes

        ws = self._weights(w, out_size)
        err, ws_buf = self.driver.cuMemAlloc(ws.nbytes)
        if int(err) != 0:
            raise RuntimeError(f"cuMemAlloc(ws) failed: {err}")

        out = np.empty((out_size, out_size), np.int32)
        try:
            self.driver.cuMemcpyHtoDAsync(self._src_buf, x.ctypes.data, x.nbytes,
                                          self.stream)
            self.driver.cuMemcpyHtoDAsync(ws_buf, ws.ctypes.data, ws.nbytes,
                                          self.stream)

            c_src = ctypes.c_void_p(int(self._src_buf))
            c_ws = ctypes.c_void_p(int(ws_buf))
            c_out = ctypes.c_void_p(int(self._out_buf))
            c_c = ctypes.c_int(c)
            c_h = ctypes.c_int(h)
            c_w = ctypes.c_int(w)
            c_ow = ctypes.c_int(out_size)
            argv = (ctypes.c_void_p * 8)(
                ctypes.cast(ctypes.byref(c_src), ctypes.c_void_p),
                ctypes.cast(ctypes.byref(c_ws), ctypes.c_void_p),
                ctypes.cast(ctypes.byref(c_out), ctypes.c_void_p),
                ctypes.cast(ctypes.byref(c_c), ctypes.c_void_p),
                ctypes.cast(ctypes.byref(c_h), ctypes.c_void_p),
                ctypes.cast(ctypes.byref(c_w), ctypes.c_void_p),
                ctypes.cast(ctypes.byref(c_ow), ctypes.c_void_p),
                None)
            threads = 128
            blocks_x = (out_size + threads - 1) // threads
            blocks_y = out_size
            err, = self.driver.cuLaunchKernel(self.func, blocks_x, blocks_y, 1,
                                              threads, 1, 1, 0, self.stream,
                                              argv, 0)
            if int(err) != 0:
                raise RuntimeError(f"cuLaunchKernel failed: {err}")
            self.driver.cuMemcpyDtoHAsync(out.ctypes.data, self._out_buf, out.nbytes,
                                          self.stream)
            self.driver.cuStreamSynchronize(self.stream)
        finally:
            self.driver.cuMemFree(ws_buf)
        return out.astype(np.int64)

    def close(self):
        for attr in ("_src_buf", "_out_buf"):
            buf = getattr(self, attr, None)
            if buf:
                try:
                    self.driver.cuMemFree(buf)
                except Exception:                              # noqa: BLE001
                    pass
                setattr(self, attr, None)
        stream = getattr(self, "stream", None)
        if stream is not None:
            try:
                self.driver.cuStreamDestroy(stream)
            except Exception:                                  # noqa: BLE001
                pass
            self.stream = None


class NumpyPostprocess:
    """Pure-numpy reference implementation (always available)."""

    def __init__(self, device=0):
        self.device = device
        self._closed = False

    def label_map(self, logits: np.ndarray, out_size: int = 512) -> np.ndarray:
        return numpy_label_map(logits, out_size)

    def close(self):
        self._closed = True


def make_postprocess(kind="auto", device=0):
    """``kind`` in {"auto", "cuda", "numpy"}; "auto" prefers CUDA and falls back."""
    if kind in ("cuda", "auto"):
        try:
            return CudaPostprocess(device=device)
        except Exception as e:                                 # noqa: BLE001
            if kind == "cuda":
                raise
            print(f"[postprocess] CUDA path unavailable ({e}); using numpy", flush=True)
    return NumpyPostprocess(device=device)
