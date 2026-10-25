# `receiver_trt/` — the TensorRT receiver

The receiving half of the split=mid deployment:

```
*.rtff  ──ctypes libsscodec.so──▶  y_hat [1,192,16,16]
                                   │
                                   ▼  TensorRT engine (receiver_fp32 / receiver_fp16)
                                logits [1,150,128,128]
                                   │
                                   ▼  bilinear 128→512 (align_corners=False) + argmax
                             label map [512,512]  ──▶  mIoU vs gt_labels.npy
```

The decode is **not** part of the TensorRT graph, on purpose: the rANS payload
is coded against per-element CDF indexes that the sender derived from the
deterministic C++ hyper-synthesis `h_s`.  Recomputing `h_s` inside the network
would not reproduce the sender's indexes (the Hailo int8 `h_s` and the FP32
`h_s` disagree on ~17 % of the elements — see `cpp/README.md` §1) and the rANS
decoder would desynchronise.  `libsscodec.so` therefore owns the decode, and
the engine starts at `y_hat`.

---

## 1. Environment

| item | value |
|---|---|
| venv | `/root/autodl-tmp/venvs/trt` (Python 3.12.3) |
| TensorRT | **`tensorrt-cu12==10.16.1.11`** (cu12 build, `tensorrt_cu12_bindings` / `tensorrt_cu12_libs`) |
| CUDA runtime bindings | `cuda-python==13.0.0` (`cuda-bindings==13.0.3`) |
| numpy | `1.26.4` |
| onnxruntime (self-test only) | `onnxruntime==1.20.1` |
| GPU | NVIDIA A800-SXM4-80GB, compute capability 8.0 |
| driver | 580.126.09 |
| CUDA runtime reported by `cuda-python` | 13.0 (the cu12 TensorRT wheels ship their own cu12 runtime) |
| host CUDA toolkit | `/usr/local/cuda-12.8` |

### Why TensorRT 10 and not 11

TensorRT **11.4.0.106** removed `BuilderFlag.FP16` / `INT8` (strongly-typed
networks only), so the weak-typed FP16-flag path does not exist there.  The venv
was switched to the pinned **`tensorrt-cu12==10.16.1.11`**:

```bash
export TMPDIR=/root/autodl-tmp/tmp
/root/autodl-tmp/venvs/trt/bin/pip uninstall -y tensorrt tensorrt-cu13 \
    tensorrt_cu13_bindings tensorrt_cu13_libs
/root/autodl-tmp/venvs/trt/bin/pip install --cache-dir /root/autodl-tmp/pipcache \
    tensorrt-cu12==10.16.1.11
```

The cu12 wheels run fine on driver 580 and on the A800 (sm_80).  The fallback
plan (TRT 11 + a strongly-typed FP16 ONNX export) was **not** needed.

---

## 2. Files

| file | role |
|---|---|
| `export_receiver_onnx.py` | exports the static-shape receiver ONNX (torch venv) — *unchanged, already working* |
| `build_engine.py` | builds the FP32 / FP16 TensorRT engines + timing cache + metadata JSON |
| `receiver_trt.py` | the runtime: ctypes codec decode + TensorRT inference + label map |
| `postprocess.py` | bilinear 128→512 + argmax (fused CUDA kernel, numpy reference) |
| `eval_miou.py` | end-to-end mIoU over the sender bitstreams + per-stage latency |
| `bench_latency.py` | clean per-image latency benchmark (rounds per engine, one engine per process) |
| `selftest_receiver.py` | codec / engine / post-processing correctness checks |
| `pytorch_reference_eval.py` | the PyTorch FP32 reference on the **same** bitstreams (torch venv) |

---

## 3. Commands

### 3.1 Build the engines

```bash
cd /root/semantic-sensing-hailo-deployment
export TMPDIR=/root/autodl-tmp/tmp
PKG=/root/autodl-tmp/deploy_artifacts/package/receiver

# FP32
/root/autodl-tmp/venvs/trt/bin/python receiver_trt/build_engine.py \
    --precision fp32 --workspace 4096 --out-dir $PKG

# FP16 (FP16 everywhere except the numerically sensitive layers, which stay FP32)
/root/autodl-tmp/venvs/trt/bin/python receiver_trt/build_engine.py \
    --precision fp16 --workspace 4096 --out-dir $PKG

# full-FP16 variant (optional; NOT built or evaluated for the shipped results)
/root/autodl-tmp/venvs/trt/bin/python receiver_trt/build_engine.py \
    --precision fp16 --fp16-all --name receiver_fp16all --out-dir $PKG
```

The `--fp32-layers` pattern list (default) pins these ONNX node groups to FP32
in the FP16 build:

* `/synth/g_s_trunk` — the IGDN synthesis (divisive normalisation: `sqrt` of a
  conv sum amplifies activation errors),
* `/synth/head_h` — IGDN + the up-sampled high-branch mixing,
* `attn_h` — the cross-attention softmax that feeds the K/V memory,
* `attn_l` — the external (double-normalised) attention.

**54 layers** are pinned.  `CONSTANT` layers are skipped automatically: they
carry Int64 shape tensors and TensorRT rejects them with
`cannot use precision Float with weights of type Int64`.

The engine bytes are **not** bit-reproducible across builds (TensorRT embeds a
per-build UUID), but the rebuilt engines were verified to produce
**bit-identical logits and label maps** on 20 bitstreams.

> ⚠️ **This holds only for a rebuild that reuses the shipped timing cache**
> (`--precision fp32` / `--precision fp16` into the package dir, which loads
> `receiver_*.timing.cache`).  Re-verified here: a warm FP32 rebuild and a warm
> FP16 rebuild both reproduce the shipped engines **bit-exactly**
> (`max_abs_logit_diff = 0.0`, `max_abs_label_diff = 0`).
>
> A **cold** rebuild (fresh timing cache, no warm-up) picks different FP16
> kernel selections and is therefore *not* bit-identical: the FP16 engine
> differs by up to 0.25 in the logits and changes some label pixels.  The
> effect on the metric is negligible (mIoU 0.31172 vs 0.31179 on the first 300
> images, +0.02 %), but the **shipped numbers were all produced with the warm
> (cached) engines** — so rebuild with the shipped `.timing.cache` present to
> reproduce them exactly.

`receiver_*.build.log` holds the build metadata JSON followed by every
TensorRT log message from the build (ONNX opset, activation / weight memory,
block assignment, timing-cache serialisation).

### 3.2 Run one bitstream

```bash
/root/autodl-tmp/venvs/trt/bin/python receiver_trt/receiver_trt.py \
    --engine /root/autodl-tmp/deploy_artifacts/package/receiver/receiver_fp16.engine \
    --bitstream /root/autodl-tmp/deploy_artifacts/package/sender_out/bitstreams/ADE_val_00000001.rtff \
    --out-npy /root/autodl-tmp/tmp/pred01.npy
```

### 3.3 Evaluate (mIoU + latency)

```bash
PKG=/root/autodl-tmp/deploy_artifacts/package
/root/autodl-tmp/venvs/trt/bin/python receiver_trt/eval_miou.py \
    --engine $PKG/receiver/receiver_fp32.engine \
    --out-json $PKG/receiver/miou_fp32.json \
    --save-conf $PKG/receiver/conf_fp32.npy
```

(`--limit N` evaluates only the first N images.)

### 3.4 Self-test

```bash
/root/autodl-tmp/venvs/trt/bin/python receiver_trt/selftest_receiver.py \
    --n 12 --out-json /root/autodl-tmp/deploy_artifacts/package/receiver/selftest.json
```

### 3.5 Latency benchmark (clean per-image numbers)

```bash
# one engine per process so the two never share the GPU
for E in fp32 fp16; do
  /root/autodl-tmp/venvs/trt/bin/python receiver_trt/bench_latency.py \
      --engines $PKG/receiver/receiver_$E.engine \
      --n 100 --rounds 4 --warmup 20 \
      --out-json $PKG/receiver/latency_benchmark_$E.json
done

# the two per-engine JSONs are then merged into latency_benchmark.json
```

### 3.6 The PyTorch FP32 reference on the same bitstreams

```bash
/root/Semantic-sensing/venv-hailo/bin/python receiver_trt/pytorch_reference_eval.py \
    --n 2000 \
    --out-json /root/autodl-tmp/deploy_artifacts/package/receiver/miou_pytorch_fp32.json \
    --save-conf /root/autodl-tmp/deploy_artifacts/package/receiver/conf_pytorch_fp32.npy
```

---

## 4. Results (all measured on this machine)

### 4.1 Engines

| | FP32 | FP16 (sensitive layers pinned to FP32) |
|---|---|---|
| TensorRT | 10.16.1.11 | 10.16.1.11 |
| engine size | 65.1 MB | 33.4 MB |
| layers pinned to FP32 | – | 54 |
| build time (cold) | 77.0 s | 263.9 s |
| build time (warm timing cache) | **3.3 s** | **8.4 s** |
| timing cache | `receiver_fp32.timing.cache` (148 KB) | `receiver_fp16.timing.cache` (574 KB) |
| metadata | `receiver_fp32.json` | `receiver_fp16.json` |

Both metadata JSONs record the TensorRT version, the CUDA runtime version, the
GPU name, the compute capability and the driver version.

### 4.2 End-to-end mIoU on the 2000 sender bitstreams

| pipeline | images | mIoU | pixel acc | per-image mIoU (mean) |
|---|---|---|---|---|
| **TRT FP32** | 2000 | **0.350205** | 0.762236 | 0.351582 |
| **TRT FP16** (54 layers pinned FP32) | 2000 | **0.350220** | 0.762224 | 0.351579 |
| **PyTorch FP32** (same bitstreams) | 2000 | **0.350209** | 0.762236 | 0.351540 |
| PyTorch FP32 *sender arm* (`dump_reference_500.py`, pre-existing) | 2000 | 0.350361 | 0.762714 | 0.349067 |

The TRT FP32 engine reproduces the PyTorch FP32 receiver to **4e-6 mIoU** on
the identical 2000 bitstreams — i.e. the ONNX export + TensorRT build is
numerically lossless at the metric level.

The FP16 engine is **not worse** than FP32 here (0.350220 vs 0.350205, a
+1.5e-5 difference, i.e. 0.004 % relative): the 54 pinned FP32 layers cover
exactly the numerically fragile parts (the IGDN `sqrt` normalisation and the
attention softmaxes), so the remaining FP16 layers do not move the metric.
Individual per-pixel logits do drift (max abs diff 0.59, argmax agreement
97.07 % vs ORT) but the argmax noise is spatially uncorrelated and washes out
in the global confusion matrix.

> Note on the two PyTorch rows: the *receiver-arm* PyTorch number (0.350209)
> is the one that is directly comparable to the TRT numbers — it runs the same
> `libsscodec.so`-decoded bitstreams through the PyTorch `ReceiverNet`.  The
> *sender-arm* number (0.350361) comes from the pre-existing
> `dump_reference_500.py` and reconstructs the receiver input as
> `round(y − means) + means` in PyTorch instead of decoding the rANS payload,
> hence the small systematic offset.

### 4.3 Latency (per image)

Two independent measurements are reported:

* the **2000-image evaluation runs** (`miou_fp32.json` / `miou_fp16.json`),
  which also do the mIoU bookkeeping on the CPU;
* a **dedicated benchmark** (`bench_latency.py`, 4 rounds × 100 bitstreams per
  engine = **400 timed samples per engine**, each engine measured in its **own
  process** so the two never share the GPU, `latency_benchmark.json`).  This is
  the clean per-image latency quoted below.

| stage | engine | median | mean | p95 |
|---|---|---|---|---|
| decode (`ss_decode`, CPU) | FP32 | 181.1 ms | 181.3 ms | 182.2 ms |
| decode (`ss_decode`, CPU) | FP16 | 181.0 ms | 181.3 ms | 182.1 ms |
| **TRT inference** | **FP32** | **2.99 ms** | 2.98 ms | 4.33 ms |
| **TRT inference** | **FP16** | **2.48 ms** | 2.60 ms | 3.49 ms |
| post-process (128→512 + argmax, fused CUDA) | FP32 | 1.83 ms | 1.86 ms | 2.11 ms |
| post-process (128→512 + argmax, fused CUDA) | FP16 | 1.84 ms | 1.87 ms | 2.11 ms |
| end-to-end (decode + infer + post) | FP32 | 185.8 ms | 186.1 ms | 188.6 ms |
| end-to-end (decode + infer + post) | FP16 | 185.3 ms | 185.8 ms | 187.5 ms |

FP16 inference is **1.21× faster** than FP32 (2.48 ms vs 2.99 ms median).
The per-round medians were very stable (FP32: 3.004 / 2.996 / 2.994 /
2.971 ms; FP16: 2.506 / 2.482 / 2.462 / 2.474 ms).

The **decode dominates** at 181 ms/image (97 % of the end-to-end budget): it is
the single-threaded C++ rANS decoder plus the deterministic `h_s`.  The
eager-PyTorch FP32 reference inference on the same GPU takes 5.72 ms, so the
TensorRT FP32 engine is ~1.9× faster than eager PyTorch on this graph (the
graph is dominated by many small elementwise ops, so the speed-up is modest).

Throughput of the full 2000-image runs: 5.32 img/s (FP32) and 5.34 img/s
(FP16) end-to-end, decode included.

### 4.4 Engine parity vs ONNX Runtime (12 images, `selftest.json`)

| comparison | cosine | argmax agreement | max abs diff |
|---|---|---|---|
| ORT vs **TRT FP32** | 0.999999957 | **99.67 %** | 0.092 |
| ORT vs **TRT FP16** | 0.999999205 | 97.07 % | 0.586 |
| TRT FP32 vs TRT FP16 | 0.999999214 | 97.05 % | 0.540 |

(The ONNX itself already differs from eager PyTorch by the same order — see
`export_report.json`: cosine 0.999999987, argmax 99.92 %.)

### 4.5 Codec correctness (`selftest.json`)

| check | result |
|---|---|
| decoded `y_hat` vs the sender's own reconstruction `round(y − means_cpp) + means_cpp` | **exact** (cosine 1.0, max abs diff 0.0, **0 symbol mismatches**) |
| decoded `z_hat` vs the Hailo sender dump's `z_hat` | **exact** (max abs diff 0.0) |
| C++ `h_s` means vs the **PyTorch FP32** `h_s` means (same `z_hat`) | max abs diff 8.6e-3, mean 2.0e-4 |
| C++ `h_s` means vs the **Hailo int8** `h_s` means | max abs diff 2.1e-1 |
| fused CUDA post-process vs the numpy reference | **identical** (0 differing pixels) |

The last two rows are the documented `h_s` gap, not a bug: they are exactly why
the receiver decodes with the C++ library instead of recomputing `h_s` in the
network.

---

## 5. Implementation notes

### 5.1 The codec (`receiver_trt.py`, class `SsCodec`)

Plain `ctypes` binding of `cpp/include/sscodec.h`:

| C entry point | use |
|---|---|
| `ss_open` / `ss_close` | open `codec_tables.bin`, release the handle |
| `ss_decode` | RTFF container → `y_hat` (the deployed path) |
| `ss_decode_z` | z-stream only (self-test) |
| `ss_hyper_synthesis` | the deterministic `h_s` → means / scales / CDF indexes (self-test) |
| `ss_version` | "sscodec 1.0.0" |

`ss_decode` returns `y_hat = symbols + means`, i.e. exactly the tensor the
sender's receiver path reconstructs with `torch.round(y − means) + means`
(verified bit-exactly, §4.5).

### 5.2 TensorRT execution (`receiver_trt.py`, class `TrtEngine`)

TensorRT 10's name-based API: `set_tensor_address` + `execute_async_v3` on a
non-default CUDA stream, with `cuda-python` for the device buffers and the
H2D / D2H copies.  The output tensor is `float32` in both builds (the graph's
output is not quantised).

### 5.3 Post-processing (`postprocess.py`)

The metric code resizes the logits to the GT resolution before the argmax
(`SegMeter.update`), so 128×128 → 512×512 is needed.  A pure-numpy separable
resize costs ~200 ms/image for a 150-class map, which would dominate the
pipeline, so the default path is a **single fused CUDA kernel** (compiled at
runtime with NVRTC) that does the bilinear interpolation and the argmax in one
pass and writes only the 512×512 label map (~2 ms).  The numpy path is kept as
a reference and as the automatic fallback if NVRTC / the driver API is
unavailable; the two are verified to produce **identical** label maps.

`align_corners=False` semantics: `src = (q + 0.5) · in/out − 0.5`, clamped to
`[0, in−1]`, linear blend of the four neighbours.  The numpy implementation was
checked against `torch.nn.functional.interpolate(..., align_corners=False)`:
max abs diff 4.8e-7, **identical** label maps.

### 5.4 The metric convention

Taken verbatim from `/root/Semantic-sensing/src/utils/metrics.py`
(`SegMeter.update` / `SegMeter.compute`):

* bilinear `align_corners=False` resize of the logits to the GT resolution,
* `pred = logits.argmax(1)`,
* pixels whose GT label is `ignore_index` (255) are dropped,
* the confusion matrix is accumulated over the **whole** set and `mIoU` is the
  mean IoU over the classes with `union > 0` (150/150 classes are present in
  the 2000-image validation set).

`eval_miou.py` also reports the per-image mIoU mean/std, which is what
`dump_reference_500.py` reports, so the two are directly comparable.

---

## 6. Produced artifacts

All under `/root/autodl-tmp/deploy_artifacts/package/receiver/` (157 MB total):

| file | size | content |
|---|---|---|
| `receiver.onnx` | 64.0 MB | the static-shape receiver ONNX (input from the previous stage) |
| `receiver_fp32.engine` | 65.1 MB | FP32 TensorRT engine (2326 layers) |
| `receiver_fp16.engine` | 33.4 MB | FP16 engine, 54 sensitive layers pinned to FP32 |
| `receiver_fp32.timing.cache` | 148 KB | serialised timing cache (FP32) |
| `receiver_fp16.timing.cache` | 574 KB | serialised timing cache (FP16) |
| `receiver_fp32.json` / `receiver_fp16.json` | 1.2 / 3.2 KB | build metadata: TRT version, CUDA runtime, GPU name, compute capability, driver, pinned layer names, build time, engine/timing-cache sizes |
| `receiver_fp32.build.log` / `receiver_fp16.build.log` | 3 / 5 KB | the build metadata plus every TensorRT log message (ONNX opset, memory usage, block assignment, timing-cache serialisation) |
| `miou_fp32.json` / `miou_fp16.json` | 1.9 KB | end-to-end mIoU + latency over the 2000 bitstreams |
| `miou_pytorch_fp32.json` | 0.7 KB | the PyTorch FP32 reference on the same bitstreams |
| `latency_benchmark.json` | 4.6 KB | the clean per-image latency benchmark (4 rounds × 100 images = 400 samples per engine, one engine per process) |
| `conf_fp32.npy` / `conf_fp16.npy` / `conf_pytorch_fp32.npy` | 180 KB each | the 150×150 global confusion matrices |
| `selftest.json` | 19 KB | the self-test report (§4.4, §4.5) |
| `export_report.json` | 1.3 KB | the pre-existing ONNX-export validation report |

Source files live in `/root/semantic-sensing-hailo-deployment/receiver_trt/`
(`build_engine.py`, `receiver_trt.py`, `postprocess.py`, `eval_miou.py`,
`bench_latency.py`, `selftest_receiver.py`, `pytorch_reference_eval.py`,
`README.md`).

---

## 7. Known limitations / what was **not** done

* **The engines are machine-bound.**  A TensorRT engine is tied to the TRT
  version and the GPU architecture it was built on (TRT 10.16.1.11, sm_80,
  driver 580).  They must be rebuilt on the actual receiver box.
* **The 128→512 head-resolution mismatch.**  The exported graph ends at 128×128
  while the native seg head emits 256×256, so the deployed path upsamples
  256→128 (in-graph) and then 128→512 (post-process).  A single direct
  256→512 resize would not give the same labels; the code follows the deployed
  graph, and the metric code resizes whatever the head emits.
* **The decode is CPU-bound** (181 ms/image, single-threaded C++).  It is not
  parallelised and is by far the largest term in the end-to-end latency.  A
  multi-threaded / batched rANS decoder would be the obvious next step.
* **The full-FP16 engine** (`--fp16-all`) was not built or evaluated; the
  shipped FP16 engine keeps the 54 numerically sensitive layers in FP32.
* **No INT8 engine** was built (the task only asked for FP32/FP16), and no
  calibration was performed.
* **The sender-side Hailo device path was not exercised here** — the receiver
  only consumes the packaged bitstreams, so the HailoRT backend itself remains
  compile-checked-only (as documented in `cpp/README.md` §6).
* The channel has **no FEC / CRC / retransmission** (documented in
  `cpp/README.md`): a single flipped bit desynchronises the rANS decoder.
