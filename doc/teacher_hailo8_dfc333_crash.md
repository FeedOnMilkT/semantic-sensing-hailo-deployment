# Teacher (DAPPM/SegFormer-style) on Hailo-8 — DFC 3.33 compile crash: root cause and current status

Date: 2026-10-25
Scope: the *teacher* segmentation network (input `1x3x512x512`, output logits `1x150x128x128`).
Toolchain: `hailo-dataflow-compiler 3.33.0` / `hailo_sdk_client 3.33.0` (conda env `/root/miniconda3/envs/hailo-dfc`).

## 1. Summary

The teacher **cannot be compiled as one Hailo-8 graph** with DFC 3.33: the compiler
subprocess aborts with `SIGABRT` (exit code `-6`) and produces *no* stderr, no
`errors.pb`, no assertion text — the SDK only reports
`BackendAllocatorException: Compilation failed with unexpected crash`.

The crash is **not** caused by the ONNX rewriting. It is caused by the
**wide attention MatMul + softmax pair** in `layer4/attn_h` (and the structurally
identical pair in `layer5/attn_h`).

A **split deployment does work**: cutting the teacher into 4 segments lets 2 of the
4 segments compile to HEF today, and the whole chain reproduces ORT FP32 logits
with cosine `0.99999999` under `SDK_FP_OPTIMIZED`. Under `SDK_QUANTIZED` the
measured cosine against ORT FP32 is **0.8799** (below the 0.9 target).

## 2. Exact trigger of the DFC 3.33 compile crash

### 2.1 The crashing construct

Isolating node ranges of the staticised teacher ONNX and compiling each range
separately pins the trigger to the attention block of `layer4`:

| isolated range | nodes | outcome |
|---|---|---|
| `[220,222]` | `attn_h/Reshape`, `attn_h/MatMul` (QK), `attn_h/Mul` (scale) | `IndexError: list index out of range` |
| `[221,223]` | `attn_h/MatMul` (QK), `attn_h/Mul`, `attn_h/Softmax` | `UnsupportedModelError: Unexpected input shapes at matmul layer blk/matmul1` |
| `[220,224]` | + `attn_h/MatMul_1` (AV) | `IndexError: list index out of range` |
| `[210,225]` | whole `layer4/attn_h` block incl. `Reshape_3` | **optimize OK → compile `CRASH(-6 SIGABRT)`** |
| `[210,225]` (baseline, no rewrite) | same | **compile `CRASH(-6 SIGABRT)`** |

So the *only* range that both optimizes and then crashes the compiler is the
attention block that contains **both** MatMuls around the softmax.

In the optimized HAR that block is:

```
blk/format_conversion1  [-1,64,64,128] -> [-1,1,4096,128] x2     # K/V flatten to 4096
blk/format_conversion4  [-1,12,12,128] -> [-1,1,144,128]
blk/linear_matmul1      [-1,1,4096,128]
blk/concat_matmul1      [-1,1,4096,128] + [-1,1,4096,1] -> [-1,1,4096,129]
blk/matmul1             [-1,1,144,128] x [-1,1,4096,129] -> [-1,1,144,4096]   # QK
blk/reduce_max_softmax1 [-1,1,144,4096] -> [-1,1,1,4096]                    # smart softmax
blk/ew_sub_softmax1 / ne_activation / reduce_sum_softmax1 / ew_mult_softmax1
blk/format_conversion5  [-1,1,144,4096] -> [-1,1,4096,144]
blk/matmul2             [-1,1,4096,144] x [-1,1,128,144] -> [-1,1,4096,128]   # AV
blk/format_conversion6  [-1,1,4096,128] -> [-1,64,64,128]
```

The crash-inducing pair is **`blk/matmul1` (K=4096, N=144, C=128, concat'd with a
broadcast column → 129)** followed by the decomposed ("smart") softmax
(`reduce_max` → `ew_sub` → `exp` → `reduce_sum` → `ew_mult`) and then
**`blk/matmul2`**.

### 2.2 What was ruled out

Every one of these was tested on the same optimized HAR and still returns
`CRASH(-6 SIGABRT)` (`work2/s1/var/seg1_variants.json`,
`work2/s1/opt/seg1_compile_variants.json`):

* `matmul_correction` set to `zp_comp_none` on the offending `format_conversion` layers
* `matmul_correction` disabled entirely
* `smart_softmax_stats` disabled
* `optimization_level=1`
* `performance_param(compiler_optimization_level=1)`
* `control/compiler/memory` level sweeps (40/60/75)
* `context switch` (`ctxswitch`)
* `force_range_out` on `blk/matmul1`
* `a16_w16` precision mode on `blk/matmul1` / `blk/matmul2` (rejected earlier by
  the SDK: *"Unsupported value `[PrecisionMode.a16_w16]` for fields
  `['precision_mode']` in layer `blk/matmul1`"*)

Disk space was also ruled out: after freeing ~11 GB the identical crash
reproduced, and a known-good sender HAR compiles fine in the same environment,
so the toolchain is not broken.

### 2.3 Compiler diagnostics are unavailable

The DFC SDK (`hailo_sdk_client/allocator/hailo_tools_runner.py`) calls
`process.communicate()` and **discards both stdout and stderr** of the
`hailo_tools/build/compiler` binary, and on a negative return code raises only
the generic *"Compilation failed with unexpected crash"*. Monkey-patching the
runner to keep the streams (`scratch_teacher/compile_verbose.py`,
`scratch_teacher/exp_stderr/`) shows the child writes:

```
[info] Loading network parameters
[info] Starting Hailo allocation and compilation flow
<nothing>
```

and then dies with `returncode=-6 (SIGABRT)`. The abort is therefore a **hard
crash inside the proprietary compiler binary**, not a reported compile error.
**The exact internal assertion is not observable** — what is established is
*which* graph construct triggers it.

### 2.4 Other, non-crash blockers on the same block

For completeness, these are *distinct* failures from the SIGABRT and are
**not** the crash trigger:

* `N_sm_id` (Softmax → Identity) gets past parsing and optimization but then
  fails in **optimization** with
  `AccelerasUnsupportedError: layer blk/matmul2 does not support shift delta`
  (input ranges `[-987.563, 1139.387]` x `[-95753.860, 118441.555]`).
* The `layer5/attn_h` twin cannot even be shape-inferred:
  `Reshape ... input_shape_size == size was false. Input shape:{1,1,1,1},
  requested shape:{144,128}`.

### 2.5 Mutation experiments (this round)

Four attention rewrites were generated and compiled on range `[210,225]`.
**All four failed before reaching the compiler**, each for a *tooling* reason,
so they add no evidence about the crash itself:

| mutation | result | failure |
|---|---|---|
| `P_manual_sm_full` | FAIL | `ValidationError: Unrecognized attribute: axes for operator ReduceSum` (the helper emitted `axes` as an attribute; ONNX 13 requires it as an input) |
| `P_chunk_mm` | FAIL | `ValidationError: c_mut1 initializer name is not unique` |
| `P_manual_sm_chunk` | FAIL | `IndexError: list index (15) out of range` |
| `P_qk_id` | FAIL | `UnsupportedModelError: Couldn't find inputs from ONNX proto. Number of expected inputs: 5, Inputs found: 4` (ORT shape inference failed first: `Reshape_3` input `{1,1,128,128}` vs requested `{1,128,64,64}`) |

The earlier `N_*` round failed on the same class of tooling errors
(*"Nodes in a graph must be topologically sorted"* — the node rewriter appends
new nodes after the consumers), and the `M_*` round failed with
*"could not locate attention chain"* (wrong node-name constants).

**Conclusion for this round: the four requested mutation rewrites were never
actually compiled, so the question "does rewriting the attention softmax/matmul
make the crash go away?" remains open.** The crash-trigger localisation in §2.1
rests on the unmodified-graph bisection, which *is* solid.

## 3. What does work: split deployment

`split_onnx.py` cuts the teacher at 4 boundaries (see
`/root/autodl-tmp/scratch_teacher/segs4/segments.json`):

| segment | node range | inputs | compile result |
|---|---|---|---|
| seg0 | `[0,158]` | `image` | **OK** -> `seg0.hef` (5.28 MiB) |
| seg1 | `[159,261]` | `layer3.Add`, `layer2.Add` | **CRASH(-6 SIGABRT)** |
| seg2 | `[262,422]` | `layer5.Add_1`, `relu_5` | FAIL `format_conversion4: Agent infeasible` |
| seg3 | `[423,434]` | 4 neck Resize tensors | **OK** -> `seg3.hef` (15.57 MiB) |

seg1 fails because it contains the `layer4/attn_h` crashing block; seg2 fails
with an allocator *infeasibility* (`Agent infeasible` on `format_conversion4`),
which is a **different, non-crash** failure that is in principle fixable by
re-cutting.

Note: the segments were compiled from the **optimized HAR**, so seg1's crash is
the same §2 trigger.

## 4. Measured numbers (DFC emulator, `SDK_QUANTIZED`)

Measured with `scripts/eval_teacher_split_chain.py` (see §5), 2 validation images
from `/root/deploy_artifacts/data/val_manifest.json`, reference = ORT FP32 on
`/root/deploy_artifacts/hailo/teacher_fix/teacher_rw.onnx`.

| context | cos_mean | cos_min | cos_max | argmax agree | n |
|---|---|---|---|---|---|
| `SDK_FP_OPTIMIZED` | **0.9999999883211963** | 0.9999999835006281 | 0.9999999931417645 | 0.99481 | 2 |
| `SDK_QUANTIZED` | **0.879862769296762** | 0.8708348782667885 | 0.8888906603267354 | 0.03830 | 2 |

Interpretation:

* The FP number proves the **graph surgery is faithful** — the split, the
  staticised resize and the NHWC/NCHW conventions reproduce ORT FP32 to
  numerical noise. Any remaining loss is pure int8 quantization error.
* The quantized number **0.8799 < 0.9** target. For reference, the *unsplit*
  whole-network HARs measured earlier: `teacher_fast` -> cos_mean **0.9295**
  (min 0.9130) and the QFT variant `teacher_rw` -> cos_mean **0.9765**
  (min 0.9674), both at n=8. The split chain is *worse* than the unsplit QFT
  model because the int8 boundary tensors are re-quantized at every split.

**These 2-image numbers are a measurement of the split chain, not of a fixed
rewrite.** The four requested rewrites never compiled, so no "rewrite + cosine"
pair exists.

## 5. Artifacts

| what | where |
|---|---|
| split-chain evaluator (new) | `scripts/eval_teacher_split_chain.py` (this repo) |
| segment boundary definition | `/root/autodl-tmp/scratch_teacher/segs4/segments.json` |
| segment ONNX | `/root/autodl-tmp/scratch_teacher/segs4/seg{0..3}.onnx` |
| optimized HARs | `/root/autodl-tmp/scratch_teacher/work2/s{0..3}/seg{0..3}_optimized.har` |
| compiled HEFs (seg0, seg3) | `/root/autodl-tmp/scratch_teacher/work2/s0/seg0.hef`, `.../s3/seg3.hef` |
| measured JSONs | `/root/autodl-tmp/scratch_teacher/eval_chain_fp.json`, `.../eval_chain_quant.json` |
| crash-bisect logs | `/root/autodl-tmp/scratch_teacher/b*.log`, `.../blocks_*.log` |
| mutation logs | `/root/autodl-tmp/scratch_teacher/{N,P}_*.log`, `.../M*.log` |

## 6. Not verified / open items

1. **No compile-passing attention rewrite exists yet.** The four mutations failed
   in the ONNX-rewriting tooling, not in the compiler. Fixing the rewriter
   (ONNX-13 `axes` as input, unique initializer names, topological insertion,
   correct attention-node names) is the prerequisite for re-running them.
2. **seg1 still has no HEF**, so the current split chain is *not* deployable as-is
   — the measured 0.8799 comes from the emulator, which can run an optimized HAR
   without a compiled HEF.
3. **seg2's "Agent infeasible"** was not chased; re-cutting around the
   `layer5/attn_h` block may resolve it.
4. **The compiler's internal abort reason is unobservable** (empty stderr).
   Attribution to `matmul1 + smart-softmax + matmul2` is by bisection, not by
   reading the compiler's assertion.
5. **n=2 images** for the split-chain measurement; the unsplit references are n=8.
6. The `layer5/attn_h` twin block has an **unresolved ONNX shape-inference
   failure** (`Reshape {1,1,1,1} -> {144,128}`), which will also need a rewrite.
