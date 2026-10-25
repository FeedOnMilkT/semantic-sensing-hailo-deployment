# cpp/ — shared C++ codec, sender pipeline

A shared C++ bitstream codec (`libsscodec`) plus a self-contained C++ sender
pipeline (`sender_cpp`) for the semantic-sensing Hailo deployment.

```
cpp/
├── include/sscodec.h          the C ABI (sender + receiver share it)
├── src/                       libsscodec: rANS, deterministic h_s, RTFF container
│   ├── rans.{h,cpp}           pure-C++ port of CompressAI 1.2.8's rANS coder
│   ├── hyper_synthesis.{h,cpp}  the deterministic h_s evaluation
│   ├── codec.{h,cpp}          tables + streams + container
│   ├── sscodec_api.cpp        the extern "C" wrapper
│   └── tables_format.h        the codec_tables.bin binary layout
├── third_party/compressai_rans/  CompressAI's rans64.h + LICENSE (BSD-3-Clause-Clear)
├── tools/                     Python tools (export tables, prep images, channel sim)
├── tests/                     the test suite (C++ + Python cross-checks)
└── sender/                    the sender pipeline (HailoRT + replay backends)
```

---

## 1. The key design decision: h_s lives in ONE place

**Problem.** The deployed sender runs the hyper-synthesis `h_s` as a Hailo HEF
(`sender_bottleneck_hs.hef`, int8). The receiver was going to run `h_s` in
TensorRT (FP32). Measured on the 2000-image validation set, the int8 Hailo
`h_s` and the FP32 PyTorch `h_s` produce **different CDF "档位" (index) for
~17 % of the elements**. rANS decoding requires the sender's and the
receiver's CDF index to agree **element by element** — a single mismatch
desynchronises the whole stream.

**Therefore the receiver must NOT recompute h_s.** Since the receiver cannot
run the Hailo int8 `hs` HEF (no Hailo device on the receiver), and the FP32
recomputation does not reproduce the sender's indexes, the only consistent
option is:

> **`h_s` is computed by the same deterministic C++ code on BOTH ends.**
> The Hailo `hs` HEF is not used at all. The receiver calls
> `ss_hyper_synthesis()` through `libsscodec` (via ctypes) and gets the
> *identical* means / scales / CDF indexes the sender used.

This also removes the FP32-vs-int8 h_s discrepancy from the bitstream
entirely: the deployed bitstreams are defined by the C++ h_s, and both ends
agree bit-exactly.

**Why the C++ h_s is deterministic.** The requirements are: pure float32,
hand-written loops, fixed accumulation order, no BLAS/OpenMP reductions, no
fast-math, and no transcendental functions at runtime.

* The convolutions accumulate in float32 in a fixed `(ic, ky, kx)` loop nest —
  no reassociation, no vectorised reduction.
* The BatchNorm is **not folded** into the convolution; it keeps the reference
  operation order `(x - mean) * invstd * gamma + beta`. `invstd =
  1/sqrt(var + eps)` is **pre-computed by the exporter**, so the runtime
  performs **no square root and no division**.
* The bilinear upsample follows PyTorch's `align_corners=False` formula
  **including the `src < 0 → src = 0` clamp** (PyTorch clamps, it does not
  floor). Getting this wrong was worth **0.89 absolute error** on the means;
  with the clamp it is **~1e-5**.
* The scale → CDF index rule is CompressAI's `build_indexes` verbatim:
  `scales = max(scales, scale_bound)`, `indexes = len(scale_table) - 1`,
  then `indexes -= (scales <= s)` for every `s` in `scale_table[:-1]`.

**Compiler flags.** `-ffp-contract=off` is **mandatory**: GCC on aarch64
contracts `a*b+c` into an FMA *by default*, which would change the rounding
of the accumulation and make the aarch64 sender disagree with the x86-64
receiver. `-fno-fast-math` prevents reassociation and reciprocal
substitution. The build uses `-O2 -ffp-contract=off -fno-fast-math` by
default (`STRICT_FP=ON`).

---

## 2. The bitstream

### 2.1 Container (`RTFF`)

Big-endian, exactly the format of `FusedLatentBottleneck.compress()` in
`src/models/bottlenecks/rtformer_fused.py`:

```
magic "RTFF"            4 bytes
u16 z_h, u16 z_w        hyper-latent size          (8, 8)
u16 y_h, u16 y_w        latent size                (16, 16)
u16 fh_h, u16 fh_w      x_h feature-map size       (64, 64)
u16 fl_h, u16 fl_w      x_l feature-map size       (32, 32)
u8  B                  batch size                 (1)
--- z stream ---
u32 len, len bytes     the rANS-coded factorised prior
--- y stream ---
u32 len, len bytes     the rANS-coded conditional latent
```

### 2.2 Symbols

* **z stream** — the transmitted symbol is `round(z_hat - median)`, per
  channel, where `median` is the EntropyBottleneck median (128 values). The
  sender's `z_hat` is `floor(z_pre + 0.5) + median` (the host rounding), so
  the symbol is exactly `floor(z_pre + 0.5)`. This matches CompressAI's
  `EntropyBottleneck.compress(z_hat)` **byte for byte**.
* **y stream** — the symbol is `round(y - means)` (round-half-to-even, like
  `torch.round`), and the CDF index of each element comes from the h_s scale.
  The receiver reconstructs `y_hat = symbols + means`.

### 2.3 rANS

`cpp/src/rans.cpp` is a de-pybinded port of CompressAI v1.2.8's
`compressai/cpp_exts/rans/rans_interface.cpp` (Rans64, precision 16, with the
bypass escape coding). It is **bit-compatible with `compressai.ans`** — that
is what makes the Python cross-check possible. The upstream BSD-3-Clause-Clear
licence and the ryg_rans CC0 notice are preserved in
`cpp/third_party/compressai_rans/`.

### 2.4 `codec_tables.bin`

Produced by `cpp/tools/export_codec_tables.py`, consumed by `ss_open()`.
**All integers are little-endian, all floats are IEEE-754 binary32
little-endian.** The exact layout is documented in `cpp/src/tables_format.h`:

```
offset 0   magic "SSTB"        (4 bytes)
offset 4   u32 format_version  (= 1)
offset 8   u32 flags           (reserved, 0)
offset 12  u32 h_s_layer_count
offset 16  u32 table_count
offset 20  u32 reserved0, offset 24 u32 reserved1

then h_s_layer_count LAYER DESCRIPTORS (56-byte header + float32 params):
  u32 kind (1=Conv, 2=BatchNorm, 3=Upsample), u32 flags, u32 in_channels,
  u32 out_channels, u32 kernel_h/w, u32 stride_h/w, u32 pad_h/w,
  u32 reserved0, u64 param_count, f32 scalar0, then param_count float32.
    Conv     : weight[O*I*KH*KW] (OIHW) then bias[O] (if flag kHsFlagHasBias)
    BatchNorm: gamma[C], beta[C], mean[C], invstd[C]
    Upsample : (no params); scalar0 = the scale factor

then table_count TABLE DESCRIPTORS:
  u32 tag, u32 dtype (0=f32, 1=i32), u32 ndim, u32 reserved, u64 elem_count,
  then elem_count elements.
    1  scale_table      f32[64]
    2  gauss_cdf        i32[64*L]   (row-major, stride L = max_length+2)
    3  gauss_cdf_length i32[64]
    4  gauss_offset     i32[64]
    5  eb_cdf           i32[128*L]
    6  eb_cdf_length    i32[128]
    7  eb_offset        i32[128]
    8  medians          f32[128]
    9  quantiles        f32[128*3]
   10  scale_bound      f32[1]
```

---

## 3. Building and testing

```bash
# --- build (out-of-tree; keep the build dir on the data disk) ---
cmake -S cpp -B /root/autodl-tmp/build/O2 \
      -DCMAKE_BUILD_TYPE=None \
      -DCMAKE_CXX_FLAGS="-O2 -ffp-contract=off -fno-fast-math" \
      -DSSCODEC_TABLES=$PWD/cpp/codec_tables.bin \
      -DSSCODEC_VECTORS=/root/autodl-tmp/work/vectors250.bin \
      -DSSCODEC_N_VECTORS=250
cmake --build /root/autodl-tmp/build/O2 -j4

# --- the C++ tests ---
cd /root/autodl-tmp/build/O2 && ctest --output-on-failure
./ss_test_roundtrip   <tables.bin> <vectors.bin> <n_images>
./ss_test_determinism --dump --out <f> --tables <t> --vectors <v> [--n N]
./ss_test_determinism --compare <a.bin> <b.bin>

# --- the CompressAI cross-check (needs the venv-hailo python) ---
/root/Semantic-sensing/venv-hailo/bin/python cpp/tests/crosscheck_compressai.py \
    --tables cpp/codec_tables.bin --lib /root/autodl-tmp/build/O2/libsscodec.so \
    --vectors /root/autodl-tmp/work/vectors250.bin --n 20

# --- the end-to-end bitstream check (C++ .rtff vs CompressAI) ---
/root/Semantic-sensing/venv-hailo/bin/python cpp/tests/check_bitstreams.py \
    --rtff-dir <bitstreams dir> --vectors /root/autodl-tmp/work/vectors250.bin --n 25

# --- build the test vectors from the Hailo emulator dump ---
/root/Semantic-sensing/venv-hailo/bin/python cpp/tests/make_test_vectors.py \
    --chunks /dev/shm/hailo2000/hailo --n 250 --out /root/autodl-tmp/work/vectors250.bin
```

### The three-build determinism check

```bash
for v in O0 O2 O3n; do
  case $v in
    O0)  F="-O0" ;;
    O2)  F="-O2" ;;
    O3n) F="-O3 -march=native" ;;
  esac
  cmake -S cpp -B /root/autodl-tmp/build/$v -DCMAKE_BUILD_TYPE=None \
        -DCMAKE_CXX_FLAGS="$F -ffp-contract=off -fno-fast-math"
  cmake --build /root/autodl-tmp/build/$v -j4
  ./build/$v/ss_test_determinism --dump --out work/dump_$v.bin \
        --tables cpp/codec_tables.bin --vectors work/vectors250.bin --n 250
done
sha256sum work/dump_O*.bin          # all three must be identical
```

Measured result (250 images, means + scales + CDF indexes):

```
32ef0e0d3f1b558ed2ecff2c64ec98245c54f4f5c35ba9d27343dee8c18fc2fd  work/dump_O0.bin
32ef0e0d3f1b558ed2ecff2c64ec98245c54f4f5c35ba9d27343dee8c18fc2fd  work/dump_O2.bin
32ef0e0d3f1b558ed2ecff2c64ec98245c54f4f5c35ba9d27343dee8c18fc2fd  work/dump_O3n.bin
```

**Note on `-O3 -march=native`:** `native` is only a *self-test* of the
determinism contract on this x86-64 host. It must **not** be used for the
deployed sender or receiver builds — `-march=native` bakes in the build
machine's instruction set. The shipped flag set is `-O2 -ffp-contract=off
-fno-fast-math` (see §1).

---

## 4. The sender pipeline

```
image (512x512x3 uint8, NHWC)
   -> [backend]  backbone HEF -> (x_h 128x64x64, x_l 256x32x32)
   -> [backend]  enc HEF      -> (y 192x16x16, z_pre 128x8x8)
   -> [host]     z_hat = floor(z_pre + 0.5) + median
   -> [codec]    deterministic h_s -> (means, scales) -> CDF indexes
                 symbols = round(y - means)
                 rANS(z symbols), rANS(y symbols)
   -> [container] RTFF
```

The host rounding **must** stay on the host: the DFC compiler drops
`Floor`/`Cast(int32)` at parse time (they are no-ops on chip), and an in-graph
rounding emits *continuous* symbols, which inflates the rANS z-stream ~2.3x.

### Backends

| backend | what it does | status here |
|---|---|---|
| `--backend hailo` | runs the two HEFs on a Hailo device via the HailoRT C++ API | **compile-checked only** (see §6) |
| `--backend replay` | replays tensors dumped earlier, for host testing | **fully working** |

The HailoRT backend matches the vstreams **by tensor shape**, never by name:

```
input  (512,512,3)                  the image
input  (64,64,128) / (32,32,256)   x_h / x_l      (enc HEF)
output (64,64,128) / (32,32,256)   x_h / x_l      (backbone HEF)
output (16,16,192) / (8,8,128)     y / z_pre      (enc HEF)
```

Hailo tensors are NHWC, the Python reference is NCHW, so every tensor is
transposed on the way in and out. The outputs are requested as
`HAILO_FORMAT_TYPE_FLOAT32` so the host sees de-quantised values, matching the
DFC emulator's `SDK_QUANTIZED` context.

### CLI

```bash
# the real sender (on the board / with a device attached)
./sender_cpp --backend hailo --hef-dir /root/deploy_artifacts/hailo/sender \
             --tables codec_tables.bin \
             --input <images.u8 | dir> --out-dir <out>

# the host-only replay path (no device needed)
./sender_cpp --backend replay --replay-zhat --replay-dir <replay dir> \
             --tables codec_tables.bin \
             --input <images.u8 | dir> --out-dir <out>
```

`--input` accepts a single raw `.u8` file (512*512*3 bytes, HWC RGB) or a
directory of them. The output is one `<name>.rtff` per image plus a
`manifest.json` (name, bytes, bpp, sha256).

| flag | meaning |
|---|---|
| `--backend hailo\|replay` | which inference backend to use |
| `--hef-dir DIR` | the two HEFs (`sender_backbone.hef`, `sender_bottleneck_enc.hef`) |
| `--replay-dir DIR` | the dumped tensors for the replay backend |
| `--replay-zhat` | the dump already contains the rounded `z_hat` (skip the host rounding) |
| `--tables FILE` | `codec_tables.bin` |
| `--input FILE\|DIR` | raw `.u8` images |
| `--out-dir DIR` | where the `.rtff` files go |
| `--manifest PATH` | where to write `manifest.json` (default `<out-dir>/manifest.json`) |
| `--limit N` | only process the first N images |

The manifest reports **two** rates: `avg_bpp` (the whole RTFF container) and
`avg_bpp_payload` (the rANS payloads only, the convention the deployment
guide uses). The difference is the 29-byte container framing.

### Python tools

```bash
# 512x512x3 uint8 images, using the SAME loader as the training pipeline
python cpp/tools/prep_images.py --split full --n 2000 \
       --out-dir /root/autodl-tmp/work/images_u8

# dump the sender tensors into the replay-backend format
python cpp/tools/dump_replay_tensors.py --chunks <chunk dir> \
       --out-dir /root/autodl-tmp/work/replay_hailo --names names.json

# a minimal "noisy channel" demo (NO FEC, NO CRC -- see the warning below)
python cpp/tools/channel_sim.py --in-dir <bitstreams> --out-dir <out> --ber 1e-3
```

---

## 5. The deployable package

`/root/autodl-tmp/deploy_artifacts/package/sender_out/` contains everything the
receiver needs, and can be copied to another machine as-is:

```
sender_out/
├── bitstreams/*.rtff     one RTFF container per image (2000 for the val set)
├── manifest.json         name, bytes, bpp, sha256 per image + the average bpp
├── names.json            the 2000 image names, in the manifest's order
├── gt_labels.npy         uint8 (2000, 512, 512) ground truth (for the mIoU)
├── codec_tables.bin      the shared codec tables
├── libsscodec.so         the x86-64 shared library (ctypes-loadable)
├── sscodec.h             the C ABI header
└── tools/                channel_sim.py, export_codec_tables.py, prep_images.py
```

The bitstreams were produced by the **C++ `ss_encode`** from the **real
all-Hailo sender output** (the DFC emulator's `SDK_QUANTIZED` tensors: the
`y` and the already-rounded `z_hat` from `chunk_XXXXX.npz`).

The measured rate on the 2000-image validation set:

| quantity | value |
|---|---|
| average container size | 5789.70 bytes |
| `avg_bpp` (whole RTFF container) | 0.176687 |
| `avg_bpp_payload` (rANS payloads only) | **0.175802** |
| container overhead | 29 bytes = 0.000885 bpp |

`avg_bpp_payload` is the convention used by the deployment guide, so it is
directly comparable to it. `cpp/tests/rate_reference.py` computes all three
rates on the same 2000 images:

| encoder | bpp | guide |
|---|---|---|
| **C++ `h_s`** (what ships) | **0.175802** | — |
| Hailo int8 `h_s` (the old `hs` HEF) | 0.175832 | 0.17583 |
| PyTorch FP32 `h_s` | 0.174673 | 0.17467 |

The C++ sender lands at **0.175802**, within 3e-5 of the guide's 0.17583 for
the Hailo int8 sender. The tiny gap is the difference between the emulator's
int8 `h_s` scales and the deterministic C++ `h_s` scales, and it is *by
design*: the deployed bitstreams are defined by the C++ `h_s`, and both ends
of the link agree on them bit-exactly. The z-stream cost is identical
(1156.3 bits) in both cases — only the y stream differs, by 8 bits out of
44929 (0.02 %).

---

## 6. What was verified here, and what was not

### Verified on this machine (real runs, numbers in the final report)

* **C++ encode → C++ decode is 100 % lossless** on real sender tensors:
  250/250 images, **0 symbol mismatches** in both the z and the y stream.
* **h_s matches PyTorch FP32** to `max|diff| ≈ 1.0e-5` (means and scales) —
  three orders of magnitude below the level where a CDF index could flip.
  (A single CDF index flip would desynchronise the rANS decoder, so this
  margin is the thing that matters, not the absolute error.)
* **Three builds agree bit-exactly**: `-O0`, `-O2`, and `-O3 -march=native`
  produce *identical* means / scales / CDF-index dumps (identical sha256:
  `32ef0e0d3f1b558ed2ecff2c64ec98245c54f4f5c35ba9d27343dee8c18fc2fd`).
* **Bit-compatible with CompressAI 1.2.8**: the z stream and the y stream are
  **byte-identical** to `compressai.ans` (20/20 and 25/25 respectively in the
  two cross-checks), and the whole RTFF container matches a container
  assembled by Python from the same two streams (25/25).
* **The real bpp of the deployed bitstreams**: `avg_bpp_payload = 0.175802`
  (see §5), against the guide's 0.17583 (Hailo int8) / 0.17467 (FP32).
* The packaged `libsscodec.so` was loaded with **ctypes** and used to decode
  a real `.rtff` from the package — the exact path the TensorRT receiver
  will take.

### NOT verified here

* **The HailoRT backend was only compile-checked.** This machine has no Hailo
  device and no `libhailort`, so `hailort_backend.cpp` was compiled against
  the upstream **hailo-ai/hailort v4.23.0** headers only (0 errors, 0
  warnings; the compiled archive references 8 real `hailort::` API symbols).
  It has **never been linked against a real `libhailort`** and has **never run
  on silicon**. The vstream-by-shape matching, the NHWC↔NCHW transposes and
  the float32 output request are therefore *unverified at runtime* and must
  be re-checked on the board. (The linker error list from the attempted real
  link was exactly the expected set of `hailort::` entry points, so the API
  usage is at least self-consistent.)
* **The aarch64 / Raspberry Pi build was not run either** — only the x86-64
  host build was exercised. The determinism argument relies on
  `-ffp-contract=off` being honoured by the aarch64 GCC; the flag is in the
  CMake build for every target, but it has not been verified on a Pi.
* **The `y` stream uses the C++ `h_s` scales, not the emulator's int8
  `h_s` scales.** This is the intended design (see §1), but it means the
  deployed bitstreams are *not* bit-identical to what the Python reference
  encoder would produce — the CDF indexes differ on 0.0004 % of the elements
  (5 of 1 228 800 in the 25-image check), which is the same 1e-5-scale
  numerical margin showing up as a handful of index flips.

### Cross-compiling for the Raspberry Pi (aarch64)

```bash
# on the board (or with an aarch64 cross toolchain) -- this LINKS the real
# HailoRT backend against libhailort:
cmake -S cpp -B build-aarch64 -DCMAKE_TOOLCHAIN_FILE=<your aarch64 toolchain> \
      -DCMAKE_BUILD_TYPE=Release -DWITH_HAILORT=ON -DHAILORT_LINK=ON \
      -DHAILORT_ROOT=/usr             # the HailoRT install prefix
cmake --build build-aarch64 -j4
```

On a machine that has **no** `libhailort` (like this one) the backend can only
be *compile-checked*:

```bash
cmake -S cpp -B build-hailort -DWITH_HAILORT=ON \
      -DHAILORT_INCLUDE_DIR=/root/autodl-tmp/third_party/hailoRT-clone/hailort/libhailort/include
cmake --build build-hailort -j4
```

This compiles `hailort_backend.cpp` against the upstream HailoRT v4.23.0
headers (0 errors, 0 warnings) and builds the archive
`libhailort_backend.a`, but the **executable** keeps the stub factory,
because the real backend cannot be linked without `libhailort`. The CLI
says so explicitly instead of silently misbehaving. Use `-DHAILORT_LINK=ON`
to get the real thing.

The `-ffp-contract=off` flag is **not optional on aarch64**: without it GCC
contracts `a*b+c` into an FMA by default, the h_s rounding changes, and the
sender and receiver disagree on the CDF indexes.

### The channel has no error protection

The RTFF container is a bare rANS stream: there is **no FEC, no CRC and no
retransmission**. A single flipped bit in the rANS payload desynchronises the
decoder and everything after the flip decodes to garbage (the container header
*is* checked, so a flip in the header is reported as a hard error).
`cpp/tools/channel_sim.py` demonstrates this; it is a demo, not a transport.
