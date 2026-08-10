# Board inference runtime

This directory contains the minimal source closure recovered from the frozen Phase-2 OC4 Raspberry Pi deployment snapshot dated 2026-08-05. It is the code executed on the board, separated from host-side deployment and evaluation tooling.

## Files

- `hybrid_ort_runtime_runner.py`: current mainline runner. It executes `sender_backbone`, `bn_g_s`, and the receiver tail with HailoRT, while the four analysis/hyperprior modules run with ONNX Runtime on the CPU.
- `fullmodule_runtime_runner.py`: shared tensor framing, HailoRT engine, entropy codec, and runtime primitives used by the hybrid runner. It also retains the historical all-HEF runner.
- `rans_fast/`: pybind11/C++ rANS implementation compiled locally on the Raspberry Pi.

The runtime manifest, HEFs, ONNX models, codec tables, compiled `_rans_fast` extension, datasets, and inference results are intentionally not included.

## Observed board environment

The frozen deployment was inspected with:

- Python 3.13.5
- HailoRT Python API 4.23.0
- ONNX Runtime 1.28.0
- NumPy 2.5.1

These versions record the deployed environment; they are not a promise that arbitrary newer versions are compatible.

## Build the entropy codec

Install `setuptools` and `pybind11` in the board runtime environment, then build beside the Python wrapper:

```bash
cd board_runtime/rans_fast
python setup_ext.py build_ext --inplace
```

The resulting shared library is generated locally and ignored by Git.

## Run

Place the immutable artifacts described by the runtime manifest next to their manifest-defined paths, then start the framed stdin/stdout service:

```bash
cd board_runtime
python hybrid_ort_runtime_runner.py \
  --serve \
  --manifest /path/to/runtime_manifest.json
```

The process implements a binary framed protocol with `hello`, `infer`, and `shutdown` operations. It is normally driven by the host-side board evaluation/deployment client rather than used interactively.

## Provenance

The six source files were copied from the formal OC4 board deployment and compared by SHA-256 with the corresponding host-side deployment sources at the time of import. All six matched byte-for-byte. No board-specific hostname, key, manifest, model, or result was copied into this directory.
