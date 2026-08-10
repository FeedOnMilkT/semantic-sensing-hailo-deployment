# Semantic Sensing · Hailo-8 Deployment

Host-side deployment tooling for the RTFormer mid-split semantic-sensing model. The repository covers Hailo DFC parsing, native validation, PTQ, HEF compilation, hybrid Hailo/ONNX Runtime packaging, and Raspberry Pi board evaluation.

Model training and ONNX/handoff export live in the companion `semantic-sensing` project. This repository consumes immutable exported artifacts; it does not contain checkpoints, datasets, ONNX models, HAR files, or HEF files.

## Current runtime boundary

The Phase-2 mainline is a hybrid Hailo/ORT runtime:

- Hailo-8 HEF: `sender_backbone`, `bn_g_s`, receiver stage4/stage5, DAPPM, and receiver head.
- Raspberry Pi CPU with ONNX Runtime FP32 by default: `bn_align_h`, `bn_g_a`, `bn_h_a`, and `bn_h_s`.
- Raspberry Pi CPU: rounding, CDF/index construction, rANS, and byte-container handling.

The historical nine/ten-module all-Hailo chain remains diagnostic only. The hardware target is Hailo-8 with DFC 3.x; DFC 5.x targets are not supported by this project.

## Repository layout

```text
hailo_deploy/    reusable deployment library
  io/            handoff validation, calibration loading, hashes, metrics
  gates/         parse/native/PTQ/quantized/compile gates
  artifacts/     run layout and SDK log helpers
pipelines/       thin orchestration entry points
board_eval/      Raspberry Pi deployment and Gate-8 evaluation tools
scripts/         compatibility wrappers and legacy diagnostics
alls/            maintained DFC model scripts
configs/         candidate-specific DFC model scripts
doc/             version-sensitive deployment notes
tests/           host-side unit tests
```

Each run uses a new `out_root` and refuses to overwrite an existing result. Store generated artifacts outside the checkout, for example under `/data/deployment-runs` or a Hailo Suite shared volume.

## Environment

Run parse/PTQ/compile commands in a Hailo Software Suite environment compatible with Hailo-8 and DFC 3.x. Keep the PyTorch training/export environment separate.

Set the companion training repository when a command needs model or codec source:

```bash
export SEMANTIC_SENSING_ROOT=/path/to/semantic-sensing
```

Large inputs are passed explicitly as handoff, manifest, ONNX, calibration, or output paths. Do not copy them into this repository.

## Main workflows

Run commands from the repository root.

```bash
# Parse and SDK_NATIVE validation.
python pipelines/parse_native.py --profile demo3
python pipelines/parse_native.py --profile fused --only bn_g_s receiver_stage4

# Real PTQ for one module from an immutable handoff.
python pipelines/module_ptq.py fullmodule \
  --handoff-root /path/to/handoff \
  --module receiver_stage4 \
  --parse-har /path/to/receiver_stage4_hailo_model.har \
  --out-dir /path/to/runs/example/ptq/receiver_stage4

# Historical diagnostic full-module chain.
python pipelines/fullmodule_chain.py \
  --handoff-root /path/to/handoff \
  --out-root /path/to/runs/example \
  --cuda-visible-devices 0

# Freeze, deploy, and evaluate a board runtime.
python pipelines/board_gate8.py \
  --out-root /path/to/runs/board-example \
  --manifest /path/to/runtime-manifest.json \
  --ssh-target user@hailo-board \
  --identity-file /path/to/id_ed25519 \
  --known-hosts /path/to/known_hosts
```

If the board is reached through a proxy, pass `--proxy-command` explicitly. Board-specific hostnames, private keys, and known-hosts files must remain outside the repository.

Compatibility wrappers under `scripts/` delegate to the corresponding `pipelines/` entry points. New work should use `pipelines/` and `hailo_deploy/` directly.

## Validation

The dependency-light host test is:

```bash
PYTHONPATH=. python -m unittest discover -s tests -v
```

Hailo SDK, ONNX Runtime, and board tests require their respective environments and immutable external artifacts. See [the deployment guide](doc/hailo_deployment_guide.md) and [board evaluation notes](board_eval/README.md).

## Artifact policy

Generated `.onnx`, `.har`, `.hef`, checkpoints, calibration arrays, logs, and run directories are intentionally ignored. Preserve their SHA-256 values, tool versions, manifests, and evaluation reports alongside the external artifact bundle.
