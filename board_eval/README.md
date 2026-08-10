# Board evaluation

This directory contains the reproducible Gate-8 workflow for a final Hailo-8
HEF.  It deliberately separates three references for every named tensor:

1. x86 ONNX Runtime FP32;
2. the x86 `SDK_QUANTIZED` result from the exact HAR used to compile the HEF;
3. the raw board HailoRT result.

The Raspberry Pi keeps only a HEF, this runner, and a small JSON manifest.
Evaluation tensors are length-prefixed `np.savez_compressed` frames over one
persistent SSH session.  They are never written to Pi storage.

## Commands

All artifacts are identified by the immutable manifest produced first:

```bash
python board_eval/freeze_manifest.py --help
python board_eval/deploy.py --help
python board_eval/evaluate_receiver_stage45.py --help
```

`tail_metrics.py` is intentionally run in the training container, because it
loads the frozen PyTorch tail after the Stage45 boundary.  A Stage45-only HEF
therefore reports a **hybrid** mIoU.  Do not label it as full-board mIoU.

The host report separates board inference time from the SSH/Tailscale control
round-trip.  The latter is not a sender-to-receiver deployment latency.
