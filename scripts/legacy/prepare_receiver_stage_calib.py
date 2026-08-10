#!/usr/bin/env python3
"""Create cascade-correct calibration handoffs for receiver stage4/stage5."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import onnxruntime as ort


LOADER = '''"""Load NHWC float32 calibration tensors."""
import json
from pathlib import Path
import numpy as np
_ROOT = Path(__file__).resolve().parent
META = json.loads((_ROOT / "meta.json").read_text())
def load_calib(limit=None):
    count = META["sample_count"] if limit is None else min(limit, META["sample_count"])
    result = {}
    for spec in META["tensor_mapping"].values():
        root = (_ROOT / spec["directory"]).resolve()
        arrays = [np.load(root / f"{i:05d}.npy", allow_pickle=False) for i in range(count)]
        result[spec["onnx_name"]] = np.stack(arrays).astype(np.float32, copy=False)
    return result
'''


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path(
        "/local/shared_with_docker/calib/receiver_stage45"))
    parser.add_argument("--onnx-dir", type=Path, default=Path(
        "/local/shared_with_docker/onnx/modules"))
    parser.add_argument("--out-root", type=Path, default=Path(
        "/local/shared_with_docker/calib"))
    args = parser.parse_args()
    source_meta = json.loads((args.source / "meta.json").read_text())
    count = int(source_meta["sample_count"])
    stage4_root = args.out_root / "receiver_stage4"
    stage5_root = args.out_root / "receiver_stage5"
    stage4_root.mkdir(parents=True, exist_ok=True)
    stage5_root.mkdir(parents=True, exist_ok=True)
    for name in ("stage4_high", "stage4_low"):
        directory = stage5_root / name
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir()

    onnx4 = args.onnx_dir / "receiver_stage4.onnx"
    onnx5 = args.onnx_dir / "receiver_stage5.onnx"
    session = ort.InferenceSession(str(onnx4), providers=["CPUExecutionProvider"])
    for index in range(count):
        x_h = np.load(args.source / "x_h" / f"{index:05d}.npy")
        x_l = np.load(args.source / "x_l" / f"{index:05d}.npy")
        high, low = session.run(None, {
            "x_h": x_h.transpose(2, 0, 1)[None],
            "x_l": x_l.transpose(2, 0, 1)[None],
        })
        np.save(stage5_root / "stage4_high" / f"{index:05d}.npy",
                high[0].transpose(1, 2, 0).astype(np.float32))
        np.save(stage5_root / "stage4_low" / f"{index:05d}.npy",
                low[0].transpose(1, 2, 0).astype(np.float32))
        if (index + 1) % 128 == 0:
            print(f"stage5 calibration {index + 1}/{count}", flush=True)

    common = {
        "sample_count": count,
        "checkpoint": source_meta["checkpoint"],
        "preprocessing": source_meta["preprocessing"],
        "codec": source_meta["codec"],
        "roundtrip_alignment": source_meta["roundtrip_alignment"],
        "cascade_source": str(args.source),
    }
    meta4 = common | {
        "onnx": {"path": str(onnx4), "sha256": sha256(onnx4), "opset": 13},
        "tensor_mapping": {
            "x_h": {"onnx_name": "x_h", "directory": "../receiver_stage45/x_h",
                    "shape_nhwc": [count, 64, 64, 128]},
            "x_l": {"onnx_name": "x_l", "directory": "../receiver_stage45/x_l",
                    "shape_nhwc": [count, 32, 32, 256]},
        },
    }
    meta5 = common | {
        "onnx": {"path": str(onnx5), "sha256": sha256(onnx5), "opset": 13},
        "upstream": {"module": "receiver_stage4", "onnx_sha256": sha256(onnx4)},
        "tensor_mapping": {
            "stage4_high": {"onnx_name": "stage4_high", "directory": "stage4_high",
                            "shape_nhwc": [count, 64, 64, 128]},
            "stage4_low": {"onnx_name": "stage4_low", "directory": "stage4_low",
                           "shape_nhwc": [count, 16, 16, 512]},
        },
    }
    for root, meta in ((stage4_root, meta4), (stage5_root, meta5)):
        (root / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        (root / "load_calib.py").write_text(LOADER)


if __name__ == "__main__":
    main()
