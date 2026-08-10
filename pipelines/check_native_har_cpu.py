#!/usr/bin/env python3
"""Run one SDK_NATIVE vs ORT check with GPU visibility disabled."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hailo_deploy import emulate  # noqa: E402
from hailo_deploy.config import get_profile  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--har", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite: {args.out}")
    module = get_profile("fused").modules_by_name["bn_g_s"]
    result = emulate.native_check(module, args.har, args.onnx)
    payload = result.as_dict()
    payload["execution_policy"] = {
        "CUDA_VISIBLE_DEVICES": "", "purpose": "CPU/FP32 recheck after GPU TF32 mismatch"
    }
    temporary = args.out.with_name(f".{args.out.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, args.out)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if result.status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
