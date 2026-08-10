#!/usr/bin/env python3
"""Repair only stale pre-rename output paths in an older board-eval report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensor-eval", type=Path, required=True)
    args = parser.parse_args()
    root = args.tensor_eval
    report_path = root / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS":
        raise ValueError("only a successful evaluation may be repaired")
    expected = {
        "fp_onnx": str(root / "fp_onnx_outputs"),
        "board": str(root / "board_outputs"),
    }
    for name, path in expected.items():
        if not Path(path).is_dir():
            raise FileNotFoundError(path)
        report.setdefault("outputs", {})[name] = path
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (root / "path_repair.json").write_text(
        json.dumps({"status": "PASS", "repaired": expected}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASS", "report": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
