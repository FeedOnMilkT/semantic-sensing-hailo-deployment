#!/usr/bin/env python3
"""Repair staged-to-final artifact paths in completed PTQ result metadata.

The initial runner wrote paths while artifacts were still in its atomic staging
directory. This migration changes only those paths after checking each final
artifact against the recorded SHA-256, and writes a separate audit report.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hailo_deploy.io.hash import sha256

FIELDS = ("model_script", "quantized_har", "hef", "compiled_har", "auto_alls")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ptq-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite: {args.out}")
    changes = []
    for result_path in sorted(args.ptq_root.glob("*/result.json")):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "PASS":
            continue
        module_changes = []
        for field in FIELDS:
            item = result.get(field)
            if not isinstance(item, dict) or "path" not in item or "sha256" not in item:
                continue
            final = result_path.parent / Path(item["path"]).name
            if not final.is_file():
                raise FileNotFoundError(f"{result.get('module')}/{field}: {final}")
            actual = sha256(final)
            if actual != item["sha256"]:
                raise ValueError(f"{result.get('module')}/{field}: SHA mismatch")
            if item["path"] != str(final):
                module_changes.append({"field": field, "from": item["path"], "to": str(final), "sha256": actual})
                item["path"] = str(final)
        if module_changes:
            result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
            changes.append({"module": result.get("module"), "result": str(result_path), "changes": module_changes})
    report = {"kind": "ptq_result_path_repair", "status": "PASS", "changed_modules": changes,
              "module_count": len(changes), "notes": ["Only metadata paths changed.", "Every target artifact SHA-256 matched before updating its path."]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "modules": len(changes)}, indent=2))


if __name__ == "__main__":
    main()
