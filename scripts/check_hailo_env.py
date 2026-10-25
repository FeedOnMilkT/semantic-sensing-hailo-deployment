#!/usr/bin/env python3
"""Report whether this Python can run the Hailo-8 DFC deployment tools."""
from __future__ import annotations

import importlib.metadata
import importlib.util
import platform
import sys


def main() -> int:
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"Platform: {platform.platform()}")

    required = ("hailo_sdk_client", "onnx", "onnxruntime", "numpy")
    missing = []
    for module in required:
        spec = importlib.util.find_spec(module)
        location = spec.origin if spec else "MISSING"
        print(f"{module}: {location}")
        if spec is None:
            missing.append(module)

    for distribution in ("hailo-sdk-client", "hailo-model-optimization", "hailo-platform"):
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            version = "not installed"
        print(f"{distribution}: {version}")

    if missing:
        print("\nHailo deployment commands are not ready; missing: " + ", ".join(missing))
        print("Install DFC 3.33.x for Hailo-8 from the authorized Hailo Software Suite package.")
        return 1

    print("\nCore Hailo Python imports are available.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
