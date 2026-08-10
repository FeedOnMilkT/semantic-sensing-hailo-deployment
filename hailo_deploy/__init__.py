"""hailo_deploy — Hailo-8 deployment library (parse → PTQ → compile → board gates).

Library = reusable capabilities. Orchestration lives in ``pipelines/``.
Authority for DFC APIs: ``doc/hailo_deployment_guide.md``.
"""

from . import config, emulate, onnx_io, parse, reporting

__all__ = [
    "config",
    "onnx_io",
    "parse",
    "emulate",
    "reporting",
]
