#!/usr/bin/env python3
"""Thin wrapper → scripts/legacy/export_receiver_gfa_scale_candidate.py"""
from __future__ import annotations
import runpy
import sys
from pathlib import Path
sys.argv[0] = str(Path(__file__).resolve().parent / "legacy" / "export_receiver_gfa_scale_candidate.py")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
runpy.run_path(sys.argv[0], run_name="__main__")
