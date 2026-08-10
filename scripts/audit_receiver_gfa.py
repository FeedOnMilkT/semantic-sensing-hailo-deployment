#!/usr/bin/env python3
"""Thin wrapper → scripts/legacy/audit_receiver_gfa.py"""
from __future__ import annotations
import runpy
import sys
from pathlib import Path
sys.argv[0] = str(Path(__file__).resolve().parent / "legacy" / "audit_receiver_gfa.py")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
runpy.run_path(sys.argv[0], run_name="__main__")
