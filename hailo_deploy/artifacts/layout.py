"""Artifact layout helpers: one run → one out_root, refuse overwrite."""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunLayout:
    """Canonical subdirs under a single ``out_root``."""

    out_root: Path

    @property
    def har(self) -> Path:
        return self.out_root / "har"

    @property
    def hef(self) -> Path:
        return self.out_root / "hef"

    @property
    def ptq(self) -> Path:
        return self.out_root / "ptq"

    @property
    def results(self) -> Path:
        return self.out_root / "results"

    @property
    def reports(self) -> Path:
        return self.out_root / "reports"

    @property
    def logs(self) -> Path:
        return self.out_root / "logs"

    @property
    def evaluation(self) -> Path:
        return self.out_root / "evaluation"

    def ensure(self, *names: str) -> None:
        targets = names or ("har", "hef", "ptq", "results", "reports", "logs", "evaluation")
        for name in targets:
            getattr(self, name).mkdir(parents=True, exist_ok=True)


def refuse_existing(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def stage_dir(out_dir: Path) -> Path:
    """Create a sibling temp directory for atomic PASS/FAIL rename."""
    temporary = out_dir.parent / f".{out_dir.name}.tmp-{os.getpid()}"
    refuse_existing(temporary)
    temporary.mkdir(parents=True)
    return temporary


def commit_or_fail(temporary: Path, out_dir: Path, passed: bool) -> Path:
    """Atomically publish ``temporary`` as ``out_dir`` or ``out_dir.FAILED``."""
    if passed:
        os.replace(temporary, out_dir)
        return out_dir
    failed = out_dir.parent / f"{out_dir.name}.FAILED"
    if failed.exists():
        shutil.rmtree(temporary, ignore_errors=True)
        raise FileExistsError(failed)
    os.replace(temporary, failed)
    return failed
