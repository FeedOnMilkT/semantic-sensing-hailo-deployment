"""Fused mid-split profile — thin wrapper over ``config.get_profile("fused")``.

Legacy scripts still call ``activate()`` to mutate ``hailo_deploy.config``
globals. New code should prefer ``get_profile("fused")`` / ``apply_profile``.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import config
from .config import FUSED_MODULES, Module, get_profile

# Re-export names expected by legacy fused scripts.
ONNX_DIR = Path()
MANIFEST_PATH = Path()
ARTIFACTS = Path()
HAR_DIR = Path()
HEF_DIR = Path()
RESULTS_DIR = Path()
REPORTS_DIR = Path()
ALLS_DIR = config.PROJECT_ROOT / "alls"
MODULES: list[Module] = list(FUSED_MODULES)
MODULES_BY_NAME = {m.name: m for m in MODULES}
CODEC_MODULES = ["bn_align_h", "bn_g_a", "bn_g_s"]


def _sync_exports(profile: config.DeployProfile) -> None:
    global ONNX_DIR, MANIFEST_PATH, ARTIFACTS, HAR_DIR, HEF_DIR, RESULTS_DIR, REPORTS_DIR
    global MODULES, MODULES_BY_NAME, ALLS_DIR
    ONNX_DIR = profile.onnx_dir
    MANIFEST_PATH = profile.manifest_path or (profile.onnx_dir / "manifest.json")
    ARTIFACTS = profile.artifacts
    HAR_DIR = profile.har_dir
    HEF_DIR = profile.hef_dir
    RESULTS_DIR = profile.results_dir
    REPORTS_DIR = profile.reports_dir
    ALLS_DIR = profile.alls_dir or (config.PROJECT_ROOT / "alls")
    MODULES = list(profile.modules)
    MODULES_BY_NAME = {m.name: m for m in MODULES}


# Initialize exports from the fused profile without mutating demo3 globals yet.
_sync_exports(get_profile("fused"))


def alls_for(name: str) -> Path | None:
    """fused 每模块可选 .alls(命名 alls/fused_<name>.alls);存在才返回。"""
    p = ALLS_DIR / f"fused_{name}.alls"
    return p if p.is_file() else None


def load_manifest(path: Path | None = None) -> dict | None:
    path = path or MANIFEST_PATH
    if not path.is_file():
        return None
    with open(path) as f:
        return json.load(f)


def check_against_manifest(path: Path | None = None) -> list[str]:
    """交叉校验 fused MODULES 与 manifest 的模块名/risk 是否一致。空 = 一致。"""
    manifest = load_manifest(path)
    if manifest is None:
        return [f"manifest 缺失:{path or MANIFEST_PATH}(无法交叉校验)"]
    issues: list[str] = []
    man_by_name = {m["name"]: m for m in manifest.get("modules", [])}
    cfg_names, man_names = set(MODULES_BY_NAME), set(man_by_name)
    if cfg_names != man_names:
        only_cfg = sorted(cfg_names - man_names)
        only_man = sorted(man_names - cfg_names)
        if only_cfg:
            issues.append(f"config_fused 有但 manifest 无:{only_cfg}")
        if only_man:
            issues.append(f"manifest 有但 config_fused 无:{only_man}")
    for name in sorted(cfg_names & man_names):
        cfg_risk = MODULES_BY_NAME[name].risk
        man_risk = man_by_name[name].get("hailo_risk")
        if man_risk and cfg_risk != man_risk:
            issues.append(f"{name} risk 不一致:config_fused={cfg_risk} manifest={man_risk}")
    return issues


def activate() -> None:
    """Apply the fused profile onto ``hailo_deploy.config`` globals (idempotent)."""
    profile = config.apply_profile("fused")
    _sync_exports(profile)
