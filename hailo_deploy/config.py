"""集中配置:硬件分线、路径、模块风险顺序登记表 + 命名 profile。

改路径/加模块只动这里。Prefer ``get_profile("demo3"|"fused"|"fullmodule")`` over
mutating globals; ``apply_profile`` / ``config_fused.activate`` remain for
legacy scripts that still read module-level ``config.ONNX_DIR`` etc.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

# --- 硬件分线(guide §1,记死)-----------------------------------------------
# Hailo-8 -> DFC 3.x(当前 3.33.x)-> hw_arch="hailo8"。
# 绝不用 DFC 5.x / hailo10 / hailo15(那是别的硬件)。
HW_ARCH = "hailo8"

# --- 路径 --------------------------------------------------------------------
PKG_ROOT = Path(__file__).resolve().parent            # .../deplyment-demo/hailo_deploy
PROJECT_ROOT = PKG_ROOT.parent                        # .../deplyment-demo

# ONNX 来源:默认上游 backbone-demo 的导出目录(tools/export_modules.py 产物,
# 已 onnxsim + ORT 对齐)。正式在 suite 容器跑可用环境变量改指
# /local/shared_with_docker/onnx(start-hailo-suite.sh 会把 ONNX 拷进去)。
ONNX_DIR = Path(os.environ.get(
    "HAILO_ONNX_DIR", "/workspace/demo/backbone-demo/deploy/onnx"))

ARTIFACTS = PROJECT_ROOT / "artifacts"
HAR_DIR = ARTIFACTS / "har"          # parse 产出的 *_hailo_model.har
HEF_DIR = ARTIFACTS / "hef"          # compile 产出的 *.hef + *.auto.alls
RESULTS_DIR = ARTIFACTS / "results"  # 结构化 json 结果(parse / native 矩阵)
REPORTS_DIR = ARTIFACTS / "reports"  # markdown 报告

# 数值比对随机输入的种子(可复现)
SEED = 1234

FULLMODULE_NAMES = (
    "sender_backbone", "bn_align_h", "bn_g_a", "bn_h_a", "bn_h_s", "bn_g_s",
    "receiver_stage4", "receiver_stage5", "receiver_dappm", "receiver_head",
)


@dataclass(frozen=True)
class Module:
    """一个待验证的子图模块。"""
    name: str                       # registry 名(唯一)+ HAR 前缀
    risk: str                       # 风险等级标签,仅用于报告可读性
    desc: str                       # 它在验什么
    onnx_stem: str | None = None    # ONNX 文件名(不含 .onnx);None 则用 name
    #   解耦原因:不同骨干可能有同名 ONNX(如两个 full_backbone.onnx),
    #   registry 名加前缀去重,onnx_stem 仍指向真实文件名。

    @property
    def onnx_name(self) -> str:
        return f"{self.onnx_stem or self.name}.onnx"

    @property
    def har_name(self) -> str:
        return f"{self.name}_hailo_model.har"


# 按风险顺序(demo3_prompt「落地顺序」/ guide §7),先小后大隔离风险。
# 一个失败不挡其余 —— 一轮跑完拿到完整 pass/fail 矩阵。
DEMO3_MODULES: list[Module] = [
    Module("conv_sanity",    "low",  "stem+stage1~3 纯卷积,隔离环境,零算子风险"),
    Module("gfa",            "high", "#1 风险:GFA grouped double-normalization"),
    Module("cross_res_attn", "high", "cross-res 注意力,裸 matmul→MHA 映射(要盯)"),
    Module("rtformer_stage", "mid",  "一个完整 stepped block(GFA+CA+2×FFN)"),
    Module("full_backbone",  "mid",  "整网骨干(DAPPM/上采样/动态轴残留一起验)"),
    Module("full_net_seg_depth", "mid",
           "整网 backbone + seg/depth 头(头尾部算子集火:Resize/Sigmoid)"),
    # --- MiT-B0 (SegFormer 编码器) 模块,ONNX 在 onnx_mit/(用 HAILO_ONNX_DIR 指定)---
    Module("mit_patch_embed",   "low",  "OverlapPatchEmbed:stride-Conv + LayerNorm",
           onnx_stem="patch_embed"),
    Module("mit_attn_sr",       "high", "#1:SR 注意力(非对称 Q/KV + 5D permute→Gather)",
           onnx_stem="attn_sr"),
    Module("mit_attn_full",     "high", "无 SR 全注意力(stage4,5D permute→Gather)",
           onnx_stem="attn_full"),
    Module("mit_mix_ffn",       "mid",  "Mix-FFN:Linear + 3×3 dwConv + GELU(Erf)",
           onnx_stem="mix_ffn"),
    Module("mit_block",         "mid",  "完整 transformer block(LN+attn+MixFFN+残差)",
           onnx_stem="block"),
    Module("mit_full_backbone", "mid",  "整个 MiT-B0 编码器(4 stage,540 nodes)",
           onnx_stem="full_backbone"),
]

FUSED_MODULES: list[Module] = [
    Module("sender_backbone", "low",
           "stem+stage1~3 纯卷积 sender;Conv/Add/Relu/Resize,demo3 conv_sanity 同类已过"),
    Module("bn_align_h", "high",
           "align_h stride-2 + GDN(Pow/Sqrt/Div/Mul);Hailo 无原生 GDN → 新验证重点"),
    Module("bn_g_a", "high",
           "concat(aligned_x_h,x_l)->g_a 首块 GDN + 后续 Conv;与 bn_align_h 同风险族"),
    Module("bn_h_a", "low",
           "hyper analysis y->z,标准 Conv+ReLU 栈,无 GDN"),
    Module("bn_h_s", "low",
           "hyper synthesis z_hat->means/scales,Conv+ReLU+Resize;高斯反量化在 CPU 不在图内"),
    Module("bn_g_s", "high",
           "y_hat->x_h/x_l 重建:BoundedIGDN×3(Pow/Sqrt/Mul/Clip)+bilinear Resize;"
           "receiver 侧最大 GDN 风险 → 新验证重点"),
    Module("receiver_stage4", "medium",
           "x_h/x_l->stage4_high/stage4_low;RTFormer GFA+cross-res attention;"
           "输出必须原样级联到 receiver_stage5"),
    Module("receiver_stage5", "medium",
           "stage4_high/stage4_low->high_feat/low_feat;RTFormer GFA+cross-res attention;"
           "输出必须原样级联到 receiver_dappm"),
    Module("receiver_dappm", "low",
           "DAPPM(s32) 上下文融合到 s8;Conv/Pool/Resize,demo3 full_backbone 同类已过"),
    Module("receiver_head", "low",
           "passthrough neck + identity decoder + seg 头(3x3+1x1,up×8);"
           "Conv/BN/ReLU/Resize,demo3 full_net_seg_depth 头尾已过 parse"),
]

# Active demo3 defaults (mutated only by apply_profile for legacy callers).
MODULES: list[Module] = list(DEMO3_MODULES)
MODULES_BY_NAME = {m.name: m for m in MODULES}


@dataclass(frozen=True)
class DeployProfile:
    """Immutable deployment profile — prefer passing this over mutating globals."""

    name: str
    onnx_dir: Path
    artifacts: Path
    modules: tuple[Module, ...]
    manifest_path: Path | None = None
    alls_dir: Path | None = None

    @property
    def har_dir(self) -> Path:
        return self.artifacts / "har"

    @property
    def hef_dir(self) -> Path:
        return self.artifacts / "hef"

    @property
    def results_dir(self) -> Path:
        return self.artifacts / "results"

    @property
    def reports_dir(self) -> Path:
        return self.artifacts / "reports"

    @property
    def modules_by_name(self) -> dict[str, Module]:
        return {m.name: m for m in self.modules}

    def ensure_dirs(self) -> None:
        for d in (self.har_dir, self.hef_dir, self.results_dir, self.reports_dir):
            d.mkdir(parents=True, exist_ok=True)

    def select(self, names: Sequence[str] | None) -> list[Module]:
        if not names:
            return list(self.modules)
        unknown = [n for n in names if n not in self.modules_by_name]
        if unknown:
            raise KeyError(f"未知模块 {unknown};可选:{list(self.modules_by_name)}")
        wanted = set(names)
        return [m for m in self.modules if m.name in wanted]

    def with_onnx_dir(self, onnx_dir: Path) -> DeployProfile:
        return replace(self, onnx_dir=onnx_dir)

    def with_artifacts(self, artifacts: Path) -> DeployProfile:
        return replace(self, artifacts=artifacts)


def _demo3_profile() -> DeployProfile:
    return DeployProfile(
        name="demo3",
        onnx_dir=Path(os.environ.get(
            "HAILO_ONNX_DIR", "/workspace/demo/backbone-demo/deploy/onnx")),
        artifacts=PROJECT_ROOT / "artifacts",
        modules=tuple(DEMO3_MODULES),
    )


def _fused_profile() -> DeployProfile:
    onnx_dir = Path(os.environ.get(
        "HAILO_ONNX_DIR", "/local/shared_with_docker/onnx/modules"))
    artifacts = Path(os.environ.get(
        "HAILO_ARTIFACTS_DIR",
        "/local/shared_with_docker/deplyment-demo/artifacts")) / "fused"
    return DeployProfile(
        name="fused",
        onnx_dir=onnx_dir,
        artifacts=artifacts,
        modules=tuple(FUSED_MODULES),
        manifest_path=Path(os.environ.get("HAILO_MANIFEST", str(onnx_dir / "manifest.json"))),
        alls_dir=PROJECT_ROOT / "alls",
    )


def _fullmodule_profile() -> DeployProfile:
    """Same mid-split module set as fused; paths typically overridden by handoff."""
    base = _fused_profile()
    return replace(base, name="fullmodule")


def get_profile(name: str) -> DeployProfile:
    profiles = {
        "demo3": _demo3_profile,
        "fused": _fused_profile,
        "fullmodule": _fullmodule_profile,
    }
    if name not in profiles:
        raise KeyError(f"unknown profile {name!r}; choose from {sorted(profiles)}")
    return profiles[name]()


def apply_profile(profile: DeployProfile | str) -> DeployProfile:
    """Compat shim: copy a profile onto module-level config globals."""
    if isinstance(profile, str):
        profile = get_profile(profile)
    global ONNX_DIR, ARTIFACTS, HAR_DIR, HEF_DIR, RESULTS_DIR, REPORTS_DIR
    global MODULES, MODULES_BY_NAME
    ONNX_DIR = profile.onnx_dir
    ARTIFACTS = profile.artifacts
    HAR_DIR = profile.har_dir
    HEF_DIR = profile.hef_dir
    RESULTS_DIR = profile.results_dir
    REPORTS_DIR = profile.reports_dir
    MODULES = list(profile.modules)
    MODULES_BY_NAME = {m.name: m for m in MODULES}
    return profile


def ensure_dirs() -> None:
    for d in (HAR_DIR, HEF_DIR, RESULTS_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def select(names: list[str] | None) -> list[Module]:
    """按名字筛选模块,保持登记表里的风险顺序;名字未知则报错。"""
    if not names:
        return list(MODULES)
    unknown = [n for n in names if n not in MODULES_BY_NAME]
    if unknown:
        raise KeyError(f"未知模块 {unknown};可选:{list(MODULES_BY_NAME)}")
    return [m for m in MODULES if m.name in set(names)]
