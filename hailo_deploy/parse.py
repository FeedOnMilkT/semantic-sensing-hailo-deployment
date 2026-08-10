"""阶段1 Parse(Translation):ONNX -> Hailo HN -> HAR(guide §3/§4)。

只做 parse,不 optimize/compile —— 先把「算子能否过 parser」这一个问题隔离出来
(demo3_prompt 落地顺序 step 2-3)。核心是验 #1 风险 GFA double-norm。

护栏:ClientRunner / translate_onnx_model 的用法严格照 guide §4,不臆造参数。
干净静态 ONNX 默认不传 start/end/net_input_shapes(让它自动推断);只有 parse
报错时才回退到「带 net_input_shapes 重试」这一条 guide 明示的逃生通道。
"""
from __future__ import annotations

import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path

from . import config
from .onnx_io import signature


@dataclass
class ParseResult:
    name: str
    status: str                       # PASS / FAIL / SKIP
    har_path: str | None = None
    op_types: list[str] = field(default_factory=list)  # HN 层类型,对照 Supported Layers
    used_net_input_shapes: bool = False                # 是否动用了逃生通道
    error: str | None = None
    desc: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def _hn_op_types(runner) -> list[str]:
    """从 parse 后的 HN 取层类型集合,用于人工对照 guide §7 Supported Layers。"""
    try:
        layers = runner.get_hn_dict().get("layers", {})
        return sorted({v.get("type", "?") for v in layers.values()})
    except Exception:
        return []


def parse_module(module: config.Module,
                 onnx_dir: Path,
                 har_dir: Path) -> ParseResult:
    """对单个模块跑 parse。一个模块失败不抛出,返回带 error 的 FAIL 结果。"""
    # 延迟 import:只有真正 parse 时才需要 suite 容器里的 SDK。
    from hailo_sdk_client import ClientRunner

    onnx_path = onnx_dir / module.onnx_name
    if not onnx_path.is_file():
        return ParseResult(module.name, "SKIP",
                           error=f"缺 ONNX:{onnx_path}", desc=module.desc)

    # 1) 干净路径:不带可选参数,让 parser 自动推断(guide §4)
    runner = ClientRunner(hw_arch=config.HW_ARCH)
    try:
        runner.translate_onnx_model(str(onnx_path), module.name)
        used_shapes = False
    except Exception as first_err:
        # 2) 逃生通道:动态轴没固定住才需要 net_input_shapes(guide §4)。
        #    用 ONNX 内省出的静态 shape 显式喂进去,换个全新 runner 重试。
        try:
            sig = signature(onnx_path)
            net_input_shapes = {name: shape for name, shape in sig.inputs}
            runner = ClientRunner(hw_arch=config.HW_ARCH)
            runner.translate_onnx_model(
                str(onnx_path), module.name, net_input_shapes=net_input_shapes)
            used_shapes = True
        except Exception:
            # 两条路都崩 —— 记录干净路径的原始错(更能反映真实算子坑)。
            return ParseResult(
                module.name, "FAIL",
                error=f"{type(first_err).__name__}: {first_err}\n"
                      f"--- 带 net_input_shapes 重试也失败 ---\n{traceback.format_exc()}",
                desc=module.desc)

    har_path = har_dir / module.har_name
    runner.save_har(str(har_path))
    return ParseResult(
        module.name, "PASS",
        har_path=str(har_path),
        op_types=_hn_op_types(runner),
        used_net_input_shapes=used_shapes,
        desc=module.desc,
    )


def parse_all(modules: list[config.Module]) -> list[ParseResult]:
    config.ensure_dirs()
    results: list[ParseResult] = []
    for m in modules:
        try:
            r = parse_module(m, config.ONNX_DIR, config.HAR_DIR)
        except Exception as e:  # 兜底:连 SDK import 都崩等
            r = ParseResult(m.name, "FAIL",
                            error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
                            desc=m.desc)
        results.append(r)
    return results
