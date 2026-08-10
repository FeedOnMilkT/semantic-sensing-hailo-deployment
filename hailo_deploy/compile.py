"""阶段3 Compile(→ HEF)+ 读 # of Contexts —— 用【dummy 随机校准】只看编译规模。

⚠️ 重要边界:本模块的量化用**随机数据**,目的【仅】是把状态机推到 compile()、
读出 **# of Contexts**(网络对 26 TOPS 是否偏大的早期信号,guide §8/§9)。
- # of Contexts 由架构/资源分配决定,**几乎与校准数据的值无关** → dummy 足够可信。
- 掉点数字【无意义】,不看、不画 bpp-mIoU。真实掉点需真校准集(另案)。
- 跑的是默认 a8_w8;真实部署若给敏感层上 16-bit,占用只增不减 → 本结果是**乐观下界**。

状态机:load HAR → load_model_script(level0) → optimize(随机 calib) → compile() → 读 contexts。
"""
from __future__ import annotations

import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

from . import config
from .onnx_io import signature, nchw_to_nhwc
from .emulate import _hn_layers_by_role   # 复用:HN 层 -> original_names


@dataclass
class CompileResult:
    name: str
    status: str                       # PASS / FAIL / SKIP
    num_contexts: int | None = None   # = len(integrated_graph.contexts_graph.nodes)
    hef_bytes: int | None = None
    hef_path: str | None = None
    auto_alls_path: str | None = None
    n_calib: int = 0
    opt_level: int = 0
    error: str | None = None
    notes: list[str] = field(default_factory=list)
    desc: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def _random_calib(runner, sig, n: int, seed: int, notes: list[str]):
    """造 n 张随机 NHWC dummy 校准。单输入 -> array;多输入 -> {hn_input: array}。"""
    rng = np.random.default_rng(seed)
    # (n, C, H, W) -> NHWC
    nhwc = {name: nchw_to_nhwc(rng.standard_normal((n, *shape[1:])).astype(np.float32))
            for name, shape in sig.inputs}
    if len(nhwc) == 1:
        return next(iter(nhwc.values()))
    # 多输入:按 original_names 配 HN 输入层名(同 emulate 的做法)
    hn_inputs = _hn_layers_by_role(runner, "input")
    onnx_names = [n for n, _ in sig.inputs]
    feed, leftover = {}, list(hn_inputs)
    for onnx_name in onnx_names:
        match = next((h for h, origs in hn_inputs.items() if onnx_name in origs), None)
        if match is None:
            match = leftover[onnx_names.index(onnx_name)]
            notes.append(f"calib 输入按顺序兜底:{onnx_name} -> {match}")
        feed[match] = nhwc[onnx_name]
    return feed


def num_contexts(runner) -> int | None:
    """从 compile 后 backend 内存里的序列化 integrated graph 读 context 数。

    runner._sdk_backend._integrated_graph 是 ProtoIntegratedHWGraph 的序列化 bytearray;
    # of Contexts = len(contexts_graph.nodes)(每个 node 含一个 context,已核对 proto)。
    """
    from hailo_sdk_client.allocator import integrated_hw_graph_pb2 as ig
    raw = getattr(runner._sdk_backend, "_integrated_graph", None)
    if not raw:
        return None
    g = ig.ProtoIntegratedHWGraph()
    g.ParseFromString(bytes(raw))
    return len(g.contexts_graph.nodes)


def compile_module(module: config.Module,
                   har_path: Path,
                   onnx_path: Path,
                   n_calib: int = 8,
                   opt_level: int = 0,
                   seed: int = config.SEED) -> CompileResult:
    from hailo_sdk_client import ClientRunner

    notes: list[str] = []
    if not har_path.is_file():
        return CompileResult(module.name, "SKIP", error=f"缺 HAR:{har_path}",
                             desc=module.desc)
    try:
        sig = signature(onnx_path)
        runner = ClientRunner(har=str(har_path))
        calib = _random_calib(runner, sig, n_calib, seed, notes)

        # 仅设 optimization_level=0(最快,精度无所谓);默认 a8_w8,不加其它 .alls
        runner.load_model_script(
            f"model_optimization_flavor(optimization_level={opt_level}, "
            f"compression_level=0)\n")
        runner.optimize(calib)              # 随机校准:只为推到 compile
        hef = runner.compile()

        n_ctx = num_contexts(runner)
        hef_path = config.HEF_DIR / f"{module.name}.hef"
        hef_path.write_bytes(hef)
        auto_path = config.HEF_DIR / f"{module.name}.auto.alls"
        try:
            runner.save_autogen_allocation_script(str(auto_path))
        except Exception as e:
            notes.append(f"auto.alls 保存失败(不阻断):{type(e).__name__}: {e}")
            auto_path = None

        return CompileResult(
            module.name, "PASS", num_contexts=n_ctx, hef_bytes=len(hef),
            hef_path=str(hef_path),
            auto_alls_path=str(auto_path) if auto_path else None,
            n_calib=n_calib, opt_level=opt_level, notes=notes, desc=module.desc)
    except Exception as e:
        return CompileResult(module.name, "FAIL", n_calib=n_calib, opt_level=opt_level,
                             error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
                             notes=notes, desc=module.desc)


def compile_all(modules: list[config.Module],
                n_calib: int = 8, opt_level: int = 0) -> list[CompileResult]:
    config.ensure_dirs()
    results: list[CompileResult] = []
    for m in modules:
        har = config.HAR_DIR / m.har_name
        onnx = config.ONNX_DIR / m.onnx_name
        results.append(compile_module(m, har, onnx, n_calib, opt_level))
    return results
