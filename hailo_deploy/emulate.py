"""NATIVE emulator 数值对齐(验收线 ②,guide §5.1)。

逻辑:同一份固定输入,分别喂
  - ONNXRuntime(NCHW)  = 黄金参考(host 上代理 PyTorch)
  - Hailo SDK_NATIVE(NHWC)= parse 后的 HN 参考实现
比 max|diff|。NATIVE 不一致 => parse 坏了(算子翻译/切分),不是量化问题。
不涉及校准、不涉及量化 —— 只喂一份样本,纯验 parse 数值语义。

⚠️ 已知脆弱点(会如实写进报告):
  - Hailo HN 的输入/输出层名 ≠ ONNX 原名,多输入/多输出时要做映射。这里按
    「ONNX 名是否为 HN 名子串」匹配,失败回退到「按 shape 匹配」,再不行按顺序。
  - 任何映射不确定都标注在 notes,不假造对齐数字。
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

from . import config
from .onnx_io import IOSig, signature, make_inputs, run_onnxruntime, nchw_to_nhwc, nhwc_to_nchw


@dataclass
class NativeResult:
    name: str
    status: str                                  # PASS / MISMATCH / FAIL / SKIP
    max_abs_diff: float | None = None            # 所有输出里最大的 max|diff|
    per_output: dict[str, float] = field(default_factory=dict)
    tol: float = 1e-3                            # 与 notes 一致的对齐阈
    notes: list[str] = field(default_factory=list)
    error: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def _hn_layers_by_role(runner, role: str) -> dict[str, list[str]]:
    """取 HN 中 input/output 层 -> 其 original_names(ONNX 原名)。

    findings §1.3:HN 层名(input_layer1/output_layer2)≠ONNX 原名,但每层带
    original_names 记录原名,据此可【按名精确配】,不靠顺序猜。
    """
    layers = runner.get_hn_dict().get("layers", {})
    return {k: list(v.get("original_names") or [])
            for k, v in layers.items() if role in str(v.get("type", "")).lower()}


def _build_native_feed(runner, sig: IOSig, inputs_nchw: dict[str, np.ndarray],
                       notes: list[str]):
    """把 NCHW 输入转 NHWC,并对齐到 Hailo NATIVE infer 期望的格式。

    单输入:直接给 array。多输入:按 HN original_names 配 {hn_input_name: array}。
    """
    nhwc = {name: nchw_to_nhwc(arr) for name, arr in inputs_nchw.items()}
    if len(nhwc) == 1:
        return next(iter(nhwc.values()))

    # 多输入:按名配(原名在哪个 HN 输入层的 original_names 里)
    hn_inputs = _hn_layers_by_role(runner, "input")     # {hn_name: [orig...]}
    onnx_names = [n for n, _ in sig.inputs]
    feed: dict[str, np.ndarray] = {}
    leftover_hn = list(hn_inputs)
    for onnx_name in onnx_names:
        match = next((h for h, origs in hn_inputs.items() if onnx_name in origs), None)
        if match is None:                                # 退回顺序兜底
            match = leftover_hn[onnx_names.index(onnx_name)]
            notes.append(f"输入按顺序兜底:{onnx_name} -> {match}(original_names 未对上)")
        feed[match] = nhwc[onnx_name]
    return feed


def _to_array_list(native_res) -> list[np.ndarray]:
    """Hailo infer 的返回在不同版本可能是 array / list / dict,统一成 list。"""
    if isinstance(native_res, dict):
        return [np.asarray(v) for v in native_res.values()]
    if isinstance(native_res, (list, tuple)):
        return [np.asarray(v) for v in native_res]
    return [np.asarray(native_res)]


def _match_outputs_by_shape(ort_outs: dict[str, np.ndarray],
                            native_nchw: list[np.ndarray],
                            notes: list[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """ORT 输出(有名)与 Hailo 输出按 shape 配对(兜底路径)。"""
    pairs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    remaining = list(native_nchw)
    for oname, oarr in ort_outs.items():
        hit = next((h for h in remaining if h.shape == oarr.shape), None)
        if hit is None:
            notes.append(f"输出 {oname}{list(oarr.shape)} 在 Hailo 端找不到同 shape 对应")
            continue
        remaining.remove(hit)
        pairs[oname] = (oarr, hit)
    if remaining:
        notes.append(f"Hailo 端还剩 {len(remaining)} 个输出没配上(shape "
                     f"{[list(r.shape) for r in remaining]})")
    return pairs


def _match_outputs(runner, ort_outs: dict[str, np.ndarray], native_res,
                   notes: list[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """优先按名配(infer 返回 dict 时,用 HN original_names 对到 ONNX 输出名);
    拿不到可靠名再退回 shape 配(findings §1.3)。"""
    if isinstance(native_res, dict):
        hn_out = _hn_layers_by_role(runner, "output")    # {hn_name: [orig...]}
        pairs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        ok = True
        for hn_name, arr in native_res.items():
            orig = next((o for o in hn_out.get(hn_name, []) if o in ort_outs), None)
            if orig is None:
                ok = False
                break
            pairs[orig] = (ort_outs[orig], nhwc_to_nchw(np.asarray(arr)))
        if ok and len(pairs) == len(ort_outs):
            notes.append("输出按名配(HN original_names ↔ ONNX 输出名)")
            return pairs
        notes.append("infer 返回 dict 但 original_names 未能全配,退回 shape 兜底")
    native_nchw = [nhwc_to_nchw(a) for a in _to_array_list(native_res)]
    notes.append("输出按 shape 兜底配")
    return _match_outputs_by_shape(ort_outs, native_nchw, notes)


def native_check(module: config.Module,
                 har_path: Path,
                 onnx_path: Path,
                 seed: int = config.SEED,
                 tol: float = 1e-3) -> NativeResult:
    from hailo_sdk_client import ClientRunner, InferenceContext

    notes: list[str] = []
    try:
        sig = signature(onnx_path)
        inputs_nchw = make_inputs(sig, seed)

        # 黄金参考
        ort_outs = run_onnxruntime(onnx_path, inputs_nchw)

        # Hailo NATIVE
        runner = ClientRunner(har=str(har_path))
        feed = _build_native_feed(runner, sig, inputs_nchw, notes)
        with runner.infer_context(InferenceContext.SDK_NATIVE) as ctx:
            native_res = runner.infer(ctx, feed)

        pairs = _match_outputs(runner, ort_outs, native_res, notes)

        if not pairs:
            return NativeResult(module.name, "FAIL", tol=tol, notes=notes,
                                error="没有任何输出能在两端配对,无法比较")

        per_output = {name: float(np.max(np.abs(o - h))) for name, (o, h) in pairs.items()}
        worst = max(per_output.values())
        status = "PASS" if worst < tol else "MISMATCH"
        return NativeResult(module.name, status, max_abs_diff=worst,
                            per_output=per_output, tol=tol, notes=notes)
    except Exception as e:
        import traceback
        return NativeResult(module.name, "FAIL", tol=tol, notes=notes,
                            error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


def native_check_all(parse_results) -> list[NativeResult]:
    """只对 parse PASS 的模块做 NATIVE 比对。"""
    results: list[NativeResult] = []
    for pr in parse_results:
        module = config.MODULES_BY_NAME[pr.name]
        if pr.status != "PASS" or not pr.har_path:
            results.append(NativeResult(pr.name, "SKIP",
                                        notes=[f"parse 未 PASS({pr.status}),跳过"]))
            continue
        r = native_check(module, Path(pr.har_path),
                         config.ONNX_DIR / module.onnx_name)
        results.append(r)
    return results
