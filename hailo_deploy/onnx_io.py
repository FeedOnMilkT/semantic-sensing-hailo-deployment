"""ONNX I/O 内省 + ONNXRuntime 参考推理 + NCHW/NHWC 转换。

NATIVE emulator 比对的「黄金参考」就是 ONNXRuntime(host 上 = PyTorch 的代理,
notes 已验 max|torch-ort|~1e-6)。Hailo NATIVE 应与之一致(guide §5.1)。

布局约定:ONNX/PyTorch 是 NCHW;Hailo infer 吃 NHWC(guide §5.2)。转换在这里集中。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


@dataclass
class IOSig:
    inputs: list[tuple[str, list[int]]]   # [(name, NCHW shape)]
    outputs: list[tuple[str, list[int]]]  # [(name, shape)]


def signature(onnx_path: Path) -> IOSig:
    """读 ONNX 的输入/输出名与静态 shape。要求已是静态 shape(导出时固定)。"""
    m = onnx.load(str(onnx_path))

    def shp(t) -> list[int]:
        dims = []
        for x in t.type.tensor_type.shape.dim:
            if x.dim_value <= 0:
                raise ValueError(
                    f"{onnx_path.name} 的 {t.name} 含动态轴 {x.dim_param!r};"
                    "DFC 需静态 shape(guide §4)。请用 export_modules.py 固定 shape。")
            dims.append(x.dim_value)
        return dims

    return IOSig(
        inputs=[(i.name, shp(i)) for i in m.graph.input],
        outputs=[(o.name, shp(o)) for o in m.graph.output],
    )


def make_inputs(sig: IOSig, seed: int) -> dict[str, np.ndarray]:
    """造一份固定随机输入(NCHW, float32)。仅用于数值对齐,不是校准集。"""
    rng = np.random.default_rng(seed)
    return {name: rng.standard_normal(shape).astype(np.float32)
            for name, shape in sig.inputs}


def run_onnxruntime(onnx_path: Path,
                    inputs_nchw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """跑 ORT,返回 {输出名: NCHW array}。CPU provider 足够(只比数值)。"""
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out_names = [o.name for o in sess.get_outputs()]
    outs = sess.run(out_names, inputs_nchw)
    return dict(zip(out_names, outs))


def nchw_to_nhwc(a: np.ndarray) -> np.ndarray:
    """(N,C,H,W) -> (N,H,W,C)。非 4D 原样返回。"""
    return np.transpose(a, (0, 2, 3, 1)) if a.ndim == 4 else a


def nhwc_to_nchw(a: np.ndarray) -> np.ndarray:
    """(N,H,W,C) -> (N,C,H,W)。非 4D 原样返回。"""
    return np.transpose(a, (0, 3, 1, 2)) if a.ndim == 4 else a
