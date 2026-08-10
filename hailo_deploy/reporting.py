"""把 parse / native 结果汇成结构化 json + 人读 markdown 报告。"""
from __future__ import annotations

import json
from pathlib import Path

from . import config


def _mark(status: str) -> str:
    return {"PASS": "✅", "FAIL": "❌", "SKIP": "⚠️", "MISMATCH": "🟡"}.get(status, "?")


def dump_json(parse_results, native_results, path: Path) -> None:
    payload = {
        "hw_arch": config.HW_ARCH,
        "onnx_dir": str(config.ONNX_DIR),
        "parse": [r.as_dict() for r in parse_results],
        "native": [r.as_dict() for r in native_results],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def _matrix_lines(parse_results, nat_by_name) -> list[str]:
    out = ["| 模块 | 风险 | parse | NATIVE max\\|diff\\| | 判定 |",
           "|---|---|---|---|---|"]
    for pr in parse_results:
        m = config.MODULES_BY_NAME[pr.name]
        nat = nat_by_name.get(pr.name)
        diff, nat_mark = "—", ""
        if nat is not None:
            if nat.max_abs_diff is not None:
                diff = f"{nat.max_abs_diff:.2e}"
            nat_mark = f"{_mark(nat.status)} {nat.status}"
        out.append(f"| `{pr.name}` | {m.risk} | {_mark(pr.status)} {pr.status} "
                   f"| {diff} | {nat_mark} |")
    return out


def _parse_detail_lines(parse_results, head="###") -> list[str]:
    out = []
    for pr in parse_results:
        out.append(f"{head} `{pr.name}` — {pr.desc}")
        out.append(f"- 状态:{_mark(pr.status)} **{pr.status}**")
        if pr.used_net_input_shapes:
            out.append("- ⚠️ 干净路径失败,动用了 `net_input_shapes` 逃生通道(guide §4)")
        if pr.op_types:
            out.append(f"- HN 算子类型:`{', '.join(pr.op_types)}`")
        if pr.har_path:
            out.append(f"- HAR:`{pr.har_path}`")
        if pr.error:
            out.append(f"- 错误:\n```\n{pr.error.strip()}\n```")
        out.append("")
    return out


def _native_detail_lines(native_results, head="###") -> list[str]:
    out = []
    for nr in native_results:
        out.append(f"{head} `{nr.name}`")
        out.append(f"- 状态:{_mark(nr.status)} **{nr.status}**(阈 {nr.tol:.0e})")
        if nr.per_output:
            for k, v in nr.per_output.items():
                out.append(f"  - `{k}`: max|diff| = {v:.2e}")
        for n in nr.notes:
            out.append(f"- 注:{n}")
        if nr.error:
            out.append(f"- 错误:\n```\n{nr.error.strip()}\n```")
        out.append("")
    return out


def render_markdown(parse_results, native_results, path: Path) -> str:
    nat_by_name = {r.name: r for r in native_results}
    lines: list[str] = []
    lines.append("# Demo 3 · Hailo DFC parse + NATIVE 对齐结果\n")
    lines.append(f"- hw_arch: `{config.HW_ARCH}`(DFC 3.x / Hailo-8,guide §1)")
    lines.append(f"- ONNX 来源:`{config.ONNX_DIR}`")
    lines.append("- 范围:仅 parse + NATIVE(不做校准/量化/compile)\n")
    lines.append("## 结果矩阵\n")
    lines += _matrix_lines(parse_results, nat_by_name)
    lines.append("")
    lines.append("## parse 细节\n")
    lines += _parse_detail_lines(parse_results)
    lines.append("## NATIVE 对齐细节\n")
    lines += _native_detail_lines(native_results)

    text = "\n".join(lines)
    path.write_text(text)
    return text


def append_compile_section(compile_results, title: str, path: Path) -> str:
    """把一轮 compile(dummy 校准,只看规模)结果作为新一节【追加】到报告。"""
    lines = ["", "", "---", "", f"## {title}", "",
             "> ⚠️ dummy 随机校准,**仅看编译可行性 + # of Contexts**;掉点无意义。"
             "默认 a8_w8 → 是真实(可能上 16-bit)的乐观下界。", ""]
    lines.append("| 模块 | compile | # of Contexts | HEF 大小 | 备注 |")
    lines.append("|---|---|---|---|---|")
    for r in compile_results:
        ctx = "—" if r.num_contexts is None else (
            f"**{r.num_contexts}**" + (" ✅单" if r.num_contexts == 1 else " ⚠️多"))
        hef = "—" if r.hef_bytes is None else f"{r.hef_bytes/1e6:.1f} MB"
        note = (r.error.splitlines()[0] if r.error else "; ".join(r.notes)) or ""
        lines.append(f"| `{r.name}` | {_mark(r.status)} {r.status} | {ctx} | {hef} | {note} |")
    lines.append("")
    for r in compile_results:
        if r.error:
            lines.append(f"#### `{r.name}` 错误")
            lines.append(f"```\n{r.error.strip()}\n```")
            lines.append("")
    section = "\n".join(lines)
    existing = path.read_text() if path.exists() else ""
    path.write_text(existing + section)
    return section


def append_section(parse_results, native_results, title: str, path: Path) -> str:
    """把一轮结果作为新一节【追加】到已有报告(同矩阵格式),不覆盖旧内容。"""
    nat_by_name = {r.name: r for r in native_results}
    lines = ["", "", "---", "", f"## {title}", ""]
    lines += _matrix_lines(parse_results, nat_by_name)
    lines.append("")
    lines.append("**parse 细节**")
    lines.append("")
    lines += _parse_detail_lines(parse_results, head="####")
    lines.append("**NATIVE 对齐细节**")
    lines.append("")
    lines += _native_detail_lines(native_results, head="####")
    section = "\n".join(lines)

    existing = path.read_text() if path.exists() else ""
    path.write_text(existing + section)
    return section
