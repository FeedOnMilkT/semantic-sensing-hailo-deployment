#!/usr/bin/env python3
"""Fused mid-split · compile 规模 sweep —— dummy 随机校准,只读 # of Contexts。

⚠️ 量化用随机数据,掉点无意义;只验「能否编译 + 切几个 context」(guide §8/§9)。

挂靠 demo hailo_deploy:复用 compile._random_calib / compile.num_contexts /
onnx_io.signature / reporting.append_compile_section;唯一增量 = 对 attention 模块
receiver_stage45 自动追加 alls/fused_receiver_stage45.alls(force_range_in + §6.3),
绕过随机校准在 attn@v matmul 的退化 range(shift delta)。demo compile.py 一字未改。

  python scripts/run_fused_compile.py --only sender_backbone receiver_stage45
  python scripts/run_fused_compile.py                       # 全部 9 模块

产物:artifacts/fused/hef/<name>.hef + .auto.alls,artifacts/fused/results/fused_compile.json,
报告追加进 artifacts/fused/reports/pipeline.md。
"""
import argparse
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hailo_deploy import config_fused
from hailo_deploy import compile as cmpl
from hailo_deploy import config, reporting
from hailo_deploy.compile import CompileResult
from hailo_deploy.onnx_io import signature


def compile_one(module, n_calib: int, opt_level: int) -> CompileResult:
    """单模块 compile:复用 demo compile 的 calib/contexts,叠加 fused 每模块 .alls。"""
    notes: list[str] = []
    har = config.HAR_DIR / module.har_name
    onnx = config.ONNX_DIR / module.onnx_name
    if not har.is_file():
        return CompileResult(module.name, "SKIP", error=f"缺 HAR:{har}", desc=module.desc)
    try:
        # 延迟 import:SDK 缺失变 FAIL 结果而非中断整轮 sweep(一失败不停)。
        from hailo_sdk_client import ClientRunner

        sig = signature(onnx)
        runner = ClientRunner(har=str(har))
        calib = cmpl._random_calib(runner, sig, n_calib, config.SEED, notes)

        script = (f"model_optimization_flavor(optimization_level={opt_level}, "
                  f"compression_level=0)\n")
        alls = config_fused.alls_for(module.name)
        if alls is not None:
            script += "\n" + alls.read_text() + "\n"
            notes.append(f"追加每模块 .alls:{alls.name}")

        runner.load_model_script(script)
        runner.optimize(calib)
        hef = runner.compile()

        n_ctx = cmpl.num_contexts(runner)
        hef_path = config.HEF_DIR / f"{module.name}.hef"
        hef_path.write_bytes(hef)
        auto = config.HEF_DIR / f"{module.name}.auto.alls"
        try:
            runner.save_autogen_allocation_script(str(auto))
        except Exception as e:
            notes.append(f"auto.alls 保存失败(不阻断):{type(e).__name__}: {e}")
            auto = None

        return CompileResult(
            module.name, "PASS", num_contexts=n_ctx, hef_bytes=len(hef),
            hef_path=str(hef_path), auto_alls_path=str(auto) if auto else None,
            n_calib=n_calib, opt_level=opt_level, notes=notes, desc=module.desc)
    except Exception as e:
        return CompileResult(module.name, "FAIL", n_calib=n_calib, opt_level=opt_level,
                             error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
                             notes=notes, desc=module.desc)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="只编这些模块")
    ap.add_argument("--n-calib", type=int, default=8, help="随机校准张数(level0 够用)")
    ap.add_argument("--opt-level", type=int, default=0, help="optimization_level(0 最快)")
    ap.add_argument("--section", default="Fused mid-split · compile 规模(dummy 校准)")
    ap.add_argument("--results-name", default="fused_compile")
    args = ap.parse_args()

    config_fused.activate()
    modules = config.select(args.only)
    config.ensure_dirs()
    print(f"== fused compile sweep · hw={config.HW_ARCH} · onnx={config.ONNX_DIR} · "
          f"level={args.opt_level} · calib={args.n_calib}(随机)==")
    print(f"   模块:{[m.name for m in modules]}\n")

    results = [compile_one(m, args.n_calib, args.opt_level) for m in modules]
    for r in results:
        ctx = "—" if r.num_contexts is None else r.num_contexts
        print(f"  [{r.status}] {r.name:20s} contexts={ctx}"
              + (f"  hef={r.hef_bytes/1e6:.1f}MB" if r.hef_bytes else "")
              + (f"  ERR={r.error.splitlines()[0]}" if r.error else ""))

    json_path = config.RESULTS_DIR / f"{args.results_name}.json"
    json_path.write_text(json.dumps(
        {"hw_arch": config.HW_ARCH, "onnx_dir": str(config.ONNX_DIR),
         "note": "dummy random calibration; contexts/HEF-size only, accuracy meaningless",
         "compile": [r.as_dict() for r in results]}, ensure_ascii=False, indent=2))
    reporting.append_compile_section(results, args.section, config.REPORTS_DIR / "pipeline.md")
    print(f"\nJSON:{json_path}\n已追加节「{args.section}」到 {config.REPORTS_DIR / 'pipeline.md'}")

    return 1 if any(r.status == "FAIL" for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
