#!/usr/bin/env python3
"""Demo 3 · compile 规模 sweep —— dummy 随机校准,只读 # of Contexts。

⚠️ 量化用随机数据,掉点无意义;只验「能否编译 + 切几个 context」(guide §8/§9)。

  # RTFormer 那几个(ONNX 在默认目录)
  python scripts/run_compile.py --only conv_sanity gfa cross_res_attn rtformer_stage \
                                       full_backbone full_net_seg_depth \
         --section "RTFormer compile 规模(dummy 校准)" --results-name rtformer_compile

  # MiT 整网(ONNX 在 onnx_mit)
  HAILO_ONNX_DIR=/workspace/demo/backbone-demo/deploy/onnx_mit \
    python scripts/run_compile.py --only mit_full_backbone \
      --section "MiT-B0 compile 规模(dummy 校准)" --results-name mit_compile

产物:artifacts/hef/<name>.hef + .auto.alls,results/<name>.json,报告追加进 pipeline.md。
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if __name__ == "__main__":
    from run_hailo_logged import reexec_if_unlogged
    reexec_if_unlogged(Path(__file__), sys.argv[1:])

from hailo_deploy import config, compile as cmpl, reporting


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="只编这些模块")
    ap.add_argument("--n-calib", type=int, default=8, help="随机校准张数(level0 够用)")
    ap.add_argument("--opt-level", type=int, default=0, help="optimization_level(0 最快)")
    ap.add_argument("--section", default="compile 规模(dummy 校准)")
    ap.add_argument("--results-name", default="compile")
    args = ap.parse_args()

    modules = config.select(args.only)
    config.ensure_dirs()
    print(f"== compile sweep · hw={config.HW_ARCH} · onnx={config.ONNX_DIR} · "
          f"level={args.opt_level} · calib={args.n_calib}(随机)==")
    print(f"   模块:{[m.name for m in modules]}\n")

    results = cmpl.compile_all(modules, n_calib=args.n_calib, opt_level=args.opt_level)
    for r in results:
        ctx = "—" if r.num_contexts is None else r.num_contexts
        print(f"  [{r.status}] {r.name:20s} contexts={ctx}"
              + (f"  hef={r.hef_bytes/1e6:.1f}MB" if r.hef_bytes else "")
              + (f"  ERR={r.error.splitlines()[0]}" if r.error else ""))

    json_path = config.RESULTS_DIR / f"{args.results_name}.json"
    import json
    json_path.write_text(json.dumps(
        {"hw_arch": config.HW_ARCH, "onnx_dir": str(config.ONNX_DIR),
         "note": "dummy random calibration; contexts/HEF-size only, accuracy meaningless",
         "compile": [r.as_dict() for r in results]}, ensure_ascii=False, indent=2))
    reporting.append_compile_section(results, args.section, config.REPORTS_DIR / "pipeline.md")
    print(f"\nJSON:{json_path}\n已追加节「{args.section}」到 pipeline.md")

    return 1 if any(r.status == "FAIL" for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
