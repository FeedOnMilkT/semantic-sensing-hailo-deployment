#!/usr/bin/env python3
"""Parse → NATIVE pipeline for demo3 or fused profiles."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hailo_deploy import config, emulate, parse, reporting


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--profile",
        choices=["demo3", "fused", "fullmodule"],
        default="demo3",
        help="module/path profile (default: demo3)",
    )
    ap.add_argument("--only", nargs="*", default=None, help="只跑这些模块")
    ap.add_argument("--no-native", action="store_true", help="只 parse,跳过 NATIVE 对齐")
    ap.add_argument(
        "--section",
        default=None,
        help="给定则【追加】一节到 pipeline.md(不覆盖),并把 json 写成独立文件",
    )
    ap.add_argument(
        "--results-name",
        default=None,
        help="追加模式下 results/<name>.json 的文件名",
    )
    ap.add_argument("--onnx-dir", type=Path, default=None)
    ap.add_argument("--artifacts-dir", type=Path, default=None)
    args = ap.parse_args(argv)

    profile = config.get_profile(args.profile)
    if args.onnx_dir is not None:
        profile = profile.with_onnx_dir(args.onnx_dir)
    if args.artifacts_dir is not None:
        profile = profile.with_artifacts(args.artifacts_dir)
    config.apply_profile(profile)

    modules = profile.select(args.only)
    profile.ensure_dirs()

    print(f"== parse/NATIVE · profile={profile.name} · hw_arch={config.HW_ARCH} · onnx={profile.onnx_dir} ==")
    print(f"   modules: {[m.name for m in modules]}\n")

    print("--- Gate 2: parse ---")
    parse_results = parse.parse_all(modules)
    for r in parse_results:
        print(
            f"  [{r.status}] {r.name}"
            + (f"  ops={','.join(r.op_types)}" if r.op_types else "")
            + (f"  ERR={r.error.splitlines()[0]}" if r.error else "")
        )

    if args.no_native:
        native_results = [
            emulate.NativeResult(r.name, "SKIP", notes=["--no-native"]) for r in parse_results
        ]
    else:
        print("\n--- Gate 3: NATIVE ---")
        native_results = emulate.native_check_all(parse_results)
        for r in native_results:
            d = f"{r.max_abs_diff:.2e}" if r.max_abs_diff is not None else "—"
            print(
                f"  [{r.status}] {r.name}  max|diff|={d}"
                + (f"  ERR={r.error.splitlines()[0]}" if r.error else "")
            )

    md_path = profile.reports_dir / "pipeline.md"
    if args.section:
        slug = args.results_name or (
            "_".join(m.name for m in modules) if len(modules) <= 2 else "section"
        )
        json_path = profile.results_dir / f"{slug}.json"
        reporting.dump_json(parse_results, native_results, json_path)
        reporting.append_section(parse_results, native_results, args.section, md_path)
        print(f"\n已【追加】节「{args.section}」到:{md_path}\nJSON:{json_path}")
    else:
        json_path = profile.results_dir / "pipeline.json"
        reporting.dump_json(parse_results, native_results, json_path)
        reporting.render_markdown(parse_results, native_results, md_path)
        print(f"\n报告:{md_path}\nJSON:{json_path}")
    n_fail = sum(1 for r in parse_results if r.status == "FAIL")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
