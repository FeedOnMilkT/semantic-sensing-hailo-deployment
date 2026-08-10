#!/usr/bin/env python3
"""Freeze and deploy the two bounded CPU/Hailo bottleneck fallback arms."""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from board_eval.deploy import common_options, stream_upload
from hailo_deploy.io.hash import sha256


SEQUENCE = (
    "sender_backbone", "bn_align_h", "bn_g_a", "bn_h_a", "bn_h_s", "bn_g_s",
    "receiver_stage4", "receiver_stage5", "receiver_dappm", "receiver_head",
)
ARMS = {
    "hs_cpu": {"bn_h_s"},
    "gs_only_hailo": {"bn_align_h", "bn_g_a", "bn_h_a", "bn_h_s"},
}


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def checked_file(record: dict, label: str) -> Path:
    path = Path(record["path"]).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256(path)
    if actual != record["sha256"]:
        raise ValueError(f"{label} SHA mismatch: {actual} != {record['sha256']}")
    return path


def build_manifest(
    arm: str,
    selection: dict,
    onnx_manifest: dict,
    onnx_root: Path,
    codec_dir: Path,
    ort_threads: int,
) -> tuple[dict, dict[str, Path], dict[str, Path]]:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    cpu_modules = ARMS[arm]
    onnx_source = {item["name"]: item for item in onnx_manifest["modules"]}
    if set(onnx_source) != set(SEQUENCE):
        raise ValueError("ONNX manifest does not contain the frozen ten-module sequence")
    if set(selection["modules"]) != set(SEQUENCE):
        raise ValueError("selection manifest does not contain the frozen ten-module sequence")
    if selection["checkpoint_sha256"] != onnx_manifest["ckpt_sha256"]:
        raise ValueError("selection/ONNX checkpoint mismatch")

    hefs: dict[str, Path] = {}
    onnx_files: dict[str, Path] = {}
    modules: list[dict] = []
    for name in SEQUENCE:
        selected = selection["modules"][name]
        source = onnx_source[name]
        spec = {
            "name": name,
            "input_names": source["input_names"],
            "output_names": source["output_names"],
            "in_shapes_nchw": source["in_shapes_nchw"],
            "out_shapes_nchw": source["out_shapes_nchw"],
        }
        if name in cpu_modules:
            onnx_path = (onnx_root / f"{name}.onnx").resolve()
            if not onnx_path.is_file() or sha256(onnx_path) != source["sha256"]:
                raise ValueError(f"{name}: ONNX file missing or SHA mismatch")
            onnx_files[name] = onnx_path
            spec.update({
                "engine": "ort_cpu_fp32",
                "onnx_filename": f"onnx/{name}.onnx",
                "onnx_sha256": source["sha256"],
                "onnx_bytes": onnx_path.stat().st_size,
            })
        else:
            hef = checked_file(selected["hef"], f"{name} HEF")
            hefs[name] = hef
            spec.update({
                "engine": "hailort_float32_vstream",
                "hef_filename": f"hef/{name}.hef",
                "hef_sha256": selected["hef"]["sha256"],
                "hef_bytes": hef.stat().st_size,
                "num_contexts": selected.get("num_contexts"),
            })
        modules.append(spec)

    codec_metadata = codec_dir / "codec_runtime.json"
    codec_tables = codec_dir / "codec_tables.npz"
    for path in (codec_metadata, codec_tables):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = {
        "schema_version": 1,
        "kind": "semantic_sensing_pi_hybrid_ort_runtime",
        "arm": arm,
        "checkpoint_sha256": selection["checkpoint_sha256"],
        "selection_candidate_fingerprint": selection["candidate_fingerprint"],
        "sequence": list(SEQUENCE),
        "cpu_modules": sorted(cpu_modules),
        "hailo_modules": [name for name in SEQUENCE if name not in cpu_modules],
        "ort_threads": ort_threads,
        "modules": modules,
        "codec": {
            "metadata_filename": codec_metadata.name,
            "metadata_sha256": sha256(codec_metadata),
            "tables_filename": codec_tables.name,
            "tables_sha256": sha256(codec_tables),
        },
        "thermal_guard": {
            "max_start_temp_c": 70.0,
            "resume_temp_c": 60.0,
            "poll_seconds": 2.0,
            "max_wait_seconds": 600.0,
        },
        "timing_policy": {
            "serial_stages": True,
            "no_overlap": True,
            "temperature_cooldown_excluded_from_total_pipeline_ms": True,
        },
    }
    return manifest, hefs, onnx_files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--onnx-manifest", type=Path, required=True)
    parser.add_argument("--codec-runtime", type=Path, required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ort-threads", type=int, default=4)
    parser.add_argument("--remote-root", default=".local/share/semantic-sensing-board/hybrid-ort")
    parser.add_argument(
        "--remote-ort-python",
        default="/home/sidney/.local/share/semantic-sensing-board/venv-ort128/bin/python",
    )
    parser.add_argument("--ssh-target", required=True)
    parser.add_argument("--identity-file", type=Path)
    parser.add_argument("--proxy-command")
    parser.add_argument("--known-hosts", type=Path)
    args = parser.parse_args()
    if args.ort_threads <= 0:
        raise ValueError("--ort-threads must be positive")
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite deployment report: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    selection_path = args.selection_manifest.resolve()
    onnx_manifest_path = args.onnx_manifest.resolve()
    codec_dir = args.codec_runtime.resolve()
    selection = load_json(selection_path)
    onnx_manifest = load_json(onnx_manifest_path)
    onnx_root = onnx_manifest_path.parent
    manifests: dict[str, dict] = {}
    all_hefs: dict[str, Path] = {}
    all_onnx: dict[str, Path] = {}
    for arm in ARMS:
        manifest, hefs, onnx_files = build_manifest(
            arm, selection, onnx_manifest, onnx_root, codec_dir, args.ort_threads
        )
        manifests[arm] = manifest
        all_hefs.update(hefs)
        all_onnx.update(onnx_files)

    remote_dir = f"{args.remote_root.rstrip('/')}/{args.run_tag}"
    opts = common_options(args)
    # A unique run-tag is mandatory; never mix artifacts with an older run.
    run(["ssh", *opts, args.ssh_target,
         f"test ! -e {shlex.quote(remote_dir)} && mkdir -p {shlex.quote(remote_dir)}"])

    runner = Path(__file__).with_name("hybrid_ort_runtime_runner.py")
    base_runner = Path(__file__).with_name("fullmodule_runtime_runner.py")
    rans_root = Path("/workspace/semantic-sensing/src/codec/rans_fast")
    rans_sources = (
        rans_root / "__init__.py", rans_root / "setup_ext.py",
        rans_root / "cpp" / "rans64.h", rans_root / "cpp" / "rans_fast.cpp",
    )
    for path in (runner, base_runner, *rans_sources):
        if not path.is_file():
            raise FileNotFoundError(path)

    with tempfile.TemporaryDirectory(dir=args.out.parent) as temporary_raw:
        temporary = Path(temporary_raw)
        uploads: list[tuple[Path, str]] = [
            (runner, "hybrid_ort_runtime_runner.py"),
            (base_runner, "fullmodule_runtime_runner.py"),
            (codec_dir / "codec_runtime.json", "codec_runtime.json"),
            (codec_dir / "codec_tables.npz", "codec_tables.npz"),
            (rans_root / "__init__.py", "rans_fast/__init__.py"),
            (rans_root / "setup_ext.py", "rans_fast/setup_ext.py"),
            (rans_root / "cpp" / "rans64.h", "rans_fast/cpp/rans64.h"),
            (rans_root / "cpp" / "rans_fast.cpp", "rans_fast/cpp/rans_fast.cpp"),
        ]
        for arm, manifest in manifests.items():
            path = temporary / f"runtime_manifest_{arm}.json"
            path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
            uploads.append((path, path.name))
        uploads.extend((path, f"hef/{name}.hef") for name, path in sorted(all_hefs.items()))
        uploads.extend((path, f"onnx/{name}.onnx") for name, path in sorted(all_onnx.items()))
        for source, target in uploads:
            stream_upload(source, f"{remote_dir}/{target}", opts, args.ssh_target)

    build = run([
        "ssh", *opts, args.ssh_target,
        f"cd {shlex.quote(remote_dir + '/rans_fast')} && "
        f"python3 setup_ext.py build_ext --inplace && "
        f"{shlex.quote(args.remote_ort_python)} -c "
        f"{shlex.quote('import _rans_fast; print(_rans_fast.__doc__)')}",
    ])
    preflight_code = (
        "import sys, onnxruntime as ort, numpy as np; "
        "sys.path.append('/usr/lib/python3/dist-packages'); import hailo_platform; "
        "print(ort.__version__, np.__version__, hailo_platform.__file__)"
    )
    preflight = run([
        "ssh", *opts, args.ssh_target,
        f"{shlex.quote(args.remote_ort_python)} -c {shlex.quote(preflight_code)}",
    ])

    report = {
        "kind": "pi_hybrid_ort_runtime_deployment",
        "status": "PASS",
        "run_tag": args.run_tag,
        "remote_dir": remote_dir,
        "remote_ort_python": args.remote_ort_python,
        "checkpoint_sha256": selection["checkpoint_sha256"],
        "selection_manifest": {"path": str(selection_path), "sha256": sha256(selection_path)},
        "onnx_manifest": {"path": str(onnx_manifest_path), "sha256": sha256(onnx_manifest_path)},
        "arms": {
            arm: {
                "manifest_filename": f"runtime_manifest_{arm}.json",
                "cpu_modules": manifest["cpu_modules"],
                "hailo_modules": manifest["hailo_modules"],
            }
            for arm, manifest in manifests.items()
        },
        "remote_artifacts": {
            "hefs": {name: {"sha256": sha256(path), "bytes": path.stat().st_size}
                     for name, path in sorted(all_hefs.items())},
            "onnx": {name: {"sha256": sha256(path), "bytes": path.stat().st_size}
                     for name, path in sorted(all_onnx.items())},
        },
        "rans_build_stdout": build.stdout,
        "rans_build_stderr": build.stderr,
        "preflight_stdout": preflight.stdout,
        "preflight_stderr": preflight.stderr,
    }
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": "PASS", "out": str(args.out), "remote_dir": remote_dir}, indent=2))


if __name__ == "__main__":
    main()
