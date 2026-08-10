#!/usr/bin/env python3
"""Deploy the complete 10-HEF ConvBN runtime plus rANS assets to the Pi.

The checkpoint is deliberately not an input here. The board receives only HEFs,
the immutable codec table bundle and the aarch64 rANS extension sources.
"""
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


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def parse_hefs(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        name, separator, raw = value.partition("=")
        if not separator or not name or not raw:
            raise ValueError(f"--hef must be MODULE=PATH, got {value!r}")
        if name not in SEQUENCE:
            raise ValueError(f"unknown module in --hef: {name}")
        if name in result:
            raise ValueError(f"duplicate HEF for {name}")
        path = Path(raw).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        result[name] = path
    missing = [name for name in SEQUENCE if name not in result]
    if missing:
        raise ValueError(f"missing HEFs for {missing}")
    return result


def runtime_manifest(onnx_manifest: Path, codec_dir: Path, hefs: dict[str, Path]) -> dict:
    onnx = json.loads(onnx_manifest.read_text(encoding="utf-8"))
    codec_path = codec_dir / "codec_runtime.json"
    tables = codec_dir / "codec_tables.npz"
    codec = json.loads(codec_path.read_text(encoding="utf-8"))
    if not tables.is_file():
        raise FileNotFoundError(tables)
    if onnx.get("ckpt_sha256") != codec.get("checkpoint", {}).get("sha256"):
        raise ValueError("ONNX/codec checkpoint SHA mismatch")
    source = {item["name"]: item for item in onnx.get("modules", [])}
    if set(source) != set(SEQUENCE):
        raise ValueError("ONNX manifest does not contain exactly the full runtime modules")
    modules = []
    for name in SEQUENCE:
        item = source[name]
        modules.append({
            "name": name,
            "hef_filename": f"{name}.hef",
            "hef_sha256": sha256(hefs[name]),
            "hef_bytes": hefs[name].stat().st_size,
            "input_names": item["input_names"], "output_names": item["output_names"],
            "in_shapes_nchw": item["in_shapes_nchw"], "out_shapes_nchw": item["out_shapes_nchw"],
        })
    return {
        "schema_version": 1,
        "kind": "semantic_sensing_pi_fullmodule_runtime",
        "checkpoint_sha256": onnx["ckpt_sha256"],
        "sequence": list(SEQUENCE), "modules": modules,
        "codec": {
            "metadata_filename": codec_path.name, "metadata_sha256": sha256(codec_path),
            "tables_filename": tables.name, "tables_sha256": sha256(tables),
        },
        "runtime_filename": "fullmodule_runtime_runner.py",
        "rans_extension": "rans_fast/_rans_fast*.so compiled on target from project source",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx-manifest", type=Path, required=True)
    parser.add_argument("--codec-runtime", type=Path, required=True)
    parser.add_argument("--hef", action="append", required=True, help="repeat MODULE=PATH for all ten modules")
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--remote-root", default=".local/share/semantic-sensing-board/fullruntime")
    parser.add_argument("--ssh-target", required=True)
    parser.add_argument("--identity-file", type=Path)
    parser.add_argument("--proxy-command")
    parser.add_argument("--known-hosts", type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite deployment report: {args.out}")
    hefs = parse_hefs(args.hef)
    manifest = runtime_manifest(args.onnx_manifest.resolve(), args.codec_runtime.resolve(), hefs)
    remote_dir = f"{args.remote_root.rstrip('/')}/{args.run_tag}"
    opts = common_options(args)
    runner = Path(__file__).with_name("fullmodule_runtime_runner.py")
    rans_root = Path("/workspace/semantic-sensing/src/codec/rans_fast")
    rans_sources = (rans_root / "__init__.py", rans_root / "setup_ext.py",
                    rans_root / "cpp" / "rans64.h", rans_root / "cpp" / "rans_fast.cpp")
    for source in (runner, *rans_sources):
        if not source.is_file():
            raise FileNotFoundError(source)
    run(["ssh", *opts, args.ssh_target, f"mkdir -p {shlex.quote(remote_dir)}"])
    with tempfile.TemporaryDirectory(dir=args.out.parent) as temporary:
        local_manifest = Path(temporary) / "runtime_manifest.json"
        local_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        codec_dir = args.codec_runtime.resolve()
        uploads: list[tuple[Path, str]] = [
            (runner, "fullmodule_runtime_runner.py"), (local_manifest, "runtime_manifest.json"),
            (codec_dir / "codec_runtime.json", "codec_runtime.json"),
            (codec_dir / "codec_tables.npz", "codec_tables.npz"),
            (rans_root / "__init__.py", "rans_fast/__init__.py"),
            (rans_root / "setup_ext.py", "rans_fast/setup_ext.py"),
            (rans_root / "cpp" / "rans64.h", "rans_fast/cpp/rans64.h"),
            (rans_root / "cpp" / "rans_fast.cpp", "rans_fast/cpp/rans_fast.cpp"),
        ]
        uploads.extend((hefs[name], f"{name}.hef") for name in SEQUENCE)
        for source, target_name in uploads:
            stream_upload(source, f"{remote_dir}/{target_name}", opts, args.ssh_target)
    # pybind11 is headers-only; install into the Pi user's site, then build the
    # extension beside the runner. This does not install torch/CompressAI.
    rans_import = (
        f"import sys; sys.path.insert(0, {remote_dir + '/rans_fast'!r}); "
        "import _rans_fast; print(_rans_fast.__doc__)"
    )
    build = run([
        "ssh", *opts, args.ssh_target,
        f"python3 -m pip install --user --break-system-packages --disable-pip-version-check pybind11 && "
        f"cd {shlex.quote(remote_dir + '/rans_fast')} && python3 setup_ext.py build_ext --inplace && "
        f"python3 -c {shlex.quote(rans_import)}",
    ])
    remote_hefs: dict[str, dict] = {}
    for item in manifest["modules"]:
        remote_path = f"{remote_dir}/{item['hef_filename']}"
        remote_sha = run(["ssh", *opts, args.ssh_target, f"sha256sum {shlex.quote(remote_path)}"]).stdout.split()[0]
        if remote_sha != item["hef_sha256"]:
            raise ValueError(f"Pi HEF hash mismatch for {item['name']}: {remote_sha}")
        remote_hefs[item["name"]] = {"sha256": remote_sha, "bytes": item["hef_bytes"]}
    preflight = run([
        "ssh", *opts, args.ssh_target,
        f"python3 -c {shlex.quote('import hailo_platform, numpy; print(hailo_platform.__file__)')}",
    ])
    report = {
        "kind": "pi_fullmodule_runtime_deployment", "status": "PASS", "run_tag": args.run_tag,
        "remote_dir": remote_dir, "checkpoint_sha256": manifest["checkpoint_sha256"],
        "runtime_manifest": manifest, "remote_hefs": remote_hefs,
        "rans_build_stdout": build.stdout, "rans_build_stderr": build.stderr,
        "preflight_stdout": preflight.stdout, "preflight_stderr": preflight.stderr,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": report["status"], "out": str(args.out), "remote_dir": remote_dir}, indent=2))


if __name__ == "__main__":
    main()
