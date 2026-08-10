#!/usr/bin/env python3
"""Deploy the four formal FP32/FP16 mixed-runtime arms and one local Pi dataset."""
from __future__ import annotations

import argparse
import copy
import json
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import onnx
from onnx import TensorProto
from onnxruntime.transformers.float16 import convert_float_to_float16

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from board_eval.deploy import common_options, stream_upload
from board_eval.deploy_hybrid_ort_runtime import ARMS, build_manifest, load_json
from hailo_deploy.io.hash import sha256


FORMAL_ARMS = (
    "hs_cpu_fp32",
    "gs_only_hailo_fp32",
    "hs_cpu_fp16",
    "gs_only_hailo_fp16",
)
FP16_CPU_MODULES = ("bn_align_h", "bn_g_a", "bn_h_a", "bn_h_s")


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    print(json.dumps({"command": shlex.join(command)}), flush=True)
    return subprocess.run(
        command, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )


def convert_fp16(source: Path, destination: Path) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    graph = onnx.load(source)
    converted = convert_float_to_float16(
        graph, keep_io_types=False, disable_shape_infer=True
    )
    onnx.checker.check_model(converted)
    onnx.save(converted, destination)

    def value_types(values) -> dict[str, str]:
        result = {}
        for value in values:
            element_type = value.type.tensor_type.elem_type
            result[value.name] = TensorProto.DataType.Name(element_type)
        return result

    initializer_types: dict[str, int] = {}
    for initializer in converted.graph.initializer:
        name = TensorProto.DataType.Name(initializer.data_type)
        initializer_types[name] = initializer_types.get(name, 0) + 1
    return {
        "source": {"path": str(source), "sha256": sha256(source), "bytes": source.stat().st_size},
        "converted": {
            "path": str(destination),
            "sha256": sha256(destination),
            "bytes": destination.stat().st_size,
        },
        "input_types": value_types(converted.graph.input),
        "output_types": value_types(converted.graph.output),
        "initializer_types": initializer_types,
        "node_count": len(converted.graph.node),
        "converter": {
            "keep_io_types": False,
            "disable_shape_infer": True,
            "onnx_version": onnx.__version__,
        },
    }


def make_formal_manifests(
    selection: dict,
    onnx_manifest: dict,
    onnx_root: Path,
    codec_root: Path,
    fp16_root: Path,
    ort_threads: int,
) -> tuple[dict[str, dict], dict[str, dict], dict[str, Path]]:
    base: dict[str, dict] = {}
    for arm in ARMS:
        manifest, _, _ = build_manifest(
            arm, selection, onnx_manifest, onnx_root, codec_root, ort_threads
        )
        base[arm] = manifest

    source_by_name = {item["name"]: item for item in onnx_manifest["modules"]}
    converted: dict[str, dict] = {}
    converted_paths: dict[str, Path] = {}
    for name in FP16_CPU_MODULES:
        source = (onnx_root / f"{name}.onnx").resolve()
        if sha256(source) != source_by_name[name]["sha256"]:
            raise ValueError(f"{name}: source ONNX SHA mismatch")
        destination = fp16_root / f"{name}.onnx"
        converted[name] = convert_fp16(source, destination)
        converted_paths[name] = destination

    manifests: dict[str, dict] = {}
    for formal_arm in FORMAL_ARMS:
        base_arm = formal_arm.removesuffix("_fp32").removesuffix("_fp16")
        manifest = copy.deepcopy(base[base_arm])
        manifest["arm"] = formal_arm
        manifest["precision_policy"] = (
            "native_ort_fp16" if formal_arm.endswith("_fp16") else "native_ort_fp32"
        )
        if formal_arm.endswith("_fp16"):
            for module in manifest["modules"]:
                if module["engine"] != "ort_cpu_fp32":
                    continue
                name = module["name"]
                record = converted[name]
                module.update({
                    "engine": "ort_cpu_fp16",
                    "onnx_filename": f"onnx_fp16/{name}.onnx",
                    "onnx_sha256": record["converted"]["sha256"],
                    "onnx_bytes": record["converted"]["bytes"],
                    "source_fp32_onnx_sha256": record["source"]["sha256"],
                })
        manifests[formal_arm] = manifest
    return manifests, converted, converted_paths


def upload_dataset(
    *, evaluation_root: Path, checkpoint: str, remote_root: str,
    run_tag: str, opts: list[str], ssh_target: str, manifest_path: Path,
) -> dict:
    metadata = load_json(evaluation_root / "meta.json")
    indices = [int(item) for item in metadata["val_indices"]]
    if metadata.get("sample_count") != len(indices) or len(indices) != 976:
        raise ValueError("formal evaluation set must contain exactly 976 samples")
    for directory in (evaluation_root / "images", evaluation_root / "ground_truth"):
        if len(list(directory.glob("*.npy"))) != len(indices):
            raise ValueError(f"incomplete local dataset directory: {directory}")
    dataset_manifest = {
        "schema_version": 1,
        "kind": "semantic_sensing_pi_local_evaluation_dataset",
        "checkpoint_sha256": checkpoint,
        "dataset": metadata.get("dataset"),
        "split": metadata.get("split"),
        "sample_count": len(indices),
        "val_indices": indices,
        "source_meta": {
            "path": str((evaluation_root / "meta.json").resolve()),
            "sha256": sha256(evaluation_root / "meta.json"),
        },
        "integrity_policy": "authenticated gzip/tar transfer plus exact file counts",
    }
    manifest_path.write_text(
        json.dumps(dataset_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    remote_manifest = f"{remote_root.rstrip('/')}/dataset_manifest.json"
    probe = run([
        "ssh", *opts, ssh_target,
        f"if test -f {shlex.quote(remote_manifest)}; then echo PRESENT; else echo MISSING; fi",
    ]).stdout.strip()
    if probe == "PRESENT":
        code = (
            "import json,pathlib; p=pathlib.Path(" + repr(remote_root) + "); "
            "m=json.loads((p/'dataset_manifest.json').read_text()); "
            "assert m['checkpoint_sha256']==" + repr(checkpoint) + "; "
            "assert m['sample_count']==976; "
            "assert len(list((p/'images').glob('*.npy')))==976; "
            "assert len(list((p/'ground_truth').glob('*.npy')))==976; print('REUSE')"
        )
        verified = run([
            "ssh", *opts, ssh_target,
            f"python3 -c {shlex.quote(code)}",
        ])
        return {
            "remote_root": remote_root,
            "reused": True,
            "verification_stdout": verified.stdout.strip(),
            "manifest_sha256": sha256(manifest_path),
        }
    if probe != "MISSING":
        raise RuntimeError(f"unexpected remote dataset probe: {probe!r}")

    staging = f"{remote_root}.uploading-{run_tag}"
    run([
        "ssh", *opts, ssh_target,
        f"test ! -e {shlex.quote(staging)} && mkdir -p {shlex.quote(staging)}",
    ])
    print(json.dumps({"stage": "dataset_upload", "samples": 976, "remote": staging}), flush=True)
    with tempfile.TemporaryFile() as tar_error, tempfile.TemporaryFile() as gzip_error:
        tar_process = subprocess.Popen(
            ["tar", "-C", str(evaluation_root), "-cf", "-", "images", "ground_truth"],
            stdout=subprocess.PIPE, stderr=tar_error,
        )
        assert tar_process.stdout is not None
        gzip_process = subprocess.Popen(
            ["gzip", "-1"], stdin=tar_process.stdout, stdout=subprocess.PIPE, stderr=gzip_error
        )
        tar_process.stdout.close()
        assert gzip_process.stdout is not None
        ssh_process = subprocess.Popen(
            ["ssh", *opts, ssh_target,
             f"cd {shlex.quote(staging)} && gzip -dc | tar -xf -"],
            stdin=gzip_process.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        gzip_process.stdout.close()
        ssh_stdout, ssh_stderr = ssh_process.communicate()
        gzip_return = gzip_process.wait()
        tar_return = tar_process.wait()
        if tar_return or gzip_return or ssh_process.returncode:
            tar_error.seek(0)
            gzip_error.seek(0)
            raise RuntimeError(
                "dataset upload failed: "
                f"tar={tar_return} gzip={gzip_return} ssh={ssh_process.returncode}; "
                f"tar_stderr={tar_error.read().decode(errors='replace')!r}; "
                f"gzip_stderr={gzip_error.read().decode(errors='replace')!r}; "
                f"ssh_stdout={ssh_stdout.decode(errors='replace')!r}; "
                f"ssh_stderr={ssh_stderr.decode(errors='replace')!r}"
            )
    stream_upload(manifest_path, f"{staging}/dataset_manifest.json", opts, ssh_target)
    verify_command = (
        f"test $(find {shlex.quote(staging + '/images')} -maxdepth 1 -name '*.npy' | wc -l) -eq 976 && "
        f"test $(find {shlex.quote(staging + '/ground_truth')} -maxdepth 1 -name '*.npy' | wc -l) -eq 976 && "
        f"test ! -e {shlex.quote(remote_root)} && mv {shlex.quote(staging)} {shlex.quote(remote_root)}"
    )
    run(["ssh", *opts, ssh_target, verify_command])
    return {
        "remote_root": remote_root,
        "reused": False,
        "manifest_sha256": sha256(manifest_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--onnx-manifest", type=Path, required=True)
    parser.add_argument("--codec-runtime", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fp16-root", type=Path, required=True)
    parser.add_argument("--ort-threads", type=int, default=4)
    parser.add_argument(
        "--remote-seed",
        default=".local/share/semantic-sensing-board/hybrid-ort/board-hybrid-ort-20260804T175210Z",
    )
    parser.add_argument("--remote-root", default=".local/share/semantic-sensing-board/hybrid-ort")
    parser.add_argument(
        "--remote-dataset-root",
        default=".local/share/semantic-sensing-board/datasets/convbn-gfa-gs-eval976-13dc6c4b3a3ea630",
    )
    parser.add_argument(
        "--remote-ort-python",
        default="/home/sidney/.local/share/semantic-sensing-board/venv-ort128/bin/python",
    )
    parser.add_argument("--ssh-target", required=True)
    parser.add_argument("--identity-file", type=Path)
    parser.add_argument("--proxy-command")
    parser.add_argument("--known-hosts", type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite: {args.out}")
    if args.ort_threads <= 0:
        raise ValueError("--ort-threads must be positive")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.fp16_root.mkdir(parents=True, exist_ok=True)

    selection_path = args.selection_manifest.resolve()
    onnx_manifest_path = args.onnx_manifest.resolve()
    selection = load_json(selection_path)
    onnx_manifest = load_json(onnx_manifest_path)
    manifests, conversions, converted_paths = make_formal_manifests(
        selection, onnx_manifest, onnx_manifest_path.parent,
        args.codec_runtime.resolve(), args.fp16_root.resolve(), args.ort_threads,
    )
    opts = common_options(args)
    remote_dir = f"{args.remote_root.rstrip('/')}/{args.run_tag}"
    run([
        "ssh", *opts, args.ssh_target,
        f"test -d {shlex.quote(args.remote_seed)} && test ! -e {shlex.quote(remote_dir)} && "
        f"cp -a {shlex.quote(args.remote_seed)} {shlex.quote(remote_dir)}",
    ])

    runner = Path(__file__).with_name("hybrid_ort_runtime_runner.py")
    evaluator = Path(__file__).with_name("evaluate_hybrid_ort_local_dataset.py")
    manifest_dir = args.out.parent / "runtime_manifests"
    manifest_dir.mkdir()
    uploads: list[tuple[Path, str]] = [
        (runner, "hybrid_ort_runtime_runner.py"),
        (evaluator, "evaluate_hybrid_ort_local_dataset.py"),
    ]
    for arm, manifest in manifests.items():
        path = manifest_dir / f"runtime_manifest_{arm}.json"
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        uploads.append((path, path.name))
    uploads.extend((path, f"onnx_fp16/{name}.onnx") for name, path in converted_paths.items())
    for source, target in uploads:
        stream_upload(source, f"{remote_dir}/{target}", opts, args.ssh_target)

    preflight_code = (
        "import sys,onnxruntime as ort,numpy as np; "
        "sys.path.append('/usr/lib/python3/dist-packages'); import hailo_platform; "
        "sys.path.insert(0," + repr(remote_dir + "/rans_fast") + "); import _rans_fast; "
        "print(ort.__version__,np.__version__,hailo_platform.__file__,_rans_fast.__file__)"
    )
    preflight = run([
        "ssh", *opts, args.ssh_target,
        f"{shlex.quote(args.remote_ort_python)} -c {shlex.quote(preflight_code)}",
    ])
    dataset_record = upload_dataset(
        evaluation_root=args.evaluation_root.resolve(),
        checkpoint=selection["checkpoint_sha256"],
        remote_root=args.remote_dataset_root,
        run_tag=args.run_tag,
        opts=opts,
        ssh_target=args.ssh_target,
        manifest_path=args.out.parent / "dataset_manifest.json",
    )
    report = {
        "kind": "pi_hybrid_ort_fp32_fp16_formal_deployment",
        "status": "PASS",
        "created_unix_seconds": time.time(),
        "run_tag": args.run_tag,
        "remote_dir": remote_dir,
        "remote_ort_python": args.remote_ort_python,
        "checkpoint_sha256": selection["checkpoint_sha256"],
        "remote_seed": args.remote_seed,
        "dataset": dataset_record,
        "selection_manifest": {"path": str(selection_path), "sha256": sha256(selection_path)},
        "onnx_manifest": {"path": str(onnx_manifest_path), "sha256": sha256(onnx_manifest_path)},
        "conversions": conversions,
        "arms": {
            arm: {
                "manifest_filename": f"runtime_manifest_{arm}.json",
                "manifest_sha256": sha256(manifest_dir / f"runtime_manifest_{arm}.json"),
                "cpu_modules": manifest["cpu_modules"],
                "hailo_modules": manifest["hailo_modules"],
                "precision_policy": manifest["precision_policy"],
            }
            for arm, manifest in manifests.items()
        },
        "preflight_stdout": preflight.stdout.strip(),
        "preflight_stderr": preflight.stderr.strip(),
    }
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": "PASS", "out": str(args.out), "remote_dir": remote_dir}), flush=True)


if __name__ == "__main__":
    main()
