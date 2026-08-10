#!/usr/bin/env python3
"""Evaluate one mixed ORT/Hailo arm through a persistent Pi SSH session."""
from __future__ import annotations

import argparse
import io
import json
import os
import shlex
import struct
import subprocess
import threading
import time
from pathlib import Path

import numpy as np


FRAME = struct.Struct(">Q")
MAX_FRAME_BYTES = 256 * 1024 * 1024
LARGE = [7, 10, 15, 20, 23, 30, 35, 37, 48, 51, 80, 83, 84, 90, 101, 102, 103, 109, 122]
IMAGE_PIXELS = 512 * 512


def read_exact(handle, count: int) -> bytes:
    chunks: list[bytes] = []
    while count:
        value = handle.read(count)
        if not value:
            raise EOFError("truncated board response")
        chunks.append(value)
        count -= len(value)
    return b"".join(chunks)


def read_frame(handle) -> bytes:
    header = read_exact(handle, FRAME.size)
    (size,) = FRAME.unpack(header)
    if not 0 < size <= MAX_FRAME_BYTES:
        raise ValueError(f"invalid board frame size {size}")
    return read_exact(handle, size)


def pack(op: str, metadata: dict | None = None, arrays: dict[str, np.ndarray] | None = None) -> bytes:
    buffer = io.BytesIO()
    values: dict[str, object] = {
        "op": np.asarray(op),
        "metadata": np.asarray(json.dumps(metadata or {})),
    }
    values.update(arrays or {})
    np.savez_compressed(buffer, **values)
    result = buffer.getvalue()
    if len(result) > MAX_FRAME_BYTES:
        raise ValueError("request exceeds framed transport limit")
    return result


def unpack(payload: bytes) -> tuple[str, dict, dict[str, np.ndarray]]:
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        op = str(archive["op"].item())
        metadata = json.loads(str(archive["metadata"].item()))
        arrays = {
            name: np.asarray(archive[name])
            for name in archive.files
            if name not in {"op", "metadata"}
        }
    return op, metadata, arrays


def ssh_options(args) -> list[str]:
    options = [
        "-o", "BatchMode=yes", "-o", "ConnectTimeout=30",
        "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=3",
    ]
    if args.identity_file:
        options += ["-i", str(args.identity_file), "-o", "IdentitiesOnly=yes"]
    if args.proxy_command:
        options += ["-o", f"ProxyCommand={args.proxy_command}"]
    if args.known_hosts:
        options += [
            "-o", "StrictHostKeyChecking=yes",
            "-o", f"UserKnownHostsFile={args.known_hosts}",
        ]
    return options


class BoardClient:
    def __init__(self, args, deployment: dict):
        arm = args.arm
        arm_spec = deployment["arms"][arm]
        remote_dir = deployment["remote_dir"]
        command = (
            f"{shlex.quote(deployment['remote_ort_python'])} "
            f"{shlex.quote(remote_dir + '/hybrid_ort_runtime_runner.py')} --serve "
            f"--manifest {shlex.quote(remote_dir + '/' + arm_spec['manifest_filename'])}"
        )
        self.stderr: list[bytes] = []
        self.process = subprocess.Popen(
            ["ssh", *ssh_options(args), args.ssh_target, command],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert self.process.stdin and self.process.stdout and self.process.stderr
        self.thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self.thread.start()
        self.hello = self.request("hello", {}, {})[1]
        if self.hello.get("arm") != arm:
            raise ValueError(f"board started the wrong arm: {self.hello.get('arm')} != {arm}")

    def _drain_stderr(self) -> None:
        assert self.process.stderr
        for line in self.process.stderr:
            self.stderr.append(line)

    def request(
        self, op: str, metadata: dict, arrays: dict[str, np.ndarray]
    ) -> tuple[str, dict, dict[str, np.ndarray], float]:
        assert self.process.stdin and self.process.stdout
        request = pack(op, metadata, arrays)
        start = time.perf_counter_ns()
        self.process.stdin.write(FRAME.pack(len(request)))
        self.process.stdin.write(request)
        self.process.stdin.flush()
        response = read_frame(self.process.stdout)
        elapsed_ms = (time.perf_counter_ns() - start) / 1e6
        result = unpack(response)
        if result[0] == "error":
            raise RuntimeError(f"Pi runtime error: {result[1].get('error', 'unknown error')}")
        return (*result, elapsed_ms)

    def close(self) -> str:
        try:
            if self.process.poll() is None:
                self.request("shutdown", {}, {})
        except Exception:
            self.process.terminate()
        finally:
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.thread.join(timeout=2)
        return b"".join(self.stderr).decode("utf-8", errors="replace")


def stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "min": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def update_confusion(confusion: np.ndarray, prediction: np.ndarray, target: np.ndarray) -> None:
    classes = confusion.shape[0]
    prediction = np.asarray(prediction, dtype=np.int64)
    target = np.asarray(target, dtype=np.int64)
    if prediction.shape != target.shape:
        raise ValueError(f"prediction/target mismatch: {prediction.shape} != {target.shape}")
    valid = (target >= 0) & (target < classes) & (prediction >= 0) & (prediction < classes)
    encoded = target[valid] * classes + prediction[valid]
    confusion += np.bincount(encoded, minlength=classes * classes).reshape(classes, classes)


def semantic_metrics(confusion: np.ndarray) -> dict[str, object]:
    data = confusion.astype(np.float64, copy=False)
    true = np.diag(data)
    union = data.sum(axis=1) + data.sum(axis=0) - true
    iou = np.divide(true, np.maximum(union, 1.0))
    present = union > 0
    return {
        "mIoU": float(iou[present].mean()) if np.any(present) else 0.0,
        "pixel_accuracy": float(true.sum() / max(data.sum(), 1.0)),
        "fixed_19_large_object_macro_iou": float(iou[LARGE].mean()),
        "classes_with_union": int(present.sum()),
        "per_class_iou": [float(item) for item in iou],
    }


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--arm", choices=("hs_cpu", "gs_only_hailo"), required=True)
    parser.add_argument("--mode", choices=("preflight", "latency", "accuracy"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--interval-seconds", type=float, default=1.0)
    parser.add_argument("--cosine-samples", type=int, default=0)
    parser.add_argument("--ssh-target", required=True)
    parser.add_argument("--identity-file", type=Path)
    parser.add_argument("--proxy-command")
    parser.add_argument("--known-hosts", type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    if args.limit <= 0 or args.warmup < 0 or args.interval_seconds < 0 or args.cosine_samples < 0:
        raise ValueError("invalid limit/warmup/interval/cosine values")

    deployment = json.loads(args.deployment.read_text(encoding="utf-8"))
    if deployment.get("status") != "PASS" or args.arm not in deployment.get("arms", {}):
        raise ValueError("deployment is not a PASS hybrid bundle for this arm")
    handoff = args.handoff.resolve()
    meta = json.loads((handoff / "evaluation" / "meta.json").read_text(encoding="utf-8"))
    checkpoint_sha = meta.get("checkpoint", {}).get("sha256")
    if checkpoint_sha != deployment.get("checkpoint_sha256"):
        raise ValueError("handoff/deployment checkpoint mismatch")
    indices = list(meta["val_indices"])[:args.limit]
    image_root = handoff / "evaluation" / "images"
    target_root = handoff / "evaluation" / "ground_truth"
    golden_root = handoff / "evaluation" / "golden"

    confusion = np.zeros((150, 150), dtype=np.int64)
    rows: list[dict[str, object]] = []
    client: BoardClient | None = None
    stderr = ""
    try:
        client = BoardClient(args, deployment)
        warm_image = np.load(image_root / f"{indices[0]}.npy", allow_pickle=False)
        for _ in range(args.warmup):
            client.request("infer", {}, {"image": warm_image.transpose(1, 2, 0)})

        for position, index in enumerate(indices, start=1):
            image = np.load(image_root / f"{index}.npy", allow_pickle=False)
            target = np.load(target_root / f"{index}.npy", allow_pickle=False)
            arrays: dict[str, np.ndarray] = {"image": image.transpose(1, 2, 0)}
            reference_samples: np.ndarray | None = None
            reference_path = golden_root / str(index) / "receiver_head" / "outputs" / "seg_logits.npy"
            if args.cosine_samples and reference_path.is_file():
                reference = np.load(reference_path, allow_pickle=False)[0].transpose(1, 2, 0).reshape(-1)
                count = min(args.cosine_samples, reference.size)
                sample_indices = np.linspace(0, reference.size - 1, num=count, dtype=np.int64)
                arrays["cosine_indices"] = sample_indices
                reference_samples = np.asarray(reference[sample_indices], dtype=np.float32)

            op, timing, outputs, rtt_ms = client.request("infer", {}, arrays)
            if op != "result":
                raise RuntimeError(f"unexpected board operation {op!r}")
            prediction = np.asarray(outputs["argmax"], dtype=np.int64)
            update_confusion(confusion, prediction, target)
            container = timing["stages"]["rtff_container"]
            thermal = timing.get("thermal", {})
            row: dict[str, object] = {
                "val_index": index,
                "board_stream_rtt_ms": rtt_ms,
                "board_total_pipeline_ms": float(timing["total_pipeline_ms"]),
                "thermal_initial_c": thermal.get("initial_c"),
                "thermal_start_c": thermal.get("start_c"),
                "thermal_end_c": thermal.get("end_c"),
                "cooldown_wait_ms": thermal.get("cooldown_wait_ms", 0.0),
                "payload_bytes": int(container["bytes"]),
                "z_bytes": int(container["z_bytes"]),
                "y_bytes": int(container["y_bytes"]),
                "bpp": float(container["bytes"] * 8.0 / IMAGE_PIXELS),
            }
            if reference_samples is not None:
                actual = np.asarray(outputs["logit_samples"], dtype=np.float32)
                denominator = float(
                    np.linalg.norm(reference_samples.astype(np.float64))
                    * np.linalg.norm(actual.astype(np.float64))
                )
                row["final_logit_sampled_cosine_vs_fp_codec"] = (
                    float(np.dot(reference_samples.astype(np.float64), actual.astype(np.float64)) / denominator)
                    if denominator else None
                )
                row["final_logit_sample_count"] = int(actual.size)
            for stage, stage_timing in timing["stages"].items():
                for key, value in stage_timing.items():
                    if isinstance(value, (int, float)):
                        row[f"{stage}.{key}"] = value
            rows.append(row)
            print(json.dumps({
                "arm": args.arm,
                "mode": args.mode,
                "progress": f"{position}/{len(indices)}",
                "temperature_c": thermal.get("end_c"),
                "pipeline_ms": timing["total_pipeline_ms"],
            }), flush=True)
            if position != len(indices) and args.interval_seconds:
                time.sleep(args.interval_seconds)
    finally:
        if client is not None:
            stderr = client.close()

    numeric_keys = sorted({
        key for row in rows for key, value in row.items()
        if isinstance(value, (int, float)) and key != "val_index"
    })
    total_bytes = sum(int(row["payload_bytes"]) for row in rows)
    report = {
        "kind": "pi_hybrid_ort_hailort_evaluation",
        "status": "PASS",
        "mode": args.mode,
        "arm": args.arm,
        "checkpoint_sha256": checkpoint_sha,
        "sample_count": len(rows),
        "val_indices": indices,
        "board_hello": client.hello,
        "policy": {
            "warmup": args.warmup,
            "interval_seconds": args.interval_seconds,
            "cosine_samples": args.cosine_samples,
            "serial_pipeline": True,
            "stress_test": False,
        },
        "semantic": semantic_metrics(confusion),
        "bitstream": {
            "total_bytes": total_bytes,
            "mean_bpp": total_bytes * 8.0 / (len(rows) * IMAGE_PIXELS),
            "z_bytes": sum(int(row["z_bytes"]) for row in rows),
            "y_bytes": sum(int(row["y_bytes"]) for row in rows),
        },
        "timing_and_similarity": {
            key: stats([float(row[key]) for row in rows if row.get(key) is not None])
            for key in numeric_keys
        },
        "per_sample": rows,
        "scope": [
            "Pi executes the complete image-to-segmentation path with real z/y rANS roundtrips.",
            "board_total_pipeline_ms excludes SSH/Tailscale and any thermal cooldown wait.",
            "Samples are serialized with an intentional idle interval; this is not a stress or throughput test.",
            "Accuracy mode is a 64-sample held-out screen, not the formal 976-sample acceptance result.",
        ],
    }
    if stderr:
        report["board_stderr"] = stderr
    atomic_write_json(args.out, report)
    print(json.dumps({
        "status": "PASS",
        "out": str(args.out),
        "arm": args.arm,
        "mode": args.mode,
        "semantic": report["semantic"],
    }, indent=2))


if __name__ == "__main__":
    main()
