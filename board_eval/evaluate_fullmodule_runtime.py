#!/usr/bin/env python3
"""Run golden-image mIoU/cosine and timing evaluation through the Pi full runtime.

Images, labels and FP codec references remain on the host. The Pi retains no
sample files: it receives one normalized image and returns argmax (plus optional
float32 logits for the exact final-logit cosine gate).
"""
from __future__ import annotations

import argparse
import io
import json
import shlex
import struct
import subprocess
import threading
import time
from pathlib import Path

import numpy as np


FRAME = struct.Struct(">Q")
MAX_FRAME_BYTES = 256 * 1024 * 1024


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
    values: dict[str, object] = {"op": np.asarray(op), "metadata": np.asarray(json.dumps(metadata or {}))}
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
        arrays = {name: np.asarray(archive[name]) for name in archive.files if name not in {"op", "metadata"}}
    return op, metadata, arrays


def ssh_options(args) -> list[str]:
    options = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=30",
               "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=3"]
    if args.identity_file:
        options += ["-i", str(args.identity_file), "-o", "IdentitiesOnly=yes"]
    if args.proxy_command:
        options += ["-o", f"ProxyCommand={args.proxy_command}"]
    if args.known_hosts:
        options += ["-o", "StrictHostKeyChecking=yes", "-o", f"UserKnownHostsFile={args.known_hosts}"]
    return options


class BoardClient:
    def __init__(self, args, remote_dir: str):
        command = f"python3 {shlex.quote(remote_dir + '/fullmodule_runtime_runner.py')} --serve --manifest {shlex.quote(remote_dir + '/runtime_manifest.json')}"
        self.stderr: list[bytes] = []
        self.process = subprocess.Popen(["ssh", *ssh_options(args), args.ssh_target, command],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert self.process.stdin and self.process.stdout and self.process.stderr
        self.thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self.thread.start()
        self.hello = self.request("hello", {}, {})[1]

    def _drain_stderr(self) -> None:
        assert self.process.stderr
        for line in self.process.stderr:
            self.stderr.append(line)

    def request(self, op: str, metadata: dict, arrays: dict[str, np.ndarray]) -> tuple[str, dict, dict[str, np.ndarray], float]:
        assert self.process.stdin and self.process.stdout
        request = pack(op, metadata, arrays)
        start = time.perf_counter_ns()
        self.process.stdin.write(FRAME.pack(len(request)))
        self.process.stdin.write(request)
        self.process.stdin.flush()
        response = read_frame(self.process.stdout)
        elapsed = (time.perf_counter_ns() - start) / 1e6
        result = unpack(response)
        if result[0] == "error":
            detail = result[1].get("error", "unknown error")
            raise RuntimeError(f"Pi runtime error: {detail}")
        return (*result, elapsed)

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
    return {"count": int(array.size), "mean": float(array.mean()), "p50": float(np.percentile(array, 50)),
            "p95": float(np.percentile(array, 95)), "min": float(array.min()), "max": float(array.max())}


def cosine(reference_nchw: np.ndarray, board_hwc: np.ndarray) -> float:
    board_nchw = np.asarray(board_hwc, dtype=np.float32).transpose(2, 0, 1)[None]
    if board_nchw.shape != reference_nchw.shape:
        raise ValueError(f"logit shape mismatch: board={board_nchw.shape}, reference={reference_nchw.shape}")
    a, b = reference_nchw.astype(np.float64, copy=False).ravel(), board_nchw.astype(np.float64, copy=False).ravel()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0.0:
        raise ValueError("zero-norm logit tensor")
    return float(np.dot(a, b) / denominator)


def update_confusion(confusion: np.ndarray, prediction: np.ndarray, target: np.ndarray) -> None:
    if prediction.shape != target.shape:
        raise ValueError(f"prediction/target shape mismatch: {prediction.shape} vs {target.shape}")
    classes = confusion.shape[0]
    valid = (target >= 0) & (target < classes) & (prediction >= 0) & (prediction < classes)
    encoded = target[valid].astype(np.int64) * classes + prediction[valid].astype(np.int64)
    confusion += np.bincount(encoded, minlength=classes * classes).reshape(classes, classes)


def miou(confusion: np.ndarray) -> dict[str, float | int]:
    true = np.diag(confusion).astype(np.float64)
    union = confusion.sum(axis=1) + confusion.sum(axis=0) - true
    valid = union > 0
    iou = true[valid] / union[valid]
    return {"mIoU": float(iou.mean()) if iou.size else 0.0, "classes_with_union": int(valid.sum()),
            "pixel_accuracy": float(true.sum() / confusion.sum()) if confusion.sum() else 0.0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ssh-target", required=True)
    parser.add_argument("--identity-file", type=Path)
    parser.add_argument("--proxy-command")
    parser.add_argument("--known-hosts", type=Path)
    parser.add_argument("--limit", type=int, default=8, help="first N frozen golden images")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--return-logits", action="store_true", help="return final FP32 logits to calculate exact cosine")
    parser.add_argument("--cosine-samples", type=int, default=0,
                        help="deterministically return this many final-logit elements per image for sampled cosine")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite evaluation report: {args.out}")
    if args.limit <= 0 or args.warmup < 0 or args.cosine_samples < 0:
        raise ValueError("limit must be positive, and warmup/cosine-samples must be non-negative")
    if args.return_logits and args.cosine_samples:
        raise ValueError("--return-logits and --cosine-samples are mutually exclusive")
    deployment = json.loads(args.deployment.read_text(encoding="utf-8"))
    if deployment.get("status") != "PASS":
        raise ValueError("full runtime deployment did not pass")
    handoff = args.handoff.resolve()
    golden = json.loads((handoff / "evaluation" / "golden_manifest.json").read_text(encoding="utf-8"))
    if golden.get("checkpoint_sha256") != deployment.get("checkpoint_sha256"):
        raise ValueError("golden/deployment checkpoint SHA mismatch")
    indices = list(golden["golden_indices"])[:args.limit]
    if not indices:
        raise ValueError("empty golden index set")
    confusion = np.zeros((150, 150), dtype=np.int64)
    rows: list[dict] = []
    client: BoardClient | None = None
    stderr = ""
    try:
        client = BoardClient(args, deployment["remote_dir"])
        for index in indices[:min(args.warmup, len(indices))]:
            image = np.load(handoff / "evaluation" / "golden" / str(index) / "sender_backbone" / "inputs" / "image.npy", allow_pickle=False)
            client.request("infer", {"return_logits": False}, {"image": image[0].transpose(1, 2, 0)})
        for position, index in enumerate(indices, start=1):
            image = np.load(handoff / "evaluation" / "golden" / str(index) / "sender_backbone" / "inputs" / "image.npy", allow_pickle=False)
            target = np.load(handoff / "evaluation" / "ground_truth" / f"{index}.npy", allow_pickle=False)
            request_arrays: dict[str, np.ndarray] = {"image": image[0].transpose(1, 2, 0)}
            reference_samples: np.ndarray | None = None
            if args.cosine_samples:
                reference = np.load(handoff / "evaluation" / "golden" / str(index) / "receiver_head" / "outputs" / "seg_logits.npy", allow_pickle=False)
                reference_hwc = np.asarray(reference, dtype=np.float32)[0].transpose(1, 2, 0)
                count = min(args.cosine_samples, reference_hwc.size)
                sample_indices = np.linspace(0, reference_hwc.size - 1, num=count, dtype=np.int64)
                request_arrays["cosine_indices"] = sample_indices
                reference_samples = reference_hwc.reshape(-1)[sample_indices]
            op, timing, outputs, rtt_ms = client.request("infer", {"return_logits": args.return_logits}, request_arrays)
            if op != "result":
                raise RuntimeError(f"unexpected Pi operation {op!r}")
            prediction = np.asarray(outputs["argmax"], dtype=np.int64)
            update_confusion(confusion, prediction, target)
            row: dict[str, object] = {"val_index": index, "board_stream_rtt_ms": rtt_ms,
                                      "board_total_pipeline_ms": timing["total_pipeline_ms"]}
            if args.return_logits:
                reference = np.load(handoff / "evaluation" / "golden" / str(index) / "receiver_head" / "outputs" / "seg_logits.npy", allow_pickle=False)
                row["final_logit_cosine_vs_fp_codec"] = cosine(reference, outputs["seg_logits"])
            if reference_samples is not None:
                board_samples = np.asarray(outputs["logit_samples"], dtype=np.float32)
                if board_samples.shape != reference_samples.shape:
                    raise ValueError(f"sampled-logit shape mismatch: board={board_samples.shape}, reference={reference_samples.shape}")
                a = reference_samples.astype(np.float64, copy=False)
                b = board_samples.astype(np.float64, copy=False)
                denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
                if denominator == 0.0:
                    raise ValueError("zero-norm sampled logits")
                row["final_logit_sampled_cosine_vs_fp_codec"] = float(np.dot(a, b) / denominator)
                row["final_logit_sample_count"] = int(a.size)
            for stage, value in timing["stages"].items():
                for key, number in value.items():
                    row[f"{stage}.{key}"] = number
            rows.append(row)
            print(f"evaluated {position}/{len(indices)}", flush=True)
        numeric = sorted({key for row in rows for key, value in row.items()
                          if isinstance(value, (int, float)) and key != "val_index"})
        report = {
            "kind": "pi_fullmodule_runtime_golden_evaluation", "status": "PASS",
            "checkpoint_sha256": deployment["checkpoint_sha256"], "sample_count": len(rows), "val_indices": indices,
            "board_hello": client.hello, "return_logits": bool(args.return_logits),
            "cosine_samples_per_image": int(args.cosine_samples),
            "semantic": miou(confusion), "timing_and_similarity": {key: stats([float(row[key]) for row in rows if key in row]) for key in numeric},
            "per_sample": rows,
            "scope": ["Pi computes the full ten-HEF sequence and real z/y rANS roundtrips.",
                      "board_total_pipeline_ms includes per-module activation switching and CPU entropy coding.",
                      "board_stream_rtt_ms additionally includes SSH/Tailscale transfer and is not target inference latency."],
        }
        if args.cosine_samples:
            report["scope"].append("final_logit_sampled_cosine_vs_fp_codec uses evenly spaced final-logit elements; it is not an exact whole-tensor cosine.")
    finally:
        if client is not None:
            stderr = client.close()
    if stderr:
        report["board_stderr"] = stderr
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": report["status"], "out": str(args.out), "semantic": report["semantic"]}, indent=2))


if __name__ == "__main__":
    main()
