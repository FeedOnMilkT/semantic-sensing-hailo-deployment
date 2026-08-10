#!/usr/bin/env python3
"""Run held-out receiver_stage45 FP32/quantized/board tensor evaluation.

Inputs and labels stay on x86.  A single board subprocess keeps the HEF loaded
and receives exactly one `(x_h, x_l)` activation pair per framed request.
Outputs are retained on x86 for the later frozen-tail mIoU pass.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import shlex
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hailo_deploy.io.hash import sha256

FRAME = struct.Struct(">Q")


def read_frame(handle) -> bytes:
    header = handle.read(FRAME.size)
    if len(header) != FRAME.size:
        raise EOFError("truncated board response header")
    (size,) = FRAME.unpack(header)
    if not 0 < size <= 64 * 1024 * 1024:
        raise ValueError(f"invalid board response size: {size}")
    pieces, remaining = [], size
    while remaining:
        chunk = handle.read(remaining)
        if not chunk:
            raise EOFError("truncated board response")
        pieces.append(chunk)
        remaining -= len(chunk)
    return b"".join(pieces)


def pack(op: str, arrays: dict[str, np.ndarray] | None = None) -> bytes:
    buffer = io.BytesIO()
    values: dict[str, object] = {"op": np.asarray(op), "metadata": np.asarray("{}")}
    values.update(arrays or {})
    np.savez_compressed(buffer, **values)
    return buffer.getvalue()


def unpack(payload: bytes) -> tuple[str, dict, dict[str, np.ndarray]]:
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        op = str(archive["op"].item())
        metadata = json.loads(str(archive["metadata"].item()))
        arrays = {name: np.asarray(archive[name]) for name in archive.files if name not in {"op", "metadata"}}
    return op, metadata, arrays


def common_options(args) -> list[str]:
    # Detect a board/Tailscale path that accepts the TCP connection but stops
    # forwarding the persistent framed inference stream.
    options = [
        "-o", "BatchMode=yes", "-o", "ConnectTimeout=30",
        "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=3",
    ]
    if args.identity_file:
        options += ["-i", str(args.identity_file), "-o", "IdentitiesOnly=yes"]
    if args.proxy_command:
        options += ["-o", f"ProxyCommand={args.proxy_command}"]
    if args.known_hosts:
        options += ["-o", "StrictHostKeyChecking=yes", "-o", f"UserKnownHostsFile={args.known_hosts}"]
    return options


class BoardClient:
    def __init__(self, args, deployment: dict):
        remote = deployment["remote_dir"]
        command = f"python3 {shlex.quote(remote + '/board_runner.py')} --serve --manifest {shlex.quote(remote + '/manifest.json')}"
        self.stderr: list[bytes] = []
        self.process = subprocess.Popen(
            ["ssh", *common_options(args), args.ssh_target, command],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        assert self.process.stdin and self.process.stdout and self.process.stderr
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()
        self.hello = self.request("hello", {})[1]

    def _drain_stderr(self) -> None:
        assert self.process.stderr
        for line in self.process.stderr:
            self.stderr.append(line)

    def request(self, op: str, arrays: dict[str, np.ndarray]) -> tuple[str, dict, dict[str, np.ndarray], float]:
        assert self.process.stdin and self.process.stdout
        payload = pack(op, arrays)
        start = time.perf_counter_ns()
        self.process.stdin.write(FRAME.pack(len(payload)))
        self.process.stdin.write(payload)
        self.process.stdin.flush()
        response = read_frame(self.process.stdout)
        elapsed_ms = (time.perf_counter_ns() - start) / 1e6
        result_op, metadata, result_arrays = unpack(response)
        if result_op == "error":
            raise RuntimeError(f"board runner error: {metadata.get('error')}")
        return result_op, metadata, result_arrays, elapsed_ms

    def close(self) -> str:
        try:
            if self.process.poll() is None:
                self.request("shutdown", {})
        except Exception:
            self.process.terminate()
        finally:
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self._stderr_thread.join(timeout=2)
        return b"".join(self.stderr).decode("utf-8", errors="replace")


def canonical(value: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if value.shape == (1, *shape):
        value = value[0]
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"invalid {name}: {value.shape}")
    return np.ascontiguousarray(value)


def tensor_metrics(reference: np.ndarray, value: np.ndarray) -> dict[str, float | None]:
    delta = np.asarray(value, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    ref_flat, value_flat = reference.astype(np.float64, copy=False).ravel(), value.astype(np.float64, copy=False).ravel()
    denom = float(np.linalg.norm(ref_flat) * np.linalg.norm(value_flat))
    cosine = None if denom == 0.0 else float(np.dot(ref_flat, value_flat) / denom)
    return {
        "cosine": cosine,
        "mae": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "max_abs": float(np.max(np.abs(delta))),
    }


def summarize(rows: list[dict], key: str) -> dict[str, float | int | None]:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return {"count": 0, "mean": None, "min": None, "p50": None, "p95": None, "p99": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {"count": int(array.size), "mean": float(array.mean()), "min": float(array.min()),
            "p50": float(np.percentile(array, 50)), "p95": float(np.percentile(array, 95)),
            "p99": float(np.percentile(array, 99)), "max": float(array.max())}


def ort_session(onnx: Path, provider: str) -> tuple[ort.InferenceSession, str]:
    available = ort.get_available_providers()
    if provider == "auto":
        provider = "cuda" if "CUDAExecutionProvider" in available else "cpu"
    if provider == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(f"CUDAExecutionProvider unavailable: {available}")
        providers: list[object] = [("CUDAExecutionProvider", {"use_tf32": "0"}), "CPUExecutionProvider"]
    elif provider == "cpu":
        providers = ["CPUExecutionProvider"]
    else:
        raise ValueError(provider)
    return ort.InferenceSession(str(onnx), providers=providers), provider


def ort_infer(session: ort.InferenceSession, x_h: np.ndarray, x_l: np.ndarray) -> dict[str, np.ndarray]:
    feed = {
        "x_h": np.ascontiguousarray(x_h.transpose(2, 0, 1)[None]),
        "x_l": np.ascontiguousarray(x_l.transpose(2, 0, 1)[None]),
    }
    raw = dict(zip([item.name for item in session.get_outputs()], session.run(None, feed)))
    expected = {"high_feat": (64, 64, 128), "low_feat": (16, 16, 512)}
    result: dict[str, np.ndarray] = {}
    for name, shape in expected.items():
        value = raw.get(name)
        if value is None:
            candidates = [array for array in raw.values() if tuple(np.asarray(array).shape) == (1, shape[2], shape[0], shape[1])]
            if len(candidates) != 1:
                raise ValueError(f"cannot map ONNX output {name}: {[(k, np.asarray(v).shape) for k, v in raw.items()]}")
            value = candidates[0]
        value = np.asarray(value, dtype=np.float32)
        if value.shape == (1, shape[2], shape[0], shape[1]):
            value = value[0].transpose(1, 2, 0)
        result[name] = canonical(value, shape, f"onnx:{name}")
    return result


def load_input(root: Path, index: int) -> tuple[np.ndarray, np.ndarray]:
    x_h = canonical(np.load(root / "x_h" / f"{index:05d}.npy", allow_pickle=False), (64, 64, 128), "x_h")
    x_l = canonical(np.load(root / "x_l" / f"{index:05d}.npy", allow_pickle=False), (32, 32, 256), "x_l")
    return x_h, x_l


def load_quantized(root: Path, index: int) -> dict[str, np.ndarray]:
    return {
        "high_feat": canonical(np.load(root / "high_feat" / f"{index:05d}.npy", allow_pickle=False),
                               (64, 64, 128), "quantized:high_feat"),
        "low_feat": canonical(np.load(root / "low_feat" / f"{index:05d}.npy", allow_pickle=False),
                              (16, 16, 512), "quantized:low_feat"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--ssh-target", required=True)
    parser.add_argument("--identity-file", type=Path)
    parser.add_argument("--proxy-command")
    parser.add_argument("--known-hosts", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ort-provider", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None,
                        help="first N held-out samples; use 8 for the golden gate")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite: {args.out}")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    deployment = json.loads(args.deployment.read_text(encoding="utf-8"))
    if sha256(Path(manifest["candidate"]["hef"]["path"])) != deployment["hef"]["sha256"]:
        raise ValueError("deployment HEF differs from frozen manifest")
    indices = list(manifest["evaluation"]["val_indices"])
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        indices = indices[:args.limit]
    if not indices:
        raise ValueError("empty evaluation indices")
    input_root = Path(manifest["evaluation"]["input_root"])
    quant_root = Path(manifest["evaluation"]["quantized_root"])
    session, provider = ort_session(Path(manifest["candidate"]["onnx"]["path"]), args.ort_provider)
    temporary = args.out.parent / f".{args.out.name}.tmp-{os.getpid()}"
    temporary.mkdir(parents=True)
    fp_root, board_root = temporary / "fp_onnx_outputs", temporary / "board_outputs"
    for root in (fp_root, board_root):
        (root / "high_feat").mkdir(parents=True)
        (root / "low_feat").mkdir(parents=True)
    client: BoardClient | None = None
    rows: list[dict] = []
    try:
        client = BoardClient(args, deployment)
        for index in indices[:min(args.warmup, len(indices))]:
            x_h, x_l = load_input(input_root, index)
            client.request("infer", {"x_h": x_h, "x_l": x_l})
        for position, index in enumerate(indices, start=1):
            sample_start = time.perf_counter_ns()
            x_h, x_l = load_input(input_root, index)
            ort_start = time.perf_counter_ns()
            fp = ort_infer(session, x_h, x_l)
            ort_ms = (time.perf_counter_ns() - ort_start) / 1e6
            quant_start = time.perf_counter_ns()
            quantized = load_quantized(quant_root, index)
            quant_load_ms = (time.perf_counter_ns() - quant_start) / 1e6
            op, board_timing, board, stream_rtt_ms = client.request("infer", {"x_h": x_h, "x_l": x_l})
            if op != "result":
                raise ValueError(f"unexpected board op: {op}")
            row: dict[str, object] = {
                "val_index": index, "ort_ms": ort_ms, "quantized_cache_load_ms": quant_load_ms,
                "board_stream_rtt_ms": stream_rtt_ms, "board_infer_wall_ms": board_timing.get("infer_wall_ms"),
                "board_serialize_ms": board_timing.get("serialize_ms"),
                "host_orchestration_e2e_ms": (time.perf_counter_ns() - sample_start) / 1e6,
            }
            for name in ("high_feat", "low_feat"):
                np.save(fp_root / name / f"{index:05d}.npy", fp[name])
                np.save(board_root / name / f"{index:05d}.npy", board[name])
                for label, reference, value in (
                    ("quantized_vs_fp32", fp[name], quantized[name]),
                    ("board_vs_fp32", fp[name], board[name]),
                    ("board_vs_quantized", quantized[name], board[name]),
                ):
                    for metric, value_metric in tensor_metrics(reference, value).items():
                        row[f"{name}.{label}.{metric}"] = value_metric
            rows.append(row)
            if position % 25 == 0 or position == len(indices) or len(indices) <= 8:
                print(f"evaluated {position}/{len(indices)}", flush=True)
        summary: dict[str, object] = {}
        numeric_keys = sorted({key for row in rows for key, value in row.items() if isinstance(value, (float, int)) and key != "val_index"})
        for key in numeric_keys:
            summary[key] = summarize(rows, key)
        (temporary / "per_sample.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
        report = {
            "kind": "receiver_stage45_board_tensor_evaluation", "status": "PASS",
            "run_tag": manifest["run_tag"], "sample_count": len(indices), "val_indices": indices,
            "calibration_overlap": sorted(set(indices) & set(manifest["calibration"]["val_indices"])),
            "ort_provider": provider, "onnxruntime": ort.__version__, "board_hello": client.hello,
            "manifest_sha256": sha256(args.manifest), "deployment_sha256": sha256(args.deployment),
            # ``temporary`` is atomically renamed below.  Persist the final
            # locations, otherwise tail_metrics.py would follow stale .tmp
            # paths after a successful evaluation.
            "outputs": {"fp_onnx": str(args.out / "fp_onnx_outputs"),
                        "board": str(args.out / "board_outputs"),
                        "host_quantized": str(quant_root)},
            "timing_and_similarity": summary,
            "limitations": [
                "board_stream_rtt_ms includes SSH/Tailscale orchestration and is not target sender-receiver latency",
                "mIoU is produced by tail_metrics.py using the frozen x86 receiver tail for this Stage45-only HEF",
            ],
        }
        (temporary / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, args.out)
        print(json.dumps({"out": str(args.out), "sample_count": len(indices), "provider": provider}, indent=2))
    except BaseException:
        import shutil
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    finally:
        stderr = client.close() if client is not None else ""
        if args.out.exists() and stderr:
            (args.out / "board_runner.stderr.log").write_text(stderr, encoding="utf-8")


if __name__ == "__main__":
    main()
