#!/usr/bin/env python3
"""Pi-side causal attribution runtime for the two ``bn_g_s`` outputs.

The permanent runtime topology is preserved: the four upstream bottleneck
modules use ORT FP32 and every non-bottleneck graph uses its selected Hailo
HEF.  Only ``bn_g_s`` is evaluated twice for the exact same decoded y_hat:
once with the selected HEF and once with the frozen FP32 ONNX.  The receiver
tail is then run for all four high/low branch splices.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np

from hybrid_ort_runtime_runner import HybridRuntime, OrtModuleEngine, cpu_frequency_khz, cpu_temperature_c


CPU_MODULES = ("bn_align_h", "bn_g_a", "bn_h_a", "bn_h_s")
HAILO_MODULES = (
    "sender_backbone", "bn_g_s", "receiver_stage4", "receiver_stage5",
    "receiver_dappm", "receiver_head",
)
BRANCHES = {
    "B00_fp_high_fp_low": ("fp", "fp"),
    "B10_hailo_high_fp_low": ("hailo", "fp"),
    "B01_fp_high_hailo_low": ("fp", "hailo"),
    "B11_hailo_high_hailo_low": ("hailo", "hailo"),
}


def tensor_drift(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    candidate64 = np.asarray(candidate, dtype=np.float64).reshape(-1)
    reference64 = np.asarray(reference, dtype=np.float64).reshape(-1)
    delta = candidate64 - reference64
    denominator = float(np.linalg.norm(candidate64) * np.linalg.norm(reference64))
    return {
        "cosine": float(np.dot(candidate64, reference64) / denominator) if denominator else 0.0,
        "mae": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "max_abs": float(np.max(np.abs(delta))),
    }


class BranchAttributionRuntime(HybridRuntime):
    def __init__(self, manifest_path: Path):
        super().__init__(manifest_path)
        if tuple(self.manifest.get("cpu_modules", ())) != CPU_MODULES:
            raise ValueError(
                f"permanent ORT boundary mismatch: {self.manifest.get('cpu_modules')} != {CPU_MODULES}"
            )
        if tuple(self.manifest.get("hailo_modules", ())) != HAILO_MODULES:
            raise ValueError("permanent Hailo boundary mismatch")
        expected_engines = {
            **{name: "ort_cpu_fp32" for name in CPU_MODULES},
            **{name: "hailort_float32_vstream" for name in HAILO_MODULES},
        }
        if self.engine_kinds != expected_engines:
            raise ValueError(f"engine topology mismatch: {self.engine_kinds}")

        gs_spec = next(spec for spec in self.manifest["modules"] if spec["name"] == "bn_g_s")
        fp_spec = dict(gs_spec)
        fp_spec.update({
            "engine": "ort_cpu_fp32",
            "onnx_filename": str(self.manifest["branch_attribution"]["bn_g_s_fp32_onnx_filename"]),
        })
        self.gs_fp32 = OrtModuleEngine(self.root, fp_spec, self.ort_threads)

    def hello(self) -> dict:
        hello = super().hello()
        hello["branch_attribution"] = {
            "branch_contract": {
                "x_h_hat": "128x64x64 high-spatial-resolution branch",
                "x_l_hat": "256x32x32 low-spatial-resolution branch",
            },
            "splices": BRANCHES,
            "fp32_bn_g_s": str(self.gs_fp32.onnx_path),
            "same_decoded_y_hat_for_both_bn_g_s_engines": True,
        }
        return hello

    def _tail(self, x_h: np.ndarray, x_l: np.ndarray) -> tuple[np.ndarray, dict]:
        timings: dict[str, dict] = {}
        stage4, timings["receiver_stage4"] = self.modules["receiver_stage4"].infer(
            {"x_h": x_h, "x_l": x_l}
        )
        stage5, timings["receiver_stage5"] = self.modules["receiver_stage5"].infer({
            "stage4_high": stage4["stage4_high"], "stage4_low": stage4["stage4_low"],
        })
        s8, timings["receiver_dappm"] = self.modules["receiver_dappm"].infer({
            "high_feat": stage5["high_feat"], "low_feat": stage5["low_feat"],
        })
        logits, timings["receiver_head"] = self.modules["receiver_head"].infer({"s8": s8["s8"]})
        return np.asarray(np.argmax(logits["seg_logits"], axis=-1), dtype=np.uint8), timings

    def infer_branches(self, image: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
        total_start = time.perf_counter_ns()
        guard = self._thermal_guard()
        shared: dict[str, dict] = {}
        sender = self._module("sender_backbone", {"image": image}, shared)
        aligned = self._module("bn_align_h", {"x_h": sender["x_h"]}, shared)
        y = self._module(
            "bn_g_a", {"aligned_x_h": aligned["aligned_x_h"], "x_l": sender["x_l"]}, shared
        )["latent_y"]
        z = self._module("bn_h_a", {"latent_y": y}, shared)["latent_z"]
        z_hat, z_stream, z_timing = self.codec.z_roundtrip(z)
        shared["rans_z"] = z_timing
        hyper = self._module("bn_h_s", {"latent_z_hat": z_hat}, shared)
        y_hat, y_stream, y_timing = self.codec.y_roundtrip(y, hyper["means"], hyper["scales"])
        shared["rans_y"] = y_timing
        _container, container_timing = self.codec.container(
            z_stream, y_stream, z, y, sender["x_h"], sender["x_l"]
        )
        shared["rtff_container"] = container_timing

        # Both engines receive this exact array object.  No second entropy
        # decode or intermediate re-materialization is permitted here.
        hailo_recon, hailo_timing = self.modules["bn_g_s"].infer({"latent_y_hat": y_hat})
        fp_recon, fp_timing = self.gs_fp32.infer({"latent_y_hat": y_hat})
        expected = {"x_h_hat": (64, 64, 128), "x_l_hat": (32, 32, 256)}
        for name, shape in expected.items():
            if hailo_recon[name].shape != shape or fp_recon[name].shape != shape:
                raise ValueError(
                    f"semantic branch contract failed for {name}: "
                    f"hailo={hailo_recon[name].shape}, fp32={fp_recon[name].shape}, expected={shape}"
                )

        predictions: dict[str, np.ndarray] = {}
        tail_timings: dict[str, dict] = {}
        for arm, (high_source, low_source) in BRANCHES.items():
            high = hailo_recon["x_h_hat"] if high_source == "hailo" else fp_recon["x_h_hat"]
            low = hailo_recon["x_l_hat"] if low_source == "hailo" else fp_recon["x_l_hat"]
            predictions[arm], tail_timings[arm] = self._tail(high, low)

        self.calls += 1
        return predictions, {
            "call": self.calls,
            "shared_stages": shared,
            "bn_g_s": {"hailo": hailo_timing, "fp32": fp_timing},
            "tail_by_splice": tail_timings,
            "drift_hailo_minus_fp32": {
                "x_h_hat_high_128x64x64": tensor_drift(
                    hailo_recon["x_h_hat"], fp_recon["x_h_hat"]
                ),
                "x_l_hat_low_256x32x32": tensor_drift(
                    hailo_recon["x_l_hat"], fp_recon["x_l_hat"]
                ),
            },
            "exact_input_gate": {
                "same_array_object": True,
                "latent_y_hat_sha256": hashlib.sha256(
                    np.ascontiguousarray(y_hat).view(np.uint8)
                ).hexdigest(),
            },
            "thermal": {
                **guard,
                "end_c": cpu_temperature_c(),
                "cpu_frequency_khz": cpu_frequency_khz(),
            },
            "total_experiment_ms": (time.perf_counter_ns() - total_start) / 1e6,
        }

    def close(self) -> None:
        if getattr(self, "gs_fp32", None) is not None:
            try:
                self.gs_fp32.close()
            except Exception:
                pass
            self.gs_fp32 = None
        super().close()
