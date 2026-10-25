#!/usr/bin/env python3
"""Export the RECEIVER side of the split=mid student to static-shape ONNX.

Graph (exactly the deployed receiver path, nothing else):

    y_hat [1,192,16,16]            (decoded symbols + means; NO h_s inside!)
      -> bottleneck g_s synthesis  (g_s_trunk + head_l + head_h, IGDN)
      -> x_h_hat [1,128,64,64], x_l_hat [1,256,32,32]
      -> teacher receiver tail     (stage4/stage5 EABlocks + DAPPM + neck + seg head)
      -> logits [1,150,256,256]    (native head resolution)
      -> Resize -> logits [1,150,OUT,OUT]   (OUT default 128)

Notes
-----
* The hyper-prior (h_a / h_s / entropy bottleneck) is NOT part of this graph.  The
  decoder side of ``libsscodec`` reproduces ``h_s`` deterministically in C++ and
  hands us the already-decoded ``y_hat`` (symbols + means).  Re-implementing h_s in
  ONNX/TensorRT would break the rANS decode (Hailo-int8 and FP32 disagree on 17% of
  the scale-bin choices), so the engine MUST consume the decoded y_hat.
* IGDN (CompressAI ``GDN(inverse=True)``) is re-implemented with *folded* constants
  (beta/gamma reparametrisers are constants at inference), which removes the custom
  autograd ``LowerBound`` from the graph and gives the TensorRT builder stable,
  inspectable layer names (``.../IGDN/...``) for per-layer precision control.
* The SegHead natively emits 256x256 (upscale x4 from the 64x64 neck feature).  The
  deployed evaluation resizes logits to the 512x512 GT resolution, so the exported
  ``logits`` is resized to ``--out-size`` (default 128) by an explicit Resize node.

Run (torch venv, NOT the TRT venv):

    /root/Semantic-sensing/venv-hailo/bin/python \
        receiver_trt/export_receiver_onnx.py \
        --out /root/autodl-tmp/deploy_artifacts/package/receiver/receiver.onnx
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_ROOT = os.environ.get("SEMANTIC_SENSING_ROOT", "/root/Semantic-sensing")
for p in (TRAIN_ROOT, os.path.join(TRAIN_ROOT, "deploy")):
    if p not in sys.path:
        sys.path.insert(0, p)

from export_sender import load_student  # noqa: E402

DEFAULT_CKPT = os.path.join(TRAIN_ROOT, "ckpt", "student_mid_b0.03_s2.pth")
DEFAULT_OUT = "/root/autodl-tmp/deploy_artifacts/package/receiver/receiver.onnx"


# --------------------------------------------------------------------------- #
# IGDN with folded constants (numerically identical to compressai GDN inverse)
# --------------------------------------------------------------------------- #
class IGDN(nn.Module):
    """CompressAI ``GDN(inverse=True)`` with the reparametrisers folded.

    compressai:  norm = conv2d(x^2, gamma', beta');  out = x * sqrt(norm)
    where gamma' = gamma_reparam(gamma) >= 0 and beta' = beta_reparam(beta) > 0.
    Both are constants in eval mode, so they are folded into buffers here.  The
    export therefore contains Mul -> Conv(1x1) -> Sqrt -> Mul, all TRT-friendly.
    """

    def __init__(self, gdn):
        super().__init__()
        assert getattr(gdn, "inverse", False), "IGDN expects an inverse GDN (IGDN)"
        c = int(gdn.beta.numel())
        with torch.no_grad():
            beta = gdn.beta_reparam(gdn.beta).reshape(-1).to(torch.float32)
            gamma = gdn.gamma_reparam(gdn.gamma).reshape(c, c, 1, 1).to(torch.float32)
        self.register_buffer("gamma_c", gamma.contiguous())
        self.register_buffer("beta_c", beta.contiguous())

    def forward(self, x):
        t = F.conv2d(x * x, self.gamma_c, self.beta_c)
        return x * torch.sqrt(t)


def _convert_gdn(module):
    """Recursively replace every compressai GDN(inverse=True) by :class:`IGDN`."""
    from compressai.layers import GDN

    for name, child in list(module.named_children()):
        if isinstance(child, GDN):
            if getattr(child, "inverse", False):
                setattr(module, name, IGDN(child))
            else:
                raise RuntimeError("unexpected forward GDN in the receiver synthesis path")
        else:
            _convert_gdn(child)
    return module


# --------------------------------------------------------------------------- #
# static-shape AdaptiveMaxPool2d replacement (ONNX-export safe, bit-exact)
# --------------------------------------------------------------------------- #
class StaticAdaptiveMaxPool2d(nn.Module):
    """Exact, static-shape stand-in for ``nn.AdaptiveMaxPool2d`` (max mode).

    PyTorch's adaptive max pool uses, for output index ``i`` of a size-``S`` axis:

        start = floor(i * S / out)        end = ceil((i + 1) * S / out)

    so the windows have *different* sizes and overlap -> no single ``MaxPool2d``
    can express them, and the ATen op has no ONNX symbolic (torch.onnx raises
    ``Unsupported: ONNX export of operator adaptive_max_pool2d``).

    We factorise the 2-D window max into
        (a) per-COLUMN-window maxima  (slice + Maximum)  -> [B,C,H,ow]
        (b) per-ROW-window maxima of (a) (slice + Maximum) -> [B,C,oh,ow]
    which is bit-exact w.r.t. ATen (max is exact, no interpolation) and uses only
    Slice / Maximum / Stack -- all natively supported by ONNX Runtime and TensorRT.

    The window bounds are computed from the *static* input size, so the exported
    graph has fixed shapes (the receiver tail is fully convolutional at fixed
    resolution for a fixed input size).
    """

    def __init__(self, output_size):
        super().__init__()
        self.output_size = tuple(int(v) for v in output_size)

    @staticmethod
    def _windows(size, out):
        return [((i * size) // out, -(-((i + 1) * size) // out)) for i in range(out)]

    def forward(self, x):
        h, w = x.size(-2), x.size(-1)
        if not isinstance(h, int):
            h, w = int(h), int(w)
        if not isinstance(h, int) or not isinstance(w, int):
            raise RuntimeError("StaticAdaptiveMaxPool2d requires a static spatial size")
        oh, ow = self.output_size
        cols = []
        for (s, e) in self._windows(w, ow):
            sl = x[:, :, :, s:e]
            acc = sl[:, :, :, 0]
            for k in range(1, e - s):
                acc = torch.maximum(acc, sl[:, :, :, k])
            cols.append(acc)
        cw = torch.stack(cols, dim=-1)                       # [B,C,H,ow]
        rows = []
        for (s, e) in self._windows(h, oh):
            sl = cw[:, :, s:e, :]
            acc = sl[:, :, 0, :]
            for k in range(1, e - s):
                acc = torch.maximum(acc, sl[:, :, k, :])
            rows.append(acc)
        return torch.stack(rows, dim=-2)                    # [B,C,oh,ow]


def replace_adaptive_pools(module):
    """Swap every ``nn.AdaptiveMaxPool2d`` for the static equivalent (in-place).

    The training repo (``/root/Semantic-sensing``) is NOT modified: this runs on
    the in-memory copy owned by the exporter only.
    """
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.AdaptiveMaxPool2d):
            setattr(module, name, StaticAdaptiveMaxPool2d(child.output_size))
            n += 1
        else:
            n += replace_adaptive_pools(child)
    return n


def check_static_pool_exact(device):
    """Self-check: the static rewrite must be bit-exact vs torch's adaptive max pool."""
    out = {}
    for size in (32, 16, 12, 8):
        x = torch.randn(2, 3, size, size, device=device)
        ref = nn.AdaptiveMaxPool2d((12, 12)).to(device)(x)
        got = StaticAdaptiveMaxPool2d((12, 12)).to(device)(x)
        out[f"{size}->12"] = {
            "max_abs_diff": float((ref - got).abs().max()),
            "exact": bool(torch.equal(ref, got)),
        }
    return out


# --------------------------------------------------------------------------- #
# static-shape patches for the RTFormer receiver tail
# --------------------------------------------------------------------------- #
# The stock tail derives every spatial size from ``x.shape`` at runtime.  Under
# the TorchScript ONNX exporter that turns into *dynamic* dims, and the first
# dynamic dim poisons the whole downstream graph (dynamic ``groups`` in the
# cross-attention Conv, dynamic ``Resize`` sizes, dynamic ``Reshape`` shapes),
# which torch.onnx cannot express.
#
# The receiver tail is fully convolutional and its sizes are FIXED for a fixed
# input resolution, so we re-implement the four affected forwards with explicit
# integer sizes.  The parameters are the loaded ones (no copy, no re-init), so
# the math is unchanged.  These patches are applied to the in-memory copy only;
# the training repo is never modified.
def _static_double_normalize(self, x):
    b, c, h, w = self._sa_shape
    x = x.reshape(b, self.num_heads, c // self.num_heads, h * w)
    x = x.softmax(dim=-1)
    x = x / (x.sum(dim=2, keepdim=True) + 1e-6)
    return x.reshape(b, c, h, w)


def _static_external_attention_forward(self, x, cross_k=None, cross_v=None):
    x = self.norm(x)
    if not self.use_cross_kv:
        x = F.conv2d(x, self.k, stride=2 if not self.same_in_out_chs else 1)
        x = self._double_normalize(x)
        return F.conv2d(x, self.v)
    b, c, h, w = self._sa_shape
    x = x.reshape(1, b * c, h, w)
    x = F.conv2d(x, cross_k, groups=b)
    x = (x.reshape(b, self.inter_channels, h, w) * self.inter_channels ** -0.5)
    x = x.softmax(dim=1).reshape(1, b * self.inter_channels, h, w)
    x = F.conv2d(x, cross_v, groups=b)
    return x.reshape(b, c, h, w)


def _static_ea_block_forward(self, inputs):
    x_h, x_l = inputs
    x_l_res = self.attn_shortcut_l(x_l) if self.proj_flag else x_l
    x_l = x_l_res + self.drop_path(self.attn_l(x_l))
    x_l = x_l + self.drop_path(self.mlp_l(x_l))

    x_h = x_h + F.interpolate(self.compression(x_l), size=self._sa_xh_hw,
                              mode="bilinear")
    kv = self.cross_kv(x_l)
    cross_k, cross_v = kv.chunk(2, dim=1)
    c = self._sa_cross_c
    cross_k = cross_k.permute(0, 2, 3, 1).reshape(-1, c, 1, 1)
    cross_v = cross_v.reshape(-1, self.cross_size * self.cross_size, 1, 1)
    x_h = x_h + self.drop_path(self.attn_h(x_h, cross_k, cross_v))
    x_h = x_h + self.drop_path(self.mlp_h(x_h))
    if self.use_injection:
        x_l = x_l + self.down(x_h)
    return x_h, x_l


def _static_dappm_forward(self, x):
    size = self._sa_size
    up = lambda y: F.interpolate(y, size=size, mode="bilinear")
    x0 = self.scale0(x)
    x1 = self.process1(up(self.scale1(x)) + x0)
    x2 = self.process2(up(self.scale2(x)) + x1)
    x3 = self.process3(up(self.scale3(x)) + x2)
    x4 = self.process4(up(self.scale4(x)) + x3)
    return self.compression(torch.cat((x0, x1, x2, x3, x4), dim=1)) + self.shortcut(x)


def _static_head_feature(self, x_h, x_l):
    context = self.dappm(x_l)
    context = F.interpolate(context, size=self._sa_xh_hw, mode="bilinear")
    return torch.cat((x_h, context), dim=1)


def patch_tail_static(tail, x_h_hw, x_l_hw, batch=1):
    """Install the static-shape forwards on the RTFormer receiver tail.

    ``x_h_hw`` / ``x_l_hw`` are the spatial sizes of the reconstructed features
    that enter the tail ([1,128,64,64] and [1,256,32,32] for the 512x512 input);
    ``batch`` is the (static) batch size of the exported graph.

    NOTE on dynamic batch: the cross-attention uses a *grouped* convolution with
    ``groups = batch`` (one group per sample), and ONNX requires ``groups`` to be
    a compile-time constant.  A dynamic batch axis is therefore NOT expressible
    for this graph -- export a static batch instead (``--batch-size N``).
    """
    import types

    n = 0
    xh_hw = tuple(x_h_hw)
    xl_hw = tuple(x_l_hw)
    for blk in (tail.layer4, tail.layer5):
        out_l = blk.attn_l.out_channels
        in_h = blk.attn_h.in_channels
        out_h = blk.attn_h.out_channels
        # attn_l: stride-2 conv when in/out channels differ, else stride 1
        stride = 1 if blk.attn_l.same_in_out_chs else 2
        sl_hw = (xl_hw[0] // stride, xl_hw[1] // stride)
        blk.attn_l._sa_shape = (batch, out_l, sl_hw[0], sl_hw[1])
        blk.attn_l._double_normalize = types.MethodType(_static_double_normalize, blk.attn_l)
        blk.attn_l.forward = types.MethodType(_static_external_attention_forward, blk.attn_l)
        # attn_h: cross-KV attention on the high branch
        blk.attn_h._sa_shape = (batch, in_h, xh_hw[0], xh_hw[1])
        blk.attn_h.forward = types.MethodType(_static_external_attention_forward, blk.attn_h)
        blk._sa_xh_hw = xh_hw
        blk._sa_cross_c = out_h
        blk.forward = types.MethodType(_static_ea_block_forward, blk)
        xl_hw = sl_hw
        n += 3
    # DAPPM runs on the low branch of stage 5
    tail.dappm._sa_size = xl_hw
    tail.dappm.forward = types.MethodType(_static_dappm_forward, tail.dappm)
    # adaptive avg pool -> global mean (exact for a 1x1 output)
    tail.dappm.scale4[0] = _GlobalMeanPool()
    tail._sa_xh_hw = xh_hw
    tail._head_feature = types.MethodType(_static_head_feature, tail)
    n += 2
    return n


class _GlobalMeanPool(nn.Module):
    """``nn.AdaptiveAvgPool2d((1,1))`` written as an explicit mean."""

    def forward(self, x):
        return x.mean(dim=(-2, -1), keepdim=True)


# --------------------------------------------------------------------------- #
# exportable receiver
# --------------------------------------------------------------------------- #
class Synthesizer(nn.Module):
    """``bottleneck._synthesize`` with folded-IGDN weights (same math)."""

    def __init__(self, bn):
        super().__init__()
        self.g_s_trunk = _convert_gdn(copy.deepcopy(bn.g_s_trunk))
        self.head_l = copy.deepcopy(bn.head_l)
        self.head_h = _convert_gdn(copy.deepcopy(bn.head_h))

    def forward(self, y_hat, hw_h, hw_l):
        t = self.g_s_trunk(y_hat)
        x_l = self.head_l(t)
        x_h = self.head_h(torch.cat([t, x_l], dim=1))
        if x_h.shape[-2:] != tuple(hw_h):
            x_h = F.interpolate(x_h, size=tuple(hw_h), mode="bilinear", align_corners=False)
        if x_l.shape[-2:] != tuple(hw_l):
            x_l = F.interpolate(x_l, size=tuple(hw_l), mode="bilinear", align_corners=False)
        return x_h, x_l


class ReceiverNet(nn.Module):
    """y_hat -> logits (g_s synthesis + teacher receiver tail + resize)."""

    def __init__(self, student, out_size=128, hw_h=(64, 64), hw_l=(32, 32), batch=1):
        super().__init__()
        assert student.split == "mid", f"this exporter targets the mid split (got {student.split})"
        self.synth = Synthesizer(student.bottleneck)
        # deep copies: the stock student (used for the reference path) stays intact
        self.tail = copy.deepcopy(student.teacher.backbone)
        self.neck = copy.deepcopy(student.teacher.neck)
        self.decoder = copy.deepcopy(student.teacher.decoder)
        self.seg_head = copy.deepcopy(student.teacher.heads["seg"])
        self.n_pools = replace_adaptive_pools(self.tail)
        self.n_patches = patch_tail_static(self.tail, hw_h, hw_l, batch=batch)
        self.hw_h = tuple(hw_h)
        self.hw_l = tuple(hw_l)
        self.out_size = int(out_size)

    def forward(self, y_hat):
        x_h, x_l = self.synth(y_hat, self.hw_h, self.hw_l)
        feats = self.tail.forward_receiver_mid(x_h, x_l)
        d = self.decoder(self.neck(feats))
        logits = self.seg_head(d)                      # [B,150,256,256] native
        if self.out_size != logits.shape[-1]:
            logits = F.interpolate(logits, size=(self.out_size, self.out_size),
                                   mode="bilinear", align_corners=False)
        return logits


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def cosine(a, b):
    a = np.asarray(a, np.float64).reshape(-1)
    b = np.asarray(b, np.float64).reshape(-1)
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def parity_report(name, ref, got):
    ref = np.asarray(ref, np.float32)
    got = np.asarray(got, np.float32)
    diff = np.abs(ref - got)
    return {
        "name": name,
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "cosine": cosine(ref, got),
        "argmax_agree": float((ref.argmax(-1) == got.argmax(-1)).mean()),
    }


def build_inputs(batch, device, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(batch, 192, 16, 16, generator=g).to(device)


def real_y_hat(ref_dir, n, device):
    """y_hat = round(y - means) + means from a sender dump.

    Accepts either the FP32 reference dir (``ref_y.npy`` / ``ref_means.npy``) or a
    Hailo sender dump dir (``chunk_*.npz`` with ``y`` / ``means``).
    """
    import glob
    chunks = sorted(glob.glob(os.path.join(ref_dir, "chunk_*.npz")))
    if chunks:
        ys, ms = [], []
        for c in chunks:
            z = np.load(c)
            ys.append(z["y"])
            ms.append(z["means"])
        y = np.concatenate(ys, 0)[:n]
        means = np.concatenate(ms, 0)[:n]
    else:
        y = np.load(os.path.join(ref_dir, "ref_y.npy"), mmap_mode="r")[:n]
        means = np.load(os.path.join(ref_dir, "ref_means.npy"), mmap_mode="r")[:n]
    y_hat = np.round(y - means) + means
    return torch.from_numpy(np.ascontiguousarray(y_hat)).to(device)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--out-size", type=int, default=128,
                    help="output logits resolution (default 128)")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="static batch size of the exported graph")
    ap.add_argument("--dynamic-batch", action="store_true",
                    help="(unsupported) the cross-attention uses groups=batch, so the "
                         "batch axis must stay static")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--skip-ort", action="store_true")
    ap.add_argument("--ref-dir", default="/dev/shm/hailo2000/ref",
                    help="sender dump used for the real-input parity check "
                         "(FP32 ref dir with ref_y.npy, or a Hailo dir with chunk_*.npz)")
    ap.add_argument("--n-check", type=int, default=4)
    ap.add_argument("--report", default=None, help="optional JSON report path")
    a = ap.parse_args(argv)

    device = torch.device(a.device)
    torch.manual_seed(0)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)

    print(f"[export] loading student from {a.ckpt}", flush=True)
    student, cfg = load_student(a.ckpt)
    student = student.to(device).eval()
    net = ReceiverNet(student, out_size=a.out_size, batch=a.batch_size).to(device).eval()

    sample = build_inputs(a.batch_size, device)
    with torch.no_grad():
        torch_logits = net(sample).float().cpu().numpy()
    print(f"[export] torch logits {torch_logits.shape} "
          f"(native head 256x256 resized to {a.out_size}x{a.out_size})", flush=True)

    # ---- reference path: the stock deployment code (stock compressai GDN) ----
    with torch.no_grad():
        xh, xl = student.bottleneck._synthesize(sample, (64, 64), (32, 32))
        stock = student._receiver((xh, xl), {})["seg"].float()
        if stock.shape[-1] != a.out_size:
            stock = F.interpolate(stock, size=(a.out_size, a.out_size),
                                  mode="bilinear", align_corners=False)
    stock = stock.cpu().numpy()
    report = {"torch_vs_stock_path": parity_report("torch_wrapper_vs_stock_receiver",
                                                   stock, torch_logits)}
    print(f"[export] static-pool replacements: {net.n_pools}", flush=True)
    report["static_pool_exactness"] = check_static_pool_exact(device)
    print(f"[export] static pool exactness: {report['static_pool_exactness']}", flush=True)
    print(f"[export] wrapper vs stock receiver: {report['torch_vs_stock_path']}", flush=True)

    # ---- ONNX export --------------------------------------------------------
    dynamic_axes = None
    if a.dynamic_batch:
        raise SystemExit("--dynamic-batch is not supported: the RTFormer cross-attention "
                         "uses a grouped convolution with groups=batch, and ONNX requires "
                         "a constant 'groups'.  Use --batch-size N (static) instead.")
    print(f"[export] tracing ONNX (opset {a.opset}, batch {a.batch_size}) ...", flush=True)
    t0 = time.time()
    with torch.no_grad():
        torch.onnx.export(
            net,
            sample,
            a.out,
            input_names=["y_hat"],
            output_names=["logits"],
            opset_version=a.opset,
            do_constant_folding=True,
            dynamic_axes=dynamic_axes,
            dynamo=False,
        )
    export_s = time.time() - t0
    print(f"[export] wrote {a.out} in {export_s:.1f}s "
          f"({os.path.getsize(a.out) / 1e6:.1f} MB)", flush=True)

    import onnx
    model = onnx.load(a.out)
    onnx.checker.check_model(model)
    ops = sorted({n.op_type for n in model.graph.node})
    print(f"[export] onnx ok, {len(model.graph.node)} nodes, ops={ops}", flush=True)

    # ---- ORT parity ---------------------------------------------------------
    ort_info = None
    if not a.skip_ort:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(a.out, so, providers=["CPUExecutionProvider"])
        feed = {"y_hat": sample.float().cpu().numpy()}
        t0 = time.time()
        got = sess.run(["logits"], feed)[0]
        ort_s = time.time() - t0
        ort_info = parity_report("onnxruntime_vs_pytorch", torch_logits, got)
        ort_info["ort_cpu_seconds"] = ort_s
        report["onnxruntime_vs_pytorch"] = ort_info
        print(f"[export] ORT vs PyTorch: {ort_info}", flush=True)

    # ---- real-input parity (sender dump -> round(y-means)+means) -----------
    real_info = None
    if os.path.isdir(a.ref_dir):
        try:
            y_hat = real_y_hat(a.ref_dir, a.n_check, device)
            # the exported graph has a STATIC batch (groups=batch in the
            # cross-attention), so images are checked one by one
            per_img = []
            with torch.no_grad():
                for i in range(y_hat.shape[0]):
                    per_img.append(net(y_hat[i:i + 1]).float().cpu().numpy())
            t_ref = np.concatenate(per_img, 0)
            if ort_info is not None:
                sess = ort.InferenceSession(a.out, so, providers=["CPUExecutionProvider"])
                per_img = []
                for i in range(y_hat.shape[0]):
                    per_img.append(sess.run(
                        ["logits"], {"y_hat": y_hat[i:i + 1].float().cpu().numpy()})[0])
                o_ref = np.concatenate(per_img, 0)
                real_info = parity_report("onnxruntime_vs_pytorch_real_yhat", t_ref, o_ref)
                report["real_yhat"] = real_info
                print(f"[export] real-y_hat parity ({y_hat.shape[0]} imgs): {real_info}",
                      flush=True)
        except Exception as e:                                   # noqa: BLE001
            print(f"[export] real-input parity skipped: {e}", flush=True)

    report.update({
        "onnx": os.path.abspath(a.out),
        "onnx_mb": os.path.getsize(a.out) / 1e6,
        "opset": a.opset,
        "out_size": a.out_size,
        "batch_size": a.batch_size,
        "dynamic_batch": bool(a.dynamic_batch),
        "export_seconds": export_s,
        "input_shape": list(sample.shape),
        "output_shape": list(torch_logits.shape),
        "ckpt": a.ckpt,
        "torch": torch.__version__,
        "onnxruntime": ort_info is not None,
    })
    if a.report:
        os.makedirs(os.path.dirname(os.path.abspath(a.report)), exist_ok=True)
        with open(a.report, "w") as f:
            json.dump(report, f, indent=1)
        print(f"[export] report -> {a.report}", flush=True)
    print(json.dumps({k: v for k, v in report.items()
                      if k in ("torch_vs_stock_path", "onnxruntime_vs_pytorch", "real_yhat")},
                     indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
