#!/usr/bin/env python3
"""Shared helpers for the teacher wide-attention (``attn_h``) Hailo-8 work.

Contents
--------
* ``selfcheck_onnx``   -- the gate every rewritten ONNX has to pass *before* DFC
                          is touched: onnx.checker -> ORT load + run -> numeric
                          comparison against a reference model.
* ``dfc_pipeline``     -- parse -> optimize -> compile for one ONNX, crash tolerant,
                          result reported as a dict (status OK / FAIL / CRASH).
* small graph-building utilities that keep initializer / node names unique.

Everything here is import-safe in both the system python (onnx only) and the
``hailo-dfc`` env (DFC imported lazily).
"""
from __future__ import annotations

import json
import os
import signal
import time

import numpy as np

MEAN255 = [123.675, 116.28, 103.53]
STD255 = [58.395, 57.12, 57.375]


# --------------------------------------------------------------------------- #
# ONNX self-check gate
# --------------------------------------------------------------------------- #
def _rand_inputs(model, rng, scale=1.0):
    import onnx
    feeds = {}
    init = {i.name for i in model.graph.initializer}
    for vi in model.graph.input:
        if vi.name in init:
            continue
        shp = [d.dim_value for d in vi.type.tensor_type.shape.dim]
        feeds[vi.name] = (rng.standard_normal(shp) * scale).astype(np.float32)
    return feeds


def ort_run(path_or_model, feeds, providers=("CPUExecutionProvider",)):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    if not isinstance(path_or_model, (str, bytes)):
        path_or_model = path_or_model.SerializeToString()
    sess = ort.InferenceSession(path_or_model, so, providers=list(providers))
    names = [o.name for o in sess.get_outputs()]
    return dict(zip(names, sess.run(None, feeds)))


def check_names_unique(model):
    """Every node name, node output and initializer name must be unique."""
    g = model.graph
    problems = []
    seen = set()
    for i in g.initializer:
        if i.name in seen:
            problems.append(f"duplicate initializer {i.name}")
        seen.add(i.name)
    outs = set(x.name for x in g.input)
    for n in g.node:
        for o in n.output:
            if o in outs or o in seen and o not in [x.name for x in g.input]:
                problems.append(f"duplicate tensor name {o}")
            outs.add(o)
    names = [n.name for n in g.node if n.name]
    if len(names) != len(set(names)):
        problems.append("duplicate node names")
    return problems


def check_topological(model):
    g = model.graph
    avail = {x.name for x in g.input} | {x.name for x in g.initializer}
    bad = []
    for i, n in enumerate(g.node):
        for t in n.input:
            if t and t not in avail:
                bad.append((i, n.name, t))
        avail.update(n.output)
    return bad


def selfcheck_onnx(path, ref_path=None, n_trials=2, seed=0, tol=1e-4,
                   input_scale=1.0, exact=True, verbose=True):
    """Gate for a rewritten ONNX.  Returns a dict; ``ok`` is the verdict.

    Steps: checker (full, with data prop) -> unique names -> topological order
    -> shape inference -> ORT load/run -> numeric diff vs ``ref_path``.
    """
    import onnx
    res = dict(path=path, ok=False)
    m = onnx.load(path)
    onnx.checker.check_model(m, full_check=False)
    res["checker"] = "OK"
    probs = check_names_unique(m)
    if probs:
        res["names"] = probs[:5]
        return res
    bad = check_topological(m)
    if bad:
        res["topo"] = bad[:5]
        return res
    onnx.shape_inference.infer_shapes(m, strict_mode=True)
    res["shape_inference"] = "OK"
    rng = np.random.default_rng(seed)
    max_diff, max_rel = 0.0, 0.0
    for t in range(n_trials):
        feeds = _rand_inputs(m, rng, input_scale)
        out = ort_run(path, feeds)
        res["ort"] = "OK"
        if ref_path:
            ref = ort_run(ref_path, feeds)
            for k in ref:
                a, b = ref[k].astype(np.float64), out[k].astype(np.float64)
                d = float(np.abs(a - b).max())
                max_diff = max(max_diff, d)
                max_rel = max(max_rel, d / (float(np.abs(a).max()) + 1e-12))
    if ref_path:
        res["max_abs_diff"] = max_diff
        res["max_rel_diff"] = max_rel
        res["tol"] = tol
        res["ok"] = (max_diff <= tol) if exact else True
    else:
        res["ok"] = True
    if verbose:
        print("[selfcheck]", json.dumps(res), flush=True)
    return res


# --------------------------------------------------------------------------- #
# DFC pipeline
# --------------------------------------------------------------------------- #
def _install_crash_diag():
    import hailo_sdk_client.allocator.hailo_tools_runner as R
    orig = R._handle_hailo_tools_error

    def patched(err, process):
        rc = process.returncode
        nm = ""
        if rc is not None and rc < 0:
            try:
                nm = signal.Signals(-rc).name
            except Exception:
                pass
        print(f"[DIAG] compiler subprocess returncode={rc} {nm}", flush=True)
        return orig(err, process)

    R._handle_hailo_tools_error = patched


def hn_input_layers(runner):
    hn = runner.get_hn()
    if isinstance(hn, str):
        hn = json.loads(hn)
    out = []
    for lname, v in hn["layers"].items():
        if v.get("type") == "input_layer":
            out.append((lname, (v.get("original_names") or [lname])[0],
                        v.get("output_shapes", [[1, 1, 1, 1]])[0]))
    return out


def dfc_pipeline(onnx_path, name, out_dir, calib_fn, *, script_lines=(),
                 opt_level=2, comp_level=0, calib_n=64, do_compile=True,
                 hw_arch="hailo8", start=None, end=None):
    """parse -> optimize -> compile.  Never raises; returns result dict.

    ``calib_fn(input_layers)`` receives ``[(hailo_name, onnx_name, shape)]`` and
    returns the dict ``{hailo_name: ndarray}`` for ``runner.optimize``.
    """
    os.makedirs(out_dir, exist_ok=True)
    os.chdir(out_dir)
    _install_crash_diag()
    from hailo_sdk_client import ClientRunner
    res = dict(name=name, onnx=onnx_path)
    har_p = os.path.join(out_dir, f"{name}_parsed.har")
    har_o = os.path.join(out_dir, f"{name}_optimized.har")
    hef = os.path.join(out_dir, f"{name}.hef")

    t0 = time.time()
    try:
        r = ClientRunner(hw_arch=hw_arch)
        kw = {}
        if start:
            kw["start_node_names"] = start
        if end:
            kw["end_node_names"] = end
        r.translate_onnx_model(onnx_path, name, **kw)
        r.save_har(har_p)
        res["parse"] = "OK"
    except Exception as e:  # noqa
        res["parse"] = f"FAIL {type(e).__name__}: {str(e)[:300]}"
        return _finish(res, out_dir, name)
    res["parse_s"] = round(time.time() - t0, 1)

    t0 = time.time()
    try:
        r2 = ClientRunner(har=har_p, hw_arch=hw_arch)
        lines = [
            f"model_optimization_flavor(optimization_level={opt_level}, compression_level={comp_level}, batch_size=1)",
            f"model_optimization_config(calibration, batch_size=1, calibset_size={calib_n})",
            "model_optimization_config(checker_cfg, policy=disabled)",
        ] + list(script_lines)
        r2.load_model_script("\n".join(lines) + "\n")
        calib = calib_fn(hn_input_layers(r2))
        r2.optimize(calib)
        r2.save_har(har_o)
        res["optimize"] = "OK"
    except Exception as e:  # noqa
        res["optimize"] = f"FAIL {type(e).__name__}: {str(e)[:300]}"
        return _finish(res, out_dir, name)
    res["optimize_s"] = round(time.time() - t0, 1)
    res["har_opt"] = har_o

    if do_compile:
        t0 = time.time()
        try:
            r3 = ClientRunner(har=har_o, hw_arch=hw_arch)
            blob = r3.compile()
            with open(hef, "wb") as f:
                f.write(blob)
            res["compile"] = "OK"
            res["hef"] = hef
            res["hef_mib"] = round(len(blob) / 2**20, 2)
        except Exception as e:  # noqa
            msg = str(e)
            res["compile"] = ("CRASH(-6 SIGABRT)" if "unexpected crash" in msg
                              else f"FAIL {type(e).__name__}: {msg[:300]}")
        res["compile_s"] = round(time.time() - t0, 1)
    return _finish(res, out_dir, name)


def _finish(res, out_dir, name):
    with open(os.path.join(out_dir, f"{name}_result.json"), "w") as f:
        json.dump(res, f, indent=2)
    print("[RESULT]", json.dumps(res), flush=True)
    return res
