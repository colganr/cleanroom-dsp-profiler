"""Verifies bit-exact parity between Pure-Python feature evaluators and compiled Pure-Rust `cdylib`."""

import unittest
import numpy as np

from amp_profiler.nonlinear_fit import (
    apply_post_inv_shelf,
    eval_amp_features_py,
    eval_pedal_features_py,
)
from amp_profiler.rust_bridge import get_rust_bridge
from amp_profiler.stimulus import SR_DEFAULT, generate_procedural_guitar_di


def test_amp_180ch_python_rust_parity():
    bridge = get_rust_bridge(allow_missing=True)
    if bridge is None:
        raise unittest.SkipTest("dsp-core-rust cdylib not built")

    u = generate_procedural_guitar_di(duration_s=0.25, sr=SR_DEFAULT, peak=0.45)
    phi_py, sat0_py, pp_py, pz_py = eval_amp_features_py(u, sr=SR_DEFAULT)
    phi_rs, sat0_rs, pp_rs, pz_rs = bridge.eval_amp_features(u, sr=SR_DEFAULT)

    assert abs(pp_py - pp_rs) < 1e-6
    assert abs(pz_py - pz_rs) < 1e-6
    assert float(np.max(np.abs(sat0_py - sat0_rs))) < 2e-5
    assert float(np.max(np.abs(phi_py - phi_rs))) < 2e-5

    rng = np.random.default_rng(123)
    w = rng.normal(0.0, 0.05, size=180).astype(np.float32)
    y_py = apply_post_inv_shelf(sat0_py + phi_py @ w, pp_py, pz_py)
    y_rs = bridge.render_amp_core(u, w, sr=SR_DEFAULT)
    assert float(np.max(np.abs(y_py - y_rs))) < 5e-5


def test_pedal_126ch_python_rust_parity():
    bridge = get_rust_bridge(allow_missing=True)
    if bridge is None:
        raise unittest.SkipTest("dsp-core-rust cdylib not built")

    x = generate_procedural_guitar_di(duration_s=0.25, sr=SR_DEFAULT, peak=0.25)
    u_pre = (x * 4.0).astype(np.float32)

    for sub_dry in (False, True):
        psi_py, ub_py = eval_pedal_features_py(x, u_pre, subtract_dry=sub_dry)
        psi_rs, ub_rs = bridge.eval_pedal_features(x, u_pre, subtract_dry=sub_dry)
        assert float(np.max(np.abs(ub_py - ub_rs))) < 2e-5
        assert float(np.max(np.abs(psi_py - psi_rs))) < 2e-5
