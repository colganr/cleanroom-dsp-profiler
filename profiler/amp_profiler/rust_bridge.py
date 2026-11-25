"""
Cross-Platform Optional C-ABI Bridge to Compiled `dsp-core-rust` (`libcleanroom_dsp`).

When `dsp-core-rust` is compiled (`cargo build --release`), this bridge runs the exact
Pure-Rust feature evaluators and renderers via `ctypes` for 50-100x faster profiling and
guaranteed bit-exact Python <-> Rust parity. Falls back cleanly to pure Python if unavailable.
"""

from __future__ import annotations

import ctypes
import os
import platform
from pathlib import Path
import numpy as np

AMP_CHANNELS = 180
PEDAL_CHANNELS = 126
PEDAL_FIR_LEN = 256

_FP = ctypes.POINTER(ctypes.c_float)


def find_rust_cdylib() -> Path | None:
    """Searches `CLEANROOM_DSP_LIB` env var and standard `dsp-core-rust/target/release` locations."""
    env_p = os.environ.get("CLEANROOM_DSP_LIB", "").strip()
    if env_p and Path(env_p).is_file():
        return Path(env_p)

    sys_name = platform.system()
    if sys_name == "Darwin":
        fname = "libcleanroom_dsp.dylib"
    elif sys_name == "Windows":
        fname = "cleanroom_dsp.dll"
    else:
        fname = "libcleanroom_dsp.so"

    here = Path(__file__).resolve()
    candidates = [
        here.parents[2] / "dsp-core-rust" / "target" / "release" / fname,
        here.parents[2] / "dsp-core-rust" / "target" / "debug" / fname,
        Path.cwd() / "dsp-core-rust" / "target" / "release" / fname,
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


class RustDspBridge:
    """Wraps the compiled `cleanroom_dsp` C ABI library."""

    def __init__(self, lib_path: str | Path | None = None):
        p = Path(lib_path) if lib_path else find_rust_cdylib()
        if p is None or not p.is_file():
            raise FileNotFoundError(
                "Compiled cleanroom_dsp library not found. Build it with `cd dsp-core-rust && cargo build --release`."
            )
        self.path = p
        self.lib = ctypes.CDLL(str(p))

        self.lib.cleanroom_eval_amp_features.argtypes = [
            _FP, ctypes.c_size_t, ctypes.c_float,
            ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
            _FP, _FP, _FP,
        ]
        self.lib.cleanroom_eval_amp_features.restype = ctypes.c_int

        self.lib.cleanroom_render_amp_core.argtypes = [
            _FP, ctypes.c_size_t, ctypes.c_float,
            ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
            _FP, _FP,
        ]
        self.lib.cleanroom_render_amp_core.restype = ctypes.c_int

        self.lib.cleanroom_eval_pedal_features.argtypes = [
            _FP, _FP, ctypes.c_size_t, ctypes.c_int, _FP, _FP,
        ]
        self.lib.cleanroom_eval_pedal_features.restype = ctypes.c_int

        self.lib.cleanroom_render_pedal_engine.argtypes = [
            _FP, ctypes.c_size_t, ctypes.c_int, ctypes.c_float, ctypes.c_float,
            _FP, _FP, _FP, _FP,
        ]
        self.lib.cleanroom_render_pedal_engine.restype = ctypes.c_int

        self.lib.cleanroom_eval_envelope_filter.argtypes = [
            _FP, ctypes.c_size_t, ctypes.c_float,
            ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_int,
            _FP, _FP,
        ]
        self.lib.cleanroom_eval_envelope_filter.restype = ctypes.c_int

    def eval_amp_features(
        self,
        u_in: np.ndarray,
        sr: float = 48_000.0,
        cp0: float = 0.212,
        cn0: float = 0.227,
        knee: float = 2.9,
        bloom_alpha: float = 0.18,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        u = np.ascontiguousarray(u_in, dtype=np.float32)
        n = len(u)
        phi = np.zeros((n, AMP_CHANNELS), dtype=np.float32)
        sat0 = np.zeros(n, dtype=np.float32)
        shelf = np.zeros(2, dtype=np.float32)
        rc = self.lib.cleanroom_eval_amp_features(
            u.ctypes.data_as(_FP), n, float(sr),
            float(cp0), float(cn0), float(knee), float(bloom_alpha),
            phi.ctypes.data_as(_FP), sat0.ctypes.data_as(_FP), shelf.ctypes.data_as(_FP),
        )
        if rc != 0:
            raise RuntimeError(f"cleanroom_eval_amp_features failed with code {rc}")
        return phi, sat0, float(shelf[0]), float(shelf[1])

    def render_amp_core(
        self,
        u_in: np.ndarray,
        weights_180: np.ndarray,
        sr: float = 48_000.0,
        cp0: float = 0.212,
        cn0: float = 0.227,
        knee: float = 2.9,
        bloom_alpha: float = 0.18,
    ) -> np.ndarray:
        u = np.ascontiguousarray(u_in, dtype=np.float32)
        w = np.ascontiguousarray(weights_180, dtype=np.float32)
        y = np.zeros(len(u), dtype=np.float32)
        rc = self.lib.cleanroom_render_amp_core(
            u.ctypes.data_as(_FP), len(u), float(sr),
            float(cp0), float(cn0), float(knee), float(bloom_alpha),
            w.ctypes.data_as(_FP), y.ctypes.data_as(_FP),
        )
        if rc != 0:
            raise RuntimeError(f"cleanroom_render_amp_core failed with code {rc}")
        return y

    def eval_pedal_features(
        self,
        x_in: np.ndarray,
        u_pre: np.ndarray,
        subtract_dry: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        x = np.ascontiguousarray(x_in, dtype=np.float32)
        u = np.ascontiguousarray(u_pre, dtype=np.float32)
        n = min(len(x), len(u))
        psi = np.zeros((n, PEDAL_CHANNELS), dtype=np.float32)
        ub = np.zeros(n, dtype=np.float32)
        rc = self.lib.cleanroom_eval_pedal_features(
            x.ctypes.data_as(_FP), u.ctypes.data_as(_FP), n,
            1 if subtract_dry else 0,
            psi.ctypes.data_as(_FP), ub.ctypes.data_as(_FP),
        )
        if rc != 0:
            raise RuntimeError(f"cleanroom_eval_pedal_features failed with code {rc}")
        return psi, ub

    def render_pedal_engine(
        self,
        x_in: np.ndarray,
        pre_ir: np.ndarray,
        post_ir: np.ndarray,
        weights_126: np.ndarray,
        subtract_dry: bool = False,
        level_gain: float = 1.0,
        mix: float = 1.0,
    ) -> np.ndarray:
        x = np.ascontiguousarray(x_in, dtype=np.float32)
        pre = np.ascontiguousarray(pre_ir, dtype=np.float32)
        post = np.ascontiguousarray(post_ir, dtype=np.float32)
        w = np.ascontiguousarray(weights_126, dtype=np.float32)
        y = np.zeros(len(x), dtype=np.float32)
        rc = self.lib.cleanroom_render_pedal_engine(
            x.ctypes.data_as(_FP), len(x), 1 if subtract_dry else 0,
            float(level_gain), float(mix),
            pre.ctypes.data_as(_FP), post.ctypes.data_as(_FP),
            w.ctypes.data_as(_FP), y.ctypes.data_as(_FP),
        )
        if rc != 0:
            raise RuntimeError(f"cleanroom_render_pedal_engine failed with code {rc}")
        return y

    def eval_envelope_filter(
        self,
        x_in: np.ndarray,
        params_14: np.ndarray,
        sr: float = 48_000.0,
        drive_knob: float = 0.60,
        peak_q_scale: float = 1.0,
        slow_mult: float = 1.0,
        sweep_down: bool = False,
    ) -> np.ndarray:
        x = np.ascontiguousarray(x_in, dtype=np.float32)
        p = np.ascontiguousarray(params_14, dtype=np.float32)
        if len(p) != 14:
            raise ValueError(f"Expected 14 envelope filter params, got {len(p)}")
        y = np.zeros(len(x), dtype=np.float32)
        rc = self.lib.cleanroom_eval_envelope_filter(
            x.ctypes.data_as(_FP), len(x), float(sr),
            float(drive_knob), float(peak_q_scale), float(slow_mult),
            1 if sweep_down else 0,
            p.ctypes.data_as(_FP), y.ctypes.data_as(_FP),
        )
        if rc != 0:
            raise RuntimeError(f"cleanroom_eval_envelope_filter failed with code {rc}")
        return y


_CACHED_BRIDGE: RustDspBridge | None = None
_BRIDGE_PROBED = False


def get_rust_bridge(allow_missing: bool = True) -> RustDspBridge | None:
    global _CACHED_BRIDGE, _BRIDGE_PROBED
    if not _BRIDGE_PROBED:
        _BRIDGE_PROBED = True
        try:
            _CACHED_BRIDGE = RustDspBridge()
        except FileNotFoundError:
            _CACHED_BRIDGE = None
    if _CACHED_BRIDGE is None and not allow_missing:
        raise FileNotFoundError("cleanroom_dsp cdylib is not built")
    return _CACHED_BRIDGE
