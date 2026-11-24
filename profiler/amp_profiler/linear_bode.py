"""
Linear Transfer-Function Extraction & Wiener-Hammerstein Cepstral Factorization.

Implements:
  - Settled bipolar symmetric small-signal impulse capture (`±1e-4`), cancelling DC/bias drift
    and preventing high-gain clipping during linear Bode measurement.
  - Hard-saturated log-spaced sine-sweep quadrature lock-in measurement of `H_post(f)` and
    FFT deconvolution `H_pre(f) = H_tot(f) / H_post(f)`.
  - Real-cepstrum Hilbert folding into causal minimum-phase FIRs (matching `cepstral_fir.rs`).
  - Non-additive `2^K` front-end switch-combination tensor capture.
  - Coupled 3D `(Bass x Middle x Treble)` tone-stack tensor capture indexed by cap switches.
  - 21-point Gain / Master / Presence bright-cap taper sweeps.
"""

from __future__ import annotations

import math
import numpy as np
from .backends.base import TargetBackend

SR_DEFAULT = 48_000
NUM_FREQS = 64
FREQS_64 = np.logspace(math.log10(20.0), math.log10(20_000.0), NUM_FREQS).astype(np.float32)


def capture_bipolar_ir(
    backend: TargetBackend,
    ir_len: int = 1024,
    onset: int = 0,
    imp_amp: float = 1e-4,
    pre_pad: int = 128,
) -> np.ndarray:
    """
    Captures a linear small-signal impulse response using symmetric bipolar impulses:
      h[n] = 0.5 * (y_+[n] - y_-[n]) / imp_amp
    """
    total = pre_pad + onset + ir_len + 256
    imp_pos = np.zeros(total, dtype=np.float32)
    imp_neg = np.zeros(total, dtype=np.float32)
    imp_pos[pre_pad] = imp_amp
    imp_neg[pre_pad] = -imp_amp

    backend.settle(full=False)
    y_pos = backend.render_mono(imp_pos)
    backend.settle(full=False)
    y_neg = backend.render_mono(imp_neg)

    start = pre_pad + onset
    ir = 0.5 * (y_pos[start : start + ir_len].astype(np.float64) - y_neg[start : start + ir_len].astype(np.float64)) / imp_amp
    return ir.astype(np.float32)


def ir_to_bode64(ir: np.ndarray, sr: int = SR_DEFAULT, fft_n: int = 4096) -> np.ndarray:
    """Evaluates magnitude response (in dB) of an impulse response on `FREQS_64`."""
    spec = np.abs(np.fft.rfft(ir.astype(np.float64), n=fft_n)) + 1e-9
    freqs = np.fft.rfftfreq(fft_n, d=1.0 / sr)
    mag = np.interp(FREQS_64.astype(np.float64), freqs, spec)
    return (20.0 * np.log10(np.maximum(mag, 1e-9))).astype(np.float32)


def synthesize_min_phase_ir(
    base_ir: np.ndarray,
    delta_db_64: np.ndarray,
    sr: int = SR_DEFAULT,
    polarity: float = 1.0,
    out_len: int | None = None,
    fade_len: int = 64,
    fft_n: int = 2048,
) -> np.ndarray:
    """
    Synthesizes a causal minimum-phase FIR by folding `delta_db_64` via the real cepstrum
    and convolving with `base_ir`. Matches `dsp-core-rust/src/cepstral_fir.rs`.
    """
    if out_len is None:
        out_len = len(base_ir)
    half = fft_n // 2
    freqs_fft = np.arange(half + 1, dtype=np.float64) * sr / fft_n
    d_db = np.interp(freqs_fft, FREQS_64.astype(np.float64), delta_db_64.astype(np.float64))

    cep = np.zeros(fft_n, dtype=np.complex128)
    cep[: half + 1] = d_db * (math.log(10.0) / 20.0)
    cep[half + 1 :] = cep[1:half][::-1]

    c_time = np.fft.ifft(cep).real
    c_min = np.zeros(fft_n, dtype=np.float64)
    c_min[0] = c_time[0]
    c_min[1:half] = 2.0 * c_time[1:half]
    c_min[half] = c_time[half]
    h_min = np.exp(np.fft.fft(c_min))

    base_pad = np.zeros(fft_n, dtype=np.float64)
    base_pad[: min(len(base_ir), fft_n)] = base_ir[: min(len(base_ir), fft_n)]
    h = np.fft.ifft(np.fft.fft(base_pad) * h_min).real[:out_len] * polarity

    if fade_len > 1 and out_len >= fade_len:
        fade = np.ones(out_len, dtype=np.float64)
        fade[out_len - fade_len :] = 0.5 * (1.0 + np.cos(np.pi * np.arange(fade_len) / (fade_len - 1)))
        h *= fade
    return h.astype(np.float32)


def measure_post_ir_hard_sat(
    backend: TargetBackend,
    post_len: int = 64,
    drive_amp: float = 0.30,
    n_tones: int = 32,
    sr: int = SR_DEFAULT,
) -> np.ndarray:
    """
    Measures the post-clipping filter `H_post(f)` via hard-saturated quadrature lock-in sine tones.

    When a nonlinear stage is driven hard into its rails, its fundamental output amplitude
    converges toward a frequency-flat square-wave fundamental `(4/pi) * rail`, so output
    magnitude variation across frequency directly reveals the post-clipping filter `H_post(f)`.
    """
    test_freqs = np.logspace(math.log10(30.0), math.log10(16_000.0), n_tones)
    amps = np.zeros(n_tones, dtype=np.float64)
    dur_n = int(0.08 * sr)
    skip_n = int(0.025 * sr)
    t = np.arange(dur_n, dtype=np.float64) / sr
    t_eval = t[skip_n:]

    for idx, f in enumerate(test_freqs):
        tone = (drive_amp * np.sin(2.0 * math.pi * f * t)).astype(np.float32)
        backend.settle(full=False)
        y = backend.render_mono(tone).astype(np.float64)[skip_n:]
        # Quadrature lock-in fundamental extraction
        c = np.cos(2.0 * math.pi * f * t_eval)
        s = np.sin(2.0 * math.pi * f * t_eval)
        a_cos = 2.0 * np.mean(y * c)
        a_sin = 2.0 * np.mean(y * s)
        amps[idx] = math.hypot(a_cos, a_sin)

    # Normalize to 0 dB at ~400 Hz passband reference so gain stays in H_pre
    ref_idx = int(np.argmin(np.abs(test_freqs - 400.0)))
    ref_amp = max(amps[ref_idx], 1e-7)
    post_db_tones = np.clip(20.0 * np.log10(np.maximum(amps, 1e-7) / ref_amp), -36.0, 12.0)
    post_db_64 = np.interp(FREQS_64.astype(np.float64), test_freqs, post_db_tones).astype(np.float32)

    dirac = np.zeros(post_len, dtype=np.float32)
    dirac[0] = 1.0
    ir_post = synthesize_min_phase_ir(dirac, post_db_64, sr=sr, out_len=post_len, fade_len=12)
    return ir_post


def deconvolve_pre_from_total(
    h_tot: np.ndarray,
    h_post: np.ndarray,
    pre_len: int = 1024,
    eps: float = 1e-4,
) -> np.ndarray:
    """Computes `H_pre(f) = H_tot(f) / H_post(f)` so `H_pre * H_post == H_tot` linearly."""
    nfft = 1 << (max(len(h_tot), len(h_post), pre_len) * 2 - 1).bit_length()
    H_tot = np.fft.rfft(h_tot.astype(np.float64), n=nfft)
    H_post = np.fft.rfft(h_post.astype(np.float64), n=nfft)
    H_pre = H_tot * np.conj(H_post) / (np.abs(H_post) ** 2 + eps * eps)
    h_pre = np.fft.irfft(H_pre, n=nfft)[:pre_len]
    fade = np.ones(pre_len, dtype=np.float64)
    if pre_len >= 64:
        fade[-64:] = 0.5 * (1.0 + np.cos(np.pi * np.arange(64) / 63.0))
    return (h_pre * fade).astype(np.float32)


def capture_switch_combo_tensor(
    backend: TargetBackend,
    switch_pids: list[int],
    ref_bode_64: np.ndarray,
    onset: int = 0,
    imp_amp: float = 1e-4,
) -> np.ndarray:
    """
    Captures exact non-additive dB deltas across all `2^K` combinations of coupled
    front-end passive switches (`shape = (2**K, 64)`).
    """
    k = len(switch_pids)
    n_combos = 1 << k
    combos = np.zeros((n_combos, NUM_FREQS), dtype=np.float32)
    for mask in range(n_combos):
        for bit, pid in enumerate(switch_pids):
            backend.set_param(pid, 1.0 if ((mask >> bit) & 1) else 0.0)
        backend.settle(full=False)
        ir = capture_bipolar_ir(backend, ir_len=1024, onset=onset, imp_amp=imp_amp)
        combos[mask] = ir_to_bode64(ir, sr=backend.sr) - ref_bode_64
    return combos


def capture_tonestack_3d(
    backend: TargetBackend,
    pid_bass: int,
    pid_mid: int,
    pid_treble: int,
    onset: int = 0,
    imp_amp: float = 1e-4,
    grid_pts: list[float] | None = None,
) -> np.ndarray:
    """Captures a coupled `5x5x5` `(Bass x Middle x Treble)` Bode tensor (`shape = (5,5,5,64)`)."""
    pts = grid_pts or [0.0, 0.25, 0.5, 0.75, 1.0]
    n = len(pts)
    cube = np.zeros((n, n, n, NUM_FREQS), dtype=np.float32)
    for bi, b in enumerate(pts):
        backend.set_param(pid_bass, b)
        for mi, m in enumerate(pts):
            backend.set_param(pid_mid, m)
            for ti, t in enumerate(pts):
                backend.set_param(pid_treble, t)
                ir = capture_bipolar_ir(backend, ir_len=1024, onset=onset, imp_amp=imp_amp)
                cube[bi, mi, ti] = ir_to_bode64(ir, sr=backend.sr)
    return cube


def capture_knob_bode_21(
    backend: TargetBackend,
    pid: int,
    onset: int = 0,
    imp_amp: float = 1e-4,
    min_val: float = 0.01,
) -> np.ndarray:
    """Captures a 21-point `[0.0 .. 1.0]` Bode taper curve (`shape = (21, 64)`)."""
    table = np.zeros((21, NUM_FREQS), dtype=np.float32)
    for i in range(21):
        val = max(min_val, i * 0.05)
        backend.set_param(pid, val)
        ir = capture_bipolar_ir(backend, ir_len=1024, onset=onset, imp_amp=imp_amp)
        table[i] = ir_to_bode64(ir, sr=backend.sr)
    return table
