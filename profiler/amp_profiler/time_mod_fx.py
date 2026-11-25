"""
Clean-Room Profilers for Time-Based, Modulation & Dynamics FX:

  1. Spring / Plate / Hall / Room Reverb (`profile_reverb_svd`):
     - Low-rank SVD compression of 21-point Decay IR sweeps (`4` orthogonal basis vectors)
     - Causal Tikhonov-smoothed Toeplitz Wet Tone FIRs (`64` taps)
     - 21-point Dry/Wet Mix gain law
  2. Harmonic / Optical Tube Tremolo (`profile_harmonic_tremolo`):
     - 2 kHz carrier RMS-envelope FFT + parabolic peak interpolation for exact LFO Speed (Hz)
     - Non-overlapping impulse-train SVD dual crossover FIR decomposition (`psi_0`, `psi_1`)
     - Phase-locked nonlinear saturation calibration at `Intensity = 0.0`
  3. Analog BBD & Studio Digital Delay (`profile_delay_bbd` + `fit_bbd_30ch_saturation`):
     - Impulse echo timing law + exact hidden Extra Loop Delay offset (`R1->R2` vs `Dry->R1`)
     - Causal Toeplitz-smoothed Echo Hardness / BBD Clock-Aperture loop FIRs (`64` taps)
     - 30-channel zero-slope BBD bucket-rail saturation solver
  4. Envelope Filter / Auto-Wah (`profile_envelope_filter`):
     - Small-signal `1e-3` impulse resonant `Q` & DC gain sweep
     - 14-parameter Vactrol + TPT 2-pole SVF multi-corner coordinate/subspace optimization
       (Rust FFI accelerated when `libcleanroom_dsp` is compiled)
  5. Studio Compressor (`profile_compressor`):
     - Stepped-sine static transfer curve fitting (`threshold_db`, `ratio`, `knee_db`, `makeup_db`)
     - Step-envelope dynamic attack & release time-constant estimation (`attack_ms`, `release_ms`)
"""

from __future__ import annotations

import math
import numpy as np
from .backends.base import TargetBackend
from .nonlinear_fit import soft_sat_vec, solve_scale_penalized_ridge
from .rust_bridge import get_rust_bridge

SR_DEFAULT = 48_000


def _solve_causal_toeplitz_fir(
    ref_sig: np.ndarray,
    target_sig: np.ndarray,
    fir_len: int = 64,
    ridge: float = 2e-6,
    smooth: float = 2e-5,
    fade_len: int = 8,
) -> tuple[np.ndarray, float]:
    """
    Solves a causal `fir_len`-tap filter `h` minimizing `||conv(ref_sig, h) - target_sig||^2`
    with Tikhonov ridge + first-difference smoothness regularization.
    """
    n = min(len(ref_sig), len(target_sig))
    r = np.asarray(ref_sig[:n], dtype=np.float64)
    y = np.asarray(target_sig[:n], dtype=np.float64)

    X = np.zeros((n, fir_len), dtype=np.float64)
    for k in range(fir_len):
        X[k:, k] = r[: n - k]

    D = np.eye(fir_len) - np.diag(np.ones(fir_len - 1), 1)
    R = ridge * np.eye(fir_len) + smooth * (D.T @ D)
    h = np.linalg.solve(X.T @ X + R, X.T @ y)

    if fade_len > 1 and fir_len >= fade_len:
        h[-fade_len:] *= 0.5 * (1.0 + np.cos(np.pi * np.arange(fade_len) / (fade_len - 1)))

    pred = X @ h
    corr = float(np.dot(y, pred) / (np.linalg.norm(y) * np.linalg.norm(pred) + 1e-12)) * 100.0
    return h.astype(np.float32), corr


# =============================================================================
# 1. SPRING / PLATE / HALL / ROOM REVERB (SVD BASIS + TOEPLITZ TONE FIR)
# =============================================================================

def profile_reverb_svd(
    backend: TargetBackend,
    pid_decay: int,
    pid_tone: int | None = None,
    pid_mix: int | None = None,
    onset: int = 0,
    ir_len: int = 16_384,
    n_basis: int = 4,
    n_steps: int = 21,
    tone_fir_len: int = 64,
    imp_amp: float = 0.08,
) -> dict:
    """
    Profiles a reverb module into `n_basis` SVD impulse-response basis vectors,
    `n_steps` decay projection weights, optional causal Wet Tone FIRs, and Dry/Wet Mix laws.
    """
    pre_pad = 256
    total = pre_pad + onset + ir_len + 256
    imp = np.zeros(total, dtype=np.float32)
    imp[pre_pad] = imp_amp
    start = pre_pad + onset

    # 1. Mix law (dry vs wet across [0..1])
    dry_gain_table = np.ones(n_steps, dtype=np.float32)
    wet_gain_table = np.linspace(0.0, 1.0, n_steps, dtype=np.float32)
    if pid_mix is not None:
        backend.set_param(pid_decay, 0.5)
        if pid_tone is not None:
            backend.set_param(pid_tone, 1.0)
        # Measure 100% dry & 100% wet references
        backend.set_param(pid_mix, 0.0)
        backend.settle(full=False)
        y_dry = backend.render_mono(imp[: start + 2048]).astype(np.float64) / imp_amp
        dry_ref = max(abs(float(y_dry[start])), 1e-6)

        backend.set_param(pid_mix, 1.0)
        backend.settle(full=False)
        y_wet_full = backend.render_mono(imp[: start + 4096]).astype(np.float64) / imp_amp
        wet_tail_ref = y_wet_full[start + 32 : start + 4096]
        wet_norm = float(np.linalg.norm(wet_tail_ref)) + 1e-12

        for idx, m_val in enumerate(np.linspace(0.0, 1.0, n_steps)):
            backend.set_param(pid_mix, float(m_val))
            backend.settle(full=False)
            ym = backend.render_mono(imp[: start + 4096]).astype(np.float64) / imp_amp
            # Estimate wet gain from tail (zero dry overlap), dry gain from tap 0 residual
            tail_m = ym[start + 32 : start + 4096]
            w_g = float(np.dot(tail_m, wet_tail_ref) / (wet_norm * wet_norm))
            d_g = (float(ym[start]) - w_g * float(y_wet_full[start])) / dry_ref
            dry_gain_table[idx] = max(0.0, d_g)
            wet_gain_table[idx] = max(0.0, w_g)

    # 2. Decay sweep -> SVD compression into `n_basis` orthogonal IRs
    if pid_mix is not None:
        backend.set_param(pid_mix, 1.0)
    if pid_tone is not None:
        backend.set_param(pid_tone, 1.0)

    M_dec = np.zeros((ir_len, n_steps), dtype=np.float64)
    for idx, d_val in enumerate(np.linspace(0.0, 1.0, n_steps)):
        backend.set_param(pid_decay, float(d_val))
        backend.settle(full=False)
        y = backend.render_mono(imp).astype(np.float64) / imp_amp
        M_dec[:, idx] = y[start : start + ir_len]

    U, S, Vt = np.linalg.svd(M_dec, full_matrices=False)
    k_use = min(n_basis, len(S))
    basis = np.zeros((n_basis, ir_len), dtype=np.float32)
    weights = np.zeros((n_steps, n_basis), dtype=np.float32)
    basis[:k_use] = U[:, :k_use].T.astype(np.float32)
    weights[:, :k_use] = (S[:k_use, None] * Vt[:k_use, :]).T.astype(np.float32)

    M_rec = basis.T.astype(np.float64) @ weights.T.astype(np.float64)
    corrs = [
        float(np.dot(M_dec[:, i], M_rec[:, i]) / (np.linalg.norm(M_dec[:, i]) * np.linalg.norm(M_rec[:, i]) + 1e-12)) * 100.0
        for i in range(n_steps)
    ]

    # 3. Wet Tone FIR sweep (relative to Tone = 1.0)
    tone_firs = np.zeros((n_steps, tone_fir_len), dtype=np.float32)
    tone_firs[:, 0] = 1.0
    tone_corrs: list[float] = []
    if pid_tone is not None:
        backend.set_param(pid_decay, 0.65)
        backend.set_param(pid_tone, 1.0)
        backend.settle(full=False)
        eval_n = min(ir_len, 8192)
        y_t10 = backend.render_mono(imp[: start + eval_n]).astype(np.float64) / imp_amp
        ref_t10 = y_t10[start : start + eval_n]

        for idx, t_val in enumerate(np.linspace(0.0, 1.0, n_steps)):
            if idx == n_steps - 1:
                tone_corrs.append(100.0)
                continue
            backend.set_param(pid_tone, float(t_val))
            backend.settle(full=False)
            yt = backend.render_mono(imp[: start + eval_n]).astype(np.float64) / imp_amp
            h_t, c_t = _solve_causal_toeplitz_fir(ref_t10, yt[start : start + eval_n], fir_len=tone_fir_len)
            tone_firs[idx] = h_t
            tone_corrs.append(c_t)

    return {
        "basis": basis,
        "decay_weights": weights,
        "tone_firs": tone_firs,
        "dry_gain": dry_gain_table,
        "wet_gain": wet_gain_table,
        "svd_min_corr_pct": round(min(corrs), 4),
        "svd_mean_corr_pct": round(float(np.mean(corrs)), 4),
        "tone_min_corr_pct": round(min(tone_corrs), 4) if tone_corrs else 100.0,
    }


# =============================================================================
# 2. HARMONIC / OPTICAL TREMOLO (CARRIER DEMOD + IMPULSE-TRAIN SVD)
# =============================================================================

def profile_harmonic_tremolo(
    backend: TargetBackend,
    pid_speed: int,
    pid_intensity: int,
    onset: int = 0,
    fir_len: int = 128,
    n_steps: int = 11,
    sr: int = SR_DEFAULT,
) -> dict:
    """
    Profiles an LFO-modulated harmonic or optical tremolo:
      1. Sweeps `Speed` using a 2 kHz carrier -> RMS envelope FFT + parabolic peak interpolation.
      2. Sweeps `Intensity` with a non-overlapping impulse train (200 imp/sec) -> SVD extracts
         dual crossover basis FIRs (`psi_0`, `psi_1`) and modulation envelopes (`c0`, `c1`).
    """
    # 1. Carrier envelope demodulation for LFO speed law
    n_sp = int(3.0 * sr)
    t_sp = np.arange(n_sp, dtype=np.float64) / sr
    carrier = (0.05 * np.sin(2.0 * math.pi * 2000.0 * t_sp)).astype(np.float32)

    backend.set_param(pid_intensity, 0.85)
    speeds_x = np.linspace(0.0, 1.0, n_steps)
    freq_meas = np.zeros(n_steps, dtype=np.float64)
    rms_win = np.ones(48, dtype=np.float64) / 48.0

    for idx, sp in enumerate(speeds_x):
        backend.set_param(pid_speed, float(sp))
        backend.settle(full=False)
        y = backend.render_mono(carrier).astype(np.float64)[onset:]
        env = np.sqrt(np.convolve(y * y, rms_win, mode="valid"))
        seg = env[int(0.2 * sr) :]
        seg = seg - np.mean(seg)
        spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
        pk = int(np.argmax(spec[1:])) + 1
        if 1 <= pk < len(spec) - 1:
            y0 = math.log(float(spec[pk - 1]) + 1e-12)
            y1 = math.log(float(spec[pk]) + 1e-12)
            y2 = math.log(float(spec[pk + 1]) + 1e-12)
            denom = y0 - 2.0 * y1 + y2
            delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
        else:
            delta = 0.0
        freq_meas[idx] = max(0.05, float((pk + delta) * sr / len(seg)))

    # 2. Non-overlapping impulse-train SVD crossover capture across Intensity
    step = max(fir_len + 64, 240)
    n_train = int(2.0 * sr)
    imp_train = np.zeros(n_train, dtype=np.float32)
    imp_idx = np.arange(512, n_train - step - onset, step)
    imp_amp = 0.01
    imp_train[imp_idx] = imp_amp

    backend.set_param(pid_speed, 0.50)
    snaps_by_i: list[np.ndarray] = []
    all_snaps = []

    for inten in np.linspace(0.0, 1.0, n_steps):
        backend.set_param(pid_intensity, float(inten))
        backend.settle(full=False)
        y = backend.render_mono(imp_train).astype(np.float64) / imp_amp
        snaps = np.stack([y[i + onset : i + onset + fir_len] for i in imp_idx], axis=0)
        snaps_by_i.append(snaps)
        all_snaps.append(snaps)

    A = np.vstack(all_snaps)  # (total_snaps, fir_len)
    _, S, Vt = np.linalg.svd(A, full_matrices=False)
    energy_2 = float(np.sum(S[:2] ** 2) / max(float(np.sum(S ** 2)), 1e-12)) * 100.0
    psi_0 = Vt[0].astype(np.float32)
    psi_1 = Vt[1].astype(np.float32) if Vt.shape[0] > 1 else np.zeros(fir_len, dtype=np.float32)

    c0_mean = np.zeros(n_steps, dtype=np.float32)
    c0_amp = np.zeros(n_steps, dtype=np.float32)
    c1_mean = np.zeros(n_steps, dtype=np.float32)
    c1_amp = np.zeros(n_steps, dtype=np.float32)

    for idx, snaps in enumerate(snaps_by_i):
        proj0 = snaps @ psi_0.astype(np.float64)
        proj1 = snaps @ psi_1.astype(np.float64)
        c0_mean[idx] = float(np.mean(proj0))
        c0_amp[idx] = 0.5 * float(np.max(proj0) - np.min(proj0))
        c1_mean[idx] = float(np.mean(proj1))
        c1_amp[idx] = 0.5 * float(np.max(proj1) - np.min(proj1))

    return {
        "freq_hz": freq_meas.astype(np.float32),
        "psi_0": psi_0,
        "psi_1": psi_1,
        "c0_mean": c0_mean,
        "c0_amp": c0_amp,
        "c1_mean": c1_mean,
        "c1_amp": c1_amp,
        "svd_2basis_energy_pct": round(energy_2, 4),
    }


# =============================================================================
# 3. ANALOG BBD & STUDIO DIGITAL DELAY (TIMING, EXTRA LOOP OFFSET, TOEPLITZ, BBD SAT)
# =============================================================================

def profile_delay_bbd(
    backend: TargetBackend,
    pid_time: int,
    pid_feedback: int,
    pid_mix: int,
    pid_tone_hardness: int | None = None,
    onset: int = 0,
    fir_len: int = 64,
    n_steps: int = 11,
    sr: int = SR_DEFAULT,
) -> dict:
    """
    Profiles an analog BBD or digital delay:
      - Discovers exact Delay Time law (`delay_samples`) and hidden `loop_extra_delay` (`R1->R2` - `Dry->R1`)
      - Captures `64`-tap Base Wet reconstruction FIR and Toeplitz-deconvolved Loop/Hardness FIRs
      - Maps Feedback gain curve (`fb_gain`)
    """
    pre_pad = 256
    # 1. Measure dry amplitude at Mix = 0
    backend.set_param(pid_mix, 0.0)
    backend.set_param(pid_feedback, 0.0)
    backend.set_param(pid_time, 0.25)
    if pid_tone_hardness is not None:
        backend.set_param(pid_tone_hardness, 1.0)
    backend.settle(full=False)

    imp_short = np.zeros(4096, dtype=np.float32)
    imp_short[pre_pad] = 0.1
    y_dry = backend.render_mono(imp_short).astype(np.float64)
    dry_pk = pre_pad + onset
    dry_amp = float(y_dry[dry_pk]) / 0.1
    if abs(dry_amp) < 1e-6:
        dry_pk = int(np.argmax(np.abs(y_dry)))
        dry_amp = float(y_dry[dry_pk]) / 0.1

    # 2. Measure R1 and R2 echo timings to discover exact Delay Time + Extra Loop Delay Offset
    backend.set_param(pid_mix, 1.0)
    backend.set_param(pid_feedback, 0.55)
    backend.set_param(pid_time, 0.20)
    backend.settle(full=False)

    imp_long = np.zeros(int(1.5 * sr), dtype=np.float32)
    imp_long[pre_pad] = 0.05
    y_fb = backend.render_mono(imp_long).astype(np.float64) / (0.05 * dry_amp)

    search_r1 = y_fb[dry_pk + 64 :]
    r1_rel = int(np.argmax(np.abs(search_r1))) + 64
    r1_abs = dry_pk + r1_rel
    # Find causal start of R1 within 16 samples before peak (or peak-aligned tap)
    r1_pk_val = abs(float(y_fb[r1_abs]))
    search_r2_start = r1_abs + max(64, r1_rel // 2)
    search_r2 = y_fb[search_r2_start : min(len(y_fb), r1_abs + r1_rel + 256)]
    r2_abs = search_r2_start + int(np.argmax(np.abs(search_r2)))
    loop_extra_delay = int((r2_abs - r1_abs) - r1_rel)

    # 3. Sweep Delay Time law across `n_steps`
    backend.set_param(pid_feedback, 0.0)
    delay_samples = np.zeros(n_steps, dtype=np.float32)
    for idx, t_val in enumerate(np.linspace(0.0, 1.0, n_steps)):
        backend.set_param(pid_time, float(t_val))
        backend.settle(full=False)
        yt = backend.render_mono(imp_long).astype(np.float64)
        pk_t = int(np.argmax(np.abs(yt[dry_pk + 32 :]))) + 32
        delay_samples[idx] = float(pk_t)

    # 4. Capture Base Wet FIR at ref time = 0.20, FB = 0, Tone/Hardness = 1.0
    backend.set_param(pid_time, 0.20)
    backend.set_param(pid_feedback, 0.0)
    if pid_tone_hardness is not None:
        backend.set_param(pid_tone_hardness, 1.0)
    backend.settle(full=False)
    y_r1 = backend.render_mono(imp_long[: dry_pk + r1_rel + 1024]).astype(np.float64) / (0.05 * dry_amp)
    # Causal start of R1 (first tap > 0.1% of peak within 16 samples before peak)
    win_pre = y_r1[max(0, r1_abs - 16) : r1_abs + 1]
    lead_off = int(np.argmax(np.abs(win_pre) > r1_pk_val * 1e-3))
    r1_start = max(0, r1_abs - 16) + lead_off
    wet_base_fir = y_r1[r1_start : r1_start + fir_len].copy().astype(np.float32)
    wet_base_fir[-8:] *= 0.5 * (1.0 + np.cos(np.pi * np.arange(8) / 7.0))

    # 5. Capture Loop / Hardness FIRs via causal Toeplitz deconvolution
    loop_firs = np.zeros((n_steps, fir_len), dtype=np.float32)
    loop_firs[:, 0] = 1.0
    loop_corrs: list[float] = []
    if pid_tone_hardness is not None:
        ref_r1 = y_r1[r1_start : r1_start + 512]
        for idx, h_val in enumerate(np.linspace(0.0, 1.0, n_steps)):
            backend.set_param(pid_tone_hardness, float(h_val))
            backend.settle(full=False)
            yh = backend.render_mono(imp_long[: r1_start + 512]).astype(np.float64) / (0.05 * dry_amp)
            h_fir, c_fir = _solve_causal_toeplitz_fir(ref_r1, yh[r1_start : r1_start + 512], fir_len=fir_len)
            loop_firs[idx] = h_fir
            loop_corrs.append(c_fir)
        backend.set_param(pid_tone_hardness, 1.0)
    else:
        # Deconvolve R2 from R1 at FB = 0.55 to extract the feedback loop filter directly
        r1_win = y_fb[r1_start : r1_start + 512]
        r2_start = r1_start + r1_rel + loop_extra_delay
        r2_win = y_fb[r2_start : r2_start + 512]
        fb_norm = max(float(np.linalg.norm(r2_win) / (np.linalg.norm(r1_win) + 1e-12)), 1e-6)
        h_loop, c_loop = _solve_causal_toeplitz_fir(r1_win, r2_win / fb_norm, fir_len=fir_len)
        loop_firs[:] = h_loop
        loop_corrs.append(c_loop)

    # 6. Sweep Feedback Gain law (||R2|| / ||R1||)
    fb_gain = np.zeros(n_steps, dtype=np.float32)
    r2_start = r1_start + r1_rel + loop_extra_delay
    for idx, fb_v in enumerate(np.linspace(0.0, 1.0, n_steps)):
        backend.set_param(pid_feedback, float(fb_v))
        backend.settle(full=False)
        yf = backend.render_mono(imp_long[: r2_start + 512]).astype(np.float64) / (0.05 * dry_amp)
        n1 = float(np.linalg.norm(yf[r1_start : r1_start + 512])) + 1e-12
        n2 = float(np.linalg.norm(yf[r2_start : r2_start + 512]))
        fb_gain[idx] = float(n2 / n1)

    return {
        "dry_gain": float(dry_amp),
        "delay_samples": delay_samples,
        "loop_extra_delay": loop_extra_delay,
        "wet_base_fir": wet_base_fir,
        "loop_firs": loop_firs,
        "fb_gain": fb_gain,
        "loop_min_corr_pct": round(min(loop_corrs), 4) if loop_corrs else 100.0,
    }


def fit_bbd_30ch_saturation(
    u_in: np.ndarray,
    y_target: np.ndarray,
    cp0: float = 0.268,
    cn0: float = 0.268,
    knee: float = 2.2,
    skip_samples: int = 256,
) -> tuple[np.ndarray, float]:
    """
    Fits the 30-channel zero-slope BBD bucket-rail saturation core (`6` knee ratios x `5`
    causal dilations `[0, 1, 2, 4, 8]`) matching `dsp-core-rust/src/delay_bbd.rs`.
    """
    u = np.asarray(u_in, dtype=np.float32)
    y = np.asarray(y_target, dtype=np.float64)
    n = min(len(u), len(y))
    u, y = u[:n], y[:n]

    sat0 = soft_sat_vec(u, cp0, cn0, knee)
    ratios = [(0.55, 0.55), (0.85, 0.85), (1.35, 1.35), (0.75, 1.25), (2.20, 2.20), (4.00, 4.00)]
    delays = (0, 1, 2, 4, 8)
    cols = []
    for a, b in ratios:
        rb = (soft_sat_vec(u, cp0 * a, cn0 * b, knee) - sat0).astype(np.float64)
        for d in delays:
            if d == 0:
                cols.append(rb)
            else:
                rd = np.zeros_like(rb)
                rd[d:] = rb[:-d]
                cols.append(rd)
    X = np.column_stack(cols)
    res = y - sat0.astype(np.float64)
    s0 = min(skip_samples, n // 4)
    w = solve_scale_penalized_ridge(X[s0:], res[s0:], rel_reg=2e-4, abs_reg=1e-6)
    pred = sat0[s0:].astype(np.float64) + X[s0:] @ w
    corr = float(np.dot(y[s0:], pred) / (np.linalg.norm(y[s0:]) * np.linalg.norm(pred) + 1e-12)) * 100.0
    return w.astype(np.float32), round(corr, 4)


# =============================================================================
# 4. ENVELOPE FILTER / AUTO-WAH (SMALL-SIGNAL Q SWEEP + VACTROL/SVF OPTIMIZER)
# =============================================================================

def eval_envelope_filter_py(
    x_in: np.ndarray,
    params_14: np.ndarray,
    sr: float = SR_DEFAULT,
    drive_knob: float = 0.60,
    peak_q_scale: float = 1.0,
    slow_mult: float = 1.0,
    sweep_down: bool = False,
) -> np.ndarray:
    """Pure-Python reference matching `dsp-core-rust/src/envelope_filter.rs`."""
    p = [float(v) for v in params_14]
    f_hp, f_lp, sens_0, sens_exp, tau_atk, tau_rel, tau_opt, cv_pow, f_min, f_max, q_base, q_cv, g_base, bp_mix = p
    x = np.asarray(x_in, dtype=np.float64)
    n = len(x)
    out = np.zeros(n, dtype=np.float32)

    a_hp = math.exp(-2.0 * math.pi * f_hp / sr)
    a_lp = math.exp(-2.0 * math.pi * f_lp / sr)
    slow = max(slow_mult, 0.25)
    a_atk = math.exp(-1.0 / (max(tau_atk, 1e-4) * slow * sr))
    a_rel = math.exp(-1.0 / (max(tau_rel, 1e-3) * slow * sr))
    a_opt = math.exp(-1.0 / (max(tau_opt, 1e-4) * slow * sr))
    sens = sens_0 * ((0.15 + 0.85 * max(0.0, min(1.0, drive_knob))) ** sens_exp)
    cv_p = max(cv_pow, 0.25)
    mix_bp = max(0.0, min(1.0, bp_mix))

    sc_hp_x1 = sc_hp_y1 = sc_lp_y1 = env_rect = env_opt = ic1 = ic2 = 0.0
    for i in range(n):
        xi = float(x[i])
        sc_hp = a_hp * (sc_hp_y1 + xi - sc_hp_x1)
        sc_hp_x1 = xi
        sc_hp_y1 = sc_hp
        sc_lp_y1 = a_lp * sc_lp_y1 + (1.0 - a_lp) * sc_hp

        rect = abs(sc_lp_y1)
        c = a_atk if rect > env_rect else a_rel
        env_rect = c * env_rect + (1.0 - c) * rect
        env_opt = a_opt * env_opt + (1.0 - a_opt) * env_rect

        raw_cv = min(10.0, max(0.0, env_opt * sens))
        cv = (raw_cv / (1.0 + raw_cv)) ** cv_p
        sweep = (1.0 - cv) if sweep_down else cv
        fc = min(sr * 0.42, max(40.0, f_min + (f_max - f_min) * sweep))
        q = min(28.0, max(0.6, (q_base + q_cv * cv) * peak_q_scale))

        g = math.tan(math.pi * fc / sr)
        k = 1.0 / q
        a1 = 1.0 / (1.0 + g * (g + k))
        a2 = g * a1
        a3 = g * a2
        v3 = xi - ic2
        v1 = a1 * ic1 + a2 * v3
        v2 = ic2 + a2 * ic1 + a3 * v3
        ic1 = 2.0 * v1 - ic1
        ic2 = 2.0 * v2 - ic2
        out[i] = ((1.0 - mix_bp) * v2 + mix_bp * v1) * g_base
    return out


def profile_envelope_filter(
    backend: TargetBackend,
    pid_drive: int,
    pid_peak: int,
    cal_di: np.ndarray,
    onset: int = 0,
    n_q_steps: int = 11,
    n_rounds: int = 6,
    sweep_down: bool = False,
    seed: int = 42,
) -> dict:
    """
    Profiles a dynamic Vactrol + SVF Envelope Filter:
      1. Sweeps `Peak` with small-signal `1e-3` impulses to extract equivalent SVF Q & DC gain.
      2. Optimizes the 14 physical Vactrol+SVF parameters on disjoint dynamic DI across 3 corners
         (Rust FFI accelerated when compiled).
    """
    sr = backend.sr
    # 1. Small-signal Peak -> Q curve
    imp = np.zeros(4096 + onset + 256, dtype=np.float32)
    imp[256] = 1e-3
    backend.set_param(pid_drive, 0.5)
    peak_q = np.zeros(n_q_steps, dtype=np.float32)
    peak_g = np.zeros(n_q_steps, dtype=np.float32)
    for idx, p_val in enumerate(np.linspace(0.0, 1.0, n_q_steps)):
        backend.set_param(pid_peak, float(p_val))
        backend.settle(full=False)
        y = backend.render_mono(imp).astype(np.float64) / 1e-3
        ir = y[256 + onset : 256 + onset + 2048]
        spec = np.abs(np.fft.rfft(ir))
        pk_mag = float(np.max(spec[1:]))
        dc_mag = float(np.mean(spec[1:4])) + 1e-6
        peak_q[idx] = max(0.8, pk_mag / dc_mag)
        peak_g[idx] = dc_mag

    # 2. Capture 3 dynamic drive/peak corners on disjoint calibration DI
    x_cal = np.asarray(cal_di, dtype=np.float32)
    n = len(x_cal)
    corners = [(0.45, 0.35), (0.65, 0.65), (0.85, 0.85)]
    q_ref = float(np.interp(0.65, np.linspace(0.0, 1.0, n_q_steps), peak_q))
    targets = []
    for drv, pk in corners:
        backend.set_param(pid_drive, drv)
        backend.set_param(pid_peak, pk)
        backend.settle(full=False)
        yt = backend.render_mono(x_cal)
        yt_al = np.zeros(n, dtype=np.float64)
        yt_al[: n - onset] = yt[onset:]
        q_sc = float(np.interp(pk, np.linspace(0.0, 1.0, n_q_steps), peak_q)) / max(q_ref, 1e-6)
        targets.append((drv, q_sc, yt_al))

    bridge = get_rust_bridge(allow_missing=True)
    bounds = [
        (25.0, 220.0), (150.0, 1800.0), (6.0, 75.0), (1.0, 4.5),
        (0.001, 0.035), (0.04, 0.55), (0.001, 0.040), (0.8, 4.5),
        (100.0, 450.0), (600.0, 3200.0), (1.2, 12.0), (-0.5, 3.0),
        (0.4, 2.5), (0.0, 1.0),
    ]
    p_best = np.array(
        [65.0, 450.0, 22.0, 2.5, 0.008, 0.22, 0.014, 2.0, 200.0, 1600.0, 4.5, 0.5, 1.0, 0.25],
        dtype=np.float32,
    )

    def _eval(p_vec: np.ndarray, drv: float, q_sc: float) -> np.ndarray:
        if bridge is not None:
            return bridge.eval_envelope_filter(
                x_cal, p_vec, sr=sr, drive_knob=drv, peak_q_scale=q_sc, slow_mult=1.0, sweep_down=sweep_down
            )
        return eval_envelope_filter_py(
            x_cal, p_vec, sr=sr, drive_knob=drv, peak_q_scale=q_sc, slow_mult=1.0, sweep_down=sweep_down
        )

    s0 = min(1024, n // 4)
    def _loss(p_vec: np.ndarray) -> float:
        tot = 0.0
        for drv, q_sc, yt in targets:
            yp = _eval(p_vec, drv, q_sc).astype(np.float64)
            a, b = yt[s0 : n - onset], yp[s0 : n - onset]
            na = float(np.linalg.norm(a)) + 1e-12
            nb = float(np.linalg.norm(b)) + 1e-12
            corr = float(np.dot(a, b) / (na * nb))
            rel = float(np.linalg.norm(a - b) / na)
            tot += (1.0 - corr) + 0.35 * rel
        return tot / len(targets)

    best_loss = _loss(p_best)
    rng = np.random.default_rng(seed)
    steps = np.array([(hi - lo) * 0.20 for lo, hi in bounds], dtype=np.float32)
    rand_evals = 90 if bridge is not None else 20

    for _ in range(n_rounds):
        for dim in range(len(p_best)):
            lo, hi = bounds[dim]
            for sgn in (-1.0, 1.0, -0.4, 0.4):
                cand = p_best.copy()
                cand[dim] = float(np.clip(cand[dim] + sgn * steps[dim], lo, hi))
                lv = _loss(cand)
                if lv < best_loss:
                    best_loss = lv
                    p_best = cand
        for _ in range(rand_evals):
            cand = p_best.copy()
            mask = rng.random(len(p_best)) < 0.35
            if not np.any(mask):
                mask[rng.integers(0, len(p_best))] = True
            delta = rng.normal(0.0, 1.0, size=len(p_best)).astype(np.float32) * steps
            for d in np.where(mask)[0]:
                cand[d] = float(np.clip(cand[d] + delta[d], bounds[d][0], bounds[d][1]))
            lv = _loss(cand)
            if lv < best_loss:
                best_loss = lv
                p_best = cand
        steps *= 0.65

    # Compute mean fit correlation across the 3 corners
    corrs = []
    for drv, q_sc, yt in targets:
        yp = _eval(p_best, drv, q_sc).astype(np.float64)
        a, b = yt[s0 : n - onset], yp[s0 : n - onset]
        corrs.append(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)) * 100.0)

    return {
        "peak_q": peak_q,
        "peak_gain": peak_g,
        "params_14": p_best,
        "fit_mean_corr_pct": round(float(np.mean(corrs)), 3),
    }


# =============================================================================
# 5. STUDIO COMPRESSOR (STEPPED-SINE STATIC CURVE + STEP-ENVELOPE BALLISTICS)
# =============================================================================

def profile_compressor(
    backend: TargetBackend,
    onset: int = 0,
    sr: int = SR_DEFAULT,
) -> dict:
    """
    Profiles a static + dynamic compressor at its currently dialed settings:
      1. Stepped 1 kHz sine tones (-48 dBFS .. -2 dBFS) -> extracts `makeup_db`, `threshold_db`,
         `ratio`, and `knee_db`.
      2. Step-envelope burst (-36 dBFS -> -4 dBFS -> -36 dBFS) -> fits `attack_ms` & `release_ms`.
    """
    # 1. Static transfer curve via stepped 1 kHz sine bursts
    in_dbs = np.linspace(-48.0, -2.0, 24)
    out_dbs = np.zeros_like(in_dbs)
    dur_n = int(0.35 * sr)
    skip_n = int(0.22 * sr)  # Allow attack/release to settle completely
    t = np.arange(dur_n, dtype=np.float64) / sr
    unit_sin = np.sin(2.0 * math.pi * 1000.0 * t)

    for idx, db in enumerate(in_dbs):
        amp = 10.0 ** (db / 20.0)
        sig = (amp * unit_sin).astype(np.float32)
        backend.settle(full=False)
        y = backend.render_mono(sig).astype(np.float64)[onset + skip_n :]
        rms_in = amp / math.sqrt(2.0)
        rms_out = math.sqrt(float(np.mean(y * y)) + 1e-18)
        out_dbs[idx] = db + 20.0 * math.log10(rms_out / rms_in)

    # Low-level linear region (-48 .. -38 dBFS) gives exact makeup gain
    makeup_db = float(np.median(out_dbs[:5] - in_dbs[:5]))
    gr_dbs = (out_dbs - in_dbs) - makeup_db  # <= 0 dB

    # High-level compressed region (-12 .. -2 dBFS) gives exact asymptotic slope = 1/ratio - 1
    hi_in = in_dbs[-6:]
    hi_gr = gr_dbs[-6:]
    slope, intercept = np.polyfit(hi_in, hi_gr, 1)
    inv_ratio = max(0.02, min(1.0, 1.0 + float(slope)))
    ratio = 1.0 / inv_ratio
    threshold_db = float(-intercept / slope) if slope < -0.02 else 0.0

    # Estimate knee width from gain reduction at input == threshold_db:
    # For quadratic soft knee: GR(T) = (1/R - 1) * knee / 8  =>  knee = 8 * GR(T) / slope
    gr_at_t = float(np.interp(threshold_db, in_dbs, gr_dbs))
    knee_db = max(0.0, min(24.0, 8.0 * gr_at_t / min(slope, -1e-4))) if slope < -0.02 else 0.0

    # 2. Step-envelope burst for attack_ms & release_ms
    n_pre = int(0.25 * sr)
    n_hi = int(0.35 * sr)
    n_post = int(0.60 * sr)
    n_tot = n_pre + n_hi + n_post
    t_b = np.arange(n_tot, dtype=np.float64) / sr
    env_in = np.concatenate([
        np.full(n_pre, 10.0 ** (-36.0 / 20.0)),
        np.full(n_hi, 10.0 ** (-4.0 / 20.0)),
        np.full(n_post, 10.0 ** (-36.0 / 20.0)),
    ])
    burst = (env_in * np.sin(2.0 * math.pi * 1000.0 * t_b)).astype(np.float32)
    backend.settle(full=False)
    y_b = backend.render_mono(burst).astype(np.float64)[onset:]

    # Track 1 ms moving RMS gain curve in dB
    w_ms = max(1, int(0.001 * sr))
    ker = np.ones(w_ms, dtype=np.float64) / w_ms
    rms_y = np.sqrt(np.convolve(y_b * y_b, ker, mode="same") + 1e-18)
    rms_x = np.sqrt(np.convolve((burst[: len(y_b)].astype(np.float64)) ** 2, ker, mode="same") + 1e-18)
    gain_db = 20.0 * np.log10(rms_y / rms_x) - makeup_db

    # Attack: time from n_pre for gain_db to fall 63.2% (1 - 1/e) toward steady compressed gain
    g_steady = float(np.median(gain_db[n_pre + int(0.20 * sr) : n_pre + n_hi - w_ms]))
    atk_seg = gain_db[n_pre : n_pre + int(0.20 * sr)]
    atk_thresh = g_steady * 0.6321
    atk_hits = np.where(atk_seg <= atk_thresh)[0]
    attack_ms = float(atk_hits[0] * 1000.0 / sr) if len(atk_hits) > 0 else 10.0

    # Release: time from n_pre + n_hi for gain_db to recover 63.2% back toward 0 dB
    rel_start = n_pre + n_hi
    rel_seg = gain_db[rel_start : min(len(gain_db), rel_start + int(0.55 * sr))]
    rel_thresh = g_steady * (1.0 - 0.6321)
    rel_hits = np.where(rel_seg >= rel_thresh)[0]
    release_ms = float(rel_hits[0] * 1000.0 / sr) if len(rel_hits) > 0 else 100.0

    return {
        "threshold_db": round(threshold_db, 2),
        "ratio": round(ratio, 2),
        "knee_db": round(knee_db, 2),
        "makeup_db": round(makeup_db, 2),
        "attack_ms": round(max(0.2, attack_ms), 2),
        "release_ms": round(max(2.0, release_ms), 2),
        "in_dbs": in_dbs.astype(np.float32),
        "out_dbs": out_dbs.astype(np.float32),
    }
