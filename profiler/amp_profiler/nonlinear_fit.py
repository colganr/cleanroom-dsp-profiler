"""
Nonlinear Tube Amplifier (180-Ch Bloom + Sag + 96 kHz KONE) & Overdrive Pedal (126-Ch 3-Pole)
Feature Evaluators and Scale-Penalized Ridge Cholesky Calibration Solvers.

Uses the compiled Pure-Rust FFI bridge (`rust_bridge.py`) automatically when available for
50-100x speedup and guaranteed bit-exact parity, while providing a bit-identical Pure-Python
NumPy reference implementation so the profiler works out of the box even without `rustc`.
"""

from __future__ import annotations

import math
import numpy as np
from .reservoir import AMP_RESERVOIR_DEFAULT, PEDAL_RESERVOIR_DEFAULT
from .rust_bridge import AMP_CHANNELS, PEDAL_CHANNELS, get_rust_bridge

SR_DEFAULT = 48_000
PEDAL_SCALES = np.array([0.25, 0.55, 1.20, 2.60, 5.50, 11.50, 24.00], dtype=np.float32)
AMP_GAIN_BRACKETS_5 = [0.15, 0.35, 0.55, 0.75, 1.00]
PEDAL_GAIN_BRACKETS_9 = [0.00, 0.15, 0.30, 0.45, 0.55, 0.65, 0.75, 0.85, 1.00]


def fast_tanh_f32(x: np.ndarray | np.float32) -> np.ndarray | np.float32:
    """Bit-identical `[7/6]` Padé `fast_tanh` matching `dsp-core-rust/src/simd_ops.rs`."""
    xc = np.clip(np.asarray(x, dtype=np.float32), np.float32(-4.90), np.float32(4.90))
    x2 = xc * xc
    num = xc * (np.float32(135_135.0) + x2 * (np.float32(17_325.0) + x2 * (np.float32(378.0) + x2)))
    den = np.float32(135_135.0) + x2 * (
        np.float32(62_370.0) + x2 * (np.float32(3_150.0) + np.float32(28.0) * x2)
    )
    out = np.clip(num / den, np.float32(-1.0), np.float32(1.0))
    return out.astype(np.float32)


def soft_sat_vec(x: np.ndarray, cp: float, cn: float, knee: float = 2.9) -> np.ndarray:
    x32 = np.asarray(x, dtype=np.float32)
    c = np.where(x32 >= 0.0, np.float32(cp), np.float32(cn))
    ax = np.abs(x32)
    r = ax / c
    inv_k = np.float32(1.0 / knee)
    k32 = np.float32(knee)
    out = np.where(
        r <= np.float32(0.004),
        x32,
        np.where(
            r >= np.float32(24.0),
            np.copysign(c, x32),
            x32 / np.power(np.float32(1.0) + np.power(np.maximum(r, np.float32(1e-12)), k32), inv_k),
        ),
    )
    return out.astype(np.float32)


def sat_bloom_bounded_vec(
    x: np.ndarray,
    cp0: float = 0.212,
    cn0: float = 0.227,
    knee: float = 2.9,
    bloom_alpha: float = 0.18,
    bloom_sigma1: float = 0.16,
    bloom_sigma2: float = 0.32,
) -> np.ndarray:
    x32 = np.asarray(x, dtype=np.float32)
    tb = fast_tanh_f32(x32 / np.float32(bloom_sigma1))
    tc = fast_tanh_f32(x32 / np.float32(bloom_sigma2))
    sech2 = np.float32(1.0) - tc * tc
    v_bloom = x32 + np.float32(bloom_alpha) * x32 * (tb * tb) * sech2
    return soft_sat_vec(v_bloom, cp0, cn0, knee)


def _one_pole_iir_f32(x: np.ndarray, a: float) -> np.ndarray:
    a32 = np.float32(a)
    b32 = np.float32(1.0 - a)
    out = np.empty(len(x), dtype=np.float32)
    st = np.float32(0.0)
    for n in range(len(x)):
        st = np.float32(a32 * st + b32 * x[n])
        out[n] = st
    return out


def apply_post_inv_shelf(v_tot: np.ndarray, p_pole: float, p_zero: float) -> np.ndarray:
    """Applies the exact algebraic inverse of the pre-emphasis low-shelf."""
    out = np.empty(len(v_tot), dtype=np.float32)
    sp = np.float32(0.0)
    yp = np.float32(0.0)
    pp = np.float32(p_pole)
    pz = np.float32(p_zero)
    for n in range(len(v_tot)):
        sn = np.float32(v_tot[n])
        yn = np.float32(sn - pz * sp + pp * yp)
        sp = sn
        yp = yn
        out[n] = yn
    return out


def unapply_post_inv_shelf(y_target: np.ndarray, p_pole: float, p_zero: float) -> np.ndarray:
    """Inverts the post-shelf onto target audio `y_target` to align with pre-shelf features."""
    out = np.empty(len(y_target), dtype=np.float64)
    yp = 0.0
    sp = 0.0
    pp = float(p_pole)
    pz = float(p_zero)
    yt = np.asarray(y_target, dtype=np.float64)
    for n in range(len(yt)):
        yn = float(yt[n])
        sn = yn - pp * yp + pz * sp
        yp = yn
        sp = sn
        out[n] = sn
    return out


def eval_amp_features_py(
    u_in: np.ndarray,
    sr: float = SR_DEFAULT,
    cp0: float = 0.212,
    cn0: float = 0.227,
    knee: float = 2.9,
    bloom_alpha: float = 0.18,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Pure-Python bit-exact reference of `dsp-core-rust/src/amp_core.rs::eval_amp_features_sample`."""
    u = np.asarray(u_in, dtype=np.float32)
    n = len(u)

    if abs(sr - 48_000.0) < 0.5:
        p_pole, p_zero = 0.9889349, 0.9852012
        a35, a180, a383, a720 = 0.9954289, 0.9767144, 0.9511003, 0.9100584
        a1200, a2400 = 0.854636, 0.7303989
        a_cath, a_sag, a_grid, a_dc28 = 0.9876426, 0.9984304, 0.9979078, 0.9963415
    else:
        k = -2.0 * math.pi / max(sr, 8000.0)
        p_pole, p_zero = math.exp(k * 85.0), math.exp(k * 85.0 * 1.34)
        a35, a180, a383, a720 = math.exp(k * 35.0), math.exp(k * 180.0), math.exp(k * 383.0), math.exp(k * 720.0)
        a1200, a2400 = math.exp(k * 1200.0), math.exp(k * 2400.0)
        a_cath, a_sag, a_grid, a_dc28 = math.exp(k * 95.0), math.exp(k * 12.0), math.exp(k * 16.0), math.exp(k * 28.0)

    x_pre = np.empty(n, dtype=np.float32)
    u_prev = np.float32(0.0)
    xp_prev = np.float32(0.0)
    pp32, pz32 = np.float32(p_pole), np.float32(p_zero)
    for i in range(n):
        un = u[i]
        xp = np.float32(un - pp32 * u_prev + pz32 * xp_prev)
        u_prev = un
        xp_prev = xp
        x_pre[i] = xp

    x_lp180 = _one_pole_iir_f32(x_pre, a180)
    x_lp720 = _one_pole_iir_f32(x_pre, a720)
    x_lp2400 = _one_pole_iir_f32(x_pre, a2400)

    abs_x = np.abs(x_pre)
    cath_env = _one_pole_iir_f32(abs_x, a_cath)
    sag_env = _one_pole_iir_f32(abs_x, a_sag)
    grid_env = _one_pole_iir_f32(x_pre, a_grid)

    g_cathode = fast_tanh_f32(np.float32(0.35) * cath_env)
    g_sag = fast_tanh_f32(np.float32(0.35) * sag_env)
    bias_grid = fast_tanh_f32(np.float32(0.35) * grid_env)

    sat0 = sat_bloom_bounded_vec(x_pre, cp0, cn0, knee, bloom_alpha)
    s_lp180_0 = sat_bloom_bounded_vec(x_lp180, cp0, cn0, knee, bloom_alpha)
    s_lp720_0 = sat_bloom_bounded_vec(x_lp720, cp0, cn0, knee, bloom_alpha)
    s_lp2400_0 = sat_bloom_bounded_vec(x_lp2400, cp0, cn0, knee, bloom_alpha)

    x_rail = soft_sat_vec(x_pre, 0.95, 0.95, 2.8)
    ex1_lo = (x_rail - soft_sat_vec(x_pre, 0.18, 0.15, 2.8)).astype(np.float32)
    ex1_mid = (x_rail - soft_sat_vec(x_pre, 0.36, 0.30, 2.8)).astype(np.float32)

    lp180_ex1_lo = _one_pole_iir_f32(ex1_lo, a180)
    lp383_ex1_lo = _one_pole_iir_f32(ex1_lo, a383)
    lp180_ex1_mid = _one_pole_iir_f32(ex1_mid, a180)
    lp35_ex1_lo = _one_pole_iir_f32(ex1_lo, a35)
    lp383_ex1_mid = _one_pole_iir_f32(ex1_mid, a383)

    v2_inter_a = (x_rail - np.float32(0.85) * lp180_ex1_lo).astype(np.float32)
    v2_inter_b = (x_rail - np.float32(0.85) * lp383_ex1_lo).astype(np.float32)
    v2_inter_c = (x_rail - np.float32(0.85) * lp180_ex1_mid).astype(np.float32)

    r_asym_raw = (soft_sat_vec(x_pre, cp0 * 0.80, cn0 * 1.25, knee) - sat0).astype(np.float32)
    r_asym_dc = (r_asym_raw - _one_pole_iir_f32(r_asym_raw, a_dc28)).astype(np.float32)

    raw_r = [
        soft_sat_vec(x_pre, cp0 * 0.55, cn0 * 0.55, knee) - sat0,
        soft_sat_vec(x_pre, cp0 * 1.45, cn0 * 1.45, knee) - sat0,
        r_asym_dc,
        soft_sat_vec(x_pre, cp0 * 2.80, cn0 * 2.80, knee) - sat0,
        soft_sat_vec(x_pre, cp0 * 8.00, cn0 * 8.00, knee) - sat0,
        x_rail - sat0,
        soft_sat_vec(v2_inter_a, cp0, cn0, 3.2) - sat0,
        soft_sat_vec(v2_inter_b, cp0, cn0, 3.2) - sat0,
        soft_sat_vec(v2_inter_c, cp0, cn0, 3.2) - sat0,
        soft_sat_vec(x_lp180, cp0 * 0.65, cn0 * 0.65, knee) - s_lp180_0,
        soft_sat_vec(x_lp180, cp0 * 2.20, cn0 * 2.20, knee) - s_lp180_0,
        soft_sat_vec(x_lp720, cp0 * 0.65, cn0 * 0.65, knee) - s_lp720_0,
        soft_sat_vec(x_lp720, cp0 * 2.20, cn0 * 2.20, knee) - s_lp720_0,
        soft_sat_vec(x_lp2400, cp0 * 0.65, cn0 * 0.65, knee) - s_lp2400_0,
    ]

    phi = np.zeros((n, AMP_CHANNELS), dtype=np.float32)
    delays = (0, 1, 2, 4, 8, 16)
    for b in range(14):
        if b <= 5:
            s_lp = raw_r[b].astype(np.float32)
        elif b in (6, 7, 8, 13):
            s_lp = _one_pole_iir_f32(raw_r[b].astype(np.float32), a2400)
        else:
            s_lp = _one_pole_iir_f32(raw_r[b].astype(np.float32), a1200)
        base = b * 9
        for d_idx, d in enumerate(delays):
            if d == 0:
                phi[:, base + d_idx] = s_lp
            elif d < n:
                phi[d:, base + d_idx] = s_lp[:-d]
        phi[:, base + 6] = g_cathode * s_lp
        phi[:, base + 7] = g_sag * s_lp
        phi[:, base + 8] = bias_grid * s_lp

    # 96 kHz 2x oversampled clean-room KONE state-space (48 ch: 126..174)
    res = AMP_RESERVOIR_DEFAULT
    x_kone = soft_sat_vec(x_pre, 0.92, 0.92, 3.5)
    xk_lp720 = _one_pole_iir_f32(x_kone, a720)
    xk_lp2400 = _one_pole_iir_f32(x_kone, a2400)

    h1 = np.zeros(8, dtype=np.float32)
    h2 = np.zeros(8, dtype=np.float32)
    rx_prev = np.zeros(3, dtype=np.float32)
    rx_prev2 = np.zeros(3, dtype=np.float32)
    ry_prev = np.zeros(16, dtype=np.float32)
    rk_dc_lp = np.zeros(16, dtype=np.float32)
    rk_hist = np.zeros((n, 16), dtype=np.float32)
    b_dc28 = np.float32(1.0 - a_dc28)
    rot_idx = np.array([7, 0, 1, 2, 3, 4, 5, 6], dtype=np.int32)

    for i in range(n):
        v_cur = np.array([x_kone[i], xk_lp720[i], xk_lp2400[i]], dtype=np.float32)
        v_mid = np.float32(0.5625) * (rx_prev + v_cur) - np.float32(0.0625) * (rx_prev2 + v_cur)
        rx_prev2 = rx_prev
        rx_prev = v_cur

        gs = g_sag[i]
        bg = bias_grid[i]
        c1 = res.c1_ref * (np.float32(1.0) - np.float32(0.08) * gs)
        b1 = res.s1_bias + np.float32(0.03) * bg
        bt1 = fast_tanh_f32(b1)
        d1 = np.float32(1.0) - bt1 * bt1
        c2 = res.c2_ref * (np.float32(1.0) - np.float32(0.12) * gs)
        b2 = res.s2_bias - np.float32(0.04) * bg
        bt2 = fast_tanh_f32(b2)
        d2 = np.float32(1.0) - bt2 * bt2

        def _sub(v_sub: np.ndarray) -> np.ndarray:
            nonlocal h1, h2
            u1 = np.array([v_sub[0], v_sub[0], v_sub[0], v_sub[0], v_sub[0], v_sub[0], v_sub[1], v_sub[2]], dtype=np.float32)
            wx1 = c1 * fast_tanh_f32((res.s1_in * u1) / c1)
            uc1 = res.s1_diag * h1 + res.s1_cross * h1[rot_idx] + wx1
            h1_n = fast_tanh_f32(uc1 + b1) - bt1
            nl1 = h1_n - d1 * (c1 * fast_tanh_f32(uc1 / c1))
            h1 = h1_n

            s1_mix = np.float32(np.dot(res.s1_wout, nl1))
            v1_out = np.float32(v_sub[0] + s1_mix)
            ex_g = max(0.0, float(v1_out) - 0.35)
            v0_s2 = np.float32(float(v1_out) - 0.12 * ex_g * ex_g)
            u2 = np.array([v0_s2, v0_s2, v0_s2, v0_s2, v0_s2, v0_s2, np.float32(v_sub[1] + 0.5 * s1_mix), v_sub[2]], dtype=np.float32)
            wx2 = c2 * fast_tanh_f32((res.s2_in * u2) / c2)
            uc2 = res.s2_diag * h2 + res.s2_cross * h2[rot_idx] + wx2
            h2_n = fast_tanh_f32(uc2 + b2) - bt2
            nl2 = h2_n - d2 * (c2 * fast_tanh_f32(uc2 / c2))
            h2 = h2_n
            return np.concatenate([nl1, nl2])

        s_mid = _sub(v_mid)
        s_cur = _sub(v_cur)
        rec_k = np.float32(0.22) * ry_prev + np.float32(0.56) * s_cur + np.float32(0.22) * s_mid
        ry_prev = s_mid
        rk_dc_lp = rk_dc_lp + b_dc28 * (rec_k - rk_dc_lp)
        rk_hist[i] = rec_k - rk_dc_lp

    for k in range(16):
        base = 126 + k * 3
        r0 = rk_hist[:, k]
        phi[:, base] = r0
        if n > 2:
            phi[2:, base + 1] = r0[:-2]
        phi[:, base + 2] = g_sag * r0

    phi[:, 174] = lp35_ex1_lo
    phi[:, 175] = lp383_ex1_lo
    phi[:, 176] = lp383_ex1_mid
    phi[:, 177] = g_sag * sat0
    phi[:, 178] = g_cathode * sat0
    phi[:, 179] = bias_grid * sat0

    return phi, sat0, float(p_pole), float(p_zero)


def eval_pedal_features_py(
    x_in: np.ndarray,
    u_pre_raw: np.ndarray,
    subtract_dry: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Pure-Python bit-exact reference of `dsp-core-rust/src/pedal_core.rs::eval_pedal_features_sample`."""
    x32 = np.asarray(x_in, dtype=np.float32)
    u_raw32 = np.asarray(u_pre_raw, dtype=np.float32)
    n = min(len(x32), len(u_raw32))

    u_pre = np.float32(2.4) * fast_tanh_f32(u_raw32[:n] / np.float32(2.4))
    x_ref = (np.float32(2.4) * fast_tanh_f32(x32[:n] / np.float32(2.4))) if subtract_dry else np.zeros(n, dtype=np.float32)
    u_fb = (u_pre - x_ref).astype(np.float32)
    abs_fb = np.abs(u_fb)

    env_fast = _one_pole_iir_f32(abs_fb, 1.0 - 0.014296005)
    env_mid = _one_pole_iir_f32(abs_fb, 1.0 - 0.0018309027)
    bias_rc_raw = _one_pole_iir_f32(u_pre, 1.0 - 0.0023533995)

    g_fast = fast_tanh_f32(np.float32(0.35) * env_fast)
    g_mid = fast_tanh_f32(np.float32(0.35) * env_mid)
    bias_rc = fast_tanh_f32(np.float32(0.35) * bias_rc_raw)

    u_lp180 = _one_pole_iir_f32(u_fb, 1.0 - 0.023285626)
    u_lp720 = _one_pole_iir_f32(u_fb, 1.0 - 0.08994162)
    u_lp2400 = _one_pole_iir_f32(u_fb, 1.0 - 0.2696011)

    res = PEDAL_RESERVOIR_DEFAULT
    h = np.zeros(8, dtype=np.float32)
    rx_prev = np.zeros(3, dtype=np.float32)
    rx_prev2 = np.zeros(3, dtype=np.float32)
    ry_prev = np.zeros(8, dtype=np.float32)
    rk_dc_lp = np.zeros(8, dtype=np.float32)
    rk_blocked = np.zeros((n, 8), dtype=np.float32)
    a_dc = np.float32(0.003658489)
    rot_idx = np.array([7, 0, 1, 2, 3, 4, 5, 6], dtype=np.int32)

    for i in range(n):
        v_cur = np.array([u_fb[i], u_lp720[i], u_lp2400[i]], dtype=np.float32)
        v_mid = np.float32(0.5625) * (rx_prev + v_cur) - np.float32(0.0625) * (rx_prev2 + v_cur)
        rx_prev2 = rx_prev
        rx_prev = v_cur
        gm = g_mid[i]
        br = bias_rc[i]
        c = res.c_ref * (np.float32(1.0) - np.float32(0.08) * gm)
        b = res.bias + np.float32(0.03) * br
        bt = fast_tanh_f32(b)
        d = np.float32(1.0) - bt * bt

        def _step(v_sub: np.ndarray) -> np.ndarray:
            nonlocal h
            u_vec = np.array([v_sub[0], v_sub[0], v_sub[0], v_sub[0], v_sub[0], v_sub[0], v_sub[1], v_sub[2]], dtype=np.float32)
            wx = c * fast_tanh_f32((res.in_w * u_vec) / c)
            uc = res.diag * h + res.cross * h[rot_idx] + wx
            hn = fast_tanh_f32(uc + b) - bt
            out_s = hn - d * (c * fast_tanh_f32(uc / c))
            h = hn
            return out_s

        s_mid = _step(v_mid)
        s_cur = _step(v_cur)
        rec = np.float32(0.22) * ry_prev + np.float32(0.56) * s_cur + np.float32(0.22) * s_mid
        ry_prev = s_mid
        rk_dc_lp = rk_dc_lp + a_dc * (rec - rk_dc_lp)
        rk_blocked[i] = rec - rk_dc_lp

    b0 = np.float32(0.25)
    tb0 = fast_tanh_f32(b0)
    inv_d0 = np.float32(1.0) / (np.float32(1.0) - tb0 * tb0)

    psi = np.zeros((n, PEDAL_CHANNELS), dtype=np.float32)
    col = 0
    for s_idx, s in enumerate(PEDAL_SCALES):
        inv_s = np.float32(1.0) / s
        su_fb = s * u_fb
        su_pre = s * u_pre
        p1 = (fast_tanh_f32(su_fb) * inv_s - u_fb).astype(np.float32)
        p2 = (u_fb / np.sqrt(np.float32(1.0) + su_fb * su_fb) - u_fb).astype(np.float32)
        p3_raw = ((fast_tanh_f32(su_pre + b0) - tb0) * (inv_s * inv_d0) - u_pre).astype(np.float32)
        p3 = (p3_raw - _one_pole_iir_f32(p3_raw, 1.0 - 0.003658489)).astype(np.float32)
        rk = rk_blocked[:, s_idx]

        for d in (0, 1, 2, 4, 8, 16):
            if d == 0:
                psi[:, col] = p1
            elif d < n:
                psi[d:, col] = p1[:-d]
            col += 1
        for d in (0, 2, 8):
            if d == 0:
                psi[:, col] = rk
            elif d < n:
                psi[d:, col] = rk[:-d]
            col += 1
        psi[:, col] = p2
        col += 1
        for d in (0, 2):
            if d == 0:
                psi[:, col] = p3
            elif d < n:
                psi[d:, col] = p3[:-d]
            col += 1
        psi[:, col] = g_fast * p1
        psi[:, col + 1] = g_mid * rk
        psi[:, col + 2] = bias_rc * p1
        psi[:, col + 3] = fast_tanh_f32(s * u_lp180) * inv_s - u_lp180
        psi[:, col + 4] = fast_tanh_f32(s * u_lp720) * inv_s - u_lp720
        psi[:, col + 5] = fast_tanh_f32(s * u_lp2400) * inv_s - u_lp2400
        col += 6

    return psi, u_pre


def eval_amp_features(
    u_in: np.ndarray,
    sr: float = SR_DEFAULT,
    cp0: float = 0.212,
    cn0: float = 0.227,
    knee: float = 2.9,
    bloom_alpha: float = 0.18,
    prefer_rust: bool = True,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Evaluates 180-ch amp features via Rust FFI (fast) or Pure-Python fallback."""
    if prefer_rust:
        bridge = get_rust_bridge(allow_missing=True)
        if bridge is not None:
            return bridge.eval_amp_features(u_in, sr=sr, cp0=cp0, cn0=cn0, knee=knee, bloom_alpha=bloom_alpha)
    return eval_amp_features_py(u_in, sr=sr, cp0=cp0, cn0=cn0, knee=knee, bloom_alpha=bloom_alpha)


def eval_pedal_features(
    x_in: np.ndarray,
    u_pre: np.ndarray,
    subtract_dry: bool = False,
    prefer_rust: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluates 126-ch pedal features via Rust FFI (fast) or Pure-Python fallback."""
    if prefer_rust:
        bridge = get_rust_bridge(allow_missing=True)
        if bridge is not None:
            return bridge.eval_pedal_features(x_in, u_pre, subtract_dry=subtract_dry)
    return eval_pedal_features_py(x_in, u_pre, subtract_dry=subtract_dry)


def solve_scale_penalized_ridge(
    X: np.ndarray,
    y_residual: np.ndarray,
    sample_weights: np.ndarray | None = None,
    rel_reg: float = 2.5e-4,
    abs_reg: float = 2e-6,
    col_penalty_mult: np.ndarray | None = None,
) -> np.ndarray:
    """
    Solves the scale-penalized ridge regression system `(X^T W X + diag(lambda)) w = X^T W y`.

    Applying both mean-diagonal and per-column diagonal regularization (plus optional explicit
    scale multipliers) prevents collinear high-threshold channels from taking huge cancelling
    weights and collapsing linear scale on held-out audio.
    """
    X64 = np.asarray(X, dtype=np.float64)
    y64 = np.asarray(y_residual, dtype=np.float64)
    if sample_weights is not None:
        sw = np.asarray(sample_weights, dtype=np.float64)
        X64 = X64 * sw[:, None]
        y64 = y64 * sw

    XtX = X64.T @ X64
    Xty = X64.T @ y64
    diag = np.diag(XtX).copy()
    mean_d = max(float(np.mean(diag)), 1e-8)

    reg_vec = rel_reg * mean_d + 0.5 * rel_reg * diag + abs_reg
    if col_penalty_mult is not None:
        reg_vec = reg_vec * np.asarray(col_penalty_mult, dtype=np.float64)

    A = XtX + np.diag(reg_vec)
    return np.linalg.solve(A, Xty).astype(np.float32)


def _convolve_matrix_with_ir(mat: np.ndarray, h_post: np.ndarray) -> np.ndarray:
    """FFT-convolves every column of `mat (N, C)` with `h_post` simultaneously."""
    n = mat.shape[0]
    nfft = 1 << (n + len(h_post) - 1).bit_length()
    H = np.fft.rfft(h_post.astype(np.float64), n=nfft)
    M_F = np.fft.rfft(mat.astype(np.float64), n=nfft, axis=0)
    return np.fft.irfft(M_F * H[:, None], n=nfft, axis=0)[:n]


def fit_amp_180_weights(
    u_pre_cal: np.ndarray,
    y_target_cal: np.ndarray,
    h_post: np.ndarray | None = None,
    sr: float = SR_DEFAULT,
    skip_samples: int = 4096,
    cp0: float = 0.212,
    cn0: float = 0.227,
    knee: float = 2.9,
    bloom_alpha: float = 0.18,
    rel_reg: float = 2.5e-4,
    prefer_rust: bool = True,
) -> np.ndarray:
    """
    Fits the 180-channel Bloom + Sag + 96 kHz KONE readout weights on calibration audio.

    If `h_post` is provided, convolves both `sat0` and all 180 post-shelf basis features
    through `h_post` before solving so the solver optimizes end-to-end Wiener-Hammerstein output!
    """
    n = min(len(u_pre_cal), len(y_target_cal))
    phi, sat0, pp, pz = eval_amp_features(
        u_pre_cal[:n], sr=sr, cp0=cp0, cn0=cn0, knee=knee, bloom_alpha=bloom_alpha, prefer_rust=prefer_rust
    )

    if h_post is not None and len(h_post) > 1:
        # Build effective post filter = (inv_shelf) * h_post via its impulse response
        imp_len = max(256, len(h_post) + 128)
        imp = np.zeros(imp_len, dtype=np.float32)
        imp[0] = 1.0
        h_inv_shelf = apply_post_inv_shelf(imp, pp, pz)
        h_eff = np.convolve(h_inv_shelf.astype(np.float64), h_post.astype(np.float64))[:imp_len]
        y_base = np.convolve(sat0.astype(np.float64), h_eff)[:n]
        X_eff = _convolve_matrix_with_ir(phi, h_eff)
        r_target = y_target_cal[:n].astype(np.float64) - y_base
    else:
        v_target = unapply_post_inv_shelf(y_target_cal[:n], pp, pz)
        X_eff = phi.astype(np.float64)
        r_target = v_target - sat0.astype(np.float64)

    # Extra penalty on large-threshold channels (branches 3, 4, 5) to prevent linear scale collapse
    col_mult = np.ones(AMP_CHANNELS, dtype=np.float64)
    for high_branch in (3, 4, 5):
        col_mult[high_branch * 9 : (high_branch + 1) * 9] = 3.0

    s0 = min(skip_samples, n // 4)
    return solve_scale_penalized_ridge(
        X_eff[s0:n], r_target[s0:n], rel_reg=rel_reg, abs_reg=2e-6, col_penalty_mult=col_mult
    )


def fit_pedal_126_weights(
    x_cal: np.ndarray,
    y_target_cal: np.ndarray,
    h_pre: np.ndarray,
    h_post: np.ndarray,
    level_gain: float = 1.0,
    subtract_dry: bool = False,
    skip_samples: int = 2048,
    rel_reg: float = 3e-4,
    prefer_rust: bool = True,
) -> np.ndarray:
    """Fits the 126-channel 3-pole Zero-Slope Overdrive weights through `h_post * level_gain`."""
    n = min(len(x_cal), len(y_target_cal))
    u_pre = np.convolve(x_cal[:n].astype(np.float64), h_pre.astype(np.float64))[:n].astype(np.float32)
    psi, u_bounded = eval_pedal_features(x_cal[:n], u_pre, subtract_dry=subtract_dry, prefer_rust=prefer_rust)

    # Account for output coupling capacitor (~12 Hz DC blocker) + h_post * level_gain
    h_scaled = h_post.astype(np.float64) * float(level_gain)
    y_base = np.convolve(u_bounded.astype(np.float64), h_scaled)[:n]
    X_conv = _convolve_matrix_with_ir(psi, h_scaled)
    res = y_target_cal[:n].astype(np.float64) - y_base

    # Scale-weighted penalty: s is inverse threshold (1/thresh), so small s = wide linear-like threshold!
    col_mult = np.ones(PEDAL_CHANNELS, dtype=np.float64)
    for s_idx, s in enumerate(PEDAL_SCALES):
        thresh = 1.0 / max(float(s), 1e-6)
        col_mult[s_idx * 18 : (s_idx + 1) * 18] = 1.0 + (thresh / 1.5) ** 2

    s0 = min(skip_samples, n // 4)
    return solve_scale_penalized_ridge(
        X_conv[s0:n], res[s0:n], rel_reg=rel_reg, abs_reg=5e-6, col_penalty_mult=col_mult
    )
