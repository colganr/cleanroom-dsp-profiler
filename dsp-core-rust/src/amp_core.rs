//! Universal 180-Channel Tube Amplifier Core:
//!   - Bounded Zero-Slope 12AX7 Triode Transconductance "Bloom" (`sat_bloom_bounded`)
//!   - 3-Pole (180 Hz, 720 Hz, 2400 Hz) + Coupled V1 -> V2 Interstage Pre-Saturation (126 ch)
//!   - 96 kHz 2x Oversampled Coupled 2-Triode KONE State-Space Reservoir (48 ch)
//!   - 35 Hz & 383 Hz Sag-Recovery / Coupling-Cap Discharge & Envelope Compression (6 ch)
//!   - Optional 3-Stage Invertible Pole-Zero Coupled Tube + Output-Transformer Spine (`TubeSpine`)
//!
//! Every residual channel satisfies `r'(0) = 0` and `sat_0'(0) = 1.0`, guaranteeing exact
//! preservation of the Wiener-Hammerstein small-signal Bode response `H_pre(f) * H_post(f)`.

use crate::reservoir::{AmpReservoirSpec, AMP_RESERVOIR_DEFAULT};
use crate::simd_ops::{fast_powf_pos, fast_tanh};
use std::f32::consts::PI;

pub const AMP_CHANNELS: usize = 180;
pub const AMP_BRANCHES: usize = 14;
pub const AMP_RING_LEN: usize = 32;
pub const AMP_DEFAULT_BRACKETS: [f32; 5] = [0.15, 0.35, 0.55, 0.75, 1.00];

// Precomputed 48 kHz coefficients (zero exp() in hot sample loop)
const P_POLE_48K: f32 = 0.988_934_9_f32; // exp(-2*pi*85 / 48000)
const P_ZERO_48K: f32 = 0.985_201_2_f32; // exp(-2*pi*(85*1.34) / 48000)
const A35_48K: f32 = 0.995_428_9_f32;
const A180_48K: f32 = 0.976_714_4_f32;
const A383_48K: f32 = 0.951_100_3_f32;
const A720_48K: f32 = 0.910_058_4_f32;
const A1200_48K: f32 = 0.854_636_f32;
const A2400_48K: f32 = 0.730_398_9_f32;
const A_CATH_48K: f32 = 0.987_642_6_f32; // 95 Hz (~1.68 ms cathode bypass)
const A_SAG_48K: f32 = 0.998_430_4_f32; // 12 Hz (~13.3 ms B+ power-supply sag)
const A_GRID_48K: f32 = 0.997_907_8_f32; // 16 Hz (~9.95 ms grid conduction bias)
const A_DC28_48K: f32 = 0.996_341_5_f32; // 28 Hz coupling-cap DC blocker

/// Configurable nonlinear knee & bloom shape for [`AmpCore`].
#[derive(Clone, Copy, Debug)]
pub struct AmpCoreConfig {
    pub cp0: f32,
    pub cn0: f32,
    pub knee: f32,
    pub bloom_alpha: f32,
    pub bloom_sigma1: f32,
    pub bloom_sigma2: f32,
    pub rail_limit: f32,
}

impl Default for AmpCoreConfig {
    fn default() -> Self {
        Self {
            cp0: 0.212,
            cn0: 0.227,
            knee: 2.9,
            bloom_alpha: 0.18,
            bloom_sigma1: 0.16,
            bloom_sigma2: 0.32,
            rail_limit: 0.95,
        }
    }
}

/// Rational power soft saturator with exact unit slope at `x = 0` and clean `[-cn, +cp]` rails.
#[inline(always)]
pub fn soft_sat_sample(x: f32, cp: f32, cn: f32, knee: f32, inv_knee: f32) -> f32 {
    let (c, ax) = if x >= 0.0 { (cp, x) } else { (cn, -x) };
    let r = ax / c;
    if r <= 0.004_f32 {
        return x;
    }
    if r >= 24.0_f32 {
        return c.copysign(x);
    }
    x / fast_powf_pos(1.0_f32 + fast_powf_pos(r, knee), inv_knee)
}

/// Bounded Zero-Slope 12AX7 Triode Transconductance "Bloom" Base Saturator.
///
/// - At `x -> 0`: `d/dx [sat_bloom_bounded(0)] == 1.0` strictly (zero linear leak).
/// - At mid-levels (`|x| ~ 0.15..0.35`): smooth `+1.5 dB` cold-biased triode gain expansion.
/// - At extreme drive (`|x| >> bloom_sigma2`): `sech^2 -> 0`, cleanly bounded at `[-cn, +cp]`.
#[inline(always)]
pub fn sat_bloom_bounded_sample(x: f32, cfg: &AmpCoreConfig) -> f32 {
    let tb = fast_tanh(x / cfg.bloom_sigma1);
    let tc = fast_tanh(x / cfg.bloom_sigma2);
    let sech2 = 1.0_f32 - tc * tc;
    let v_bloom = x + cfg.bloom_alpha * x * (tb * tb) * sech2;
    soft_sat_sample(v_bloom, cfg.cp0, cfg.cn0, cfg.knee, 1.0_f32 / cfg.knee)
}

/// Real-time zero-allocation state for the Universal 180-Channel Tube Amplifier Core.
#[derive(Clone, Copy)]
pub struct AmpCoreState {
    pub u_prev: f32,
    pub x_pre_prev: f32,
    pub lp180_state: f32,
    pub lp720_state: f32,
    pub lp2400_state: f32,
    pub cathode_env: f32,
    pub sag_tube_env: f32,
    pub grid_bias_env: f32,
    pub lp180_ex1_lo_state: f32,
    pub lp383_ex1_lo_state: f32,
    pub lp180_ex1_mid_state: f32,
    pub lp35_ex1_lo_state: f32,
    pub lp383_ex1_mid_state: f32,
    pub r_asym_dc_state: f32,
    pub psi_lp_state: [f32; AMP_BRANCHES],
    pub psi_delay: [[f32; AMP_RING_LEN]; AMP_BRANCHES],
    pub psi_write_pos: usize,
    pub xk_lp720_state: f32,
    pub xk_lp2400_state: f32,
    pub kone_h1: [f32; 8],
    pub kone_h2: [f32; 8],
    pub resamp_x_prev: [f32; 3],
    pub resamp_x_prev2: [f32; 3],
    pub resamp_y_prev: [f32; 16],
    pub rk_dc_lp: [f32; 16],
    pub rk_delay: [[f32; AMP_RING_LEN]; 16],
    pub sat_prev: f32,
    pub v_post_prev: f32,
    pub reservoir: AmpReservoirSpec,
}

impl Default for AmpCoreState {
    fn default() -> Self {
        Self::new(AMP_RESERVOIR_DEFAULT)
    }
}

impl AmpCoreState {
    pub const fn new(reservoir: AmpReservoirSpec) -> Self {
        Self {
            u_prev: 0.0,
            x_pre_prev: 0.0,
            lp180_state: 0.0,
            lp720_state: 0.0,
            lp2400_state: 0.0,
            cathode_env: 0.0,
            sag_tube_env: 0.0,
            grid_bias_env: 0.0,
            lp180_ex1_lo_state: 0.0,
            lp383_ex1_lo_state: 0.0,
            lp180_ex1_mid_state: 0.0,
            lp35_ex1_lo_state: 0.0,
            lp383_ex1_mid_state: 0.0,
            r_asym_dc_state: 0.0,
            psi_lp_state: [0.0; AMP_BRANCHES],
            psi_delay: [[0.0; AMP_RING_LEN]; AMP_BRANCHES],
            psi_write_pos: 0,
            xk_lp720_state: 0.0,
            xk_lp2400_state: 0.0,
            kone_h1: [0.0; 8],
            kone_h2: [0.0; 8],
            resamp_x_prev: [0.0; 3],
            resamp_x_prev2: [0.0; 3],
            resamp_y_prev: [0.0; 16],
            rk_dc_lp: [0.0; 16],
            rk_delay: [[0.0; AMP_RING_LEN]; 16],
            sat_prev: 0.0,
            v_post_prev: 0.0,
            reservoir,
        }
    }

    pub fn reset(&mut self) {
        let res = self.reservoir;
        *self = Self::new(res);
    }
}

#[inline(always)]
fn step_kone_96k_pair(
    v_mid: &[f32; 3],
    v_cur: &[f32; 3],
    gs: f32,
    bg: f32,
    res: &AmpReservoirSpec,
    h1: &mut [f32; 8],
    h2: &mut [f32; 8],
) -> ([f32; 16], [f32; 16]) {
    let sag1 = 1.0 - 0.08 * gs;
    let sag2 = 1.0 - 0.12 * gs;
    let bg1 = 0.03 * bg;
    let bg2 = -0.04 * bg;

    let mut c1 = [0.0_f32; 8];
    let mut inv_c1 = [0.0_f32; 8];
    let mut b1 = [0.0_f32; 8];
    let mut bt1 = [0.0_f32; 8];
    let mut d1 = [0.0_f32; 8];
    let mut c2 = [0.0_f32; 8];
    let mut inv_c2 = [0.0_f32; 8];
    let mut b2 = [0.0_f32; 8];
    let mut bt2 = [0.0_f32; 8];
    let mut d2 = [0.0_f32; 8];

    for k in 0..8 {
        c1[k] = res.c1_ref[k] * sag1;
        inv_c1[k] = 1.0 / c1[k];
        b1[k] = res.s1_bias[k] + bg1;
        bt1[k] = fast_tanh(b1[k]);
        d1[k] = 1.0 - bt1[k] * bt1[k];

        c2[k] = res.c2_ref[k] * sag2;
        inv_c2[k] = 1.0 / c2[k];
        b2[k] = res.s2_bias[k] + bg2;
        bt2[k] = fast_tanh(b2[k]);
        d2[k] = 1.0 - bt2[k] * bt2[k];
    }

    let mut run_sub = |v_sub: &[f32; 3]| -> [f32; 16] {
        let u1 = [
            v_sub[0], v_sub[0], v_sub[0], v_sub[0], v_sub[0], v_sub[0], v_sub[1], v_sub[2],
        ];
        let h1_old = *h1;
        let mut s_out = [0.0_f32; 16];
        let mut s1_nl_mix = 0.0_f32;

        for k in 0..8 {
            let prev_rot = h1_old[(k + 7) & 7];
            let wx1 = c1[k] * fast_tanh((res.s1_in[k] * u1[k]) * inv_c1[k]);
            let u_cell = res.s1_diag[k] * h1_old[k] + res.s1_cross[k] * prev_rot + wx1;
            let hn = fast_tanh(u_cell + b1[k]) - bt1[k];
            let u_ref = c1[k] * fast_tanh(u_cell * inv_c1[k]);
            let nl1 = hn - d1[k] * u_ref;
            h1[k] = hn;
            s_out[k] = nl1;
            s1_nl_mix += res.s1_wout[k] * nl1;
        }

        // Interstage grid-conduction clamp feeding push-pull power tubes
        let v1_out = v_sub[0] + s1_nl_mix;
        let ex_g = (v1_out - 0.35_f32).max(0.0_f32);
        let v0_s2 = v1_out - 0.12_f32 * ex_g * ex_g;
        let u2 = [
            v0_s2,
            v0_s2,
            v0_s2,
            v0_s2,
            v0_s2,
            v0_s2,
            v_sub[1] + 0.5 * s1_nl_mix,
            v_sub[2],
        ];
        let h2_old = *h2;

        for k in 0..8 {
            let prev_rot = h2_old[(k + 7) & 7];
            let wx2 = c2[k] * fast_tanh((res.s2_in[k] * u2[k]) * inv_c2[k]);
            let u_cell = res.s2_diag[k] * h2_old[k] + res.s2_cross[k] * prev_rot + wx2;
            let hn = fast_tanh(u_cell + b2[k]) - bt2[k];
            let u_ref = c2[k] * fast_tanh(u_cell * inv_c2[k]);
            let nl2 = hn - d2[k] * u_ref;
            h2[k] = hn;
            s_out[8 + k] = nl2;
        }
        s_out
    };

    let s_mid = run_sub(v_mid);
    let s_cur = run_sub(v_cur);
    (s_mid, s_cur)
}

#[inline(always)]
fn sample_rate_coeffs(sr: f32) -> (f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32) {
    if (sr - 48_000.0).abs() < 0.5 {
        (
            P_POLE_48K, P_ZERO_48K, A35_48K, A180_48K, A383_48K, A720_48K, A1200_48K, A2400_48K,
            A_CATH_48K, A_SAG_48K, A_GRID_48K, A_DC28_48K,
        )
    } else {
        let k = -2.0 * PI / sr.max(8_000.0);
        (
            (k * 85.0).exp(),
            (k * (85.0 * 1.34)).exp(),
            (k * 35.0).exp(),
            (k * 180.0).exp(),
            (k * 383.0).exp(),
            (k * 720.0).exp(),
            (k * 1200.0).exp(),
            (k * 2400.0).exp(),
            (k * 95.0).exp(),
            (k * 12.0).exp(),
            (k * 16.0).exp(),
            (k * 28.0).exp(),
        )
    }
}

/// Evaluates all `180` zero-slope amplifier features + `sat0` for one sample `u`,
/// returning `(sat0, p_pole, p_zero)` and writing the 180 features into `out_phi`.
///
/// Used directly by both [`eval_amp_core_sample`] (real-time DSP) and the C-ABI profiler bridge
/// so calibration and runtime inference are guaranteed bit-identical.
#[inline(always)]
pub fn eval_amp_features_sample(
    u: f32,
    sr: f32,
    cfg: &AmpCoreConfig,
    state: &mut AmpCoreState,
    out_phi: &mut [f32; AMP_CHANNELS],
) -> (f32, f32, f32) {
    let (p_pole, p_zero, a35, a180, a383, a720, a1200, a2400, a_cath, a_sag, a_grid, a_dc28) =
        sample_rate_coeffs(sr);

    let b35 = 1.0 - a35;
    let b180 = 1.0 - a180;
    let b383 = 1.0 - a383;
    let b720 = 1.0 - a720;
    let b1200 = 1.0 - a1200;
    let b2400 = 1.0 - a2400;
    let b_cath = 1.0 - a_cath;
    let b_sag = 1.0 - a_sag;
    let b_grid = 1.0 - a_grid;
    let b_dc28 = 1.0 - a_dc28;

    // 1. Invertible low-shelf pre-emphasis before nonlinear stage
    let x_pre = u - p_pole * state.u_prev + p_zero * state.x_pre_prev;
    state.u_prev = u;
    state.x_pre_prev = x_pre;

    // 2. 3-Pole frequency splits (180 Hz, 720 Hz, 2400 Hz)
    state.lp180_state = a180 * state.lp180_state + b180 * x_pre;
    state.lp720_state = a720 * state.lp720_state + b720 * x_pre;
    state.lp2400_state = a2400 * state.lp2400_state + b2400 * x_pre;
    let x_lp180 = state.lp180_state;
    let x_lp720 = state.lp720_state;
    let x_lp2400 = state.lp2400_state;

    // 3. Dynamic tube envelope trackers (Cathode bypass 95 Hz, B+ sag 12 Hz, Grid bias 16 Hz)
    let abs_x = x_pre.abs();
    state.cathode_env = a_cath * state.cathode_env + b_cath * abs_x;
    state.sag_tube_env = a_sag * state.sag_tube_env + b_sag * abs_x;
    state.grid_bias_env = a_grid * state.grid_bias_env + b_grid * x_pre;

    let g_cathode = fast_tanh(0.35_f32 * state.cathode_env);
    let g_sag = fast_tanh(0.35_f32 * state.sag_tube_env);
    let bias_grid = fast_tanh(0.35_f32 * state.grid_bias_env);

    // 4. Bounded Zero-Slope 12AX7 Bloom Base Saturator
    let cp0 = cfg.cp0;
    let cn0 = cfg.cn0;
    let knee = cfg.knee;
    let inv_knee = 1.0_f32 / knee;

    let sat0 = sat_bloom_bounded_sample(x_pre, cfg);
    let s_lp180_0 = sat_bloom_bounded_sample(x_lp180, cfg);
    let s_lp720_0 = sat_bloom_bounded_sample(x_lp720, cfg);
    let s_lp2400_0 = sat_bloom_bounded_sample(x_lp2400, cfg);

    // 5. Coupled V1 -> V2 interstage pre-saturation excesses (rail-bounded)
    let x_rail = soft_sat_sample(x_pre, cfg.rail_limit, cfg.rail_limit, 2.8, 1.0 / 2.8);
    let ex1_lo = x_rail - soft_sat_sample(x_pre, 0.18, 0.15, 2.8, 1.0 / 2.8);
    let ex1_mid = x_rail - soft_sat_sample(x_pre, 0.36, 0.30, 2.8, 1.0 / 2.8);
    state.lp180_ex1_lo_state = a180 * state.lp180_ex1_lo_state + b180 * ex1_lo;
    state.lp383_ex1_lo_state = a383 * state.lp383_ex1_lo_state + b383 * ex1_lo;
    state.lp180_ex1_mid_state = a180 * state.lp180_ex1_mid_state + b180 * ex1_mid;
    state.lp35_ex1_lo_state = a35 * state.lp35_ex1_lo_state + b35 * ex1_lo;
    state.lp383_ex1_mid_state = a383 * state.lp383_ex1_mid_state + b383 * ex1_mid;

    let v2_inter_a = x_rail - 0.85_f32 * state.lp180_ex1_lo_state;
    let v2_inter_b = x_rail - 0.85_f32 * state.lp383_ex1_lo_state;
    let v2_inter_c = x_rail - 0.85_f32 * state.lp180_ex1_mid_state;

    let r_asym_raw = soft_sat_sample(x_pre, cp0 * 0.80, cn0 * 1.25, knee, inv_knee) - sat0;
    state.r_asym_dc_state = a_dc28 * state.r_asym_dc_state + b_dc28 * r_asym_raw;
    let r_asym_dc = r_asym_raw - state.r_asym_dc_state;

    // 6. 14 Zero-Slope Rail-Bounded Branches x 9 Taps = 126 channels (indices 0..126)
    let raw_r = [
        soft_sat_sample(x_pre, cp0 * 0.55, cn0 * 0.55, knee, inv_knee) - sat0,
        soft_sat_sample(x_pre, cp0 * 1.45, cn0 * 1.45, knee, inv_knee) - sat0,
        r_asym_dc,
        soft_sat_sample(x_pre, cp0 * 2.80, cn0 * 2.80, knee, inv_knee) - sat0,
        soft_sat_sample(x_pre, cp0 * 8.00, cn0 * 8.00, knee, inv_knee) - sat0,
        x_rail - sat0,
        soft_sat_sample(v2_inter_a, cp0, cn0, 3.2, 1.0 / 3.2) - sat0,
        soft_sat_sample(v2_inter_b, cp0, cn0, 3.2, 1.0 / 3.2) - sat0,
        soft_sat_sample(v2_inter_c, cp0, cn0, 3.2, 1.0 / 3.2) - sat0,
        soft_sat_sample(x_lp180, cp0 * 0.65, cn0 * 0.65, knee, inv_knee) - s_lp180_0,
        soft_sat_sample(x_lp180, cp0 * 2.20, cn0 * 2.20, knee, inv_knee) - s_lp180_0,
        soft_sat_sample(x_lp720, cp0 * 0.65, cn0 * 0.65, knee, inv_knee) - s_lp720_0,
        soft_sat_sample(x_lp720, cp0 * 2.20, cn0 * 2.20, knee, inv_knee) - s_lp720_0,
        soft_sat_sample(x_lp2400, cp0 * 0.65, cn0 * 0.65, knee, inv_knee) - s_lp2400_0,
    ];

    let wp = state.psi_write_pos;
    let i0 = wp;
    let i1 = (wp + AMP_RING_LEN - 1) & (AMP_RING_LEN - 1);
    let i2 = (wp + AMP_RING_LEN - 2) & (AMP_RING_LEN - 1);
    let i4 = (wp + AMP_RING_LEN - 4) & (AMP_RING_LEN - 1);
    let i8 = (wp + AMP_RING_LEN - 8) & (AMP_RING_LEN - 1);
    let i16 = (wp + AMP_RING_LEN - 16) & (AMP_RING_LEN - 1);

    for (b, &raw_val) in raw_r.iter().enumerate() {
        let s_lp = if b <= 5 {
            raw_val
        } else if b == 6 || b == 7 || b == 8 || b == 13 {
            let v = a2400 * state.psi_lp_state[b] + b2400 * raw_val;
            state.psi_lp_state[b] = v;
            v
        } else {
            let v = a1200 * state.psi_lp_state[b] + b1200 * raw_val;
            state.psi_lp_state[b] = v;
            v
        };
        state.psi_delay[b][wp] = s_lp;

        let base = b * 9;
        out_phi[base] = state.psi_delay[b][i0];
        out_phi[base + 1] = state.psi_delay[b][i1];
        out_phi[base + 2] = state.psi_delay[b][i2];
        out_phi[base + 3] = state.psi_delay[b][i4];
        out_phi[base + 4] = state.psi_delay[b][i8];
        out_phi[base + 5] = state.psi_delay[b][i16];
        out_phi[base + 6] = g_cathode * s_lp;
        out_phi[base + 7] = g_sag * s_lp;
        out_phi[base + 8] = bias_grid * s_lp;
    }

    // 7. 96 kHz 2x Oversampled Coupled 2-Triode KONE State-Space (48 channels: 126..174)
    let x_kone = soft_sat_sample(x_pre, 0.92, 0.92, 3.5, 1.0 / 3.5);
    state.xk_lp720_state = a720 * state.xk_lp720_state + b720 * x_kone;
    state.xk_lp2400_state = a2400 * state.xk_lp2400_state + b2400 * x_kone;
    let v_cur = [x_kone, state.xk_lp720_state, state.xk_lp2400_state];
    let mut v_mid = [0.0_f32; 3];
    for c in 0..3 {
        v_mid[c] = 0.5625 * (state.resamp_x_prev[c] + v_cur[c])
            - 0.0625 * (state.resamp_x_prev2[c] + v_cur[c]);
        state.resamp_x_prev2[c] = state.resamp_x_prev[c];
        state.resamp_x_prev[c] = v_cur[c];
    }

    let res = state.reservoir;
    let (s_mid, s_cur) = step_kone_96k_pair(
        &v_mid,
        &v_cur,
        g_sag,
        bias_grid,
        &res,
        &mut state.kone_h1,
        &mut state.kone_h2,
    );

    for k in 0..16 {
        let rec_k = 0.22 * state.resamp_y_prev[k] + 0.56 * s_cur[k] + 0.22 * s_mid[k];
        state.resamp_y_prev[k] = s_mid[k];
        state.rk_dc_lp[k] += b_dc28 * (rec_k - state.rk_dc_lp[k]);
        state.rk_delay[k][wp] = rec_k - state.rk_dc_lp[k];

        let base = 126 + k * 3;
        let r0 = state.rk_delay[k][i0];
        let r2 = state.rk_delay[k][i2];
        out_phi[base] = r0;
        out_phi[base + 1] = r2;
        out_phi[base + 2] = g_sag * r0;
    }

    state.psi_write_pos = (wp + 1) & (AMP_RING_LEN - 1);

    // 8. 6 Sag-Recovery (35 Hz & 383 Hz Coupling-Cap Discharge) & Envelope Compression (174..180)
    out_phi[174] = state.lp35_ex1_lo_state;
    out_phi[175] = state.lp383_ex1_lo_state;
    out_phi[176] = state.lp383_ex1_mid_state;
    out_phi[177] = g_sag * sat0;
    out_phi[178] = g_cathode * sat0;
    out_phi[179] = bias_grid * sat0;

    (sat0, p_pole, p_zero)
}

/// Processes one sample through the Universal 180-Channel Tube Amplifier Core.
#[inline(always)]
pub fn eval_amp_core_sample(
    u: f32,
    sr: f32,
    cfg: &AmpCoreConfig,
    state: &mut AmpCoreState,
    weights: &[f32; AMP_CHANNELS],
) -> f32 {
    let mut phi = [0.0_f32; AMP_CHANNELS];
    let (sat0, p_pole, p_zero) = eval_amp_features_sample(u, sr, cfg, state, &mut phi);

    let mut nam_res = 0.0_f32;
    for k in 0..AMP_CHANNELS {
        nam_res += weights[k] * phi[k];
    }

    let tot_sat = sat0 + nam_res;
    // Exact algebraic inverse of the pre-emphasis low-shelf
    let y_out = tot_sat - p_zero * state.sat_prev + p_pole * state.v_post_prev;
    state.sat_prev = tot_sat;
    state.v_post_prev = y_out;
    y_out
}

/// Linearly interpolates `180`-channel weights across sorted gain brackets.
#[inline]
pub fn interpolate_amp_weights(
    brackets: &[f32],
    table: &[[f32; AMP_CHANNELS]],
    gain: f32,
    out: &mut [f32; AMP_CHANNELS],
) {
    let n = brackets.len().min(table.len());
    if n == 0 {
        *out = [0.0; AMP_CHANNELS];
        return;
    }
    if gain <= brackets[0] || n == 1 {
        *out = table[0];
        return;
    }
    if gain >= brackets[n - 1] {
        *out = table[n - 1];
        return;
    }
    let mut seg = 0usize;
    for s in 0..(n - 1) {
        if gain <= brackets[s + 1] {
            seg = s;
            break;
        }
    }
    let span = (brackets[seg + 1] - brackets[seg]).max(1e-6);
    let t = ((gain - brackets[seg]) / span).clamp(0.0, 1.0);
    let a = &table[seg];
    let b = &table[seg + 1];
    for k in 0..AMP_CHANNELS {
        out[k] = a[k] * (1.0 - t) + b[k] * t;
    }
}

/// Optional 3-Stage Invertible Pole-Zero Coupled Tube + Output-Transformer + Screen-Sag Spine.
///
/// Useful for cascading dual-channel or high-gain overdrive heads where V1 -> V2 cathode
/// bias shift and push-pull power-stage screen sag dominate the envelope feel.
#[derive(Clone, Copy, Debug)]
pub struct TubeSpineParams {
    pub f_p1: f32,
    pub g_dc1: f32,
    pub r1_pos: f32,
    pub r1_neg: f32,
    pub a1_soft: f32,
    pub k_bias1: f32,
    pub f_env1: f32,
    pub f_p2: f32,
    pub g_dc2: f32,
    pub r2_pos: f32,
    pub r2_neg: f32,
    pub a2_soft: f32,
    pub k_bias2: f32,
    pub r_pwr: f32,
    pub ap_soft: f32,
    pub f_px: f32,
    pub g_dcx: f32,
    pub k_sag: f32,
}

impl Default for TubeSpineParams {
    fn default() -> Self {
        Self {
            f_p1: 45.0,
            g_dc1: 0.15,
            r1_pos: 3.5,
            r1_neg: 4.5,
            a1_soft: 0.5,
            k_bias1: 0.25,
            f_env1: 30.0,
            f_p2: 24.0,
            g_dc2: 0.25,
            r2_pos: 1.8,
            r2_neg: 3.2,
            a2_soft: 0.25,
            k_bias2: 0.45,
            r_pwr: 2.4,
            ap_soft: 0.4,
            f_px: 80.0,
            g_dcx: 0.35,
            k_sag: 0.03,
        }
    }
}

#[inline(always)]
fn fast_clip_asym(z: f32, rp: f32, rn: f32, alpha_soft: f32) -> f32 {
    let r = if z >= 0.0 { rp } else { rn };
    let u = z / r;
    let u2 = u * u;
    let c2 = z / (1.0 + u2).sqrt();
    let u4 = u2 * u2;
    let c8 = z / (1.0 + u4 * u4).sqrt().sqrt().sqrt();
    alpha_soft * c2 + (1.0 - alpha_soft) * c8
}

#[derive(Clone, Copy, Default)]
pub struct TubeSpineState {
    u0_z1: f32,
    w1_z1: f32,
    s1_z1: f32,
    v1_z1: f32,
    env1: f32,
    v1_d1: f32,
    w2_z1: f32,
    s2_z1: f32,
    v2_z1: f32,
    env2: f32,
    ypi_z1: f32,
    wx_z1: f32,
    sx_z1: f32,
    yx_z1: f32,
    pwr_sag: f32,
}

impl TubeSpineState {
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Processes Stage 1 (V1) and Stage 2 (V2) coupled preamp stages (placed before `H_post`).
    #[inline(always)]
    pub fn process_preamp(&mut self, u_pre: f32, sr: f32, sp: &TubeSpineParams) -> f32 {
        let k = -2.0 * PI / sr.max(8_000.0);
        let p1 = (k * sp.f_p1).exp();
        let q1 = (1.0 - sp.g_dc1 * (1.0 - p1)).clamp(0.0, 0.9997);
        let a_env1 = 1.0 - (k * sp.f_env1).exp();

        let p2 = (k * sp.f_p2).exp();
        let q2 = (1.0 - sp.g_dc2 * (1.0 - p2)).clamp(0.0, 0.9997);

        let w1 = u_pre - p1 * self.u0_z1 + q1 * self.w1_z1;
        self.u0_z1 = u_pre;
        self.w1_z1 = w1;
        let ex1 = (w1.abs() - 0.15 * sp.r1_pos).max(0.0);
        self.env1 += a_env1 * (ex1 - self.env1);
        let z1 = w1 - sp.k_bias1 * self.env1;
        let s1 = fast_clip_asym(z1, sp.r1_pos, sp.r1_neg, sp.a1_soft)
            + fast_clip_asym(sp.k_bias1 * self.env1, sp.r1_pos, sp.r1_neg, sp.a1_soft);
        let v1 = s1 - q1 * self.s1_z1 + p1 * self.v1_z1;
        self.s1_z1 = s1;
        self.v1_z1 = v1;

        let w2 = v1 - p2 * self.v1_d1 + q2 * self.w2_z1;
        self.v1_d1 = v1;
        self.w2_z1 = w2;
        let ex2 = (w2.abs() - 0.15 * sp.r2_pos).max(0.0);
        self.env2 += a_env1 * (ex2 - self.env2);
        let z2 = w2 - sp.k_bias2 * self.env2;
        let s2 = fast_clip_asym(z2, sp.r2_pos, sp.r2_neg, sp.a2_soft)
            + fast_clip_asym(sp.k_bias2 * self.env2, sp.r2_pos, sp.r2_neg, sp.a2_soft);
        let v2 = s2 - q2 * self.s2_z1 + p2 * self.v2_z1;
        self.s2_z1 = s2;
        self.v2_z1 = v2;
        v2
    }

    /// Processes Stage 3 (Output Transformer & Push-Pull Power-Tube Screen Sag, after `H_post`).
    #[inline(always)]
    pub fn process_power_stage(&mut self, y_pi: f32, sr: f32, sp: &TubeSpineParams) -> f32 {
        let k = -2.0 * PI / sr.max(8_000.0);
        let px = (k * sp.f_px).exp();
        let qx = (1.0 - sp.g_dcx * (1.0 - px)).clamp(0.0, 0.9997);
        let a_sag = 1.0 - (k * 18.0).exp();

        let wx = y_pi - px * self.ypi_z1 + qx * self.wx_z1;
        self.ypi_z1 = y_pi;
        self.wx_z1 = wx;
        self.pwr_sag += a_sag * (wx.abs() - self.pwr_sag);
        let r_eff = sp.r_pwr / (1.0 + sp.k_sag * (self.pwr_sag / (sp.r_pwr + 1e-4)));
        let sx = fast_clip_asym(wx, r_eff, r_eff, sp.ap_soft);
        let y_amp = sx - qx * self.sx_z1 + px * self.yx_z1;
        self.sx_z1 = sx;
        self.yx_z1 = y_amp;
        y_amp
    }
}
