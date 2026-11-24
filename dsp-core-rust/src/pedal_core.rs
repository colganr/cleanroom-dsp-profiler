//! 126-Channel 3-Pole Zero-Slope Overdrive / Boost Pedal Core (`OverdriveCore`).
//!
//! Architecture (Wiener-Hammerstein):
//!   `x_in -> H_pre(Drive) -> [u_pre + Psi_126(u_pre, x_ref) · w(Drive)] -> H_post(Tone) -> Level -> DC Block -> Mix`
//!
//! The 126 zero-slope nonlinear features combine:
//! - 7 diode saturation scales (`PEDAL_SCALES`)
//! - 3-pole frequency-split feedback (`180 Hz`, `720 Hz`, `2400 Hz`)
//! - 6 causal dilations (`[0, 1, 2, 4, 8, 16]`)
//! - 96 kHz 2x oversampled 8-state clean-room KONE recurrent reservoir
//! - Fast/mid envelope & DC bias trackers (`110 Hz`, `14 Hz`, `18 Hz`)
//!
//! Supports both op-amp feedback topologies that blend dry input around the diodes
//! (`subtract_dry = true`, e.g., TS-style / Klon) and pure series clippers (`subtract_dry = false`).

use crate::reservoir::{PedalReservoirSpec, PEDAL_RESERVOIR_DEFAULT};
use crate::simd_ops::{dot_f32, fast_tanh};

pub const PEDAL_FIR_LEN: usize = 256;
pub const PEDAL_CHANNELS: usize = 126; // 7 scales x 18 features
pub const PEDAL_RING_LEN: usize = 32;

pub const PEDAL_SCALES: [f32; 7] = [0.25, 0.55, 1.20, 2.60, 5.50, 11.50, 24.00];
pub const PEDAL_GAIN_BRACKETS_9: [f32; 9] = [0.00, 0.15, 0.30, 0.45, 0.55, 0.65, 0.75, 0.85, 1.00];

const ALPHA_ENV_FAST: f32 = 0.014_296_005_f32; // 110 Hz @ 48 kHz
const ALPHA_ENV_MID: f32 = 0.001_830_902_7_f32; // 14 Hz @ 48 kHz
const ALPHA_BIAS_RC: f32 = 0.002_353_399_5_f32; // 18 Hz @ 48 kHz
const ALPHA_LP180: f32 = 0.023_285_626_f32; // 180 Hz @ 48 kHz
const ALPHA_LP720: f32 = 0.089_941_62_f32; // 720 Hz @ 48 kHz
const ALPHA_LP2400: f32 = 0.269_601_1_f32; // 2400 Hz @ 48 kHz
const ALPHA_DC_LP: f32 = 0.003_658_489_f32; // 28 Hz DC blocker @ 48 kHz

const ASYM_B0: f32 = 0.25_f32;

/// Zero-allocation feature-extraction state for the 126-channel 3-pole overdrive core.
#[derive(Clone, Copy)]
pub struct PedalFeatureState {
    pub env_fast: f32,
    pub env_mid: f32,
    pub bias_rc: f32,
    pub lp180: f32,
    pub lp720: f32,
    pub lp2400: f32,
    pub kone_h: [f32; 8],
    pub resamp_x_prev: [f32; 3],
    pub resamp_x_prev2: [f32; 3],
    pub resamp_y_prev: [f32; 8],
    pub rk_dc_lp: [f32; 8],
    pub p3_dc_lp: [f32; 7],
    pub p1_ring: [[f32; PEDAL_RING_LEN]; 7],
    pub p3_ring: [[f32; PEDAL_RING_LEN]; 7],
    pub rk_ring: [[f32; PEDAL_RING_LEN]; 7],
    pub ring_wp: usize,
    pub reservoir: PedalReservoirSpec,
}

impl Default for PedalFeatureState {
    fn default() -> Self {
        Self::new(PEDAL_RESERVOIR_DEFAULT)
    }
}

impl PedalFeatureState {
    pub const fn new(reservoir: PedalReservoirSpec) -> Self {
        Self {
            env_fast: 0.0,
            env_mid: 0.0,
            bias_rc: 0.0,
            lp180: 0.0,
            lp720: 0.0,
            lp2400: 0.0,
            kone_h: [0.0; 8],
            resamp_x_prev: [0.0; 3],
            resamp_x_prev2: [0.0; 3],
            resamp_y_prev: [0.0; 8],
            rk_dc_lp: [0.0; 8],
            p3_dc_lp: [0.0; 7],
            p1_ring: [[0.0; PEDAL_RING_LEN]; 7],
            p3_ring: [[0.0; PEDAL_RING_LEN]; 7],
            rk_ring: [[0.0; PEDAL_RING_LEN]; 7],
            ring_wp: 0,
            reservoir,
        }
    }

    pub fn reset(&mut self) {
        let res = self.reservoir;
        *self = Self::new(res);
    }
}

/// Evaluates all `126` zero-slope overdrive features for one pre-filtered sample `u_pre`
/// (with optional dry reference `x_in` when `subtract_dry == true`).
#[inline(always)]
pub fn eval_pedal_features_sample(
    x_in: f32,
    u_pre_raw: f32,
    subtract_dry: bool,
    state: &mut PedalFeatureState,
    out_psi: &mut [f32; PEDAL_CHANNELS],
) -> f32 {
    // Rail-bound input to prevent polynomial/feature blowup when pedals are stacked hot
    let u_pre = 2.4_f32 * fast_tanh(u_pre_raw / 2.4_f32);
    let x_ref = if subtract_dry {
        2.4_f32 * fast_tanh(x_in / 2.4_f32)
    } else {
        0.0_f32
    };
    let u_fb = u_pre - x_ref;
    let abs_fb = u_fb.abs();

    state.env_fast += ALPHA_ENV_FAST * (abs_fb - state.env_fast);
    state.env_mid += ALPHA_ENV_MID * (abs_fb - state.env_mid);
    state.bias_rc += ALPHA_BIAS_RC * (u_pre - state.bias_rc);

    let g_fast = fast_tanh(0.35_f32 * state.env_fast);
    let g_mid = fast_tanh(0.35_f32 * state.env_mid);
    let bias_rc = fast_tanh(0.35_f32 * state.bias_rc);

    state.lp180 += ALPHA_LP180 * (u_fb - state.lp180);
    state.lp720 += ALPHA_LP720 * (u_fb - state.lp720);
    state.lp2400 += ALPHA_LP2400 * (u_fb - state.lp2400);

    let u_lp180 = state.lp180;
    let u_lp720 = state.lp720;
    let u_lp2400 = state.lp2400;

    // 96 kHz 2x oversampled 8-state clean-room KONE reservoir
    let v_cur = [u_fb, u_lp720, u_lp2400];
    let mut v_mid = [0.0_f32; 3];
    for c in 0..3 {
        v_mid[c] = 0.5625 * (state.resamp_x_prev[c] + v_cur[c])
            - 0.0625 * (state.resamp_x_prev2[c] + v_cur[c]);
        state.resamp_x_prev2[c] = state.resamp_x_prev[c];
        state.resamp_x_prev[c] = v_cur[c];
    }

    let res = state.reservoir;
    let mut step_sub = |v_sub: &[f32; 3]| -> [f32; 8] {
        let u_vec = [
            v_sub[0], v_sub[0], v_sub[0], v_sub[0], v_sub[0], v_sub[0], v_sub[1], v_sub[2],
        ];
        let h_old = state.kone_h;
        let mut s_out = [0.0_f32; 8];
        for k in 0..8 {
            let c = res.c_ref[k] * (1.0 - 0.08 * g_mid);
            let inv_c = 1.0 / c;
            let b = res.bias[k] + 0.03 * bias_rc;
            let bt = fast_tanh(b);
            let prev_rot = h_old[(k + 7) & 7];
            let wx = c * fast_tanh((res.in_w[k] * u_vec[k]) * inv_c);
            let u_cell = res.diag[k] * h_old[k] + res.cross[k] * prev_rot + wx;
            let hn = fast_tanh(u_cell + b) - bt;
            let u_ref = c * fast_tanh(u_cell * inv_c);
            s_out[k] = hn - (1.0 - bt * bt) * u_ref;
            state.kone_h[k] = hn;
        }
        s_out
    };

    let s_mid = step_sub(&v_mid);
    let s_cur = step_sub(&v_cur);

    let mut rk_blocked = [0.0_f32; 8];
    for k in 0..8 {
        let rec = 0.22 * state.resamp_y_prev[k] + 0.56 * s_cur[k] + 0.22 * s_mid[k];
        state.resamp_y_prev[k] = s_mid[k];
        state.rk_dc_lp[k] += ALPHA_DC_LP * (rec - state.rk_dc_lp[k]);
        rk_blocked[k] = rec - state.rk_dc_lp[k];
    }

    let tb0 = fast_tanh(ASYM_B0);
    let inv_d0 = 1.0_f32 / (1.0_f32 - tb0 * tb0);

    let wp = state.ring_wp;
    let i0 = wp;
    let i1 = (wp + PEDAL_RING_LEN - 1) & (PEDAL_RING_LEN - 1);
    let i2 = (wp + PEDAL_RING_LEN - 2) & (PEDAL_RING_LEN - 1);
    let i4 = (wp + PEDAL_RING_LEN - 4) & (PEDAL_RING_LEN - 1);
    let i8 = (wp + PEDAL_RING_LEN - 8) & (PEDAL_RING_LEN - 1);
    let i16 = (wp + PEDAL_RING_LEN - 16) & (PEDAL_RING_LEN - 1);

    let mut col = 0usize;
    for (s_idx, &s) in PEDAL_SCALES.iter().enumerate() {
        let inv_s = 1.0_f32 / s;
        let su_fb = s * u_fb;
        let su_pre = s * u_pre;

        let p1 = fast_tanh(su_fb) * inv_s - u_fb;
        let p2 = u_fb / (1.0_f32 + su_fb * su_fb).sqrt() - u_fb;
        let p3_raw = (fast_tanh(su_pre + ASYM_B0) - tb0) * (inv_s * inv_d0) - u_pre;
        state.p3_dc_lp[s_idx] += ALPHA_DC_LP * (p3_raw - state.p3_dc_lp[s_idx]);
        let p3 = p3_raw - state.p3_dc_lp[s_idx];
        let rk = rk_blocked[s_idx];

        state.p1_ring[s_idx][wp] = p1;
        state.p3_ring[s_idx][wp] = p3;
        state.rk_ring[s_idx][wp] = rk;

        // 6 causal dilations of p1
        out_psi[col] = state.p1_ring[s_idx][i0];
        out_psi[col + 1] = state.p1_ring[s_idx][i1];
        out_psi[col + 2] = state.p1_ring[s_idx][i2];
        out_psi[col + 3] = state.p1_ring[s_idx][i4];
        out_psi[col + 4] = state.p1_ring[s_idx][i8];
        out_psi[col + 5] = state.p1_ring[s_idx][i16];
        // 3 causal dilations of 96 kHz KONE state rk
        out_psi[col + 6] = state.rk_ring[s_idx][i0];
        out_psi[col + 7] = state.rk_ring[s_idx][i2];
        out_psi[col + 8] = state.rk_ring[s_idx][i8];
        // Algebraic soft-knee diode term p2 + asymmetric term p3 (dilations 0, 2)
        out_psi[col + 9] = p2;
        out_psi[col + 10] = state.p3_ring[s_idx][i0];
        out_psi[col + 11] = state.p3_ring[s_idx][i2];
        // Envelope-gated diode & reservoir terms
        out_psi[col + 12] = g_fast * p1;
        out_psi[col + 13] = g_mid * rk;
        out_psi[col + 14] = bias_rc * p1;
        // 3-Pole frequency-split zero-slope diode residuals (180 Hz, 720 Hz, 2400 Hz)
        out_psi[col + 15] = fast_tanh(s * u_lp180) * inv_s - u_lp180;
        out_psi[col + 16] = fast_tanh(s * u_lp720) * inv_s - u_lp720;
        out_psi[col + 17] = fast_tanh(s * u_lp2400) * inv_s - u_lp2400;
        col += 18;
    }

    state.ring_wp = (wp + 1) & (PEDAL_RING_LEN - 1);
    u_pre
}

/// Complete real-time zero-allocation Overdrive / Boost Pedal Engine.
#[derive(Clone, Copy)]
pub struct OverdriveEngine {
    pub active: bool,
    pub subtract_dry: bool,
    pub mix: f32,
    pub level_gain: f32,
    pub active_pre_ir: [f32; PEDAL_FIR_LEN],
    pub active_post_ir: [f32; PEDAL_FIR_LEN],
    pub active_weights: [f32; PEDAL_CHANNELS],
    pre_hist: [f32; PEDAL_FIR_LEN * 2],
    pre_wp: usize,
    post_hist: [f32; PEDAL_FIR_LEN * 2],
    post_wp: usize,
    feat: PedalFeatureState,
    dc_x1: f32,
    dc_y1: f32,
}

impl Default for OverdriveEngine {
    fn default() -> Self {
        Self::new(false, PEDAL_RESERVOIR_DEFAULT)
    }
}

impl OverdriveEngine {
    pub const fn new(subtract_dry: bool, reservoir: PedalReservoirSpec) -> Self {
        let mut pre = [0.0_f32; PEDAL_FIR_LEN];
        let mut post = [0.0_f32; PEDAL_FIR_LEN];
        pre[0] = 1.0;
        post[0] = 1.0;
        Self {
            active: true,
            subtract_dry,
            mix: 1.0,
            level_gain: 1.0,
            active_pre_ir: pre,
            active_post_ir: post,
            active_weights: [0.0; PEDAL_CHANNELS],
            pre_hist: [0.0; PEDAL_FIR_LEN * 2],
            pre_wp: 0,
            post_hist: [0.0; PEDAL_FIR_LEN * 2],
            post_wp: 0,
            feat: PedalFeatureState::new(reservoir),
            dc_x1: 0.0,
            dc_y1: 0.0,
        }
    }

    pub fn reset(&mut self) {
        self.pre_hist = [0.0; PEDAL_FIR_LEN * 2];
        self.pre_wp = 0;
        self.post_hist = [0.0; PEDAL_FIR_LEN * 2];
        self.post_wp = 0;
        self.feat.reset();
        self.dc_x1 = 0.0;
        self.dc_y1 = 0.0;
    }

    /// Updates interpolated Pre-FIR, Post-FIR, Level taper, and 126-ch weights from profiled tables.
    #[allow(clippy::too_many_arguments)]
    pub fn recompute_from_tables(
        &mut self,
        drive: f32,
        tone: f32,
        level: f32,
        pre_ir_21: &[[f32; PEDAL_FIR_LEN]; 21],
        post_ir_21: &[[f32; PEDAL_FIR_LEN]; 21],
        level_gain_21: &[f32; 21],
        weights_9: &[[f32; PEDAL_CHANNELS]; 9],
    ) {
        let g = drive.clamp(0.0, 1.0);
        let t = tone.clamp(0.0, 1.0);
        let l = level.clamp(0.0, 1.0);

        let gp = g * 20.0;
        let gi = (gp as usize).min(19);
        let gf = gp - (gi as f32);
        for (dst, (&p0, &p1)) in self
            .active_pre_ir
            .iter_mut()
            .zip(pre_ir_21[gi].iter().zip(pre_ir_21[gi + 1].iter()))
        {
            *dst = (1.0 - gf) * p0 + gf * p1;
        }

        let tp = t * 20.0;
        let ti = (tp as usize).min(19);
        let tf = tp - (ti as f32);
        for (dst, (&p0, &p1)) in self
            .active_post_ir
            .iter_mut()
            .zip(post_ir_21[ti].iter().zip(post_ir_21[ti + 1].iter()))
        {
            *dst = (1.0 - tf) * p0 + tf * p1;
        }

        let lp = l * 20.0;
        let li = (lp as usize).min(19);
        let lf = lp - (li as f32);
        self.level_gain = (1.0 - lf) * level_gain_21[li] + lf * level_gain_21[li + 1];

        let mut seg = 0usize;
        for s in 0..8 {
            if g <= PEDAL_GAIN_BRACKETS_9[s + 1] {
                seg = s;
                break;
            }
        }
        let span = (PEDAL_GAIN_BRACKETS_9[seg + 1] - PEDAL_GAIN_BRACKETS_9[seg]).max(1e-6);
        let a = ((g - PEDAL_GAIN_BRACKETS_9[seg]) / span).clamp(0.0, 1.0);
        for (dst, (&w0, &w1)) in self
            .active_weights
            .iter_mut()
            .zip(weights_9[seg].iter().zip(weights_9[seg + 1].iter()))
        {
            *dst = (1.0 - a) * w0 + a * w1;
        }
    }

    #[inline(always)]
    pub fn process_sample(&mut self, x_in: f32) -> f32 {
        if !self.active {
            return x_in;
        }

        // 1. Pre-FIR (SIMD accelerated)
        let pw = (self.pre_wp + PEDAL_FIR_LEN - 1) & (PEDAL_FIR_LEN - 1);
        self.pre_wp = pw;
        self.pre_hist[pw] = x_in;
        self.pre_hist[pw + PEDAL_FIR_LEN] = x_in;
        let u_pre_raw = dot_f32(&self.pre_hist[pw..pw + PEDAL_FIR_LEN], &self.active_pre_ir);

        // 2. 126-Ch Zero-Slope 3-Pole + 96 kHz KONE Residual
        let mut psi = [0.0_f32; PEDAL_CHANNELS];
        let u_pre = eval_pedal_features_sample(
            x_in,
            u_pre_raw,
            self.subtract_dry,
            &mut self.feat,
            &mut psi,
        );
        let nl_res = dot_f32(&psi, &self.active_weights);
        let v_sat = u_pre + nl_res;

        // 3. Post-FIR (Tone)
        let qw = (self.post_wp + PEDAL_FIR_LEN - 1) & (PEDAL_FIR_LEN - 1);
        self.post_wp = qw;
        self.post_hist[qw] = v_sat;
        self.post_hist[qw + PEDAL_FIR_LEN] = v_sat;
        let wet = dot_f32(
            &self.post_hist[qw..qw + PEDAL_FIR_LEN],
            &self.active_post_ir,
        ) * self.level_gain;

        // 4. ~12 Hz output coupling capacitor (DC blocker)
        let wet_ac = 0.999_215_2_f32 * (wet - self.dc_x1) + 0.998_430_4_f32 * self.dc_y1;
        self.dc_x1 = wet;
        self.dc_y1 = wet_ac;

        let mix = self.mix.clamp(0.0, 1.0);
        (1.0 - mix) * x_in + mix * wet_ac
    }
}
