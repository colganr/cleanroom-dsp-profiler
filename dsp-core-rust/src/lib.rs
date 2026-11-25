//! Clean-Room Pure-Rust Audio DSP Core (`cleanroom-dsp`).
//!
//! Provides zero-allocation building blocks for high-accuracy black-box profiled guitar
//! amplifiers, overdrive pedals, multi-microphone speaker cabinets, and time-based,
//! modulation, and dynamics effects:
//!
//! - [`simd_ops`]: Hardware `DenormalGuard` (FTZ/DAZ), branchless `[7/6]` Padé `fast_tanh`,
//!   and runtime-dispatched `AVX2+FMA` / `SSE2` / `NEON` FIR dot products.
//! - [`cepstral_fir`]: Zero-allocation real-cepstrum Hilbert minimum-phase FIR synthesis and
//!   1D / 3D Bode tensor interpolation.
//! - [`reservoir`]: Clean-room analytical multi-scale KONE recurrent state-space reservoirs.
//! - [`amp_core`]: Universal 180-channel amplifier core (Bounded 12AX7 Triode Bloom, 3-pole
//!   `180/720/2400 Hz`, coupled `V1 -> V2`, 96 kHz KONE, `35/383 Hz` sag recovery) + optional
//!   3-stage invertible pole-zero tube/transformer spine.
//! - [`pedal_core`]: 126-channel 3-pole zero-slope Wiener-Hammerstein overdrive engine.
//! - [`cab_grid`]: Generic `ScalableCabEngine<const N_SLOTS: usize>` for 2D/3D quantized IR grids.
//! - [`spring_reverb`]: 4-basis SVD Decay compression + 64-tap Wet Tone FIR (optional 4x 12 kHz tank).
//! - [`delay_bbd`]: Hermite fractional delay + exact `loop_extra_delay` + 64-tap loop FIR + 30-ch BBD sat.
//! - [`harmonic_trem`]: 128-tap dual-band SVD crossover + 21-pt LFO curves + optional 15-ch tube sat.
//! - [`envelope_filter`]: Vactrol + Cytomic TPT 2-pole SVF envelope filter & soft-knee studio compressor.
//! - [`nam_capture`]: Calibrated `12,288`-sample conditioning stimulus for `.nam` WaveNet export.
//!
//! Also exposes a clean C ABI (`#[no_mangle] pub extern "C"`) so the exact compiled Rust DSP
//! drives DAW wrappers, standalone hosts, Python head-to-head null-test verification, and
//! Rust-accelerated feature extraction during profiling.

pub mod amp_core;
pub mod cab_grid;
pub mod cepstral_fir;
pub mod delay_bbd;
pub mod envelope_filter;
pub mod harmonic_trem;
pub mod nam_capture;
pub mod pedal_core;
pub mod reservoir;
pub mod simd_ops;
pub mod spring_reverb;

use amp_core::{
    eval_amp_core_sample, eval_amp_features_sample, AmpCoreConfig, AmpCoreState, AMP_CHANNELS,
};
use cab_grid::ScalableCabEngine;
use cepstral_fir::{log_freqs_64, synthesize_min_phase_ir, NUM_FREQS};
use pedal_core::{eval_pedal_features_sample, OverdriveEngine, PedalFeatureState, PEDAL_CHANNELS};
use simd_ops::{dot_f32, DenormalGuard};
use std::slice;

pub const HEAD_IR_LEN: usize = 1024;
pub const POST_IR_LEN: usize = 64;

/// Complete Wiener-Hammerstein Amplifier Head: `H_pre (1024-tap) -> 180-ch Bloom/Sag/KONE -> H_post (64-tap)`.
pub struct WienerHammersteinAmp {
    pub sr: f32,
    pub enabled: bool,
    pub cfg: AmpCoreConfig,
    pub active_pre_ir: [f32; HEAD_IR_LEN],
    pub active_post_ir: [f32; POST_IR_LEN],
    pub active_weights: [f32; AMP_CHANNELS],
    pre_hist: [f32; HEAD_IR_LEN * 2],
    pre_wp: usize,
    post_hist: [f32; POST_IR_LEN * 2],
    post_wp: usize,
    pub core: AmpCoreState,
}

impl Default for WienerHammersteinAmp {
    fn default() -> Self {
        Self::new(48_000.0)
    }
}

impl WienerHammersteinAmp {
    pub fn new(sr: f32) -> Self {
        let mut pre = [0.0_f32; HEAD_IR_LEN];
        let mut post = [0.0_f32; POST_IR_LEN];
        pre[0] = 1.0;
        post[0] = 1.0;
        Self {
            sr,
            enabled: true,
            cfg: AmpCoreConfig::default(),
            active_pre_ir: pre,
            active_post_ir: post,
            active_weights: [0.0; AMP_CHANNELS],
            pre_hist: [0.0; HEAD_IR_LEN * 2],
            pre_wp: 0,
            post_hist: [0.0; POST_IR_LEN * 2],
            post_wp: 0,
            core: AmpCoreState::default(),
        }
    }

    pub fn reset(&mut self) {
        self.pre_hist = [0.0; HEAD_IR_LEN * 2];
        self.pre_wp = 0;
        self.post_hist = [0.0; POST_IR_LEN * 2];
        self.post_wp = 0;
        self.core.reset();
    }

    /// Re-synthesizes `active_pre_ir` and `active_post_ir` via zero-alloc cepstral folding.
    pub fn update_filters(
        &mut self,
        base_pre_ir: &[f32; HEAD_IR_LEN],
        base_post_ir: &[f32; POST_IR_LEN],
        pre_delta_db_64: &[f32; NUM_FREQS],
        post_delta_db_64: &[f32; NUM_FREQS],
        polarity: f32,
    ) {
        let freqs = log_freqs_64();
        synthesize_min_phase_ir(
            base_pre_ir,
            &freqs,
            pre_delta_db_64,
            self.sr,
            polarity,
            64,
            &mut self.active_pre_ir,
        );
        synthesize_min_phase_ir(
            base_post_ir,
            &freqs,
            post_delta_db_64,
            self.sr,
            1.0,
            12,
            &mut self.active_post_ir,
        );
    }

    /// Processes a single mono sample with zero heap allocations.
    #[inline(always)]
    pub fn process_sample(&mut self, x_in: f32) -> f32 {
        if !self.enabled {
            return x_in;
        }
        // 1. Pre-clipping H_pre convolution (1024 taps, SIMD accelerated)
        let pw = (self.pre_wp + HEAD_IR_LEN - 1) & (HEAD_IR_LEN - 1);
        self.pre_wp = pw;
        self.pre_hist[pw] = x_in;
        self.pre_hist[pw + HEAD_IR_LEN] = x_in;
        let u_pre = dot_f32(&self.pre_hist[pw..pw + HEAD_IR_LEN], &self.active_pre_ir);

        // 2. Universal 180-channel Bloom + Sag + 96 kHz KONE nonlinear core
        let y_nl = eval_amp_core_sample(
            u_pre,
            self.sr,
            &self.cfg,
            &mut self.core,
            &self.active_weights,
        );

        // 3. Post-clipping H_post convolution (64 taps, SIMD accelerated)
        let qw = (self.post_wp + POST_IR_LEN - 1) & (POST_IR_LEN - 1);
        self.post_wp = qw;
        self.post_hist[qw] = y_nl;
        self.post_hist[qw + POST_IR_LEN] = y_nl;
        dot_f32(&self.post_hist[qw..qw + POST_IR_LEN], &self.active_post_ir)
    }

    /// Processes a contiguous block under hardware `DenormalGuard` (FTZ/DAZ).
    pub fn process_block(&mut self, input: &[f32], output: &mut [f32]) {
        let _guard = DenormalGuard::new();
        let n = input.len().min(output.len());
        for i in 0..n {
            output[i] = self.process_sample(input[i]);
        }
    }
}

/// Starter Full Guitar Rig Chain (`Overdrive -> WienerHammersteinAmp -> ScalableCabEngine<2>`).
pub struct StarterRigEngine {
    pub overdrive: OverdriveEngine,
    pub amp: WienerHammersteinAmp,
    pub cab: ScalableCabEngine<2>,
}

impl Default for StarterRigEngine {
    fn default() -> Self {
        Self::new(48_000.0)
    }
}

impl StarterRigEngine {
    pub fn new(sr: f32) -> Self {
        let mut od = OverdriveEngine::default();
        od.active = false;
        let mut cab = ScalableCabEngine::<2>::new();
        cab.active = false;
        Self {
            overdrive: od,
            amp: WienerHammersteinAmp::new(sr),
            cab,
        }
    }

    pub fn reset(&mut self) {
        self.overdrive.reset();
        self.amp.reset();
        self.cab.reset();
    }

    pub fn process_mono_to_stereo(&mut self, input: &[f32], out_l: &mut [f32], out_r: &mut [f32]) {
        let _guard = DenormalGuard::new();
        let n = input.len().min(out_l.len()).min(out_r.len());
        for i in 0..n {
            let x = self.overdrive.process_sample(input[i]);
            let y = self.amp.process_sample(x);
            let (l, r) = self.cab.process_sample(y);
            out_l[i] = l;
            out_r[i] = r;
        }
    }
}

// ============================================================================
// C ABI FFI: Rust-Accelerated Feature Evaluators & Block Processors
// ============================================================================
// Used by the Python profiler (`amp_profiler.rust_bridge`) for 50-100x faster
// feature generation with guaranteed bit-exact parity against runtime Rust DSP.

/// Evaluates all `180` zero-slope amplifier features + `sat0` across `n` samples.
///
/// # Safety
/// - `u_in` must point to `n` readable `f32` values.
/// - `phi_out` must point to `n * 180` writable `f32` values (row-major `[n][180]`).
/// - `sat0_out` must point to `n` writable `f32` values.
/// - `shelf_out` (optional, may be null) must point to `2` writable `f32` values (`[p_pole, p_zero]`).
#[no_mangle]
pub unsafe extern "C" fn cleanroom_eval_amp_features(
    u_in: *const f32,
    n: usize,
    sr: f32,
    cp0: f32,
    cn0: f32,
    knee: f32,
    bloom_alpha: f32,
    phi_out: *mut f32,
    sat0_out: *mut f32,
    shelf_out: *mut f32,
) -> i32 {
    if u_in.is_null() || phi_out.is_null() || sat0_out.is_null() || n == 0 {
        return -1;
    }
    let _guard = DenormalGuard::new();
    let xs = slice::from_raw_parts(u_in, n);
    let phi_flat = slice::from_raw_parts_mut(phi_out, n * AMP_CHANNELS);
    let sat0_s = slice::from_raw_parts_mut(sat0_out, n);

    let cfg = AmpCoreConfig {
        cp0,
        cn0,
        knee,
        bloom_alpha,
        ..AmpCoreConfig::default()
    };
    let mut state = AmpCoreState::default();
    let mut row = [0.0_f32; AMP_CHANNELS];
    let mut last_pole = 0.0_f32;
    let mut last_zero = 0.0_f32;

    for i in 0..n {
        let (s0, pp, pz) = eval_amp_features_sample(xs[i], sr, &cfg, &mut state, &mut row);
        sat0_s[i] = s0;
        let off = i * AMP_CHANNELS;
        phi_flat[off..off + AMP_CHANNELS].copy_from_slice(&row);
        last_pole = pp;
        last_zero = pz;
    }

    if !shelf_out.is_null() {
        let sh = slice::from_raw_parts_mut(shelf_out, 2);
        sh[0] = last_pole;
        sh[1] = last_zero;
    }
    0
}

/// Renders an entire signal through the 180-channel Amp Core (pre-filtered `u_in` + `weights_180`).
///
/// # Safety
/// - `u_in` must point to `n` readable `f32` values.
/// - `weights_180` must point to `180` readable `f32` values.
/// - `y_out` must point to `n` writable `f32` values.
#[no_mangle]
pub unsafe extern "C" fn cleanroom_render_amp_core(
    u_in: *const f32,
    n: usize,
    sr: f32,
    cp0: f32,
    cn0: f32,
    knee: f32,
    bloom_alpha: f32,
    weights_180: *const f32,
    y_out: *mut f32,
) -> i32 {
    if u_in.is_null() || weights_180.is_null() || y_out.is_null() || n == 0 {
        return -1;
    }
    let _guard = DenormalGuard::new();
    let xs = slice::from_raw_parts(u_in, n);
    let w_slice = slice::from_raw_parts(weights_180, AMP_CHANNELS);
    let ys = slice::from_raw_parts_mut(y_out, n);

    let mut w = [0.0_f32; AMP_CHANNELS];
    w.copy_from_slice(w_slice);

    let cfg = AmpCoreConfig {
        cp0,
        cn0,
        knee,
        bloom_alpha,
        ..AmpCoreConfig::default()
    };
    let mut state = AmpCoreState::default();
    for i in 0..n {
        ys[i] = eval_amp_core_sample(xs[i], sr, &cfg, &mut state, &w);
    }
    0
}

/// Evaluates all `126` zero-slope overdrive/pedal features across `n` samples.
///
/// # Safety
/// - `x_in` and `u_pre_in` must point to `n` readable `f32` values.
/// - `psi_out` must point to `n * 126` writable `f32` values (row-major `[n][126]`).
/// - `u_bounded_out` must point to `n` writable `f32` values.
#[no_mangle]
pub unsafe extern "C" fn cleanroom_eval_pedal_features(
    x_in: *const f32,
    u_pre_in: *const f32,
    n: usize,
    subtract_dry: i32,
    psi_out: *mut f32,
    u_bounded_out: *mut f32,
) -> i32 {
    if x_in.is_null()
        || u_pre_in.is_null()
        || psi_out.is_null()
        || u_bounded_out.is_null()
        || n == 0
    {
        return -1;
    }
    let _guard = DenormalGuard::new();
    let xs = slice::from_raw_parts(x_in, n);
    let us = slice::from_raw_parts(u_pre_in, n);
    let psi_flat = slice::from_raw_parts_mut(psi_out, n * PEDAL_CHANNELS);
    let ub_out = slice::from_raw_parts_mut(u_bounded_out, n);

    let mut state = PedalFeatureState::default();
    let mut row = [0.0_f32; PEDAL_CHANNELS];
    let sub = subtract_dry != 0;

    for i in 0..n {
        let ub = eval_pedal_features_sample(xs[i], us[i], sub, &mut state, &mut row);
        ub_out[i] = ub;
        let off = i * PEDAL_CHANNELS;
        psi_flat[off..off + PEDAL_CHANNELS].copy_from_slice(&row);
    }
    0
}

/// Renders an entire signal through the Wiener-Hammerstein `OverdriveEngine`.
///
/// # Safety
/// - `x_in` and `y_out` must point to `n` `f32` values.
/// - `pre_ir` and `post_ir` must point to `256` `f32` values.
/// - `weights_126` must point to `126` `f32` values.
#[no_mangle]
pub unsafe extern "C" fn cleanroom_render_pedal_engine(
    x_in: *const f32,
    n: usize,
    subtract_dry: i32,
    level_gain: f32,
    mix: f32,
    pre_ir: *const f32,
    post_ir: *const f32,
    weights_126: *const f32,
    y_out: *mut f32,
) -> i32 {
    if x_in.is_null()
        || pre_ir.is_null()
        || post_ir.is_null()
        || weights_126.is_null()
        || y_out.is_null()
        || n == 0
    {
        return -1;
    }
    let _guard = DenormalGuard::new();
    let xs = slice::from_raw_parts(x_in, n);
    let ys = slice::from_raw_parts_mut(y_out, n);
    let pre_s = slice::from_raw_parts(pre_ir, pedal_core::PEDAL_FIR_LEN);
    let post_s = slice::from_raw_parts(post_ir, pedal_core::PEDAL_FIR_LEN);
    let w_s = slice::from_raw_parts(weights_126, PEDAL_CHANNELS);

    let mut eng = OverdriveEngine::new(subtract_dry != 0, reservoir::PEDAL_RESERVOIR_DEFAULT);
    eng.level_gain = level_gain;
    eng.mix = mix;
    eng.active_pre_ir.copy_from_slice(pre_s);
    eng.active_post_ir.copy_from_slice(post_s);
    eng.active_weights.copy_from_slice(w_s);

    for i in 0..n {
        ys[i] = eng.process_sample(xs[i]);
    }
    0
}

/// Evaluates a 14-parameter Vactrol + TPT 2-Pole SVF Envelope Filter across `n` samples.
///
/// `params_14` order: `[f_hp_sc, f_lp_sc, sens_0, sens_exp, tau_atk_s, tau_rel_s, tau_opt_s,
/// cv_pow, f_min_hz, f_max_hz, q_base, q_cv, gain_base, bp_mix]`.
///
/// # Safety
/// - `x_in` and `y_out` must point to `n` valid `f32` elements.
/// - `params_14` must point to `14` readable `f32` elements.
#[no_mangle]
pub unsafe extern "C" fn cleanroom_eval_envelope_filter(
    x_in: *const f32,
    n: usize,
    sr: f32,
    drive_knob: f32,
    peak_q_scale: f32,
    slow_mult: f32,
    sweep_down: i32,
    params_14: *const f32,
    y_out: *mut f32,
) -> i32 {
    if x_in.is_null() || params_14.is_null() || y_out.is_null() || n == 0 {
        return -1;
    }
    let _guard = DenormalGuard::new();
    let xs = slice::from_raw_parts(x_in, n);
    let ys = slice::from_raw_parts_mut(y_out, n);
    let p = slice::from_raw_parts(params_14, 14);

    let mut eng = envelope_filter::EnvelopeFilterEngine::new(sr);
    eng.drive_knob = drive_knob;
    eng.peak_q_scale = peak_q_scale;
    eng.slow_mult = slow_mult;
    eng.params = envelope_filter::EnvelopeFilterParams {
        f_hp_sc: p[0],
        f_lp_sc: p[1],
        sens_0: p[2],
        sens_exp: p[3],
        tau_atk_s: p[4],
        tau_rel_s: p[5],
        tau_opt_s: p[6],
        cv_pow: p[7],
        f_min_hz: p[8],
        f_max_hz: p[9],
        q_base: p[10],
        q_cv: p[11],
        gain_base: p[12],
        bp_mix: p[13],
        sweep_down: sweep_down != 0,
    };

    for i in 0..n {
        ys[i] = eng.process_sample(xs[i]);
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::amp_core::sat_bloom_bounded_sample;
    use crate::cab_grid::{CabGridLayout, CAB_IR_LEN};
    use crate::nam_capture::{generate_nam_stimulus, NAM_TRACE_LEN, SINGLE_COIL_DI_PEAK};
    use crate::simd_ops::fast_tanh;

    #[test]
    fn test_fast_tanh_accuracy_and_bounds() {
        for i in -500..=500 {
            let x = (i as f32) * 0.02;
            let exact = x.tanh();
            let approx = fast_tanh(x);
            assert!(approx.is_finite());
            assert!((-1.0..=1.0).contains(&approx));
            if x.abs() <= 3.0 {
                assert!(
                    (exact - approx).abs() < 2e-6,
                    "x={x} exact={exact} approx={approx}"
                );
            }
        }
    }

    #[test]
    fn test_sat_bloom_unit_small_signal_slope_and_bounded_rail() {
        let cfg = AmpCoreConfig::default();
        let eps = 1e-4_f32;
        let slope = (sat_bloom_bounded_sample(eps, &cfg) - sat_bloom_bounded_sample(-eps, &cfg))
            / (2.0 * eps);
        assert!(
            (slope - 1.0).abs() < 1e-4,
            "Small-signal slope must be strictly 1.0, got {slope}"
        );

        // Extreme drive must asymptote cleanly to rails without exploding
        let hi_pos = sat_bloom_bounded_sample(40.0, &cfg);
        let hi_neg = sat_bloom_bounded_sample(-40.0, &cfg);
        assert!((hi_pos - cfg.cp0).abs() < 1e-3);
        assert!((hi_neg + cfg.cn0).abs() < 1e-3);
    }

    #[test]
    fn test_amp_zero_slope_features_vanish_at_small_signal() {
        let cfg = AmpCoreConfig::default();
        let mut state = AmpCoreState::default();
        let mut phi = [0.0_f32; AMP_CHANNELS];
        let eps = 1e-4_f32;
        for _ in 0..64 {
            let (s0, _, _) = eval_amp_features_sample(eps, 48_000.0, &cfg, &mut state, &mut phi);
            assert!(s0.is_finite());
        }
        let max_phi = phi.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
        assert!(
            max_phi < 1e-5,
            "Zero-slope features must vanish quadratically/cubically at 1e-4, got {max_phi}"
        );
    }

    #[test]
    fn test_cepstral_identity_and_flat_boost() {
        let freqs = log_freqs_64();
        let mut base = [0.0_f32; 256];
        base[0] = 1.0;

        // 0 dB delta reproduces exact Dirac impulse
        let zero_db = [0.0_f32; NUM_FREQS];
        let mut out_id = [0.0_f32; 256];
        synthesize_min_phase_ir(&base, &freqs, &zero_db, 48_000.0, 1.0, 16, &mut out_id);
        assert!((out_id[0] - 1.0).abs() < 1e-4);

        // +6.0206 dB flat boost doubles impulse amplitude
        let boost_db = [6.0206_f32; NUM_FREQS];
        let mut out_boost = [0.0_f32; 256];
        synthesize_min_phase_ir(&base, &freqs, &boost_db, 48_000.0, 1.0, 16, &mut out_boost);
        assert!((out_boost[0] - 2.0).abs() < 2e-3);
    }

    #[test]
    fn test_scalable_cab_quantized_roundtrip() {
        let layout = CabGridLayout {
            num_cabs: 1,
            num_mics: 1,
            n_pos: 2,
            n_dist: 2,
            n_angle: 1,
        };
        let n_irs = layout.total_irs();
        let mut bytes = vec![0u8; n_irs * CAB_IR_LEN * 2];
        let scales = vec![0.5_f32; n_irs];
        // Write Dirac peak (+32767) at tap 0 of every IR
        for ir in 0..n_irs {
            let off = ir * CAB_IR_LEN * 2;
            let q: i16 = 32767;
            bytes[off] = (q as u16 & 0xFF) as u8;
            bytes[off + 1] = ((q as u16 >> 8) & 0xFF) as u8;
        }
        let mut cab = ScalableCabEngine::<2>::new();
        cab.level_match = true;
        cab.slots[0].active = true;
        cab.slots[0].level = 1.0;
        cab.slots[0].pan = 0.5;
        cab.slots[1].active = true;
        cab.slots[1].level = 1.0;
        cab.slots[1].pan = 0.5;
        cab.rebuild_irs(&layout, &bytes, &scales);
        let (l, r) = cab.process_sample(1.0);
        assert!((l - 0.5).abs() < 1e-3);
        assert!((r - 0.5).abs() < 1e-3);
    }

    #[test]
    fn test_nam_stimulus_bounds_and_calibration() {
        let mut stim = [0.0_f32; NAM_TRACE_LEN];
        generate_nam_stimulus(&mut stim);
        let peak = stim.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
        assert!(
            peak > 0.10 && peak <= SINGLE_COIL_DI_PEAK * 1.15,
            "peak={peak}"
        );
        assert!(stim.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_starter_rig_zero_nan_processing() {
        let mut rig = StarterRigEngine::new(48_000.0);
        rig.overdrive.active = true;
        rig.amp.enabled = true;
        rig.cab.active = true;
        let inp: Vec<f32> = (0..1024)
            .map(|n| 0.15 * ((n as f32) * 0.05).sin())
            .collect();
        let mut out_l = vec![0.0_f32; 1024];
        let mut out_r = vec![0.0_f32; 1024];
        rig.process_mono_to_stereo(&inp, &mut out_l, &mut out_r);
        assert!(out_l.iter().all(|v| v.is_finite()));
        assert!(out_r.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_time_mod_dynamics_engines_zero_nan() {
        use crate::delay_bbd::DelayBbdEngine;
        use crate::envelope_filter::{EnvelopeFilterEngine, StudioCompressorEngine};
        use crate::harmonic_trem::HarmonicTremoloEngine;
        use crate::spring_reverb::SpringReverbEngine;

        let mut rev = SpringReverbEngine::new(256, true);
        let mut dly = DelayBbdEngine::new(48_000.0, 500.0);
        dly.delay_samples = 240.0;
        dly.loop_extra_delay = 16;
        dly.bbd_sat_enabled = true;
        let mut trem = HarmonicTremoloEngine::new(48_000.0);
        trem.c0_mean = 0.7;
        trem.c0_amp = 0.3;
        trem.tube_sat_enabled = true;
        let mut envf = EnvelopeFilterEngine::new(48_000.0);
        let mut comp = StudioCompressorEngine::new(48_000.0);

        // Verify soft-knee static compressor curve monotonicity
        assert!(comp.static_gain_reduction_db(-40.0).abs() < 1e-6);
        assert!(comp.static_gain_reduction_db(0.0) < -10.0);

        for n in 0..2048 {
            let x = 0.25 * ((n as f32) * 0.04).sin();
            let y1 = rev.process_sample(x);
            let y2 = dly.process_sample(x);
            let y3 = trem.process_sample(x);
            let y4 = envf.process_sample(x);
            let y5 = comp.process_sample(x);
            assert!(
                y1.is_finite()
                    && y2.is_finite()
                    && y3.is_finite()
                    && y4.is_finite()
                    && y5.is_finite()
            );
        }
    }
}
