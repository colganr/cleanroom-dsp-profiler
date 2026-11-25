//! Dynamic Vactrol + 2-Pole State-Variable Envelope Filter (`EnvelopeFilterEngine`)
//! and Soft-Knee Feed-Forward Dynamics Compressor (`StudioCompressorEngine`).
//!
//! Both engines are 100% zero-allocation in `process_sample()` and map directly onto the
//! clean-room parameter-identification routines in `profiler/amp_profiler/time_mod_fx.py`.

use std::f32::consts::PI;

/// Calibrated physical parameters for a Vactrol + 2-Pole State-Variable Envelope Filter.
#[derive(Clone, Copy, Debug)]
pub struct EnvelopeFilterParams {
    pub f_hp_sc: f32,   // Sidechain high-pass cutoff (Hz)
    pub f_lp_sc: f32,   // Sidechain low-pass cutoff (Hz)
    pub sens_0: f32,    // Base detector sensitivity gain
    pub sens_exp: f32,  // Gain-knob sensitivity exponent
    pub tau_atk_s: f32, // Rectifier attack time constant (seconds)
    pub tau_rel_s: f32, // Rectifier release time constant (seconds)
    pub tau_opt_s: f32, // Optical vactrol lag time constant (seconds)
    pub cv_pow: f32,    // Control-voltage nonlinearity exponent
    pub f_min_hz: f32,  // Resting filter cutoff (Hz)
    pub f_max_hz: f32,  // Maximum swept cutoff (Hz)
    pub q_base: f32,    // Resting SVF resonance Q
    pub q_cv: f32,      // Dynamic Q shift vs control voltage
    pub gain_base: f32, // Output trim
    pub bp_mix: f32,    // 0.0 = pure LP, 1.0 = pure BP
    pub sweep_down: bool,
}

impl Default for EnvelopeFilterParams {
    fn default() -> Self {
        Self {
            f_hp_sc: 65.0,
            f_lp_sc: 450.0,
            sens_0: 22.0,
            sens_exp: 2.8,
            tau_atk_s: 0.008,
            tau_rel_s: 0.240,
            tau_opt_s: 0.014,
            cv_pow: 2.2,
            f_min_hz: 210.0,
            f_max_hz: 1650.0,
            q_base: 4.5,
            q_cv: 0.5,
            gain_base: 1.0,
            bp_mix: 0.25,
            sweep_down: false,
        }
    }
}

/// Zero-allocation Vactrol + TPT/ZDF 2-Pole State-Variable Envelope Filter.
#[derive(Clone, Copy)]
pub struct EnvelopeFilterEngine {
    pub active: bool,
    pub sr: f32,
    pub drive_knob: f32, // 0.0 .. 1.0
    pub peak_q_scale: f32,
    pub slow_mult: f32, // 1.0 = Fast response, > 1.0 = Slow optical response
    pub params: EnvelopeFilterParams,

    sc_hp_x1: f32,
    sc_hp_y1: f32,
    sc_lp_y1: f32,
    env_rect: f32,
    env_opt: f32,
    ic1eq: f32,
    ic2eq: f32,
}

impl Default for EnvelopeFilterEngine {
    fn default() -> Self {
        Self::new(48_000.0)
    }
}

impl EnvelopeFilterEngine {
    pub fn new(sr: f32) -> Self {
        Self {
            active: true,
            sr,
            drive_knob: 0.60,
            peak_q_scale: 1.0,
            slow_mult: 1.0,
            params: EnvelopeFilterParams::default(),
            sc_hp_x1: 0.0,
            sc_hp_y1: 0.0,
            sc_lp_y1: 0.0,
            env_rect: 0.0,
            env_opt: 0.0,
            ic1eq: 0.0,
            ic2eq: 0.0,
        }
    }

    pub fn reset(&mut self) {
        self.sc_hp_x1 = 0.0;
        self.sc_hp_y1 = 0.0;
        self.sc_lp_y1 = 0.0;
        self.env_rect = 0.0;
        self.env_opt = 0.0;
        self.ic1eq = 0.0;
        self.ic2eq = 0.0;
    }

    /// Processes one sample through the sidechain vactrol detector and TPT 2-pole SVF.
    #[inline(always)]
    pub fn process_sample(&mut self, x_in: f32) -> f32 {
        if !self.active {
            return x_in;
        }
        let sr = self.sr.max(8_000.0);
        let p = &self.params;

        // 1. Sidechain band-pass shaping
        let a_hp = (-2.0 * PI * p.f_hp_sc / sr).exp();
        let sc_hp = a_hp * (self.sc_hp_y1 + x_in - self.sc_hp_x1);
        self.sc_hp_x1 = x_in;
        self.sc_hp_y1 = sc_hp;

        let a_lp = (-2.0 * PI * p.f_lp_sc / sr).exp();
        self.sc_lp_y1 = a_lp * self.sc_lp_y1 + (1.0 - a_lp) * sc_hp;

        // 2. Attack / Release rectifier + Optical Vactrol lag
        let slow = self.slow_mult.max(0.25);
        let a_atk = (-1.0 / (p.tau_atk_s.max(1e-4) * slow * sr)).exp();
        let a_rel = (-1.0 / (p.tau_rel_s.max(1e-3) * slow * sr)).exp();
        let a_opt = (-1.0 / (p.tau_opt_s.max(1e-4) * slow * sr)).exp();

        let rect = self.sc_lp_y1.abs();
        let coeff = if rect > self.env_rect { a_atk } else { a_rel };
        self.env_rect = coeff * self.env_rect + (1.0 - coeff) * rect;
        self.env_opt = a_opt * self.env_opt + (1.0 - a_opt) * self.env_rect;

        // 3. Control-voltage compression into [0, 1] sweep range
        let sens = p.sens_0 * (0.15 + 0.85 * self.drive_knob.clamp(0.0, 1.0)).powf(p.sens_exp);
        let raw_cv = (self.env_opt * sens).clamp(0.0, 10.0);
        let cv = (raw_cv / (1.0 + raw_cv)).powf(p.cv_pow.max(0.25));
        let sweep = if p.sweep_down { 1.0 - cv } else { cv };

        let fc = (p.f_min_hz + (p.f_max_hz - p.f_min_hz) * sweep).clamp(40.0, sr * 0.42);
        let q = ((p.q_base + p.q_cv * cv) * self.peak_q_scale).clamp(0.6, 28.0);

        // 4. Unconditionally stable Cytomic TPT/ZDF 2-pole State-Variable Filter
        let g = (PI * fc / sr).tan();
        let k = 1.0 / q;
        let a1 = 1.0 / (1.0 + g * (g + k));
        let a2 = g * a1;
        let a3 = g * a2;

        let v3 = x_in - self.ic2eq;
        let v1 = a1 * self.ic1eq + a2 * v3;
        let v2 = self.ic2eq + a2 * self.ic1eq + a3 * v3;
        self.ic1eq = 2.0 * v1 - self.ic1eq;
        self.ic2eq = 2.0 * v2 - self.ic2eq;

        let lp = v2;
        let bp = v1;
        let mix_bp = p.bp_mix.clamp(0.0, 1.0);
        ((1.0 - mix_bp) * lp + mix_bp * bp) * p.gain_base
    }
}

/// Zero-allocation Soft-Knee Feed-Forward Studio Dynamics Compressor (`StudioCompressorEngine`).
#[derive(Clone, Copy, Debug)]
pub struct StudioCompressorEngine {
    pub active: bool,
    pub sr: f32,
    pub threshold_db: f32,
    pub ratio: f32,
    pub knee_db: f32,
    pub attack_ms: f32,
    pub release_ms: f32,
    pub makeup_db: f32,
    pub mix: f32,

    gr_db_state: f32,
}

impl Default for StudioCompressorEngine {
    fn default() -> Self {
        Self::new(48_000.0)
    }
}

impl StudioCompressorEngine {
    pub fn new(sr: f32) -> Self {
        Self {
            active: true,
            sr,
            threshold_db: -18.0,
            ratio: 4.0,
            knee_db: 6.0,
            attack_ms: 10.0,
            release_ms: 120.0,
            makeup_db: 0.0,
            mix: 1.0,
            gr_db_state: 0.0,
        }
    }

    pub fn reset(&mut self) {
        self.gr_db_state = 0.0;
    }

    /// Computes static soft-knee gain reduction (in dB, `<= 0.0`) for input level `level_db`.
    #[inline(always)]
    pub fn static_gain_reduction_db(&self, level_db: f32) -> f32 {
        let ratio = self.ratio.max(1.0);
        let knee = self.knee_db.max(0.0);
        let over = level_db - self.threshold_db;
        let slope = 1.0 / ratio - 1.0;

        if knee <= 1e-4 {
            if over <= 0.0 {
                0.0
            } else {
                slope * over
            }
        } else if 2.0 * over <= -knee {
            0.0
        } else if 2.0 * over >= knee {
            slope * over
        } else {
            let x = over + 0.5 * knee;
            slope * (x * x) / (2.0 * knee)
        }
    }

    /// Processes a single mono sample with decoupled attack/release gain-reduction smoothing.
    #[inline(always)]
    pub fn process_sample(&mut self, x_in: f32) -> f32 {
        if !self.active {
            return x_in;
        }
        let sr = self.sr.max(8_000.0);
        let level_db = 20.0 * (x_in.abs().max(1e-7)).log10();
        let target_gr = self.static_gain_reduction_db(level_db);

        let a_atk = (-1.0 / ((self.attack_ms.max(0.1) * 0.001) * sr)).exp();
        let a_rel = (-1.0 / ((self.release_ms.max(1.0) * 0.001) * sr)).exp();
        // More negative target_gr => attacking (compressing); less negative => releasing
        let coeff = if target_gr < self.gr_db_state {
            a_atk
        } else {
            a_rel
        };
        self.gr_db_state = coeff * self.gr_db_state + (1.0 - coeff) * target_gr;

        let gain = 10.0_f32.powf((self.gr_db_state + self.makeup_db) / 20.0);
        let wet = x_in * gain;
        let m = self.mix.clamp(0.0, 1.0);
        (1.0 - m) * x_in + m * wet
    }
}
