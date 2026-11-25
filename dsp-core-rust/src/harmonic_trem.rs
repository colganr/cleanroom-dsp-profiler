//! SVD Dual-Crossover Harmonic / Optical Tube Tremolo Engine (`HarmonicTremoloEngine`).
//!
//! Represents LFO-modulated frequency-crossover tremolos (e.g. Brownface Harmonic Tremolo,
//! optical tremolo, and dual-band vibrato/modulation circuits) with:
//! - Dual `128`-tap SIMD-accelerated **SVD Crossover Basis FIRs** (`psi_0`, `psi_1` capturing
//!   `>99.99%` of instantaneous LFO crossover variance).
//! - 21-point calibrated LFO frequency law (`freq_hz_21`) and modulation envelopes
//!   (`c0_mean`, `c0_amp`, `c1_mean`, `c1_amp`).
//! - Optional zero-slope tube saturation residual (`15` channels: `3` knee ratios x `5` causal
//!   dilations `[0, 1, 2, 4, 8]`) calibrated cleanly at `Intensity = 0.0` (phase-locked).

use crate::amp_core::soft_sat_sample;
use crate::simd_ops::dot_stereo_f32;
use std::f32::consts::TAU;

pub const TREM_FIR_LEN: usize = 128;
pub const TREM_SAT_CHANNELS: usize = 15; // 3 knee ratios x 5 dilations [0, 1, 2, 4, 8]
const TREM_RING_LEN: usize = 16;

const TREM_RATIOS: [(f32, f32); 3] = [(0.60, 0.60), (0.80, 1.25), (2.00, 2.00)];

/// Zero-allocation SVD Dual-Crossover Harmonic / Optical Tremolo Engine.
pub struct HarmonicTremoloEngine {
    pub active: bool,
    pub sr: f32,
    pub lfo_freq_hz: f32,
    pub lfo_phase: f32, // [0.0, 1.0)
    pub c0_mean: f32,
    pub c0_amp: f32,
    pub c1_mean: f32,
    pub c1_amp: f32,

    pub psi_0: [f32; TREM_FIR_LEN],
    pub psi_1: [f32; TREM_FIR_LEN],

    pub tube_sat_enabled: bool,
    pub sat_cp0: f32,
    pub sat_cn0: f32,
    pub sat_knee: f32,
    pub sat_weights: [f32; TREM_SAT_CHANNELS],

    hist: [f32; TREM_FIR_LEN * 2],
    wp: usize,
    sat_ring: [[f32; TREM_RING_LEN]; 3],
    sat_wp: usize,
}

impl Default for HarmonicTremoloEngine {
    fn default() -> Self {
        Self::new(48_000.0)
    }
}

impl HarmonicTremoloEngine {
    pub fn new(sr: f32) -> Self {
        let mut psi_0 = [0.0_f32; TREM_FIR_LEN];
        psi_0[0] = 1.0;
        Self {
            active: true,
            sr,
            lfo_freq_hz: 3.5,
            lfo_phase: 0.0,
            c0_mean: 1.0,
            c0_amp: 0.0,
            c1_mean: 0.0,
            c1_amp: 0.0,
            psi_0,
            psi_1: [0.0; TREM_FIR_LEN],
            tube_sat_enabled: false,
            sat_cp0: 0.35,
            sat_cn0: 0.35,
            sat_knee: 2.2,
            sat_weights: [0.0; TREM_SAT_CHANNELS],
            hist: [0.0; TREM_FIR_LEN * 2],
            wp: 0,
            sat_ring: [[0.0; TREM_RING_LEN]; 3],
            sat_wp: 0,
        }
    }

    pub fn reset(&mut self) {
        self.hist = [0.0; TREM_FIR_LEN * 2];
        self.wp = 0;
        self.lfo_phase = 0.0;
        self.sat_ring = [[0.0; TREM_RING_LEN]; 3];
        self.sat_wp = 0;
    }

    /// Interpolates LFO speed and SVD crossover modulation curves from 21-point profiled tables.
    #[allow(clippy::too_many_arguments)]
    pub fn set_knobs_from_tables(
        &mut self,
        speed: f32,
        intensity: f32,
        freq_hz_21: &[f32; 21],
        c0_mean_21: &[f32; 21],
        c0_amp_21: &[f32; 21],
        c1_mean_21: &[f32; 21],
        c1_amp_21: &[f32; 21],
    ) {
        let interp = |tbl: &[f32; 21], x: f32| -> f32 {
            let p = x.clamp(0.0, 1.0) * 20.0;
            let i = (p as usize).min(19);
            let f = p - (i as f32);
            (1.0 - f) * tbl[i] + f * tbl[i + 1]
        };
        self.lfo_freq_hz = interp(freq_hz_21, speed).max(0.05);
        self.c0_mean = interp(c0_mean_21, intensity);
        self.c0_amp = interp(c0_amp_21, intensity);
        self.c1_mean = interp(c1_mean_21, intensity);
        self.c1_amp = interp(c1_amp_21, intensity);
    }

    /// Processes a single mono sample using simultaneous SIMD dual-basis convolution (`dot_stereo_f32`).
    #[inline(always)]
    pub fn process_sample(&mut self, x_in: f32) -> f32 {
        if !self.active {
            return x_in;
        }

        let wp = (self.wp + TREM_FIR_LEN - 1) & (TREM_FIR_LEN - 1);
        self.wp = wp;
        self.hist[wp] = x_in;
        self.hist[wp + TREM_FIR_LEN] = x_in;

        // Evaluate both 128-tap SVD crossover filters simultaneously with one SIMD pass
        let (y0, y1) = dot_stereo_f32(&self.hist[wp..wp + TREM_FIR_LEN], &self.psi_0, &self.psi_1);

        let lfo = (self.lfo_phase * TAU).sin();
        self.lfo_phase += self.lfo_freq_hz / self.sr.max(8_000.0);
        if self.lfo_phase >= 1.0 {
            self.lfo_phase -= 1.0;
        }

        let u = (self.c0_mean + self.c0_amp * lfo) * y0 + (self.c1_mean + self.c1_amp * lfo) * y1;

        if !self.tube_sat_enabled {
            return u;
        }

        let inv_k = 1.0_f32 / self.sat_knee;
        let sat0 = soft_sat_sample(u, self.sat_cp0, self.sat_cn0, self.sat_knee, inv_k);
        let swp = self.sat_wp;
        let i0 = swp;
        let i1 = (swp + TREM_RING_LEN - 1) & (TREM_RING_LEN - 1);
        let i2 = (swp + TREM_RING_LEN - 2) & (TREM_RING_LEN - 1);
        let i4 = (swp + TREM_RING_LEN - 4) & (TREM_RING_LEN - 1);
        let i8 = (swp + TREM_RING_LEN - 8) & (TREM_RING_LEN - 1);

        let mut res = 0.0_f32;
        for (b, &(a, c)) in TREM_RATIOS.iter().enumerate() {
            let rb =
                soft_sat_sample(u, self.sat_cp0 * a, self.sat_cn0 * c, self.sat_knee, inv_k) - sat0;
            self.sat_ring[b][swp] = rb;
            let base = b * 5;
            res += self.sat_weights[base] * self.sat_ring[b][i0]
                + self.sat_weights[base + 1] * self.sat_ring[b][i1]
                + self.sat_weights[base + 2] * self.sat_ring[b][i2]
                + self.sat_weights[base + 3] * self.sat_ring[b][i4]
                + self.sat_weights[base + 4] * self.sat_ring[b][i8];
        }
        self.sat_wp = (swp + 1) & (TREM_RING_LEN - 1);
        sat0 + res
    }
}
