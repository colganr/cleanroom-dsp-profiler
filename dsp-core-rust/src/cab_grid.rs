//! Scalable N-Slot 2D/3D Speaker Cabinet IR Grid Engine (`ScalableCabEngine<const N_SLOTS: usize>`).
//!
//! - Supports arbitrary microphone slot counts (`N_SLOTS = 1, 2, 4, 8, ...`).
//! - Stores `1024`-tap `48 kHz` anchor IRs quantized as little-endian `i16` + per-IR `f32` peak
//!   scale (`-96.3 dB` quantization floor, cutting embedded weight size by 50%).
//! - Performs real-time 2D bilinear (`Position x Distance`) or 3D trilinear (`Axis x Distance x Angle`)
//!   interpolation, per-slot delay (`0..48` samples), polarity inversion, constant-power stereo
//!   panning, optional `1 / N_active` level matching, and SIMD stereo FIR convolution.

use crate::simd_ops::dot_stereo_f32;
use std::f32::consts::FRAC_PI_2;

pub const CAB_IR_LEN: usize = 1024;

/// Metadata describing the layout of a quantized cabinet IR grid buffer.
#[derive(Clone, Copy, Debug)]
pub struct CabGridLayout {
    pub num_cabs: usize,
    pub num_mics: usize,
    pub n_pos: usize,
    pub n_dist: usize,
    pub n_angle: usize, // Set to 1 for 2D (Position x Distance) grids
}

impl CabGridLayout {
    pub const fn total_irs(&self) -> usize {
        self.num_cabs * self.num_mics * self.n_pos * self.n_dist * self.n_angle
    }

    #[inline(always)]
    pub const fn ir_index(
        &self,
        cab: usize,
        mic: usize,
        pos_i: usize,
        dist_i: usize,
        ang_i: usize,
    ) -> usize {
        ((((cab * self.num_mics + mic) * self.n_pos + pos_i) * self.n_dist + dist_i) * self.n_angle)
            + ang_i
    }
}

/// Decodes one sample from a little-endian `i16` quantized IR bank.
#[inline(always)]
pub fn decode_i16_ir_sample(
    grid_bytes: &[u8],
    scales: &[f32],
    ir_idx: usize,
    sample_idx: usize,
) -> f32 {
    let byte_off = (ir_idx * CAB_IR_LEN + sample_idx) * 2;
    if byte_off + 1 >= grid_bytes.len() || ir_idx >= scales.len() {
        return 0.0;
    }
    let lo = grid_bytes[byte_off] as u16;
    let hi = (grid_bytes[byte_off + 1] as u16) << 8;
    let q = (lo | hi) as i16;
    (q as f32) * (scales[ir_idx] / 32767.0_f32)
}

/// Per-microphone slot parameters.
#[derive(Clone, Copy, Debug)]
pub struct CabSlotParams {
    pub active: bool,
    pub cab_index: usize,
    pub mic_index: usize,
    pub polarity_invert: bool,
    pub position: f32, // 0.0 .. 1.0
    pub distance: f32, // 0.0 .. 1.0
    pub angle: f32,    // 0.0 .. 1.0 (ignored when n_angle == 1)
    pub delay: f32,    // 0.0 .. 1.0 -> 0..48 samples at 48 kHz
    pub pan: f32,      // 0.0 (L) .. 0.5 (C) .. 1.0 (R)
    pub level: f32,    // linear gain (1.0 = 0 dB)
}

impl Default for CabSlotParams {
    fn default() -> Self {
        Self {
            active: true,
            cab_index: 0,
            mic_index: 0,
            polarity_invert: false,
            position: 0.25,
            distance: 0.0,
            angle: 0.0,
            delay: 0.0,
            pan: 0.5,
            level: 1.0,
        }
    }
}

/// Zero-allocation scalable N-slot stereo speaker cabinet convolution engine.
pub struct ScalableCabEngine<const N_SLOTS: usize> {
    pub active: bool,
    pub level_match: bool,
    pub slots: [CabSlotParams; N_SLOTS],
    pub combined_ir_l: [f32; CAB_IR_LEN],
    pub combined_ir_r: [f32; CAB_IR_LEN],
    ir_hist: [f32; CAB_IR_LEN * 2],
    ir_wp: usize,
}

impl<const N_SLOTS: usize> Default for ScalableCabEngine<N_SLOTS> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N_SLOTS: usize> ScalableCabEngine<N_SLOTS> {
    pub fn new() -> Self {
        let mut combined_l = [0.0_f32; CAB_IR_LEN];
        let mut combined_r = [0.0_f32; CAB_IR_LEN];
        combined_l[0] = 1.0;
        combined_r[0] = 1.0;
        Self {
            active: true,
            level_match: false,
            slots: [CabSlotParams::default(); N_SLOTS],
            combined_ir_l: combined_l,
            combined_ir_r: combined_r,
            ir_hist: [0.0; CAB_IR_LEN * 2],
            ir_wp: 0,
        }
    }

    pub fn reset(&mut self) {
        self.ir_hist = [0.0; CAB_IR_LEN * 2];
        self.ir_wp = 0;
    }

    /// Re-synthesizes `combined_ir_l` and `combined_ir_r` from all active microphone slots.
    pub fn rebuild_irs(&mut self, layout: &CabGridLayout, grid_bytes: &[u8], scales: &[f32]) {
        self.combined_ir_l = [0.0_f32; CAB_IR_LEN];
        self.combined_ir_r = [0.0_f32; CAB_IR_LEN];

        let mut active_count = 0usize;
        let mut slot_ir = [0.0_f32; CAB_IR_LEN];

        for slot in &self.slots {
            if !slot.active || slot.level <= 1e-5 {
                continue;
            }
            active_count += 1;

            let cab = slot.cab_index.min(layout.num_cabs.saturating_sub(1));
            let mic = slot.mic_index.min(layout.num_mics.saturating_sub(1));

            let (p0, p1, wp) = axis_interp(slot.position, layout.n_pos);
            let (d0, d1, wd) = axis_interp(slot.distance, layout.n_dist);
            let (a0, a1, wa) = axis_interp(slot.angle, layout.n_angle);

            let i000 = layout.ir_index(cab, mic, p0, d0, a0);
            let i001 = layout.ir_index(cab, mic, p0, d0, a1);
            let i010 = layout.ir_index(cab, mic, p0, d1, a0);
            let i011 = layout.ir_index(cab, mic, p0, d1, a1);
            let i100 = layout.ir_index(cab, mic, p1, d0, a0);
            let i101 = layout.ir_index(cab, mic, p1, d0, a1);
            let i110 = layout.ir_index(cab, mic, p1, d1, a0);
            let i111 = layout.ir_index(cab, mic, p1, d1, a1);

            let c000 = (1.0 - wp) * (1.0 - wd) * (1.0 - wa);
            let c001 = (1.0 - wp) * (1.0 - wd) * wa;
            let c010 = (1.0 - wp) * wd * (1.0 - wa);
            let c011 = (1.0 - wp) * wd * wa;
            let c100 = wp * (1.0 - wd) * (1.0 - wa);
            let c101 = wp * (1.0 - wd) * wa;
            let c110 = wp * wd * (1.0 - wa);
            let c111 = wp * wd * wa;

            let pol = if slot.polarity_invert {
                -1.0_f32
            } else {
                1.0_f32
            };
            let gain = slot.level * pol;
            let delay_samples = (slot.delay.clamp(0.0, 1.0) * 48.0).round() as usize;

            for (n, dst) in slot_ir.iter_mut().enumerate() {
                if n < delay_samples {
                    *dst = 0.0;
                } else {
                    let s = n - delay_samples;
                    let v = c000 * decode_i16_ir_sample(grid_bytes, scales, i000, s)
                        + c001 * decode_i16_ir_sample(grid_bytes, scales, i001, s)
                        + c010 * decode_i16_ir_sample(grid_bytes, scales, i010, s)
                        + c011 * decode_i16_ir_sample(grid_bytes, scales, i011, s)
                        + c100 * decode_i16_ir_sample(grid_bytes, scales, i100, s)
                        + c101 * decode_i16_ir_sample(grid_bytes, scales, i101, s)
                        + c110 * decode_i16_ir_sample(grid_bytes, scales, i110, s)
                        + c111 * decode_i16_ir_sample(grid_bytes, scales, i111, s);
                    *dst = v * gain;
                }
            }

            // Constant-power stereo pan law (normalized so center = 1.0 on L & R)
            let ang = slot.pan.clamp(0.0, 1.0) * FRAC_PI_2;
            let gl = ang.cos() * std::f32::consts::SQRT_2;
            let gr = ang.sin() * std::f32::consts::SQRT_2;

            for (n, &sv) in slot_ir.iter().enumerate() {
                self.combined_ir_l[n] += sv * gl;
                self.combined_ir_r[n] += sv * gr;
            }
        }

        if self.level_match && active_count > 1 {
            let norm = 1.0 / (active_count as f32);
            for n in 0..CAB_IR_LEN {
                self.combined_ir_l[n] *= norm;
                self.combined_ir_r[n] *= norm;
            }
        }
    }

    /// Processes one mono input sample into stereo `(out_l, out_r)` via SIMD FIR convolution.
    #[inline(always)]
    pub fn process_sample(&mut self, x_in: f32) -> (f32, f32) {
        if !self.active {
            return (x_in, x_in);
        }
        let wp = (self.ir_wp + CAB_IR_LEN - 1) & (CAB_IR_LEN - 1);
        self.ir_wp = wp;
        self.ir_hist[wp] = x_in;
        self.ir_hist[wp + CAB_IR_LEN] = x_in;
        dot_stereo_f32(
            &self.ir_hist[wp..wp + CAB_IR_LEN],
            &self.combined_ir_l,
            &self.combined_ir_r,
        )
    }
}

#[inline(always)]
fn axis_interp(val: f32, knots: usize) -> (usize, usize, f32) {
    if knots <= 1 {
        return (0, 0, 0.0);
    }
    let pos = val.clamp(0.0, 1.0) * ((knots - 1) as f32);
    let i0 = (pos as usize).min(knots - 2);
    let frac = pos - (i0 as f32);
    (i0, i0 + 1, frac)
}
