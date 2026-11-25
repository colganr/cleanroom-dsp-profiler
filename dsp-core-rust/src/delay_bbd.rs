//! Hybrid Parametric + Zero-Slope Analog BBD & Studio Digital Delay Engine (`DelayBbdEngine`).
//!
//! Captures both clean digital delays (with echo hardness/tone loop filters) and dark analog
//! Bucket-Brigade Device (BBD) delays using:
//! - Fractional Hermite-interpolated delay line (`up to 2.0s @ 48 kHz`, pre-allocated in `new()`).
//! - Exact **Extra Loop Delay Offset** (`loop_extra_delay`, e.g., 25-33 samples between R1 & R2).
//! - `64`-tap SIMD-accelerated Base Wet / Anti-Alias Reconstruction FIR (`wet_base_fir`).
//! - `64`-tap SIMD-accelerated Feedback / Hardness / Clock-Aperture Loop FIR (`loop_fir`).
//! - Optional **30-Channel Zero-Slope BBD Bucket-Rail Saturation Core** (`6` knee ratios x `5`
//!   causal dilations `[0, 1, 2, 4, 8]`) preserving exact small-signal loop transfer response.

use crate::amp_core::soft_sat_sample;
use crate::simd_ops::dot_f32;

pub const DELAY_FIR_LEN: usize = 64;
pub const BBD_SAT_BRANCHES: usize = 6;
pub const BBD_SAT_CHANNELS: usize = 30; // 6 branches x 5 dilations [0, 1, 2, 4, 8]
const BBD_RING_LEN: usize = 16;

const BBD_BRANCH_RATIOS: [(f32, f32); BBD_SAT_BRANCHES] = [
    (0.55, 0.55),
    (0.85, 0.85),
    (1.35, 1.35),
    (0.75, 1.25),
    (2.20, 2.20),
    (4.00, 4.00),
];

/// Zero-allocation Analog BBD & Digital Delay Engine.
pub struct DelayBbdEngine {
    pub active: bool,
    pub sr: f32,
    pub delay_samples: f32,
    pub loop_extra_delay: usize,
    pub feedback_gain: f32,
    pub dry_gain: f32,
    pub wet_gain: f32,

    pub bbd_sat_enabled: bool,
    pub bbd_cp0: f32,
    pub bbd_cn0: f32,
    pub bbd_knee: f32,
    pub bbd_weights: [f32; BBD_SAT_CHANNELS],

    pub wet_base_fir: [f32; DELAY_FIR_LEN],
    pub loop_fir: [f32; DELAY_FIR_LEN],

    buf: Vec<f32>,
    buf_mask: usize,
    write_pos: usize,

    wet_hist: [f32; DELAY_FIR_LEN * 2],
    wet_wp: usize,
    loop_hist: [f32; DELAY_FIR_LEN * 2],
    loop_wp: usize,

    sat_ring: [[f32; BBD_RING_LEN]; BBD_SAT_BRANCHES],
    sat_wp: usize,
    prev_fb: f32,
}

impl DelayBbdEngine {
    /// Pre-allocates a power-of-two delay buffer holding at least `max_delay_ms` at `sr`.
    pub fn new(sr: f32, max_delay_ms: f32) -> Self {
        let min_samples = ((sr * max_delay_ms / 1000.0).ceil() as usize).max(2048);
        let cap = min_samples.next_power_of_two();

        let mut wet_base_fir = [0.0_f32; DELAY_FIR_LEN];
        wet_base_fir[0] = 1.0;
        let mut loop_fir = [0.0_f32; DELAY_FIR_LEN];
        loop_fir[0] = 1.0;

        Self {
            active: true,
            sr,
            delay_samples: (0.25 * sr).min((cap - 64) as f32),
            loop_extra_delay: 0,
            feedback_gain: 0.35,
            dry_gain: 1.0,
            wet_gain: 0.5,
            bbd_sat_enabled: false,
            bbd_cp0: 0.268,
            bbd_cn0: 0.268,
            bbd_knee: 2.2,
            bbd_weights: [0.0; BBD_SAT_CHANNELS],
            wet_base_fir,
            loop_fir,
            buf: vec![0.0_f32; cap],
            buf_mask: cap - 1,
            write_pos: 0,
            wet_hist: [0.0; DELAY_FIR_LEN * 2],
            wet_wp: 0,
            loop_hist: [0.0; DELAY_FIR_LEN * 2],
            loop_wp: 0,
            sat_ring: [[0.0; BBD_RING_LEN]; BBD_SAT_BRANCHES],
            sat_wp: 0,
            prev_fb: 0.0,
        }
    }

    pub fn reset(&mut self) {
        self.buf.fill(0.0);
        self.write_pos = 0;
        self.wet_hist = [0.0; DELAY_FIR_LEN * 2];
        self.wet_wp = 0;
        self.loop_hist = [0.0; DELAY_FIR_LEN * 2];
        self.loop_wp = 0;
        self.sat_ring = [[0.0; BBD_RING_LEN]; BBD_SAT_BRANCHES];
        self.sat_wp = 0;
        self.prev_fb = 0.0;
    }

    #[inline(always)]
    fn read_hermite(&self, delay_samps: f32) -> f32 {
        let max_d = (self.buf.len() - 4) as f32;
        let d = delay_samps.clamp(1.0, max_d);
        let d_int = d.floor() as usize;
        let frac = d - (d_int as f32);

        let mask = self.buf_mask;
        let wp = self.write_pos;
        let i0 = wp.wrapping_sub(d_int) & mask;
        let im1 = (i0 + 1) & mask;
        let i1 = i0.wrapping_sub(1) & mask;
        let i2 = i0.wrapping_sub(2) & mask;

        let ym1 = self.buf[im1];
        let y0 = self.buf[i0];
        let y1 = self.buf[i1];
        let y2 = self.buf[i2];

        let c0 = y0;
        let c1 = 0.5 * (y1 - ym1);
        let c2 = ym1 - 2.5 * y0 + 2.0 * y1 - 0.5 * y2;
        let c3 = 0.5 * (y2 - ym1) + 1.5 * (y0 - y1);
        ((c3 * frac + c2) * frac + c1) * frac + c0
    }

    #[inline(always)]
    fn apply_bbd_sat(&mut self, u: f32) -> f32 {
        if !self.bbd_sat_enabled {
            return u;
        }
        let inv_k = 1.0_f32 / self.bbd_knee;
        let sat0 = soft_sat_sample(u, self.bbd_cp0, self.bbd_cn0, self.bbd_knee, inv_k);
        let wp = self.sat_wp;
        let i0 = wp;
        let i1 = (wp + BBD_RING_LEN - 1) & (BBD_RING_LEN - 1);
        let i2 = (wp + BBD_RING_LEN - 2) & (BBD_RING_LEN - 1);
        let i4 = (wp + BBD_RING_LEN - 4) & (BBD_RING_LEN - 1);
        let i8 = (wp + BBD_RING_LEN - 8) & (BBD_RING_LEN - 1);

        let mut res = 0.0_f32;
        for (b, &(a, c)) in BBD_BRANCH_RATIOS.iter().enumerate() {
            let rb =
                soft_sat_sample(u, self.bbd_cp0 * a, self.bbd_cn0 * c, self.bbd_knee, inv_k) - sat0;
            self.sat_ring[b][wp] = rb;
            let base = b * 5;
            res += self.bbd_weights[base] * self.sat_ring[b][i0]
                + self.bbd_weights[base + 1] * self.sat_ring[b][i1]
                + self.bbd_weights[base + 2] * self.sat_ring[b][i2]
                + self.bbd_weights[base + 3] * self.sat_ring[b][i4]
                + self.bbd_weights[base + 4] * self.sat_ring[b][i8];
        }
        self.sat_wp = (wp + 1) & (BBD_RING_LEN - 1);
        sat0 + res
    }

    /// Processes a single mono sample with zero heap allocations.
    #[inline(always)]
    pub fn process_sample(&mut self, x_in: f32) -> f32 {
        if !self.active {
            return x_in;
        }

        // Write input + previous feedback into delay line
        let drive_in = self.apply_bbd_sat(x_in + self.prev_fb);
        let wp = self.write_pos;
        self.buf[wp] = drive_in;

        // Read primary echo tap (R1) + optional extra-offset feedback tap (R2+)
        // Subtract 1 sample from `loop_extra_delay` to account for the 1-sample `prev_fb` causal latch.
        let tap_r1 = self.read_hermite(self.delay_samples);
        let tap_fb = if self.loop_extra_delay <= 1 {
            tap_r1
        } else {
            self.read_hermite(self.delay_samples + ((self.loop_extra_delay - 1) as f32))
        };
        self.write_pos = (wp + 1) & self.buf_mask;

        // 1. Base wet reconstruction filter (SIMD)
        let ww = (self.wet_wp + DELAY_FIR_LEN - 1) & (DELAY_FIR_LEN - 1);
        self.wet_wp = ww;
        self.wet_hist[ww] = tap_r1;
        self.wet_hist[ww + DELAY_FIR_LEN] = tap_r1;
        let wet_out = dot_f32(&self.wet_hist[ww..ww + DELAY_FIR_LEN], &self.wet_base_fir);

        // 2. Echo hardness / BBD clock-aperture feedback filter (SIMD)
        let lw = (self.loop_wp + DELAY_FIR_LEN - 1) & (DELAY_FIR_LEN - 1);
        self.loop_wp = lw;
        self.loop_hist[lw] = tap_fb;
        self.loop_hist[lw + DELAY_FIR_LEN] = tap_fb;
        let fb_filt = dot_f32(&self.loop_hist[lw..lw + DELAY_FIR_LEN], &self.loop_fir);
        self.prev_fb = (fb_filt * self.feedback_gain).clamp(-4.0, 4.0);

        self.dry_gain * x_in + self.wet_gain * wet_out
    }
}
