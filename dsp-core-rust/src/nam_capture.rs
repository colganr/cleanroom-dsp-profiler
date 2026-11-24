//! Deterministic 12,288-Sample (`256 ms @ 48 kHz`) Multi-Regime NAM Capture Stimulus.
//!
//! Calibrated to passive single-coil pickup Hi-Z level (`0.160` peak, `-15.9 dBFS`) so
//! nonlinear amp and pedal chains are excited across clean, edge-of-breakup, and driven
//! operating points without artificially slamming clean headroom.

use std::f32::consts::PI;

pub const NAM_TRACE_LEN: usize = 12_288;
pub const SINGLE_COIL_DI_PEAK: f32 = 0.160;

/// Fills `out` with the deterministic 4-regime NAM conditioning stimulus at `48 kHz`.
pub fn generate_nam_stimulus(out: &mut [f32; NAM_TRACE_LEN]) {
    let sr = 48_000.0_f32;

    // 1. Samples 0..1024: Bipolar calibration impulses + low-end ramp (65 Hz -> 320 Hz)
    for (n, dst) in out.iter_mut().enumerate().take(1024) {
        let t = (n as f32) / sr;
        let frac = (n as f32) / 1024.0;
        let f = 65.0 + 255.0 * frac * frac;
        let env = (PI * frac).sin();
        *dst = 0.045 * env * (2.0 * PI * f * t).sin();
    }
    out[256] += 0.120;
    out[640] -= 0.100;

    // 2. Samples 1024..5120: Logarithmic harmonic sweep (55 Hz -> 5200 Hz) with 3-cycle dynamic swell
    let n2 = 4096usize;
    let f1 = 55.0_f32;
    let f2 = 5200.0_f32;
    let t2_total = (n2 as f32) / sr;
    let l_const = t2_total / (f2 / f1).ln();
    for i in 0..n2 {
        let t = (i as f32) / sr;
        let phase = 2.0 * PI * f1 * l_const * ((t / l_const).exp() - 1.0);
        let frac = (i as f32) / (n2 as f32);
        let amp = 0.025 + 0.115 * (0.5 - 0.5 * (6.0 * PI * frac).cos());
        let sig = 0.78 * phase.sin() + 0.22 * (2.0 * phase).sin();
        out[1024 + i] = amp * sig;
    }

    // 3. Samples 5120..8704: Dynamic multi-velocity guitar plucks (E2, G3, E4)
    let plucks: [(usize, f32, f32); 3] = [
        (5120, 82.41, 0.048),
        (6314, 196.00, 0.095),
        (7508, 329.63, 0.155),
    ];
    out[5120..8704].fill(0.0);
    for &(start, f0, peak) in &plucks {
        let dur = 1194usize;
        for j in 0..dur {
            if start + j >= 8704 {
                break;
            }
            let t = (j as f32) / sr;
            let attack = (t / 0.0012).min(1.0);
            let decay = (-t / 0.075).exp();
            let mut s = 0.0_f32;
            for h in 1..=6 {
                let hf = h as f32;
                let w = 1.0 / (hf * hf.sqrt());
                s += w * (2.0 * PI * f0 * hf * t + 0.3 * hf).sin();
            }
            out[start + j] += peak * attack * decay * (s * 0.65);
        }
    }

    // 4. Samples 8704..12288: Decaying A-major triad + dynamic chord articulation
    let chord_freqs = [110.0_f32, 164.81, 220.0, 277.18, 329.63, 659.25];
    for i in 0..(NAM_TRACE_LEN - 8704) {
        let t = (i as f32) / sr;
        let attack = (t / 0.002).min(1.0);
        let decay = (-t / 0.110).exp();
        let mut s = 0.0_f32;
        for (idx, &f) in chord_freqs.iter().enumerate() {
            let phase_off = (idx as f32) * 0.41;
            s += (2.0 * PI * f * t + phase_off).sin() / (1.0 + 0.25 * (idx as f32));
        }
        out[8704 + i] =
            (0.048 * attack * decay * s).clamp(-SINGLE_COIL_DI_PEAK, SINGLE_COIL_DI_PEAK);
    }
}
