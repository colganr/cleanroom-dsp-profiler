//! Zero-Allocation In-Place FFT, Real-Cepstrum Minimum-Phase FIR Synthesis,
//! and 1D/2D/3D Bode Tensor Interpolation.
//!
//! Given a base impulse response (`H_base`) and a 64-point log-spaced dB delta curve
//! (`delta_db_64`), [`synthesize_min_phase_ir`] folds the magnitude delta into a causal,
//! zero-group-delay minimum-phase FIR via the real cepstrum and convolves it with `H_base`
//! entirely on stack buffers (zero heap allocations).

use std::f32::consts::PI;

pub const NUM_FREQS: usize = 64;
pub const FFT_N: usize = 2048;

/// 64 logarithmically spaced reference frequencies from `20 Hz` to `20,000 Hz`.
pub fn log_freqs_64() -> [f32; NUM_FREQS] {
    let mut out = [0.0_f32; NUM_FREQS];
    let log_min = 20.0_f64.ln();
    let log_max = 20_000.0_f64.ln();
    for (i, dst) in out.iter_mut().enumerate() {
        let t = (i as f64) / ((NUM_FREQS - 1) as f64);
        *dst = (log_min + t * (log_max - log_min)).exp() as f32;
    }
    out
}

/// Zero-allocation radix-2 Cooley-Tukey in-place FFT (`N = 2048`).
#[inline]
pub fn fft_2048_in_place(re: &mut [f32; FFT_N], im: &mut [f32; FFT_N], inverse: bool) {
    let mut j = 0usize;
    for i in 1..FFT_N {
        let mut bit = FFT_N >> 1;
        while (j & bit) != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }

    let mut len = 2usize;
    let sign = if inverse { 1.0_f32 } else { -1.0_f32 };
    while len <= FFT_N {
        let half = len >> 1;
        let ang = sign * 2.0 * PI / (len as f32);
        let wlen_re = ang.cos();
        let wlen_im = ang.sin();
        let mut i = 0usize;
        while i < FFT_N {
            let mut w_re = 1.0_f32;
            let mut w_im = 0.0_f32;
            for k in 0..half {
                let u_re = re[i + k];
                let u_im = im[i + k];
                let v_re = re[i + k + half] * w_re - im[i + k + half] * w_im;
                let v_im = re[i + k + half] * w_im + im[i + k + half] * w_re;
                re[i + k] = u_re + v_re;
                im[i + k] = u_im + v_im;
                re[i + k + half] = u_re - v_re;
                im[i + k + half] = u_im - v_im;
                let next_w_re = w_re * wlen_re - w_im * wlen_im;
                let next_w_im = w_re * wlen_im + w_im * wlen_re;
                w_re = next_w_re;
                w_im = next_w_im;
            }
            i += len;
        }
        len <<= 1;
    }

    if inverse {
        let inv_n = 1.0 / (FFT_N as f32);
        for i in 0..FFT_N {
            re[i] *= inv_n;
            im[i] *= inv_n;
        }
    }
}

/// Binary-search linear interpolation of a 64-point frequency curve at frequency `f` (Hz).
#[inline(always)]
pub fn interp_freq_curve(freqs: &[f32; NUM_FREQS], vals: &[f32; NUM_FREQS], f: f32) -> f32 {
    if f <= freqs[0] {
        return vals[0];
    }
    if f >= freqs[NUM_FREQS - 1] {
        return vals[NUM_FREQS - 1];
    }
    let mut lo = 0usize;
    let mut hi = NUM_FREQS - 1;
    while hi - lo > 1 {
        let mid = (lo + hi) >> 1;
        if freqs[mid] <= f {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let span = (freqs[hi] - freqs[lo]).max(1e-6);
    let t = (f - freqs[lo]) / span;
    vals[lo] * (1.0 - t) + vals[hi] * t
}

/// Interpolates a 21-step Bode curve table (`[21 * NUM_FREQS]`) for normalized control `x in [0, 1]`.
#[inline(always)]
pub fn interp_21_freq(table: &[f32], x: f32, f_idx: usize) -> f32 {
    let pos = x.clamp(0.0, 1.0) * 20.0;
    let idx = (pos as usize).min(19);
    let frac = pos - (idx as f32);
    table[idx * NUM_FREQS + f_idx] * (1.0 - frac) + table[(idx + 1) * NUM_FREQS + f_idx] * frac
}

/// Optional low-end audio-taper pot warp for passive tone stacks (`x < 0.25`).
#[inline(always)]
pub fn warp_audio_pot_low(v: f32) -> f32 {
    let x = v.clamp(0.0, 1.0);
    if x < 0.25 {
        let u = x * 4.0;
        0.25 * (1.0 - (1.0 - u).powf(2.22))
    } else {
        x
    }
}

/// Trilinear interpolation on a 5x5x5 coupled `(Bass x Middle x Treble)` tone-stack cube.
#[inline(always)]
pub fn interp_3d_5x5x5(
    cube: &[f32],
    bass: f32,
    middle: f32,
    treble: f32,
    f_idx: usize,
    use_audio_warp: bool,
) -> f32 {
    let (b_eff, m_eff, t_eff) = if use_audio_warp {
        (
            warp_audio_pot_low(bass),
            warp_audio_pot_low(middle),
            warp_audio_pot_low(treble),
        )
    } else {
        (
            bass.clamp(0.0, 1.0),
            middle.clamp(0.0, 1.0),
            treble.clamp(0.0, 1.0),
        )
    };

    let pb = b_eff * 4.0;
    let ib = (pb as usize).min(3);
    let wb = pb - (ib as f32);

    let pm = m_eff * 4.0;
    let im = (pm as usize).min(3);
    let wm = pm - (im as f32);

    let pt = t_eff * 4.0;
    let it = (pt as usize).min(3);
    let wt = pt - (it as f32);

    let at =
        |b: usize, m: usize, t: usize| -> f32 { cube[((b * 5 + m) * 5 + t) * NUM_FREQS + f_idx] };

    let c00 = at(ib, im, it) * (1.0 - wb) + at(ib + 1, im, it) * wb;
    let c01 = at(ib, im, it + 1) * (1.0 - wb) + at(ib + 1, im, it + 1) * wb;
    let c10 = at(ib, im + 1, it) * (1.0 - wb) + at(ib + 1, im + 1, it) * wb;
    let c11 = at(ib, im + 1, it + 1) * (1.0 - wb) + at(ib + 1, im + 1, it + 1) * wb;

    let c0 = c00 * (1.0 - wm) + c10 * wm;
    let c1 = c01 * (1.0 - wm) + c11 * wm;
    c0 * (1.0 - wt) + c1 * wt
}

/// Synthesizes a causal minimum-phase FIR of length `out_ir.len()` (`<= FFT_N / 2`) by
/// applying a 64-point log-frequency dB delta (`delta_db_64`) to `base_ir` via the real cepstrum.
///
/// Applies a raised-cosine tail fade of `fade_len` taps and optional `polarity` (`+1.0` or `-1.0`).
pub fn synthesize_min_phase_ir(
    base_ir: &[f32],
    freqs_64: &[f32; NUM_FREQS],
    delta_db_64: &[f32; NUM_FREQS],
    sr: f32,
    polarity: f32,
    fade_len: usize,
    out_ir: &mut [f32],
) {
    let half = FFT_N / 2;
    let ln10_over_20 = 10.0_f32.ln() / 20.0;

    let mut base_re = [0.0_f32; FFT_N];
    let mut base_im = [0.0_f32; FFT_N];
    let copy_len = base_ir.len().min(FFT_N);
    base_re[..copy_len].copy_from_slice(&base_ir[..copy_len]);
    fft_2048_in_place(&mut base_re, &mut base_im, false);

    let mut cep_re = [0.0_f32; FFT_N];
    let mut cep_im = [0.0_f32; FFT_N];
    for k in 0..=half {
        let f = (k as f32) * sr / (FFT_N as f32);
        let d_db = interp_freq_curve(freqs_64, delta_db_64, f);
        cep_re[k] = d_db * ln10_over_20;
        if k > 0 && k < half {
            cep_re[FFT_N - k] = cep_re[k];
        }
    }

    // Real cepstrum c[n] = Re(IFFT(ln|H[k]|))
    fft_2048_in_place(&mut cep_re, &mut cep_im, true);

    // Causal Hilbert folding window W[n] = {1 at 0,N/2; 2 on 1..N/2; 0 on N/2+1..N}
    for k in 1..half {
        cep_re[k] *= 2.0;
        cep_im[k] = 0.0;
    }
    cep_im[0] = 0.0;
    cep_im[half] = 0.0;
    for k in (half + 1)..FFT_N {
        cep_re[k] = 0.0;
        cep_im[k] = 0.0;
    }

    fft_2048_in_place(&mut cep_re, &mut cep_im, false);

    let mut h_re = [0.0_f32; FFT_N];
    let mut h_im = [0.0_f32; FFT_N];
    for k in 0..FFT_N {
        let mag = cep_re[k].exp();
        let m_re = mag * cep_im[k].cos();
        let m_im = mag * cep_im[k].sin();
        let b_re = base_re[k];
        let b_im = base_im[k];
        h_re[k] = b_re * m_re - b_im * m_im;
        h_im[k] = b_re * m_im + b_im * m_re;
    }
    fft_2048_in_place(&mut h_re, &mut h_im, true);

    let out_len = out_ir.len().min(FFT_N);
    let fade_start = out_len.saturating_sub(fade_len);
    for (i, dst) in out_ir.iter_mut().enumerate().take(out_len) {
        let fade = if fade_len > 1 && i >= fade_start {
            let idx = (i - fade_start) as f32;
            0.5 * (1.0 + (PI * idx / ((fade_len - 1) as f32)).cos())
        } else {
            1.0
        };
        *dst = h_re[i] * polarity * fade;
    }
}
