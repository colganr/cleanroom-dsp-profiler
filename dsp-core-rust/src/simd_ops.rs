//! Zero-Allocation Cross-Platform SIMD & Fast-Math Primitives.
//!
//! - [`DenormalGuard`]: Enables hardware `FTZ` + `DAZ` (`MXCSR |= 0x8040`) on `x86_64` so
//!   Windows and Linux real-time audio threads never stall on subnormal microcode traps.
//! - [`fast_tanh`]: Branchless `[7/6]` Padé rational approximation (exact slope `1.0` at `0`,
//!   max error `< 1.02e-6` on `[-3, 3]`, bounded in `[-1, 1]`).
//! - [`dot_f32`] & [`dot_stereo_f32`]: Runtime-dispatched `AVX2+FMA` with `SSE2` fallback on
//!   `x86_64`, `NEON` on `aarch64`, and clean scalar fallback elsewhere.

#[cfg(target_arch = "x86_64")]
use std::sync::atomic::{AtomicU8, Ordering};

/// Branchless `[7/6]` Padé approximation of `tanh(x)`.
///
/// Guarantees exact unit slope (`tanh'(0) = 1.0`), odd symmetry, and strict bounds `[-1.0, 1.0]`.
#[inline(always)]
pub fn fast_tanh(x: f32) -> f32 {
    let xc = x.clamp(-4.90, 4.90);
    let x2 = xc * xc;
    let num = xc * (135_135.0 + x2 * (17_325.0 + x2 * (378.0 + x2)));
    let den = 135_135.0 + x2 * (62_370.0 + x2 * (3_150.0 + 28.0 * x2));
    (num / den).clamp(-1.0, 1.0)
}

/// Fast positive power `base^exp` for `base > 0.0`.
#[inline(always)]
pub fn fast_powf_pos(base: f32, exp: f32) -> f32 {
    base.powf(exp)
}

/// RAII guard enabling hardware Flush-To-Zero (`FTZ`) and Denormals-Are-Zero (`DAZ`) on `x86_64`.
#[cfg(target_arch = "x86_64")]
pub struct DenormalGuard {
    prev_mxcsr: u32,
}

#[cfg(target_arch = "x86_64")]
impl DenormalGuard {
    #[inline(always)]
    #[allow(deprecated)]
    pub fn new() -> Self {
        unsafe {
            let prev = std::arch::x86_64::_mm_getcsr();
            std::arch::x86_64::_mm_setcsr(prev | 0x8040);
            Self { prev_mxcsr: prev }
        }
    }
}

#[cfg(target_arch = "x86_64")]
impl Default for DenormalGuard {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(target_arch = "x86_64")]
impl Drop for DenormalGuard {
    #[inline(always)]
    #[allow(deprecated)]
    fn drop(&mut self) {
        unsafe {
            std::arch::x86_64::_mm_setcsr(self.prev_mxcsr);
        }
    }
}

/// Zero-cost no-op guard on architectures where FTZ is handled natively or unneeded.
#[cfg(not(target_arch = "x86_64"))]
pub struct DenormalGuard;

#[cfg(not(target_arch = "x86_64"))]
impl DenormalGuard {
    #[inline(always)]
    pub fn new() -> Self {
        Self
    }
}

#[cfg(not(target_arch = "x86_64"))]
impl Default for DenormalGuard {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(target_arch = "x86_64")]
static X86_SIMD_MODE: AtomicU8 = AtomicU8::new(0); // 0 = uninit, 1 = sse2, 2 = avx2+fma

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn x86_has_avx2_fma() -> bool {
    let m = X86_SIMD_MODE.load(Ordering::Relaxed);
    if m != 0 {
        return m == 2;
    }
    let has = std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma");
    X86_SIMD_MODE.store(if has { 2 } else { 1 }, Ordering::Relaxed);
    has
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len().min(b.len());
    let pa = a.as_ptr();
    let pb = b.as_ptr();
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + 32 <= n {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), acc0);
        acc1 = _mm256_fmadd_ps(
            _mm256_loadu_ps(pa.add(i + 8)),
            _mm256_loadu_ps(pb.add(i + 8)),
            acc1,
        );
        acc2 = _mm256_fmadd_ps(
            _mm256_loadu_ps(pa.add(i + 16)),
            _mm256_loadu_ps(pb.add(i + 16)),
            acc2,
        );
        acc3 = _mm256_fmadd_ps(
            _mm256_loadu_ps(pa.add(i + 24)),
            _mm256_loadu_ps(pb.add(i + 24)),
            acc3,
        );
        i += 32;
    }
    acc0 = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    while i + 8 <= n {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), acc0);
        i += 8;
    }
    let hi = _mm256_extractf128_ps(acc0, 1);
    let lo = _mm256_castps256_ps128(acc0);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let mut total = _mm_cvtss_f32(_mm_add_ss(sums, shuf2));
    while i < n {
        total += *pa.add(i) * *pb.add(i);
        i += 1;
    }
    total
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn dot_f32_sse2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len().min(b.len());
    let pa = a.as_ptr();
    let pb = b.as_ptr();
    let mut acc0 = _mm_setzero_ps();
    let mut acc1 = _mm_setzero_ps();
    let mut acc2 = _mm_setzero_ps();
    let mut acc3 = _mm_setzero_ps();
    let mut i = 0usize;
    while i + 16 <= n {
        acc0 = _mm_add_ps(
            acc0,
            _mm_mul_ps(_mm_loadu_ps(pa.add(i)), _mm_loadu_ps(pb.add(i))),
        );
        acc1 = _mm_add_ps(
            acc1,
            _mm_mul_ps(_mm_loadu_ps(pa.add(i + 4)), _mm_loadu_ps(pb.add(i + 4))),
        );
        acc2 = _mm_add_ps(
            acc2,
            _mm_mul_ps(_mm_loadu_ps(pa.add(i + 8)), _mm_loadu_ps(pb.add(i + 8))),
        );
        acc3 = _mm_add_ps(
            acc3,
            _mm_mul_ps(_mm_loadu_ps(pa.add(i + 12)), _mm_loadu_ps(pb.add(i + 12))),
        );
        i += 16;
    }
    acc0 = _mm_add_ps(_mm_add_ps(acc0, acc1), _mm_add_ps(acc2, acc3));
    while i + 4 <= n {
        acc0 = _mm_add_ps(
            acc0,
            _mm_mul_ps(_mm_loadu_ps(pa.add(i)), _mm_loadu_ps(pb.add(i))),
        );
        i += 4;
    }
    let mut lanes = [0.0_f32; 4];
    _mm_storeu_ps(lanes.as_mut_ptr(), acc0);
    let mut total = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
    while i < n {
        total += *pa.add(i) * *pb.add(i);
        i += 1;
    }
    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_stereo_f32_avx2(h: &[f32], ir_l: &[f32], ir_r: &[f32]) -> (f32, f32) {
    use std::arch::x86_64::*;
    let n = h.len().min(ir_l.len()).min(ir_r.len());
    let ph = h.as_ptr();
    let pl = ir_l.as_ptr();
    let pr = ir_r.as_ptr();
    let mut l0 = _mm256_setzero_ps();
    let mut l1 = _mm256_setzero_ps();
    let mut r0 = _mm256_setzero_ps();
    let mut r1 = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + 16 <= n {
        let vh0 = _mm256_loadu_ps(ph.add(i));
        let vh1 = _mm256_loadu_ps(ph.add(i + 8));
        l0 = _mm256_fmadd_ps(vh0, _mm256_loadu_ps(pl.add(i)), l0);
        r0 = _mm256_fmadd_ps(vh0, _mm256_loadu_ps(pr.add(i)), r0);
        l1 = _mm256_fmadd_ps(vh1, _mm256_loadu_ps(pl.add(i + 8)), l1);
        r1 = _mm256_fmadd_ps(vh1, _mm256_loadu_ps(pr.add(i + 8)), r1);
        i += 16;
    }
    l0 = _mm256_add_ps(l0, l1);
    r0 = _mm256_add_ps(r0, r1);
    while i + 8 <= n {
        let vh = _mm256_loadu_ps(ph.add(i));
        l0 = _mm256_fmadd_ps(vh, _mm256_loadu_ps(pl.add(i)), l0);
        r0 = _mm256_fmadd_ps(vh, _mm256_loadu_ps(pr.add(i)), r0);
        i += 8;
    }
    let hsum = |v: __m256| -> f32 {
        let s128 = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
        let sh = _mm_movehdup_ps(s128);
        let s = _mm_add_ps(s128, sh);
        _mm_cvtss_f32(_mm_add_ss(s, _mm_movehl_ps(s, s)))
    };
    let mut tl = hsum(l0);
    let mut tr = hsum(r0);
    while i < n {
        let hv = *ph.add(i);
        tl += hv * *pl.add(i);
        tr += hv * *pr.add(i);
        i += 1;
    }
    (tl, tr)
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn dot_stereo_f32_sse2(h: &[f32], ir_l: &[f32], ir_r: &[f32]) -> (f32, f32) {
    use std::arch::x86_64::*;
    let n = h.len().min(ir_l.len()).min(ir_r.len());
    let ph = h.as_ptr();
    let pl = ir_l.as_ptr();
    let pr = ir_r.as_ptr();
    let mut l0 = _mm_setzero_ps();
    let mut l1 = _mm_setzero_ps();
    let mut r0 = _mm_setzero_ps();
    let mut r1 = _mm_setzero_ps();
    let mut i = 0usize;
    while i + 8 <= n {
        let vh0 = _mm_loadu_ps(ph.add(i));
        let vh1 = _mm_loadu_ps(ph.add(i + 4));
        l0 = _mm_add_ps(l0, _mm_mul_ps(vh0, _mm_loadu_ps(pl.add(i))));
        r0 = _mm_add_ps(r0, _mm_mul_ps(vh0, _mm_loadu_ps(pr.add(i))));
        l1 = _mm_add_ps(l1, _mm_mul_ps(vh1, _mm_loadu_ps(pl.add(i + 4))));
        r1 = _mm_add_ps(r1, _mm_mul_ps(vh1, _mm_loadu_ps(pr.add(i + 4))));
        i += 8;
    }
    l0 = _mm_add_ps(l0, l1);
    r0 = _mm_add_ps(r0, r1);
    let mut ll = [0.0_f32; 4];
    let mut rr = [0.0_f32; 4];
    _mm_storeu_ps(ll.as_mut_ptr(), l0);
    _mm_storeu_ps(rr.as_mut_ptr(), r0);
    let mut tl = (ll[0] + ll[1]) + (ll[2] + ll[3]);
    let mut tr = (rr[0] + rr[1]) + (rr[2] + rr[3]);
    while i < n {
        let hv = *ph.add(i);
        tl += hv * *pl.add(i);
        tr += hv * *pr.add(i);
        i += 1;
    }
    (tl, tr)
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn dot_f32_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let n = a.len().min(b.len());
    let pa = a.as_ptr();
    let pb = b.as_ptr();
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);
    let mut i = 0usize;
    while i + 16 <= n {
        acc0 = vfmaq_f32(acc0, vld1q_f32(pa.add(i)), vld1q_f32(pb.add(i)));
        acc1 = vfmaq_f32(acc1, vld1q_f32(pa.add(i + 4)), vld1q_f32(pb.add(i + 4)));
        acc2 = vfmaq_f32(acc2, vld1q_f32(pa.add(i + 8)), vld1q_f32(pb.add(i + 8)));
        acc3 = vfmaq_f32(acc3, vld1q_f32(pa.add(i + 12)), vld1q_f32(pb.add(i + 12)));
        i += 16;
    }
    acc0 = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
    while i + 4 <= n {
        acc0 = vfmaq_f32(acc0, vld1q_f32(pa.add(i)), vld1q_f32(pb.add(i)));
        i += 4;
    }
    let mut total = vaddvq_f32(acc0);
    while i < n {
        total += *pa.add(i) * *pb.add(i);
        i += 1;
    }
    total
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn dot_stereo_f32_neon(h: &[f32], ir_l: &[f32], ir_r: &[f32]) -> (f32, f32) {
    use std::arch::aarch64::*;
    let n = h.len().min(ir_l.len()).min(ir_r.len());
    let ph = h.as_ptr();
    let pl = ir_l.as_ptr();
    let pr = ir_r.as_ptr();
    let mut l0 = vdupq_n_f32(0.0);
    let mut l1 = vdupq_n_f32(0.0);
    let mut r0 = vdupq_n_f32(0.0);
    let mut r1 = vdupq_n_f32(0.0);
    let mut i = 0usize;
    while i + 8 <= n {
        let vh0 = vld1q_f32(ph.add(i));
        let vh1 = vld1q_f32(ph.add(i + 4));
        l0 = vfmaq_f32(l0, vh0, vld1q_f32(pl.add(i)));
        r0 = vfmaq_f32(r0, vh0, vld1q_f32(pr.add(i)));
        l1 = vfmaq_f32(l1, vh1, vld1q_f32(pl.add(i + 4)));
        r1 = vfmaq_f32(r1, vh1, vld1q_f32(pr.add(i + 4)));
        i += 8;
    }
    l0 = vaddq_f32(l0, l1);
    r0 = vaddq_f32(r0, r1);
    while i + 4 <= n {
        let vh = vld1q_f32(ph.add(i));
        l0 = vfmaq_f32(l0, vh, vld1q_f32(pl.add(i)));
        r0 = vfmaq_f32(r0, vh, vld1q_f32(pr.add(i)));
        i += 4;
    }
    let mut tl = vaddvq_f32(l0);
    let mut tr = vaddvq_f32(r0);
    while i < n {
        let hv = *ph.add(i);
        tl += hv * *pl.add(i);
        tr += hv * *pr.add(i);
        i += 1;
    }
    (tl, tr)
}

/// Computes the SIMD-accelerated dot product of two equal-length `f32` slices.
#[inline(always)]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if x86_has_avx2_fma() {
            unsafe { dot_f32_avx2(a, b) }
        } else {
            unsafe { dot_f32_sse2(a, b) }
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { dot_f32_neon(a, b) }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
    }
}

/// Computes a simultaneous mono-to-stereo FIR convolution dot product (`(h·ir_l, h·ir_r)`).
#[inline(always)]
pub fn dot_stereo_f32(h: &[f32], ir_l: &[f32], ir_r: &[f32]) -> (f32, f32) {
    #[cfg(target_arch = "x86_64")]
    {
        if x86_has_avx2_fma() {
            unsafe { dot_stereo_f32_avx2(h, ir_l, ir_r) }
        } else {
            unsafe { dot_stereo_f32_sse2(h, ir_l, ir_r) }
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { dot_stereo_f32_neon(h, ir_l, ir_r) }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let mut tl = 0.0_f32;
        let mut tr = 0.0_f32;
        for ((&hv, &lv), &rv) in h.iter().zip(ir_l.iter()).zip(ir_r.iter()) {
            tl += hv * lv;
            tr += hv * rv;
        }
        (tl, tr)
    }
}
