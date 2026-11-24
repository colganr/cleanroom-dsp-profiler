"""
Strictly Held-Out Null-Test Verification & Scorecard Engine.

Computes unbiased verification metrics on strictly held-out audio segments:
  - Cross-correlation lag alignment (samples)
  - Waveform Correlation (%)
  - Raw Error-to-Signal Ratio (ESR dB)
  - Scale-Invariant ESR (dB) & optimal gain scalar
  - Output RMS Delta (dB)
  - Mean Spectral Error across 80 Hz - 6 kHz (dB)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import numpy as np


@dataclass(frozen=True)
class VerificationMetrics:
    label: str
    lag_samples: int
    corr_pct: float
    esr_raw_db: float
    esr_scaled_db: float
    scale_factor: float
    rms_delta_db: float
    spec_err_db: float

    def to_dict(self) -> dict:
        return asdict(self)


def compute_null_metrics(
    y_ref: np.ndarray,
    y_test: np.ndarray,
    sr: int = 48_000,
    label: str = "Held-Out Test",
    max_lag: int = 256,
    skip_samples: int = 2048,
) -> VerificationMetrics:
    """Aligns `y_test` to `y_ref` within `±max_lag` and computes full null-test metrics."""
    r = np.asarray(y_ref, dtype=np.float64)
    t = np.asarray(y_test, dtype=np.float64)
    n = min(len(r), len(t))
    r, t = r[:n], t[:n]

    best_lag = 0
    best_score = -1.0
    s0 = min(skip_samples, n // 4)
    e0 = n - max_lag - 1
    if e0 > s0 + 512 and max_lag > 0:
        ref_seg = r[s0:e0]
        ref_norm = float(np.linalg.norm(ref_seg)) + 1e-12
        for lag in range(-max_lag, max_lag + 1):
            a0 = s0 + lag
            a1 = e0 + lag
            if a0 < 0 or a1 > n:
                continue
            cand = t[a0:a1]
            c = abs(float(np.dot(ref_seg, cand))) / (ref_norm * (float(np.linalg.norm(cand)) + 1e-12))
            if c > best_score:
                best_score = c
                best_lag = lag

    if best_lag >= 0:
        a = r[s0 : n - best_lag]
        b = t[s0 + best_lag : n]
    else:
        shift = -best_lag
        a = r[s0 + shift : n]
        b = t[s0 : n - shift]

    dot_ab = float(np.dot(a, b))
    norm_a = float(np.linalg.norm(a)) + 1e-12
    norm_b = float(np.linalg.norm(b)) + 1e-12
    corr_pct = (dot_ab / (norm_a * norm_b)) * 100.0

    esr_raw = float(np.sum((a - b) ** 2)) / max(float(np.sum(a ** 2)), 1e-12)
    esr_raw_db = 10.0 * math.log10(esr_raw + 1e-12)

    scale = dot_ab / max(float(np.dot(b, b)), 1e-12)
    esr_scaled = float(np.sum((a - scale * b) ** 2)) / max(float(np.sum(a ** 2)), 1e-12)
    esr_scaled_db = 10.0 * math.log10(esr_scaled + 1e-12)

    rms_a = math.sqrt(float(np.mean(a ** 2)) + 1e-15)
    rms_b = math.sqrt(float(np.mean(b ** 2)) + 1e-15)
    rms_delta_db = 20.0 * math.log10(rms_b / rms_a)

    # Spectral MAE over active bins in [80 Hz, 6000 Hz]
    fa = np.abs(np.fft.rfft(a)) + 1e-8
    fb = np.abs(np.fft.rfft(b)) + 1e-8
    freqs = np.fft.rfftfreq(len(a), d=1.0 / sr)
    mask = (freqs >= 80.0) & (freqs <= 6000.0) & (fa > np.max(fa) * 0.02)
    spec_err_db = float(np.mean(np.abs(20.0 * np.log10(fa[mask] / fb[mask])))) if np.any(mask) else 0.0

    return VerificationMetrics(
        label=label,
        lag_samples=int(best_lag),
        corr_pct=round(corr_pct, 3),
        esr_raw_db=round(esr_raw_db, 3),
        esr_scaled_db=round(esr_scaled_db, 3),
        scale_factor=round(scale, 5),
        rms_delta_db=round(rms_delta_db, 3),
        spec_err_db=round(spec_err_db, 3),
    )


def format_scorecard_markdown(results: list[VerificationMetrics]) -> str:
    """Formats a list of verification metrics into a clean Markdown scorecard table."""
    lines = [
        "| Test Case (Strictly Held-Out) | Corr % | Raw ESR (dB) | Scaled ESR (dB) | Spec Err (dB) | ΔRMS (dB) | Lag |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for m in results:
        lines.append(
            f"| {m.label} | **{m.corr_pct:.2f}%** | {m.esr_raw_db:.2f} | {m.esr_scaled_db:.2f} "
            f"| {m.spec_err_db:.2f} | {m.rms_delta_db:+.2f} | {m.lag_samples} |"
        )
    return "\n".join(lines)
