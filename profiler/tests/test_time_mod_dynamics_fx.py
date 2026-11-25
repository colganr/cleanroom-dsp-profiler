"""
End-to-End Tests for Time-Based, Modulation & Dynamics FX Profilers:
  1. Spring Reverb SVD Basis + Causal Toeplitz Tone FIR
  2. Harmonic Tremolo Carrier Demodulation + Impulse-Train SVD
  3. Analog BBD Delay Extra-Loop-Delay Discovery + Toeplitz FIR + 30-Ch Saturation
  4. Dynamic Envelope Filter (Vactrol + TPT SVF) Multi-Corner Optimization
  5. Studio Compressor Stepped-Sine Transfer Curve & Attack/Release Ballistics
"""

import numpy as np

from amp_profiler.backends.synthetic_fx import (
    SyntheticBbdDelayBackend,
    SyntheticCompressorBackend,
    SyntheticEnvelopeFilterBackend,
    SyntheticHarmonicTremoloBackend,
    SyntheticSpringReverbBackend,
)
from amp_profiler.stimulus import SR_DEFAULT, generate_procedural_guitar_di
from amp_profiler.time_mod_fx import (
    fit_bbd_30ch_saturation,
    profile_compressor,
    profile_delay_bbd,
    profile_envelope_filter,
    profile_harmonic_tremolo,
    profile_reverb_svd,
)


def test_spring_reverb_svd_and_tone_profiler():
    rev = SyntheticSpringReverbBackend(sr=SR_DEFAULT, onset_latency=16)
    res = profile_reverb_svd(
        rev,
        pid_decay=rev.PID_DECAY,
        pid_tone=rev.PID_TONE,
        pid_mix=rev.PID_MIX,
        onset=16,
        ir_len=8192,
        n_basis=4,
        n_steps=9,
    )
    assert res["svd_min_corr_pct"] > 99.5, f"Low SVD decay fit: {res['svd_min_corr_pct']}"
    assert res["tone_min_corr_pct"] > 99.5, f"Low Toeplitz tone fit: {res['tone_min_corr_pct']}"
    assert abs(float(res["dry_gain"][0]) - 1.0) < 0.02
    assert abs(float(res["wet_gain"][-1]) - 1.0) < 0.02


def test_harmonic_tremolo_carrier_demod_and_svd():
    trem = SyntheticHarmonicTremoloBackend(sr=SR_DEFAULT, onset_latency=8)
    res = profile_harmonic_tremolo(
        trem,
        pid_speed=trem.PID_SPEED,
        pid_intensity=trem.PID_INTENSITY,
        onset=8,
        fir_len=128,
        n_steps=5,
        sr=SR_DEFAULT,
    )
    assert res["svd_2basis_energy_pct"] > 99.5, f"Low 2-basis energy: {res['svd_2basis_energy_pct']}"
    for idx, sp in enumerate(np.linspace(0.0, 1.0, 5)):
        expected_hz = trem.true_lfo_hz(float(sp))
        meas_hz = float(res["freq_hz"][idx])
        assert abs(meas_hz - expected_hz) < 0.15, f"Speed={sp}: expected {expected_hz} Hz, got {meas_hz} Hz"


def test_bbd_delay_extra_loop_offset_and_saturation():
    dly = SyntheticBbdDelayBackend(sr=SR_DEFAULT, onset_latency=12, loop_extra_delay=24)
    res = profile_delay_bbd(
        dly,
        pid_time=dly.PID_TIME,
        pid_feedback=dly.PID_FEEDBACK,
        pid_mix=dly.PID_MIX,
        pid_tone_hardness=dly.PID_TONE,
        onset=12,
        fir_len=64,
        n_steps=5,
        sr=SR_DEFAULT,
    )
    assert res["loop_extra_delay"] == 24, f"Expected extra loop delay 24, got {res['loop_extra_delay']}"
    assert res["loop_min_corr_pct"] > 99.5, f"Low loop FIR fit: {res['loop_min_corr_pct']}"

    # Verify 30-channel BBD bucket-rail saturation solver
    u = generate_procedural_guitar_di(duration_s=1.0, sr=SR_DEFAULT, peak=0.45)
    y_sat = (0.28 * np.tanh(u / 0.28)).astype(np.float32)
    w30, corr = fit_bbd_30ch_saturation(u, y_sat)
    assert len(w30) == 30
    assert corr > 99.8, f"Expected >99.8% BBD sat corr, got {corr}%"


def test_envelope_filter_vactrol_svf_profiler():
    ef = SyntheticEnvelopeFilterBackend(sr=SR_DEFAULT, onset_latency=8)
    cal_di = generate_procedural_guitar_di(duration_s=1.2, sr=SR_DEFAULT, peak=0.22, seed=99)
    res = profile_envelope_filter(
        ef,
        pid_drive=ef.PID_DRIVE,
        pid_peak=ef.PID_PEAK,
        cal_di=cal_di,
        onset=8,
        n_q_steps=5,
        n_rounds=4,
    )
    assert res["fit_mean_corr_pct"] > 97.0, f"Low envelope filter fit: {res['fit_mean_corr_pct']}%"
    assert len(res["params_14"]) == 14


def test_studio_compressor_static_and_ballistics_profiler():
    comp = SyntheticCompressorBackend(
        sr=SR_DEFAULT,
        threshold_db=-20.0,
        ratio=4.0,
        knee_db=6.0,
        attack_ms=12.0,
        release_ms=110.0,
        makeup_db=4.5,
    )
    res = profile_compressor(comp, onset=0, sr=SR_DEFAULT)
    assert abs(res["makeup_db"] - 4.5) < 0.4, f"makeup mismatch: {res}"
    assert abs(res["ratio"] - 4.0) < 0.35, f"ratio mismatch: {res}"
    assert abs(res["threshold_db"] - (-20.0)) < 2.0, f"threshold mismatch: {res}"
    assert 6.0 <= res["attack_ms"] <= 24.0, f"attack mismatch: {res}"
    assert 65.0 <= res["release_ms"] <= 170.0, f"release mismatch: {res}"
