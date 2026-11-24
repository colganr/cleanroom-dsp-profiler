"""Unit tests for stimulus generation, WAV 24-bit I/O, cepstrum FIR synthesis, cab grid & NAM v0.5.4."""

from pathlib import Path
import numpy as np

from amp_profiler.cab_grid import dequantize_cab_grid, quantize_cab_grid
from amp_profiler.linear_bode import NUM_FREQS, synthesize_min_phase_ir
from amp_profiler.nam_export import (
    distill_nam_v054,
    expected_wavenet_weight_count,
    forward_pass_nam_v054,
)
from amp_profiler.stimulus import (
    SR_DEFAULT,
    build_15s_composite_stimulus,
    load_wav_mono,
    save_wav_24bit_pcm,
)


def test_wav_24bit_roundtrip(tmp_path: Path):
    t = np.arange(4096, dtype=np.float32) / SR_DEFAULT
    sig = (0.35 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
    wav_path = tmp_path / "tone.wav"
    save_wav_24bit_pcm(wav_path, sig, sr=SR_DEFAULT)
    loaded, sr = load_wav_mono(wav_path)
    assert sr == SR_DEFAULT
    assert len(loaded) == len(sig)
    assert float(np.max(np.abs(loaded - sig))) < 1e-5


def test_15s_stimulus_slices_are_disjoint():
    stim, slices = build_15s_composite_stimulus(sr=SR_DEFAULT)
    assert len(stim) == 15 * SR_DEFAULT
    cal_a, cal_b = slices["calibration"]
    ho_a, ho_b = slices["heldout_test"]
    assert cal_b <= ho_a
    assert np.isfinite(stim).all()
    assert float(np.max(np.abs(stim[ho_a:ho_b]))) > 0.05


def test_cepstral_flat_gain_and_identity():
    base = np.zeros(256, dtype=np.float32)
    base[0] = 1.0
    out_id = synthesize_min_phase_ir(base, np.zeros(NUM_FREQS, dtype=np.float32), out_len=256, fade_len=16)
    assert abs(float(out_id[0]) - 1.0) < 1e-4

    out_6db = synthesize_min_phase_ir(base, np.full(NUM_FREQS, 6.0206, dtype=np.float32), out_len=256, fade_len=16)
    assert abs(float(out_6db[0]) - 2.0) < 2e-3


def test_cab_i16_quantization_floor():
    rng = np.random.default_rng(7)
    irs = (rng.normal(0.0, 0.25, size=(4, 1024)) * np.exp(-np.linspace(0, 5, 1024))[None, :]).astype(np.float32)
    raw, scales = quantize_cab_grid(irs)
    restored = dequantize_cab_grid(raw, scales, ir_len=1024)
    snr_db = 10.0 * np.log10(np.sum(irs ** 2) / max(float(np.sum((irs - restored) ** 2)), 1e-18))
    # 16-bit PCM full-scale is 96.3 dB; with ~12.5 dB decaying IR crest factor, expect > 80 dB SNR
    assert snr_db > 80.0


def test_nam_v054_schema_and_forward_pass():
    t = np.arange(8192, dtype=np.float32) / SR_DEFAULT
    x = (0.14 * np.sin(2.0 * np.pi * 220.0 * t) + 0.06 * np.sin(2.0 * np.pi * 550.0 * t)).astype(np.float32)
    y = np.tanh(3.0 * x).astype(np.float32) * 0.4
    doc, metrics = distill_nam_v054(x, y, preset="nano", sr=SR_DEFAULT, model_name="UnitTest")
    assert doc["version"] == "0.5.4"
    assert doc["architecture"] == "WaveNet"
    assert len(doc["weights"]) == expected_wavenet_weight_count(doc["config"])
    y_check = forward_pass_nam_v054(x, doc)
    assert np.isfinite(y_check).all()
    assert metrics["corr_pct"] > 95.0
