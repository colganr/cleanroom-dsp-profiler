"""
Unified Cross-Platform CLI for `cleanroom-dsp-profiler`.

Subcommands:
  - `discover`        : Inspect installed macOS AudioUnit parameters & onset latency.
  - `export-stimulus` : Write the 15s Zero-Leakage Composite Stimulus (24-bit WAV) for DAW/hardware re-amping.
  - `profile-wav-pair`: Profile ANY VST3/CLAP/AAX plugin or hardware rig on Windows/macOS/Linux from a WAV pair.
  - `demo`            : Run end-to-end clean-room synthetic Amp + Overdrive capture, Rust export & held-out null test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from .backends.macos_au import MacOSAudioUnitBackend
from .backends.synthetic import SyntheticAmpBackend, SyntheticOverdriveBackend
from .backends.wav_pair import WavPairBackend
from .export_rust import export_amp_rust_weights
from .linear_bode import (
    capture_bipolar_ir,
    deconvolve_pre_from_total,
    measure_post_ir_hard_sat,
)
from .nam_export import distill_nam_v054, save_nam_and_ir
from .nonlinear_fit import (
    apply_post_inv_shelf,
    eval_amp_features,
    fit_amp_180_weights,
    fit_pedal_126_weights,
)
from .rust_bridge import get_rust_bridge
from .stimulus import (
    SR_DEFAULT,
    build_15s_composite_stimulus,
    load_wav_mono,
    save_wav_24bit_pcm,
)
from .verify import compute_null_metrics, format_scorecard_markdown


def cmd_discover(args: argparse.Namespace) -> int:
    au = MacOSAudioUnitBackend(args.type, args.subtype, args.mfg, sr=args.sr)
    params = [
        {
            "pid": p.pid,
            "name": p.name,
            "min": p.min_value,
            "max": p.max_value,
            "default": p.default_value,
        }
        for p in au.list_parameters()
    ]
    out = {"parameters": params}
    if args.probe_impulse:
        onset, peak = au.probe_onset_latency()
        out["onset_latency_samples"] = onset
        out["impulse_peak"] = peak
    print(json.dumps(out, indent=2))
    return 0


def cmd_export_stimulus(args: argparse.Namespace) -> int:
    di = None
    if args.di_wav:
        di, sr = load_wav_mono(args.di_wav)
        if sr != args.sr:
            raise ValueError(f"DI WAV sample rate is {sr} Hz; expected {args.sr} Hz.")
    stim, slices = build_15s_composite_stimulus(di_audio=di, sr=args.sr, di_peak=args.di_peak)
    save_wav_24bit_pcm(args.out, stim, sr=args.sr)
    print(f"Wrote 15.0s zero-leakage composite stimulus ({len(stim)} samples @ {args.sr} Hz) -> {args.out}")
    print(json.dumps({"segment_slices": slices}, indent=2))
    return 0


def _estimate_h_tot_from_linear_sweep(x_sw: np.ndarray, y_sw: np.ndarray, ir_len: int = 1024) -> np.ndarray:
    """Deconvolves linear impulse response `H_tot` from the -36 dBFS sweep segment."""
    n = min(len(x_sw), len(y_sw))
    nfft = 1 << (n * 2 - 1).bit_length()
    X = np.fft.rfft(x_sw[:n].astype(np.float64), n=nfft)
    Y = np.fft.rfft(y_sw[:n].astype(np.float64), n=nfft)
    mag2 = np.abs(X) ** 2
    reg = max(float(np.max(mag2)) * 1e-5, 1e-12)
    H = (Y * np.conj(X)) / (mag2 + reg)
    ir = np.fft.irfft(H, n=nfft)[:ir_len]
    fade = np.ones(ir_len, dtype=np.float64)
    if ir_len >= 64:
        fade[-64:] = 0.5 * (1.0 + np.cos(np.pi * np.arange(64) / 63.0))
    return (ir * fade).astype(np.float32)


def _estimate_h_post_from_sweeps(
    y_lin: np.ndarray, y_drv: np.ndarray, post_len: int = 64, sr: int = SR_DEFAULT
) -> np.ndarray:
    """Estimates post-clipping filter `H_post` from linear vs. driven sweep spectral ratio."""
    n = min(len(y_lin), len(y_drv))
    nfft = 2048
    win = np.hanning(min(n, nfft))
    # Average Welch power spectrum across both sweeps
    step = nfft // 2
    p_lin = np.zeros(nfft // 2 + 1, dtype=np.float64)
    p_drv = np.zeros(nfft // 2 + 1, dtype=np.float64)
    cnt = 0
    for pos in range(0, n - nfft, step):
        p_lin += np.abs(np.fft.rfft(y_lin[pos : pos + nfft] * win)) ** 2
        p_drv += np.abs(np.fft.rfft(y_drv[pos : pos + nfft] * win)) ** 2
        cnt += 1
    p_lin = np.sqrt(p_lin / max(cnt, 1)) + 1e-8
    p_drv = np.sqrt(p_drv / max(cnt, 1)) + 1e-8

    freqs = np.fft.rfftfreq(nfft, d=1.0 / sr)
    # Driven spectrum decays faster above post-clipping cutoff; isolate gentle post rolloff
    ref_bin = int(np.argmin(np.abs(freqs - 400.0)))
    ratio_db = 20.0 * np.log10(p_drv / max(p_drv[ref_bin], 1e-8))
    ratio_db = np.clip(ratio_db, -24.0, 6.0)
    # Only retain post-clipping high-frequency rolloff (above 1.5 kHz) so linear bass stays in H_pre
    hp_blend = np.clip((freqs - 1200.0) / 1800.0, 0.0, 1.0)
    post_db = np.minimum(ratio_db, 0.0) * hp_blend

    ln_mag = post_db * (np.log(10.0) / 20.0)
    full_ln = np.concatenate([ln_mag, ln_mag[-2:0:-1]])
    cep = np.fft.ifft(full_ln).real
    c_min = np.zeros_like(cep)
    c_min[0] = cep[0]
    c_min[1 : nfft // 2] = 2.0 * cep[1 : nfft // 2]
    c_min[nfft // 2] = cep[nfft // 2]
    ir = np.fft.ifft(np.exp(np.fft.fft(c_min))).real[:post_len].astype(np.float32)
    return ir


def cmd_profile_wav_pair(args: argparse.Namespace) -> int:
    wp = WavPairBackend(args.input_wav, args.output_wav)
    sr = wp.sr
    x_all = wp.x_in
    y_all = wp.y_aligned

    if len(x_all) < 12 * sr:
        raise ValueError(
            f"Expected a 15-second composite stimulus WAV (>= 12s), got {len(x_all) / sr:.2f}s."
        )

    print(f"Loaded WAV pair @ {sr} Hz (auto-aligned lag = {wp.lag} samples)")
    s_lin = slice(0, 2 * sr)
    s_drv = slice(2 * sr, 4 * sr)
    s_cal = slice(4 * sr, 8 * sr)
    s_ho = slice(8 * sr, min(len(x_all), 15 * sr))

    h_tot = _estimate_h_tot_from_linear_sweep(x_all[s_lin], y_all[s_lin], ir_len=1024)
    h_post = _estimate_h_post_from_sweeps(y_all[s_lin], y_all[s_drv], post_len=64, sr=sr)
    h_pre = deconvolve_pre_from_total(h_tot, h_post, pre_len=1024)

    u_cal = np.convolve(x_all[s_cal].astype(np.float64), h_pre.astype(np.float64))[: len(x_all[s_cal])].astype(np.float32)
    w_180 = fit_amp_180_weights(u_cal, y_all[s_cal], h_post=h_post, sr=sr)

    # Strictly held-out verification on Segment 3 (8.0s - 15.0s)
    u_ho = np.convolve(x_all[s_ho].astype(np.float64), h_pre.astype(np.float64))[: len(x_all[s_ho])].astype(np.float32)
    bridge = get_rust_bridge(allow_missing=True)
    if bridge is not None:
        y_nl_ho = bridge.render_amp_core(u_ho, w_180, sr=sr)
    else:
        phi_ho, sat0_ho, pp, pz = eval_amp_features(u_ho, sr=sr, prefer_rust=False)
        y_nl_ho = apply_post_inv_shelf(sat0_ho + phi_ho @ w_180, pp, pz)
    y_pred_ho = np.convolve(y_nl_ho.astype(np.float64), h_post.astype(np.float64))[: len(y_nl_ho)].astype(np.float32)

    m = compute_null_metrics(y_all[s_ho], y_pred_ho, sr=sr, label=args.name)
    print("\n" + format_scorecard_markdown([m]) + "\n")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rs_path = export_amp_rust_weights(out_dir / "captured_amp_weights.rs", h_pre, h_post, [0.5], w_180)
    print(f"Exported Pure-Rust weights -> {rs_path}")

    if args.export_nam:
        nam_doc, nam_m = distill_nam_v054(x_all[s_cal], y_all[s_cal], preset=args.nam_preset, sr=sr, model_name=args.name)
        nam_out, _ = save_nam_and_ir(out_dir / f"{args.name.lower().replace(' ', '_')}.nam", nam_doc, sr=sr)
        print(f"Exported NAM v0.5.4 ({nam_m['preset']}, {nam_m['weight_count']} weights, corr={nam_m['corr_pct']:.2f}%) -> {nam_out}")

    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """Runs a complete clean-room synthetic Amp + Overdrive capture & held-out verification demo."""
    sr = SR_DEFAULT
    stim, slices = build_15s_composite_stimulus(sr=sr)
    s_cal = slice(*slices["calibration"])
    s_ho = slice(*slices["heldout_test"])
    bridge = get_rust_bridge(allow_missing=True)
    accel = f"Rust FFI ({bridge.path.name})" if bridge is not None else "Pure-Python NumPy"
    print(f"=== Clean-Room DSP Profiler Demo (Accelerator: {accel}) ===\n")

    results = []

    # 1. Synthetic Tube Amp (Clean-to-Crunch Gain = 0.55)
    amp = SyntheticAmpBackend(sr=sr)
    onset_amp, _ = amp.probe_onset_latency()
    amp.set_param(SyntheticAmpBackend.PID_GAIN, 0.55)
    amp.set_param(SyntheticAmpBackend.PID_BRIGHT, 1.0)

    h_tot = capture_bipolar_ir(amp, ir_len=1024, onset=onset_amp, imp_amp=1e-4)
    h_post = measure_post_ir_hard_sat(amp, post_len=64, drive_amp=0.35, n_tones=24, sr=sr)
    h_pre = deconvolve_pre_from_total(h_tot, h_post, pre_len=1024)

    y_amp_full = amp.render_mono(stim)
    # Strip known onset latency
    y_amp = np.zeros_like(y_amp_full)
    y_amp[: len(y_amp_full) - onset_amp] = y_amp_full[onset_amp:]

    u_cal = np.convolve(stim[s_cal].astype(np.float64), h_pre.astype(np.float64))[: len(stim[s_cal])].astype(np.float32)
    w_amp = fit_amp_180_weights(u_cal, y_amp[s_cal], h_post=h_post, sr=sr)

    u_ho = np.convolve(stim[s_ho].astype(np.float64), h_pre.astype(np.float64))[: len(stim[s_ho])].astype(np.float32)
    if bridge is not None:
        y_nl_ho = bridge.render_amp_core(u_ho, w_amp, sr=sr)
    else:
        phi_ho, sat0_ho, pp, pz = eval_amp_features(u_ho, sr=sr, prefer_rust=False)
        y_nl_ho = apply_post_inv_shelf(sat0_ho + phi_ho @ w_amp, pp, pz)
    y_pred_amp = np.convolve(y_nl_ho.astype(np.float64), h_post.astype(np.float64))[: len(y_nl_ho)].astype(np.float32)
    results.append(compute_null_metrics(y_amp[s_ho], y_pred_amp, sr=sr, label="Synthetic Tube Amp (180-Ch Bloom+Sag+KONE)"))

    # 2. Synthetic Overdrive Pedal (Drive = 0.60, Tone = 0.55)
    od = SyntheticOverdriveBackend(sr=sr)
    onset_od, _ = od.probe_onset_latency()
    od.set_param(SyntheticOverdriveBackend.PID_DRIVE, 0.60)
    od.set_param(SyntheticOverdriveBackend.PID_TONE, 0.55)
    od.set_param(SyntheticOverdriveBackend.PID_LEVEL, 0.50)

    h_od_tot = capture_bipolar_ir(od, ir_len=256, onset=onset_od, imp_amp=1e-4)
    h_od_post = measure_post_ir_hard_sat(od, post_len=256, drive_amp=0.35, n_tones=24, sr=sr)
    h_od_pre = deconvolve_pre_from_total(h_od_tot, h_od_post, pre_len=256)

    y_od_full = od.render_mono(stim)
    y_od = np.zeros_like(y_od_full)
    y_od[: len(y_od_full) - onset_od] = y_od_full[onset_od:]

    w_od = fit_pedal_126_weights(stim[s_cal], y_od[s_cal], h_od_pre, h_od_post, level_gain=1.0, subtract_dry=True)
    if bridge is not None:
        y_pred_od = bridge.render_pedal_engine(stim[s_ho], h_od_pre, h_od_post, w_od, subtract_dry=True)
    else:
        from .nonlinear_fit import eval_pedal_features
        u_ho_od = np.convolve(stim[s_ho].astype(np.float64), h_od_pre.astype(np.float64))[: len(stim[s_ho])].astype(np.float32)
        psi_ho, ub_ho = eval_pedal_features(stim[s_ho], u_ho_od, subtract_dry=True, prefer_rust=False)
        y_pred_od = np.convolve((ub_ho + psi_ho @ w_od).astype(np.float64), h_od_post.astype(np.float64))[: len(ub_ho)].astype(np.float32)
    results.append(compute_null_metrics(y_od[s_ho], y_pred_od, sr=sr, label="Synthetic Overdrive Pedal (126-Ch 3-Pole)"))

    print(format_scorecard_markdown(results))

    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        rs_path = export_amp_rust_weights(out_dir / "demo_amp_weights.rs", h_pre, h_post, [0.55], w_amp, prefix="DEMO_AMP")
        nam_doc, nam_m = distill_nam_v054(stim[s_cal][:12288], y_amp[s_cal][:12288], preset="feather", sr=sr, model_name="Demo Synthetic Amp")
        nam_out, _ = save_nam_and_ir(out_dir / "demo_amp.nam", nam_doc, sr=sr)
        print(f"\nExported demo Rust weights -> {rs_path}")
        print(f"Exported demo NAM v0.5.4 ({nam_m['weight_count']} weights, corr={nam_m['corr_pct']:.2f}%) -> {nam_out}")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cleanroom-profiler",
        description="Clean-room black-box audio DSP profiling & Pure-Rust export toolkit.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_disc = sub.add_parser("discover", help="Inspect a locally installed macOS AudioUnit")
    p_disc.add_argument("--type", default="aufx", help="4-char Component Type (aufx / aumf)")
    p_disc.add_argument("--subtype", required=True, help="4-char Component SubType")
    p_disc.add_argument("--mfg", required=True, help="4-char Component Manufacturer")
    p_disc.add_argument("--sr", type=int, default=SR_DEFAULT)
    p_disc.add_argument("--probe-impulse", action="store_true")
    p_disc.set_defaults(func=cmd_discover)

    p_stim = sub.add_parser("export-stimulus", help="Export 15s zero-leakage composite WAV for DAW/hardware re-amping")
    p_stim.add_argument("--out", required=True, help="Output 24-bit WAV path")
    p_stim.add_argument("--di-wav", default=None, help="Optional custom 48 kHz dry guitar DI WAV")
    p_stim.add_argument("--di-peak", type=float, default=0.160, help="Target DI peak (0.160 single-coil, 0.280 humbucker)")
    p_stim.add_argument("--sr", type=int, default=SR_DEFAULT)
    p_stim.set_defaults(func=cmd_export_stimulus)

    p_wp = sub.add_parser("profile-wav-pair", help="Profile any plugin/hardware cross-platform from an input/output WAV pair")
    p_wp.add_argument("--input-wav", required=True)
    p_wp.add_argument("--output-wav", required=True)
    p_wp.add_argument("--out-dir", default="captures/wav_pair")
    p_wp.add_argument("--name", default="WAV Pair Capture")
    p_wp.add_argument("--export-nam", action="store_true")
    p_wp.add_argument("--nam-preset", default="standard", choices=["standard", "feather", "nano"])
    p_wp.set_defaults(func=cmd_profile_wav_pair)

    p_demo = sub.add_parser("demo", help="Run built-in clean-room synthetic Amp + Overdrive capture & verification demo")
    p_demo.add_argument("--out-dir", default=None, help="Optional directory to write exported Rust weights & .nam")
    p_demo.set_defaults(func=cmd_demo)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
