"""
15-Second Zero-Leakage Composite Calibration & Verification Stimulus + Clean-Room WAV I/O.

Stimulus layout (720,000 samples @ 48 kHz = 15.0 seconds):
  - Segment 1 (0.0s - 4.0s): Dual-level logarithmic sine sweeps (-36 dBFS linear vs. driven)
    for Wiener-Hammerstein H_pre / H_post separation.
  - Segment 2 (4.0s - 8.0s): Dynamic amplitude ramp + gated multi-frequency tone bursts
    (exciting cathode bypass ~1.5 ms, B+ sag ~12 ms, grid bias ~9 ms) + calibration guitar DI.
  - Segment 3 (8.0s - 15.0s): Strictly held-out musical verification segment (NEVER enters
    regression fitting — enforces zero test-set leakage).
"""

from __future__ import annotations

import math
import struct
import wave
from pathlib import Path
import numpy as np

SR_DEFAULT = 48_000
SINGLE_COIL_DI_PEAK = 0.160  # -15.9 dBFS passive single-coil pickup calibration
HUMBUCKER_DI_PEAK = 0.280    # -11.1 dBFS vintage humbucker calibration


def load_wav_mono(path: str | Path, target_peak: float | None = None) -> tuple[np.ndarray, int]:
    """Loads 16-bit PCM, 24-bit PCM, 32-bit PCM, or 32-bit IEEE float WAV into mono float32."""
    p = Path(path)
    raw_bytes = p.read_bytes()
    if len(raw_bytes) < 44 or raw_bytes[:4] != b"RIFF" or raw_bytes[8:12] != b"WAVE":
        raise ValueError(f"Unsupported or invalid WAV file: {p}")

    # Parse chunks cleanly to support IEEE float (fmt tag 3) and PCM (fmt tag 1)
    pos = 12
    fmt_tag = 1
    n_ch = 1
    sr = SR_DEFAULT
    bits = 16
    data_chunk = b""

    while pos + 8 <= len(raw_bytes):
        cid = raw_bytes[pos : pos + 4]
        csz = struct.unpack_from("<I", raw_bytes, pos + 4)[0]
        cstart = pos + 8
        cend = min(cstart + csz, len(raw_bytes))
        if cid == b"fmt " and csz >= 16:
            fmt_tag, n_ch, sr, _, _, bits = struct.unpack_from("<HHIIHH", raw_bytes, cstart)
        elif cid == b"data":
            data_chunk = raw_bytes[cstart:cend]
        pos = cstart + csz + (csz & 1)

    if not data_chunk:
        raise ValueError(f"No data chunk found in WAV: {p}")

    if fmt_tag == 3 and bits == 32:
        audio = np.frombuffer(data_chunk, dtype="<f4").astype(np.float32)
    elif fmt_tag == 1 and bits == 16:
        audio = np.frombuffer(data_chunk, dtype="<i2").astype(np.float32) / 32768.0
    elif fmt_tag == 1 and bits == 24:
        b = np.frombuffer(data_chunk, dtype=np.uint8).reshape(-1, 3)
        signed = (
            b[:, 0].astype(np.int32)
            | (b[:, 1].astype(np.int32) << 8)
            | (b[:, 2].astype(np.int8).astype(np.int32) << 16)
        )
        audio = signed.astype(np.float32) / 8388608.0
    elif fmt_tag == 1 and bits == 32:
        audio = np.frombuffer(data_chunk, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV format: tag={fmt_tag}, bits={bits}")

    if n_ch > 1:
        audio = audio.reshape(-1, n_ch)[:, 0].copy()

    if target_peak is not None:
        pk = float(np.max(np.abs(audio)))
        if pk > 1e-7:
            audio = (audio * (target_peak / pk)).astype(np.float32)

    return audio.astype(np.float32), int(sr)


def save_wav_24bit_pcm(path: str | Path, audio: np.ndarray, sr: int = SR_DEFAULT) -> None:
    """Writes mono float audio to hardware-compatible 24-bit signed little-endian PCM WAV."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    x = np.clip(np.asarray(audio, dtype=np.float64), -1.0, 1.0 - 1.0 / 8388608.0)
    q = np.round(x * 8388607.0).astype(np.int32)
    raw = np.empty((len(q), 3), dtype=np.uint8)
    raw[:, 0] = q & 0xFF
    raw[:, 1] = (q >> 8) & 0xFF
    raw[:, 2] = (q >> 16) & 0xFF
    with wave.open(str(p), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(3)
        wf.setframerate(int(sr))
        wf.writeframes(raw.tobytes())


def generate_procedural_guitar_di(
    duration_s: float = 12.0,
    sr: int = SR_DEFAULT,
    peak: float = SINGLE_COIL_DI_PEAK,
    seed: int = 42,
) -> np.ndarray:
    """
    Generates a clean-room, copyright-free procedural guitar-like DI signal with realistic
    pick-attack transients, multi-harmonic string overtones, palm-mute dynamics, single-note
    melodic phrases, and multi-velocity chords across E2 (82.4 Hz) to E5 (659.3 Hz).
    """
    rng = np.random.default_rng(seed)
    n_total = int(duration_s * sr)
    out = np.zeros(n_total, dtype=np.float64)

    # Sequence of musical events: (start_s, dur_s, freqs_hz, velocity, decay_s)
    events = [
        # Calibration phrase (0.0s - 4.0s)
        (0.05, 0.70, [82.41, 123.47, 164.81], 0.85, 0.24),       # E5 power chord
        (0.80, 0.45, [196.00], 0.55, 0.16),                      # G3 single note
        (1.28, 0.50, [220.00, 277.18, 329.63], 0.95, 0.20),      # A major triad (hard pick)
        (1.85, 0.35, [293.66], 0.40, 0.14),                      # D4 soft pluck
        (2.25, 0.65, [110.00, 164.81, 220.00, 329.63], 1.00, 0.28),
        (2.95, 0.40, [329.63, 392.00], 0.65, 0.15),              # Double stop
        (3.40, 0.55, [98.00, 146.83, 196.00], 0.90, 0.22),       # G5 chord
        # Held-out phrase (4.0s - 12.0s: distinct pitches, rhythms & dynamics)
        (4.10, 0.80, [146.83, 220.00, 293.66, 369.99], 0.92, 0.30),  # D major chord
        (5.00, 0.35, [440.00], 0.70, 0.14),
        (5.40, 0.35, [392.00], 0.50, 0.15),
        (5.80, 0.45, [329.63], 0.88, 0.18),
        (6.30, 0.85, [130.81, 196.00, 261.63, 329.63], 1.00, 0.32),  # Cmaj7
        (7.25, 0.30, [92.50], 0.95, 0.10),                           # Palm-mute F#2
        (7.60, 0.30, [92.50], 0.98, 0.10),
        (7.95, 0.65, [123.47, 185.00, 246.94, 369.99], 0.75, 0.26),  # B minor
        (8.70, 0.50, [493.88, 587.33], 0.60, 0.19),
        (9.30, 0.95, [82.41, 164.81, 246.94, 329.63, 493.88], 1.00, 0.40),
        (10.35, 0.60, [246.94, 311.13, 369.99], 0.45, 0.22),         # Soft clean chord
        (11.05, 0.85, [110.00, 164.81, 220.00, 277.18], 0.82, 0.30),
    ]

    for start_s, dur_s, freqs, vel, decay_s in events:
        i0 = int(start_s * sr)
        n_ev = min(int(dur_s * sr), n_total - i0)
        if n_ev <= 0:
            continue
        t = np.arange(n_ev, dtype=np.float64) / sr
        attack = np.minimum(t / 0.0018, 1.0)
        # Brief broadband pick click transient (~2.5 ms)
        pick_click = rng.normal(0.0, 0.18, n_ev) * np.exp(-t / 0.0022)
        ev_sig = pick_click * vel
        for f0 in freqs:
            # Slight natural pitch droop after hard pick attack
            f_inst = f0 * (1.0 + 0.004 * vel * np.exp(-t / 0.025))
            phase = 2.0 * math.pi * np.cumsum(f_inst) / sr
            for h in range(1, 8):
                # Higher harmonics decay faster like real wound/plain guitar strings
                h_decay = np.exp(-t / (decay_s / (1.0 + 0.38 * (h - 1))))
                h_amp = (1.0 / (h ** 1.15)) * (0.85 if h % 2 == 0 else 1.0)
                ev_sig += vel * h_amp * h_decay * np.sin(h * phase + 0.31 * h)
        # Gentle 15 ms tail fade
        fade_n = min(int(0.015 * sr), n_ev)
        ev_sig[-fade_n:] *= 0.5 * (1.0 + np.cos(np.linspace(0.0, math.pi, fade_n)))
        out[i0 : i0 + n_ev] += attack * ev_sig

    pk = float(np.max(np.abs(out))) + 1e-12
    return (out * (peak / pk)).astype(np.float32)


def build_15s_composite_stimulus(
    di_audio: np.ndarray | None = None,
    sr: int = SR_DEFAULT,
    di_peak: float = SINGLE_COIL_DI_PEAK,
) -> tuple[np.ndarray, dict[str, tuple[int, int]]]:
    """
    Constructs the mandatory 15.0-second (720,000 samples @ 48 kHz) Zero-Leakage Stimulus.

    Returns `(stimulus_f32, segment_slices)` where `segment_slices` marks exact sample ranges:
      - `sweep_linear`:  [0.0s .. 2.0s]  (-36 dBFS log sweep 20 Hz -> 20 kHz)
      - `sweep_driven`:  [2.0s .. 4.0s]  (driven log sweep 20 Hz -> 20 kHz)
      - `calibration`:   [4.0s .. 8.0s]  (dynamic ramp + tone bursts + calibration DI)
      - `heldout_test`:  [8.0s .. 15.0s] (STRICTLY HELD-OUT verification DI)
    """
    if di_audio is None:
        di = generate_procedural_guitar_di(duration_s=12.0, sr=sr, peak=di_peak)
    else:
        di = np.asarray(di_audio, dtype=np.float32)
        pk = float(np.max(np.abs(di))) + 1e-12
        di = (di * (di_peak / pk)).astype(np.float32)
        if len(di) < int(10.0 * sr):
            # Tile with polarity flip if user provided a short clip, keeping cal/test distinct
            pad = generate_procedural_guitar_di(duration_s=12.0, sr=sr, peak=di_peak)
            di = np.concatenate([di, pad])[: int(12.0 * sr)]

    stim = np.zeros(15 * sr, dtype=np.float32)

    # Segment 1a & 1b (0.0s - 4.0s): Dual-level log sine sweeps
    t_sw = np.arange(2 * sr, dtype=np.float64) / sr
    f1, f2 = 20.0, 20_000.0
    l_const = 2.0 / math.log(f2 / f1)
    phase = 2.0 * math.pi * f1 * l_const * (np.exp(t_sw / l_const) - 1.0)
    win = np.ones_like(t_sw)
    fade = int(0.02 * sr)
    win[:fade] = 0.5 * (1.0 - np.cos(np.pi * np.arange(fade) / fade))
    win[-fade:] = 0.5 * (1.0 + np.cos(np.pi * np.arange(fade) / fade))

    amp_low = 10.0 ** (-36.0 / 20.0)  # 0.01585 (-36 dBFS linear regime)
    amp_high = 0.28                   # Driven regime for H_post separation
    stim[0 : 2 * sr] = (amp_low * np.sin(phase) * win).astype(np.float32)
    stim[2 * sr : 4 * sr] = (amp_high * np.sin(phase) * win).astype(np.float32)

    # Segment 2a (4.0s - 5.8s): Dynamic amplitude ramp + gated multi-freq bursts (cath/sag/grid)
    n_dyn = int(1.8 * sr)
    t_dyn = np.arange(n_dyn, dtype=np.float64) / sr
    env = np.sin(np.pi * t_dyn / 1.8) ** 1.4
    burst = (
        0.50 * np.sin(2.0 * math.pi * 110.0 * t_dyn)
        + 0.35 * np.sin(2.0 * math.pi * 330.0 * t_dyn)
        + 0.25 * np.sin(2.0 * math.pi * 820.0 * t_dyn)
        + 0.15 * np.sin(2.0 * math.pi * 2000.0 * t_dyn)
    )
    gate_env = 0.5 + 0.5 * np.sign(np.sin(2.0 * math.pi * 3.5 * t_dyn))
    stim[4 * sr : 4 * sr + n_dyn] = (0.24 * env * gate_env * burst).astype(np.float32)

    # Segment 2b (5.8s - 8.0s): Calibration DI slice [0.0s .. 2.2s] (strictly disjoint from held-out)
    n_cal_di = int(2.2 * sr)
    stim[4 * sr + n_dyn : 8 * sr] = di[:n_cal_di]

    # Segment 3 (8.0s - 15.0s): Strictly held-out verification DI slice [4.5s .. 11.5s]
    ho_start = int(4.5 * sr)
    n_ho = 7 * sr
    stim[8 * sr : 15 * sr] = di[ho_start : ho_start + n_ho]

    slices = {
        "sweep_linear": (0, 2 * sr),
        "sweep_driven": (2 * sr, 4 * sr),
        "calibration": (4 * sr, 8 * sr),
        "heldout_test": (8 * sr, 15 * sr),
    }
    return stim, slices
