"""
Built-In Clean-Room Synthetic Reference Targets (Tube Amplifier & Overdrive Pedal).

Provides realistic, deterministic physical-modeling reference targets with:
  - Pre-clipping passive bright-cap & 3-band tone stack EQ (`H_pre`)
  - Cold-biased 12AX7 triode bloom + dynamic cathode bypass + B+ power-supply sag
  - Post-clipping output-transformer / tone-control de-fizz filter (`H_post`)
  - Realistic sample latency (`onset = 32` samples)

Used by `examples/run_synthetic_demo.sh` and CI accuracy verification so every contributor
can run the full profiling -> Rust export -> held-out null-test pipeline with zero external
commercial plugins or copyrighted audio files.
"""

from __future__ import annotations

import math
import numpy as np
from .base import ParamInfo, TargetBackend


def _one_pole_lp(x: np.ndarray, fc: float, sr: int) -> np.ndarray:
    a = math.exp(-2.0 * math.pi * fc / sr)
    b = 1.0 - a
    y = np.empty_like(x, dtype=np.float64)
    st = 0.0
    for n in range(len(x)):
        st = a * st + b * float(x[n])
        y[n] = st
    return y


def _one_pole_hp(x: np.ndarray, fc: float, sr: int) -> np.ndarray:
    return x - _one_pole_lp(x, fc, sr)


class SyntheticAmpBackend(TargetBackend):
    """
    Clean-room synthetic tube amplifier with bright switch, mid boost switch, 3-band EQ,
    Gain, Master, Presence, 12AX7 bloom, cathode bypass, B+ sag, and post-clipping de-fizz.
    """

    PID_BRIGHT = 1
    PID_MID_BOOST = 2
    PID_GAIN = 3
    PID_BASS = 4
    PID_MIDDLE = 5
    PID_TREBLE = 6
    PID_MASTER = 7
    PID_PRESENCE = 8

    def __init__(self, sr: int = 48_000, onset_latency: int = 32):
        self.sr = int(sr)
        self.onset_latency = int(onset_latency)
        self._params: dict[int, float] = {
            self.PID_BRIGHT: 0.0,
            self.PID_MID_BOOST: 0.0,
            self.PID_GAIN: 0.5,
            self.PID_BASS: 0.5,
            self.PID_MIDDLE: 0.5,
            self.PID_TREBLE: 0.5,
            self.PID_MASTER: 0.5,
            self.PID_PRESENCE: 0.5,
        }

    def list_parameters(self) -> list[ParamInfo]:
        return [
            ParamInfo(self.PID_BRIGHT, "Bright Switch", 0.0, 1.0, 0.0),
            ParamInfo(self.PID_MID_BOOST, "Mid Boost", 0.0, 1.0, 0.0),
            ParamInfo(self.PID_GAIN, "Gain", 0.0, 1.0, 0.5),
            ParamInfo(self.PID_BASS, "Bass", 0.0, 1.0, 0.5),
            ParamInfo(self.PID_MIDDLE, "Middle", 0.0, 1.0, 0.5),
            ParamInfo(self.PID_TREBLE, "Treble", 0.0, 1.0, 0.5),
            ParamInfo(self.PID_MASTER, "Master", 0.0, 1.0, 0.5),
            ParamInfo(self.PID_PRESENCE, "Presence", 0.0, 1.0, 0.5),
        ]

    def set_param(self, pid: int, value: float) -> None:
        self._params[int(pid)] = float(value)

    def get_param(self, pid: int) -> float:
        return float(self._params.get(int(pid), 0.0))

    def render_mono(self, x_in: np.ndarray) -> np.ndarray:
        x = np.asarray(x_in, dtype=np.float64)
        sr = self.sr

        bright = self._params[self.PID_BRIGHT] >= 0.5
        mid_sw = self._params[self.PID_MID_BOOST] >= 0.5
        gain = float(np.clip(self._params[self.PID_GAIN], 0.01, 1.0))
        bass = float(np.clip(self._params[self.PID_BASS], 0.0, 1.0))
        mid = float(np.clip(self._params[self.PID_MIDDLE], 0.0, 1.0))
        treb = float(np.clip(self._params[self.PID_TREBLE], 0.0, 1.0))
        master = float(np.clip(self._params[self.PID_MASTER], 0.01, 1.0))
        pres = float(np.clip(self._params[self.PID_PRESENCE], 0.0, 1.0))

        # 1. Pre-clipping passive tone stack & non-additive bright/mid switches (H_pre)
        lo = _one_pole_lp(x, 160.0, sr)
        hi_fc = 1400.0 if not mid_sw else 850.0  # Mid cap shifts treble corner down
        hi = _one_pole_hp(x, hi_fc, sr)
        band_mid = x - lo - hi

        # Non-additive switch loading (shared passive ceiling)
        sw_mid_gain = 1.35 if mid_sw else 1.0
        sw_brt_amt = (0.45 * (1.0 - 0.7 * gain)) if bright else 0.0
        if bright and mid_sw:
            sw_mid_gain *= 0.88  # Shared passive divider compression

        pre = (
            (0.65 + 0.70 * bass) * lo
            + (0.60 + 0.80 * mid) * sw_mid_gain * band_mid
            + (0.65 + 0.75 * treb + sw_brt_amt) * hi
        )
        drive_gain = 0.8 + 8.5 * (gain ** 1.65)
        u_pre = _one_pole_hp(pre * drive_gain, 55.0, sr)

        # 2. Tube nonlinearity: 12AX7 bloom + cathode bypass + B+ sag + asymmetric V1->V2 clipping
        env_cath = _one_pole_lp(np.abs(u_pre), 90.0, sr)
        env_sag = _one_pole_lp(np.abs(u_pre), 12.0, sr)
        sag_rail = 1.0 / (1.0 + 0.22 * env_sag)

        # Bounded mid-level bloom (+1.4 dB)
        bloom = u_pre * (1.0 + 0.16 * (np.tanh(u_pre / 0.18) ** 2) * (1.0 - np.tanh(u_pre / 0.36) ** 2))
        v1 = (0.28 * sag_rail) * np.tanh((bloom - 0.03 * env_cath) / (0.28 * sag_rail))
        # Interstage 383 Hz coupling-cap softening into V2 push-pull stage
        v1_excess = v1 - 0.20 * np.tanh(v1 / 0.20)
        v2_in = v1 - 0.18 * _one_pole_lp(v1_excess, 383.0, sr)
        y_nl = np.where(
            v2_in >= 0.0,
            0.22 * np.tanh(v2_in / 0.22),
            0.235 * np.tanh(v2_in / 0.235),
        )

        # 3. Post-clipping output transformer de-fizz + Presence + Master (H_post)
        fc_post = 2800.0 + 2600.0 * pres
        y_post = _one_pole_lp(y_nl, fc_post, sr) * (0.25 + 1.50 * master)

        if self.onset_latency > 0:
            out = np.zeros_like(y_post)
            out[self.onset_latency :] = y_post[: len(y_post) - self.onset_latency]
            return out.astype(np.float32)
        return y_post.astype(np.float32)


class SyntheticOverdriveBackend(TargetBackend):
    """Clean-room synthetic mid-humped soft-clipping overdrive pedal (Drive / Tone / Level)."""

    PID_DRIVE = 1
    PID_TONE = 2
    PID_LEVEL = 3

    def __init__(self, sr: int = 48_000, onset_latency: int = 16):
        self.sr = int(sr)
        self.onset_latency = int(onset_latency)
        self._params = {
            self.PID_DRIVE: 0.5,
            self.PID_TONE: 0.5,
            self.PID_LEVEL: 0.5,
        }

    def list_parameters(self) -> list[ParamInfo]:
        return [
            ParamInfo(self.PID_DRIVE, "Drive", 0.0, 1.0, 0.5),
            ParamInfo(self.PID_TONE, "Tone", 0.0, 1.0, 0.5),
            ParamInfo(self.PID_LEVEL, "Level", 0.0, 1.0, 0.5),
        ]

    def set_param(self, pid: int, value: float) -> None:
        self._params[int(pid)] = float(value)

    def get_param(self, pid: int) -> float:
        return float(self._params.get(int(pid), 0.0))

    def render_mono(self, x_in: np.ndarray) -> np.ndarray:
        x = np.asarray(x_in, dtype=np.float64)
        sr = self.sr
        drive = float(np.clip(self._params[self.PID_DRIVE], 0.0, 1.0))
        tone = float(np.clip(self._params[self.PID_TONE], 0.0, 1.0))
        level = float(np.clip(self._params[self.PID_LEVEL], 0.0, 1.0))

        # Pre-clipping feedback HPF (~720 Hz mid-hump) + Drive gain
        x_hp = _one_pole_hp(x, 720.0, sr)
        u_fb = (1.5 + 14.0 * drive) * x_hp
        # Op-amp feedback soft diode clipping + clean bleed (TS-style topology)
        u_lp_slew = _one_pole_lp(u_fb, 2200.0, sr)
        clipped = x + 0.14 * np.tanh(u_lp_slew / 0.14) + 0.015 * np.tanh((u_lp_slew + 0.08) / 0.12)
        clipped = clipped - _one_pole_lp(clipped, 14.0, sr)

        # Post-clipping Tone low-pass + output level taper
        fc_tone = 900.0 + 4800.0 * (tone ** 1.3)
        y = _one_pole_lp(clipped, fc_tone, sr) * (0.2 + 1.6 * level)

        if self.onset_latency > 0:
            out = np.zeros_like(y)
            out[self.onset_latency :] = y[: len(y) - self.onset_latency]
            return out.astype(np.float32)
        return y.astype(np.float32)
