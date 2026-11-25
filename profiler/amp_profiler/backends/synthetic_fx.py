"""
Built-In Clean-Room Synthetic Time-Based, Modulation & Dynamics Reference Targets.

Provides zero-dependency physical test targets for CI verification and user demos:
  - `SyntheticSpringReverbBackend`: Multi-mode spring tank (`Decay`, `Tone`, `Mix`)
  - `SyntheticHarmonicTremoloBackend`: Dual-band LP/HP opposite-phase harmonic tremolo (`Speed`, `Intensity`)
  - `SyntheticBbdDelayBackend`: Analog BBD delay (`Time`, `Feedback`, `Mix`, `Tone`) with 24-sample extra loop delay
  - `SyntheticEnvelopeFilterBackend`: Vactrol + TPT 2-pole SVF auto-wah (`Drive`, `Peak`)
  - `SyntheticCompressorBackend`: Soft-knee feed-forward compressor (known threshold/ratio/knee/attack/release)
"""

from __future__ import annotations

import math
import numpy as np
from .base import ParamInfo, TargetBackend
from ..time_mod_fx import eval_envelope_filter_py


class SyntheticSpringReverbBackend(TargetBackend):
    PID_DECAY = 1
    PID_TONE = 2
    PID_MIX = 3

    def __init__(self, sr: int = 48_000, onset_latency: int = 16):
        self.sr = int(sr)
        self.onset_latency = int(onset_latency)
        self._params = {self.PID_DECAY: 0.5, self.PID_TONE: 1.0, self.PID_MIX: 0.5}
        # Deterministic spring-like multi-chirp impulse templates
        rng = np.random.default_rng(101)
        self._base_noise = rng.normal(0.0, 1.0, size=16_384).astype(np.float64)
        t = np.arange(16_384, dtype=np.float64) / sr
        self._chirp = np.sin(2.0 * math.pi * (420.0 * t + 1800.0 * t * t)) * 0.35 + self._base_noise * 0.65

    def list_parameters(self) -> list[ParamInfo]:
        return [
            ParamInfo(self.PID_DECAY, "Decay", 0.0, 1.0, 0.5),
            ParamInfo(self.PID_TONE, "Tone", 0.0, 1.0, 1.0),
            ParamInfo(self.PID_MIX, "Mix", 0.0, 1.0, 0.5),
        ]

    def set_param(self, pid: int, value: float) -> None:
        self._params[int(pid)] = float(value)

    def get_param(self, pid: int) -> float:
        return float(self._params.get(int(pid), 0.0))

    def render_mono(self, x_in: np.ndarray) -> np.ndarray:
        x = np.asarray(x_in, dtype=np.float64)
        sr = self.sr
        decay = float(np.clip(self._params[self.PID_DECAY], 0.0, 1.0))
        tone = float(np.clip(self._params[self.PID_TONE], 0.0, 1.0))
        mix = float(np.clip(self._params[self.PID_MIX], 0.0, 1.0))

        t = np.arange(len(self._chirp), dtype=np.float64) / sr
        tau = 0.045 + 0.16 * decay
        ir = self._chirp * np.exp(-t / tau)
        ir[:48] = 0.0  # 1 ms pre-delay
        ir *= 0.25 / (np.linalg.norm(ir) + 1e-12)

        wet = np.convolve(x, ir, mode="full")[: len(x)]
        # Tone 1-pole lowpass (800 Hz .. 18 kHz)
        fc = 800.0 * ((18_000.0 / 800.0) ** tone)
        a = math.exp(-2.0 * math.pi * fc / sr)
        b = 1.0 - a
        st = 0.0
        wet_t = np.empty_like(wet)
        for i in range(len(wet)):
            st = a * st + b * wet[i]
            wet_t[i] = st

        dry_g = 1.0 - mix
        wet_g = mix
        y = dry_g * x + wet_g * wet_t

        if self.onset_latency > 0:
            out = np.zeros_like(y)
            out[self.onset_latency :] = y[: len(y) - self.onset_latency]
            return out.astype(np.float32)
        return y.astype(np.float32)


class SyntheticHarmonicTremoloBackend(TargetBackend):
    PID_SPEED = 1
    PID_INTENSITY = 2

    def __init__(self, sr: int = 48_000, onset_latency: int = 8):
        self.sr = int(sr)
        self.onset_latency = int(onset_latency)
        self._params = {self.PID_SPEED: 0.5, self.PID_INTENSITY: 0.7}

    def list_parameters(self) -> list[ParamInfo]:
        return [
            ParamInfo(self.PID_SPEED, "Speed", 0.0, 1.0, 0.5),
            ParamInfo(self.PID_INTENSITY, "Intensity", 0.0, 1.0, 0.7),
        ]

    def set_param(self, pid: int, value: float) -> None:
        self._params[int(pid)] = float(value)

    def get_param(self, pid: int) -> float:
        return float(self._params.get(int(pid), 0.0))

    def true_lfo_hz(self, speed: float) -> float:
        return 1.2 + 9.0 * (speed ** 2)

    def render_mono(self, x_in: np.ndarray) -> np.ndarray:
        x = np.asarray(x_in, dtype=np.float64)
        sr = self.sr
        sp = float(np.clip(self._params[self.PID_SPEED], 0.0, 1.0))
        inten = float(np.clip(self._params[self.PID_INTENSITY], 0.0, 1.0))
        f_lfo = self.true_lfo_hz(sp)

        # Dual-band crossover at 650 Hz: LP and HP modulated out of phase
        a = math.exp(-2.0 * math.pi * 650.0 / sr)
        b = 1.0 - a
        lp = np.empty_like(x)
        st = 0.0
        for i in range(len(x)):
            st = a * st + b * x[i]
            lp[i] = st
        hp = x - lp

        t = np.arange(len(x), dtype=np.float64) / sr
        lfo = np.sin(2.0 * math.pi * f_lfo * t)
        g_lo = 1.0 - 0.5 * inten * (1.0 - lfo)
        g_hi = 1.0 - 0.5 * inten * (1.0 + lfo)
        y = g_lo * lp + g_hi * hp

        if self.onset_latency > 0:
            out = np.zeros_like(y)
            out[self.onset_latency :] = y[: len(y) - self.onset_latency]
            return out.astype(np.float32)
        return y.astype(np.float32)


class SyntheticBbdDelayBackend(TargetBackend):
    PID_TIME = 1
    PID_FEEDBACK = 2
    PID_MIX = 3
    PID_TONE = 4

    def __init__(self, sr: int = 48_000, onset_latency: int = 12, loop_extra_delay: int = 24):
        self.sr = int(sr)
        self.onset_latency = int(onset_latency)
        self.loop_extra_delay = int(loop_extra_delay)
        self._params = {self.PID_TIME: 0.2, self.PID_FEEDBACK: 0.4, self.PID_MIX: 0.5, self.PID_TONE: 1.0}

    def list_parameters(self) -> list[ParamInfo]:
        return [
            ParamInfo(self.PID_TIME, "Time", 0.0, 1.0, 0.2),
            ParamInfo(self.PID_FEEDBACK, "Feedback", 0.0, 1.0, 0.4),
            ParamInfo(self.PID_MIX, "Mix", 0.0, 1.0, 0.5),
            ParamInfo(self.PID_TONE, "Tone", 0.0, 1.0, 1.0),
        ]

    def set_param(self, pid: int, value: float) -> None:
        self._params[int(pid)] = float(value)

    def get_param(self, pid: int) -> float:
        return float(self._params.get(int(pid), 0.0))

    def render_mono(self, x_in: np.ndarray) -> np.ndarray:
        x = np.asarray(x_in, dtype=np.float64)
        sr = self.sr
        t_val = float(np.clip(self._params[self.PID_TIME], 0.0, 1.0))
        fb = float(np.clip(self._params[self.PID_FEEDBACK], 0.0, 1.0)) * 0.85
        mix = float(np.clip(self._params[self.PID_MIX], 0.0, 1.0))
        tone = float(np.clip(self._params[self.PID_TONE], 0.0, 1.0))

        d0 = int(round((0.025 + 0.275 * t_val) * sr))
        # Account for the 1-sample causal `fb_prev` latch so physical R1->R2 offset == loop_extra_delay
        d_fb = d0 + max(0, self.loop_extra_delay - 1)
        fc_loop = 1100.0 + 3800.0 * tone
        a_lp = math.exp(-2.0 * math.pi * fc_loop / sr)
        b_lp = 1.0 - a_lp

        buf = np.zeros(len(x) + d_fb + 64, dtype=np.float64)
        y = np.zeros(len(x), dtype=np.float64)
        lp_st = 0.0
        fb_prev = 0.0
        for i in range(len(x)):
            # BBD soft rail saturation on input + feedback
            u = x[i] + fb_prev
            u_sat = 0.28 * math.tanh(u / 0.28)
            buf[i + d_fb] = u_sat
            tap_r1 = buf[i + d_fb - d0]
            tap_r2 = buf[i]
            lp_st = a_lp * lp_st + b_lp * tap_r2
            fb_prev = lp_st * fb
            y[i] = (1.0 - mix) * x[i] + mix * tap_r1

        if self.onset_latency > 0:
            out = np.zeros_like(y)
            out[self.onset_latency :] = y[: len(y) - self.onset_latency]
            return out.astype(np.float32)
        return y.astype(np.float32)


class SyntheticEnvelopeFilterBackend(TargetBackend):
    PID_DRIVE = 1
    PID_PEAK = 2

    def __init__(self, sr: int = 48_000, onset_latency: int = 8):
        self.sr = int(sr)
        self.onset_latency = int(onset_latency)
        self._params = {self.PID_DRIVE: 0.65, self.PID_PEAK: 0.65}
        self.true_params_14 = np.array(
            [70.0, 520.0, 28.0, 2.4, 0.007, 0.21, 0.012, 1.9, 190.0, 1850.0, 4.8, 0.7, 1.15, 0.30],
            dtype=np.float32,
        )

    def list_parameters(self) -> list[ParamInfo]:
        return [
            ParamInfo(self.PID_DRIVE, "Drive", 0.0, 1.0, 0.65),
            ParamInfo(self.PID_PEAK, "Peak", 0.0, 1.0, 0.65),
        ]

    def set_param(self, pid: int, value: float) -> None:
        self._params[int(pid)] = float(value)

    def get_param(self, pid: int) -> float:
        return float(self._params.get(int(pid), 0.0))

    def render_mono(self, x_in: np.ndarray) -> np.ndarray:
        drv = float(np.clip(self._params[self.PID_DRIVE], 0.0, 1.0))
        pk = float(np.clip(self._params[self.PID_PEAK], 0.0, 1.0))
        q_scale = 0.55 + 0.90 * pk
        y = eval_envelope_filter_py(
            x_in, self.true_params_14, sr=self.sr, drive_knob=drv, peak_q_scale=q_scale
        )
        if self.onset_latency > 0:
            out = np.zeros_like(y)
            out[self.onset_latency :] = y[: len(y) - self.onset_latency]
            return out
        return y


class SyntheticCompressorBackend(TargetBackend):
    def __init__(
        self,
        sr: int = 48_000,
        threshold_db: float = -20.0,
        ratio: float = 4.0,
        knee_db: float = 6.0,
        attack_ms: float = 12.0,
        release_ms: float = 110.0,
        makeup_db: float = 4.5,
        onset_latency: int = 0,
    ):
        self.sr = int(sr)
        self.threshold_db = threshold_db
        self.ratio = ratio
        self.knee_db = knee_db
        self.attack_ms = attack_ms
        self.release_ms = release_ms
        self.makeup_db = makeup_db
        self.onset_latency = int(onset_latency)

    def list_parameters(self) -> list[ParamInfo]:
        return []

    def set_param(self, pid: int, value: float) -> None:
        pass

    def get_param(self, pid: int) -> float:
        return 0.0

    def render_mono(self, x_in: np.ndarray) -> np.ndarray:
        x = np.asarray(x_in, dtype=np.float64)
        sr = self.sr
        a_atk = math.exp(-1.0 / (self.attack_ms * 0.001 * sr))
        a_rel = math.exp(-1.0 / (self.release_ms * 0.001 * sr))
        slope = 1.0 / self.ratio - 1.0
        knee = self.knee_db
        thresh = self.threshold_db

        # RMS-like rectified envelope in dB for smooth steady-state transfer curve
        a_det = math.exp(-1.0 / (0.004 * sr))
        sq_st = 1e-12
        gr_st = 0.0
        y = np.empty_like(x)
        for i in range(len(x)):
            xi = float(x[i])
            sq_st = a_det * sq_st + (1.0 - a_det) * (xi * xi)
            # Peak-equivalent dB of sine = 20*log10(sqrt(2*sq_st))
            lvl_db = 10.0 * math.log10(max(2.0 * sq_st, 1e-14))
            over = lvl_db - thresh
            if 2.0 * over <= -knee:
                tgt_gr = 0.0
            elif 2.0 * over >= knee:
                tgt_gr = slope * over
            else:
                z = over + 0.5 * knee
                tgt_gr = slope * (z * z) / (2.0 * knee)
            c = a_atk if tgt_gr < gr_st else a_rel
            gr_st = c * gr_st + (1.0 - c) * tgt_gr
            y[i] = xi * (10.0 ** ((gr_st + self.makeup_db) / 20.0))

        if self.onset_latency > 0:
            out = np.zeros_like(y)
            out[self.onset_latency :] = y[: len(y) - self.onset_latency]
            return out.astype(np.float32)
        return y.astype(np.float32)
