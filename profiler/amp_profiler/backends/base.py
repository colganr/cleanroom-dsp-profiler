"""Pluggable TargetBackend abstract base class for live plugins, offline WAV pairs, and hardware."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ParamInfo:
    pid: int
    name: str
    min_value: float
    max_value: float
    default_value: float


class TargetBackend(ABC):
    """Abstract interface for any black-box audio target (AU, VST3, CLAP, WAV pair, hardware)."""

    sr: int = 48_000

    @abstractmethod
    def list_parameters(self) -> list[ParamInfo]:
        """Returns all host-exposed parameters."""

    @abstractmethod
    def set_param(self, pid: int, value: float) -> None:
        """Sets parameter `pid` to `value`."""

    @abstractmethod
    def get_param(self, pid: int) -> float:
        """Reads current value of parameter `pid`."""

    def settle(self, full: bool = False) -> None:
        """Flushes internal state and allows async background loaders/smoothers to settle."""

    @abstractmethod
    def render_mono(self, x_in: np.ndarray) -> np.ndarray:
        """Processes a mono float32 array and returns the mono float32 output of equal length."""

    def probe_onset_latency(self, impulse_amp: float = 1e-3) -> tuple[int, float]:
        """
        Measures exact causal sample onset latency and peak value using a bipolar symmetric
        small-signal impulse (`0.5 * (y_+ - y_-)`).

        Bipolar subtraction cancels asymmetric tube/diode DC bias settle transients, and
        detecting the first sample exceeding `0.1%` of peak prevents low-pass filters (tone
        stacks, cab IRs) from shifting the apparent onset by 1-3 samples past tap 0.
        """
        pre_pad = 100
        imp_pos = np.zeros(2048, dtype=np.float32)
        imp_neg = np.zeros(2048, dtype=np.float32)
        imp_pos[pre_pad] = impulse_amp
        imp_neg[pre_pad] = -impulse_amp

        self.settle(full=True)
        y_pos = self.render_mono(imp_pos)
        self.settle(full=False)
        y_neg = self.render_mono(imp_neg)

        y_bi = 0.5 * (y_pos.astype(np.float64) - y_neg.astype(np.float64))
        tail = y_bi[pre_pad:]
        pk = float(np.max(np.abs(tail)))
        if pk < 1e-12:
            return 0, 0.0
        causal_onset = int(np.argmax(np.abs(tail) > pk * 1e-3))
        peak_val = float(tail[int(np.argmax(np.abs(tail)))])
        return causal_onset, peak_val
