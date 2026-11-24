"""
Cross-Platform Offline WAV-Pair & Hardware Re-Amping Backend (Windows / macOS / Linux).

Allows users on ANY OS to profile ANY commercial VST3/CLAP/AAX plugin (rendered offline in
their DAW) or any physical tube amp / pedal (captured via audio interface re-amping) using
100% of the Wiener-Hammerstein + Bloom + Sag + 180-ch KONE / 126-ch OD calibration engine.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
from ..stimulus import SR_DEFAULT, load_wav_mono
from .base import ParamInfo, TargetBackend


class WavPairBackend(TargetBackend):
    """Wraps a pre-recorded `(input_wav, output_wav)` capture pair with automatic lag alignment."""

    def __init__(self, input_wav: str | Path, output_wav: str | Path, max_lag: int = 4096):
        self.x_in, sr_in = load_wav_mono(input_wav)
        self.y_raw, sr_out = load_wav_mono(output_wav)
        if sr_in != sr_out:
            raise ValueError(f"Sample rate mismatch: input={sr_in} Hz vs output={sr_out} Hz")
        self.sr = sr_in
        self.lag = self._estimate_lag(self.x_in, self.y_raw, max_lag=max_lag)
        self.y_aligned = self._apply_lag(self.y_raw, self.lag, len(self.x_in))

    @staticmethod
    def _estimate_lag(x: np.ndarray, y: np.ndarray, max_lag: int = 4096) -> int:
        n = min(len(x), len(y), 2 * SR_DEFAULT)
        if n < 256:
            return 0
        # Use FFT cross-correlation on the first 2 seconds (or entire clip)
        nfft = 1 << (2 * n - 1).bit_length()
        X = np.fft.rfft(x[:n].astype(np.float64), nfft)
        Y = np.fft.rfft(y[:n].astype(np.float64), nfft)
        corr = np.fft.irfft(Y * np.conj(X), nfft)
        search = min(max_lag, n // 2)
        pos_corr = np.abs(corr[:search])
        best_pos = int(np.argmax(pos_corr))
        return best_pos

    @staticmethod
    def _apply_lag(y: np.ndarray, lag: int, target_len: int) -> np.ndarray:
        out = np.zeros(target_len, dtype=np.float32)
        if lag >= 0:
            avail = min(target_len, max(0, len(y) - lag))
            if avail > 0:
                out[:avail] = y[lag : lag + avail]
        else:
            shift = -lag
            avail = min(target_len - shift, len(y))
            if avail > 0:
                out[shift : shift + avail] = y[:avail]
        return out

    def list_parameters(self) -> list[ParamInfo]:
        return []

    def set_param(self, pid: int, value: float) -> None:
        pass

    def get_param(self, pid: int) -> float:
        return 0.0

    def render_mono(self, x_in: np.ndarray) -> np.ndarray:
        n = min(len(x_in), len(self.y_aligned))
        out = np.zeros(len(x_in), dtype=np.float32)
        out[:n] = self.y_aligned[:n]
        return out
