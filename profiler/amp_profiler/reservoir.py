"""
Clean-Room Analytical Multi-Scale KONE Recurrent Reservoir Parameters.

Matches `dsp-core-rust/src/reservoir.rs` bit-for-bit. Zero third-party extracted constants.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class AmpReservoirSpec:
    s1_in: np.ndarray
    s1_diag: np.ndarray
    s1_cross: np.ndarray
    s1_bias: np.ndarray
    s1_wout: np.ndarray
    c1_ref: np.ndarray
    s2_in: np.ndarray
    s2_diag: np.ndarray
    s2_cross: np.ndarray
    s2_bias: np.ndarray
    c2_ref: np.ndarray


@dataclass(frozen=True)
class PedalReservoirSpec:
    in_w: np.ndarray
    diag: np.ndarray
    cross: np.ndarray
    bias: np.ndarray
    c_ref: np.ndarray


AMP_RESERVOIR_DEFAULT = AmpReservoirSpec(
    s1_in=np.array([0.520, -0.680, 0.850, -1.040, 0.760, -0.920, 0.640, -0.810], dtype=np.float32),
    s1_diag=np.array([0.420, 0.510, 0.610, 0.700, 0.780, 0.850, 0.560, 0.740], dtype=np.float32),
    s1_cross=np.array([-0.110, 0.095, -0.085, 0.080, -0.075, 0.065, -0.090, 0.070], dtype=np.float32),
    s1_bias=np.array([0.025, -0.032, 0.018, -0.045, 0.038, -0.022, 0.030, -0.028], dtype=np.float32),
    s1_wout=np.array([0.140, -0.115, 0.125, -0.095, 0.105, -0.085, 0.090, -0.075], dtype=np.float32),
    c1_ref=np.array([0.120, 0.280, 0.650, 1.500, 3.600, 8.500, 0.450, 1.800], dtype=np.float32),
    s2_in=np.array([0.480, -0.620, 0.790, -0.960, 0.880, -1.120, 0.590, -0.740], dtype=np.float32),
    s2_diag=np.array([0.460, 0.550, 0.650, 0.740, 0.820, 0.880, 0.600, 0.770], dtype=np.float32),
    s2_cross=np.array([0.100, -0.090, 0.080, -0.070, 0.065, -0.055, 0.085, -0.065], dtype=np.float32),
    s2_bias=np.array([-0.018, 0.024, -0.015, 0.032, -0.028, 0.020, -0.022, 0.016], dtype=np.float32),
    c2_ref=np.array([0.160, 0.380, 0.850, 2.000, 4.800, 11.000, 0.600, 2.400], dtype=np.float32),
)

PEDAL_RESERVOIR_DEFAULT = PedalReservoirSpec(
    in_w=np.array([0.560, -0.440, -0.610, 0.780, -0.890, 1.080, -0.670, 0.820], dtype=np.float32),
    diag=np.array([0.400, 0.490, 0.590, 0.690, 0.790, 0.870, 0.460, 0.680], dtype=np.float32),
    cross=np.array([-0.120, 0.085, -0.075, 0.070, -0.060, 0.050, -0.080, 0.055], dtype=np.float32),
    bias=np.array([0.016, -0.022, 0.014, -0.030, 0.026, -0.018, 0.020, -0.024], dtype=np.float32),
    c_ref=np.array([0.100, 0.250, 0.600, 1.500, 3.800, 9.500, 0.450, 1.800], dtype=np.float32),
)
