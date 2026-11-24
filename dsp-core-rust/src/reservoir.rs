//! Clean-Room Analytical Multi-Scale KONE Recurrent State-Space Reservoirs.
//!
//! These reservoir matrices are constructed 100% mathematically (zero extracted third-party
//! parameters) as spectral-radius-bounded (`|diag| + |cross| < 0.96`), multi-timescale
//! Echo-State / state-space reservoirs:
//! - Diagonal poles span fast transient (~0.40) to slow coupling-cap decay (~0.92).
//! - Circulant neighbor coupling (`cross`) introduces rich damped cross-mode dynamics.
//! - Input projections (`in_w`) and gentle bias asymmetries (`bias`) excite both odd and
//!   even harmonic families while every readout uses exact zero-slope residual subtraction
//!   so small-signal linear transfer functions remain strictly unperturbed (`r'(0) = 0`).

/// Fixed reservoir parameters for a coupled 2-stage (Preamp V1 -> Power Amp V2) 16-state KONE core.
#[derive(Clone, Copy, Debug)]
pub struct AmpReservoirSpec {
    pub s1_in: [f32; 8],
    pub s1_diag: [f32; 8],
    pub s1_cross: [f32; 8],
    pub s1_bias: [f32; 8],
    pub s1_wout: [f32; 8],
    pub c1_ref: [f32; 8],
    pub s2_in: [f32; 8],
    pub s2_diag: [f32; 8],
    pub s2_cross: [f32; 8],
    pub s2_bias: [f32; 8],
    pub c2_ref: [f32; 8],
}

/// Fixed reservoir parameters for a single-stage 8-state overdrive/stompbox KONE core.
#[derive(Clone, Copy, Debug)]
pub struct PedalReservoirSpec {
    pub in_w: [f32; 8],
    pub diag: [f32; 8],
    pub cross: [f32; 8],
    pub bias: [f32; 8],
    pub c_ref: [f32; 8],
}

/// Default clean-room analytical 2-stage coupled tube amplifier reservoir.
/// Matches `amp_profiler.reservoir.AMP_RESERVOIR_DEFAULT` bit-for-bit.
pub const AMP_RESERVOIR_DEFAULT: AmpReservoirSpec = AmpReservoirSpec {
    // Stage 1: V1/V2 Preamp Cascade (multi-scale wideband + 720 Hz + 2400 Hz)
    s1_in: [
        0.520_f32, -0.680_f32, 0.850_f32, -1.040_f32, 0.760_f32, -0.920_f32, 0.640_f32, -0.810_f32,
    ],
    s1_diag: [
        0.420_f32, 0.510_f32, 0.610_f32, 0.700_f32, 0.780_f32, 0.850_f32, 0.560_f32, 0.740_f32,
    ],
    s1_cross: [
        -0.110_f32, 0.095_f32, -0.085_f32, 0.080_f32, -0.075_f32, 0.065_f32, -0.090_f32, 0.070_f32,
    ],
    s1_bias: [
        0.025_f32, -0.032_f32, 0.018_f32, -0.045_f32, 0.038_f32, -0.022_f32, 0.030_f32, -0.028_f32,
    ],
    s1_wout: [
        0.140_f32, -0.115_f32, 0.125_f32, -0.095_f32, 0.105_f32, -0.085_f32, 0.090_f32, -0.075_f32,
    ],
    c1_ref: [
        0.120_f32, 0.280_f32, 0.650_f32, 1.500_f32, 3.600_f32, 8.500_f32, 0.450_f32, 1.800_f32,
    ],

    // Stage 2: Push-Pull Power Tube & Phase Inverter Stage
    s2_in: [
        0.480_f32, -0.620_f32, 0.790_f32, -0.960_f32, 0.880_f32, -1.120_f32, 0.590_f32, -0.740_f32,
    ],
    s2_diag: [
        0.460_f32, 0.550_f32, 0.650_f32, 0.740_f32, 0.820_f32, 0.880_f32, 0.600_f32, 0.770_f32,
    ],
    s2_cross: [
        0.100_f32, -0.090_f32, 0.080_f32, -0.070_f32, 0.065_f32, -0.055_f32, 0.085_f32, -0.065_f32,
    ],
    s2_bias: [
        -0.018_f32, 0.024_f32, -0.015_f32, 0.032_f32, -0.028_f32, 0.020_f32, -0.022_f32, 0.016_f32,
    ],
    c2_ref: [
        0.160_f32, 0.380_f32, 0.850_f32, 2.000_f32, 4.800_f32, 11.000_f32, 0.600_f32, 2.400_f32,
    ],
};

/// Default clean-room analytical 8-state overdrive/stompbox reservoir.
/// Matches `amp_profiler.reservoir.PEDAL_RESERVOIR_DEFAULT` bit-for-bit.
pub const PEDAL_RESERVOIR_DEFAULT: PedalReservoirSpec = PedalReservoirSpec {
    in_w: [
        0.560_f32, -0.440_f32, -0.610_f32, 0.780_f32, -0.890_f32, 1.080_f32, -0.670_f32, 0.820_f32,
    ],
    diag: [
        0.400_f32, 0.490_f32, 0.590_f32, 0.690_f32, 0.790_f32, 0.870_f32, 0.460_f32, 0.680_f32,
    ],
    cross: [
        -0.120_f32, 0.085_f32, -0.075_f32, 0.070_f32, -0.060_f32, 0.050_f32, -0.080_f32, 0.055_f32,
    ],
    bias: [
        0.016_f32, -0.022_f32, 0.014_f32, -0.030_f32, 0.026_f32, -0.018_f32, 0.020_f32, -0.024_f32,
    ],
    c_ref: [
        0.100_f32, 0.250_f32, 0.600_f32, 1.500_f32, 3.800_f32, 9.500_f32, 0.450_f32, 1.800_f32,
    ],
};
