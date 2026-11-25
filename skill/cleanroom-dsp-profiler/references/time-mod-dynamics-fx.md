# Profiling Time-Based, Modulation & Dynamics FX

Static nonlinear models (like 180-ch KONE or `.nam` WaveNets) assume time-invariant, short-memory responses (`< 64 ms`). Reverbs, delays, LFO modulation pedals, envelope filters, and compressors violate those assumptions in specific, physically structured ways—requiring dedicated clean-room system-identification protocols.

All five engines below have matching **NumPy-only Python profilers** (`profiler/amp_profiler/time_mod_fx.py`) and **zero-allocation Pure-Rust starter engines** (`dsp-core-rust/src/`).

---

## 1. Spring, Plate, Hall & Room Reverbs (`SpringReverbEngine` + `profile_reverb_svd`)

### Why Naive Single-IR Capture Fails
Storing 21 full-length `1.0s` (`49,152`-sample) IRs across a `Decay` knob wastes `4+ MB` per tank and cannot smoothly interpolate `Decay` or `Tone` in real time.

### Proven Clean-Room Protocol
1. **Separate Mix Law First (`dry_gain_21`, `wet_gain_21`):**
   - Measure `100% Dry` (`Mix = 0.0`) and `100% Wet` (`Mix = 1.0`) reference impulses.
   - Across `21` `Mix` points, estimate wet gain from the reverb tail (`samples [32..4096]`, zero dry contamination) via dot-product projection, and extract dry gain from tap `0` residual.
2. **Low-Rank SVD Basis Compression Across `Decay` (`4` Orthogonal Basis IRs):**
   - Set `Mix = 1.0`, `Tone = 1.0`, and sweep `Decay` across `21` steps (`0.00..1.00`), capturing settled impulses (`49,152` samples = `1.024s` @ `48 kHz` for outboard spring tanks; `32,768` samples for studio plates; `8,192` samples for room mics).
   - Stack into matrix $M \in \mathbb{R}^{L \times 21}$ and compute thin SVD $M = U \Sigma V^T$.
   - Retain the top `4` basis vectors (`basis_4 = U[:, :4].T`) and `21 × 4` projection weights (`weights = (S[:4, None] * Vt[:4, :]).T`). Typically achieves **`99.92% – 99.98%`** reconstruction correlation across every decay setting!
3. **Causal Toeplitz-Deconvolved Wet Tone FIR (`tone_fir_21`, `64` taps):**
   - Fix `Decay = 0.65`, capture reference tail `ref` at `Tone = 1.0`, then sweep `Tone = 0.00..1.00` and solve the Tikhonov + smoothness-regularized causal Toeplitz system:
     $$(X^T X + \lambda I + \alpha D^T D)\, h_{\text{tone}} = X^T y_{\text{tone}}$$
   - Achieves `99.93% – 99.999%` spectral/time accuracy with a tiny `64`-tap SIMD FIR.
4. **CPU Optimization (`4×` Decimated Anti-Aliased Spring Tank):**
   - Real mechanical spring tanks roll off sharply above `4.5–5.5 kHz`. Enabling `decimate_4x = true` in `SpringReverbEngine` boxcar-decimates to `12 kHz` (`6 kHz` Nyquist) and linearly upsamples back to `48 kHz`, cutting convolution CPU by **`16×`** with zero audible loss! Use `decimate_4x = false` for bright studio plates and room mics.

---

## 2. Harmonic / Optical Tube Tremolo & Dual-Band Modulation (`HarmonicTremoloEngine` + `profile_harmonic_tremolo`)

### Why Static Sweeps Fail
A free-running LFO shifts phase continuously between captures, so naive impulse or waveform subtraction averages out to mush.

### Proven Clean-Room Protocol
1. **Exact LFO Speed (`Hz`) via 2 kHz Carrier Envelope Demodulation:**
   - Set `Intensity = 0.85` (strong modulation), feed a `3–4s` `2 kHz` sine carrier at each `Speed` knob step, take a `1 ms` moving-RMS envelope, subtract DC, apply a Hann window, and run FFT + **3-bin log-parabolic peak interpolation** to resolve exact LFO frequency to `±0.01 Hz`.
2. **Non-Overlapping Impulse-Train SVD Crossover Extraction (`psi_0`, `psi_1`):**
   - Set `Speed = 0.50` (~`3–5 Hz` = `6–10` full LFO cycles over `2.0s`).
   - Send a **non-overlapping impulse train** (`200` impulses/sec, `step = 240` samples > `128`-tap FIR length, `amp = 0.01`) across every `Intensity` step (`0.00..1.00`).
   - Slice all `~400` instantaneous `128`-tap responses per step into matrix $A$ and run SVD: the top **2 singular vectors (`psi_0`, `psi_1`)** capture **`>99.99%` of total crossover energy**!
   - Project each `Intensity` step's snapshots onto `psi_0` and `psi_1` to extract exact `c0_mean`, `c0_amp`, `c1_mean`, `c1_amp` modulation curves.
3. **Phase-Locked Tube / VCA Saturation Calibration (`Intensity = 0.0`):**
   - Set `Intensity = 0.0` so LFO modulation is strictly zero (`100%` phase-locked!), align lag against `conv(cal_stim, psi_0)`, and fit the optional zero-slope saturation residual (`15` ch or `180` ch) with scale-penalized ridge regression.

*(Tip for Analog BBD Chorus / Vibrato: use the same 2 kHz carrier demodulation to measure LFO rate, plus a `1 kHz` FM instantaneous-phase unwrap `np.unwrap(np.angle(hilbert(y)))` to extract exact peak delay-time modulation depth in milliseconds!)*

---

## 3. Analog BBD & Studio Digital Delays (`DelayBbdEngine` + `profile_delay_bbd`)

### The Hidden Trap Almost Everyone Misses: `loop_extra_delay`
In many real analog BBD and digital delay plugins, the spacing between the first echo (`R1`) and second echo (`R2`) is **NOT** equal to `Dry → R1` — anti-aliasing, clock-aperture, or DSP loop framing adds **`24–33` extra samples** (`loop_extra_delay = (R2 - R1) - (R1 - Dry)`). Ignoring this causes comb-filtering and rhythmic drift on multi-repeat tails!

### Proven Clean-Room Protocol
1. **Measure Dry Gain (`Mix = 0`), Delay Time Law (`delay_samples_21`), and `loop_extra_delay`:**
   - Capture an impulse at `Mix = 1.0`, `Feedback = 0.55`, locate `R1` and `R2` peaks, and compute `loop_extra_delay = (r2_abs - r1_abs) - (r1_abs - dry_pk)`.
   - Sweep `Time` across `21` steps at `Feedback = 0.0` to map exact sample delay law (watch for integer-ms stepping vs. continuous curves!).
2. **Capture Base Wet Reconstruction FIR (`64` taps) & Loop / Hardness / BBD Aperture FIRs (`21 × 64` taps):**
   - Extract the causal `R1` impulse (using causal `0.1%`-of-peak leading tap detection, not raw `argmax`!) as `wet_base_fir`.
   - If the delay has a `Hardness` / `Tone` knob, deconvolve each `Tone` step against `Tone = 1.0` via `_solve_causal_toeplitz_fir`. If fixed analog BBD, deconvolve `R2` against `R1` (`R2 / ||R2||` vs `R1`) to extract the exact per-repeat darkening filter!
3. **Map Feedback Gain Taper (`fb_gain_21`) & Fit 30-Ch Zero-Slope BBD Saturation (`fit_bbd_30ch_saturation`):**
   - Sweep `Feedback` (`0.00..1.00`) and compute `||R2|| / ||R1||` per step.
   - For analog BBDs (compander / bucket-rail clipping), drive the delay at `Feedback = 0.0`, `Mix = 1.0` with a multi-level calibration riff (`peak = 0.45`) and fit the `30`-channel (`6` knee ratios × `5` causal dilations `[0, 1, 2, 4, 8]`) zero-slope saturation core.

---

## 4. Dynamic Envelope Filter / Auto-Wah (`EnvelopeFilterEngine` + `profile_envelope_filter`)

### Proven Clean-Room Protocol
1. **Small-Signal `Peak` / Resonance Q Sweep (`1e-3` Impulses):**
   - Sweep `Peak` across `21` steps with tiny `1e-3` impulses (so the envelope detector barely opens) and extract equivalent 2-pole SVF resonance $Q = \max |H(f)| / |H_{\text{DC}}|$.
2. **Multi-Corner Vactrol + TPT 2-Pole SVF Optimization (`14` Physical Parameters):**
   - Capture `3` distinct `(Drive, Peak)` corners on a `1.2–1.5s` disjoint funky dynamic guitar phrase (with crisp staccato pick attacks and decays).
   - Optimize `[f_hp_sc, f_lp_sc, sens_0, sens_exp, tau_atk_s, tau_rel_s, tau_opt_s, cv_pow, f_min_hz, f_max_hz, q_base, q_cv, gain_base, bp_mix]` via multi-scale coordinate + random subspace search minimizing `(1 - corr) + 0.35 * rel_err` across all corners simultaneously.
   - Uses compiled Rust FFI (`cleanroom_eval_envelope_filter`) automatically for `~100×` faster search!
   - For `Fast` vs `Slow` response switches, freeze the `14` base parameters and 1D-sweep `slow_mult` (`[1.2 .. 8.0]`) against `Slow` target audio.

---

## 5. Studio Compressor & Parametric / Console EQ (`StudioCompressorEngine` + `profile_compressor`)

### Compressor Profiling Protocol
1. **Static Transfer Curve (Stepped `1 kHz` Sine Tones, `-48 dBFS` to `-2 dBFS`):**
   - Hold each tone for `350 ms` and measure steady-state RMS over the final `130 ms` (fully settled!).
   - Low-level linear region (`-48 .. -38 dBFS`) yields exact **`makeup_db`**.
   - High-level compressed asymptote (`-12 .. -2 dBFS`) linear fit gives **`ratio = 1 / (1 + slope)`** and **`threshold_db = -intercept / slope`**.
   - Gain reduction right at `input == threshold_db` gives quadratic soft **`knee_db = 8 * GR(T) / slope`**.
2. **Dynamic Attack & Release Ballistics (Step-Envelope Burst):**
   - Feed a `1 kHz` burst stepping `-36 dBFS` (`250 ms`) → `-4 dBFS` (`350 ms`) → `-36 dBFS` (`600 ms`).
   - Compute `1 ms` moving-RMS gain reduction in dB and read the `63.2%` (`1 - 1/e`) drop time (`attack_ms`) and `63.2%` recovery time (`release_ms`).

### Parametric / Console EQ Profiling Protocol
- Sweep each EQ band's `Gain` (`21` steps) and `Freq` (`11–21` steps) using settled **bipolar `±1e-4` impulses** (`capture_bipolar_ir`), convert to `FREQS_64` dB deltas (`ir_to_bode64`), and either fit analytic RBJ biquad `(fc, Q, gain_db)` parameters or fold directly via `synthesize_min_phase_ir`.
