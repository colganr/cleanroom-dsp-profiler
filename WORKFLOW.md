# End-to-End Profiling-to-Plugin Workflow (`WORKFLOW.md`)

This workflow takes you (or your AI coding agent) from a target amplifier, overdrive pedal, speaker cabinet, or time-based / modulation / dynamics effect to a **verified, zero-allocation Pure-Rust DSP engine** achieving `>99%` held-out waveform correlation across interactive controls.

---

## Stage 0 — Choose Your Target Category & Profiling Path

| Category | Core DSP Engine | Recommended Profiling Path |
| :--- | :--- | :--- |
| **A. Guitar / Bass Amplifier Head** | Wiener–Hammerstein `H_pre` (`1024`-tap) → **180-Ch Bloom + Sag + 96 kHz KONE** (+ optional 3-Stage Invertible Tube Spine) → `H_post` (`64`-tap) | **macOS Live AU** (full multi-knob/switch sweeps) OR **Cross-Platform WAV-Pair** (per-setting captures on Windows/macOS/Linux/Hardware) |
| **B. Overdrive / Boost Stompbox** | `H_pre` (`256`-tap `Drive`) → **126-Ch 3-Pole + 96 kHz KONE** across `9` Gain brackets → `H_post` (`256`-tap `Tone`) → `Level` → `12 Hz` DC Block | Live AU or Cross-Platform WAV-Pair |
| **C. Speaker Cabinet (2D / 3D Multi-Mic)** | `ScalableCabEngine<const N_SLOTS: usize>` (`1024`-tap `i16 + f32` quantized grid, bilinear/trilinear interpolation) | Live AU grid sweep or exported IR WAV folder |
| **D. Spring / Plate / Hall / Room Reverb** | `SpringReverbEngine` (`4`-basis SVD Decay compression + `64`-tap causal Toeplitz Wet Tone FIR + optional `4×` decimated `12 kHz` mode) | `profile_reverb_svd()` (`time_mod_fx.py`) |
| **E. Analog BBD & Digital Delay** | `DelayBbdEngine` (Hermite delay line + hidden `loop_extra_delay` + Toeplitz Hardness/BBD FIR + `30`-ch BBD bucket-rail saturation) | `profile_delay_bbd()` + `fit_bbd_30ch_saturation()` |
| **F. Harmonic / Optical Tremolo & Mod** | `HarmonicTremoloEngine` (carrier-demod LFO speed + impulse-train SVD dual crossover FIRs + phase-locked `Intensity=0` saturation) | `profile_harmonic_tremolo()` |
| **G. Envelope Filter & Compressor / EQ** | `EnvelopeFilterEngine` (Vactrol + TPT 2-pole SVF) & `StudioCompressorEngine` (soft-knee static + attack/release ballistics) | `profile_envelope_filter()` & `profile_compressor()` |

*(For detailed step-by-step capture protocols for Categories D–G, see [`skill/cleanroom-dsp-profiler/references/time-mod-dynamics-fx.md`](skill/cleanroom-dsp-profiler/references/time-mod-dynamics-fx.md) or run `cleanroom-profiler demo-fx`.)*

---

## Stage 1 — Discovery, Host Hygiene & Onset Latency

### Path A: Live Headless macOS AudioUnit
1. **Enumerate all host parameters & measure onset latency:**
   ```bash
   cleanroom-profiler discover --type aufx --subtype XXXX --mfg YYYY --probe-impulse
   ```
   *(If `aufx` returns not found, try `--type aumf`, which many multi-FX guitar suites use for MIDI-controlled FX.)*
2. **Disable internal oversampling (if exposed) and bypass all other modules** (noise gate, tuner, cab, reverb, delay, limiter) so you isolate one nonlinear block at a time.
3. **Always run a full async settle** (`backend.settle(full=True)`) on every worker process and after every channel/mode toggle. Many plugins load internal models asynchronously over the first `~300 ms` or only initialize hidden charge-pump stages after cycling their mode toggle `0 → 1 → 0`.
4. Record the exact `onset_latency_samples` (commonly `0`, `16`, `32`, `64`, or `184` samples at `48 kHz`) and verify polarity (`+1` vs `-1`).

### Path B: Cross-Platform Offline WAV-Pair (Windows / macOS / Linux / Hardware)
1. **Export the 15-second zero-leakage composite stimulus:**
   ```bash
   cleanroom-profiler export-stimulus --out captures/stim_15s.wav
   # Or pass your own 48 kHz dry guitar DI recording:
   cleanroom-profiler export-stimulus --di-wav my_dry_guitar_48k.wav --di-peak 0.160 --out captures/stim_15s.wav
   ```
   - Use `--di-peak 0.160` (`-15.9 dBFS`) for passive single-coil pickups or `--di-peak 0.280` (`-11.1 dBFS`) for vintage humbuckers. Raw `-0.4 dBFS` normalized WAVs are `+15.5 dB` too hot and will slam clean amps into unrealistic distortion!
2. **Render `captures/stim_15s.wav`** at `48,000 Hz` (24-bit or 32-bit float mono WAV, zero dithering, zero limiter/gate/reverb) through your VST3/CLAP/AAX plugin in any DAW or through your physical hardware amp/pedal, saving as `captures/wet_15s.wav`.

---

## Stage 2 — Wiener–Hammerstein Linear Factorization (`H_pre` & `H_post`)

Never place the entire small-signal impulse response $H_{\text{tot}}(f)$ before the clipper!

1. **Small-Signal Total Response $H_{\text{tot}}(f)$:**
   - Capture with **symmetric bipolar impulses (`±1e-4`)** (`capture_bipolar_ir`):
     $$h_{\text{tot}}[n] = \frac{y_+[n] - y_-[n]}{2 \times 10^{-4}}$$
   - Why `±1e-4` instead of `0.01`? High-gain amps (`+40..+50 dB` linear gain) clip a `0.01` impulse, corrupting linear EQ curves. Bipolar subtraction also cancels tube bias drift and DC offsets.
2. **Post-Clipping Filter $H_{\text{post}}(f)$:**
   - Drive the target with hard-saturated log-spaced sine tones (`measure_post_ir_hard_sat`, `amp = 0.25..0.35`) so the clipper outputs a rail-to-rail square wave whose fundamental entering $H_{\text{post}}$ is flat across frequency; extract the fundamental via quadrature lock-in and fold into a `64`-tap minimum-phase FIR.
3. **Pre-Clipping Filter $H_{\text{pre}}(f)$:**
   - Deconvolve in the FFT domain (`deconvolve_pre_from_total`):
     $$H_{\text{pre}}(f) = \frac{H_{\text{tot}}(f)}{H_{\text{post}}(f) + \epsilon}$$
   - Route **pre-clipping controls** (`Gain`, `Bright`, `Mid`, `Deep`, `EQ1/EQ2`, `Bass/Middle/Treble`, `OD Drive`) into $H_{\text{pre}}$, and **post-clipping controls** (`Master`, `Presence`, `Half Power`, `6L6/6V6`, `OD Tone`) into $H_{\text{post}}$.
4. **Interactive Control Tensors (for multi-knob live sweeps):**
   - **Non-Additive $2^K$ Passive Switches (`capture_switch_combo_tensor`):** Passive switches share a `0 dB` divider ceiling; never add independent dB deltas (`+10–12 dB` overshoot bug!). Capture all $2^K$ states.
   - **Cap-Coupled 3D Tone Stack (`capture_tonestack_3d`):** Capture `5×5×5` `(Bass × Middle × Treble)` for each switch state that alters a tone-stack capacitor (e.g., `Mid Boost` paralleling a cap across `C_treble`).
   - **21-Point Pot Tapers (`capture_knob_bode_21`):** Sweep `Gain` (captures bright-cap bleed vs. wiper position), `Master`, and `Presence` across `[0.01 .. 1.00]`.

---

## Stage 3 — Nonlinear Core Calibration (Bloom, Sag & 96 kHz KONE)

1. **Zero-Leakage Split:**
   - Train **exclusively** on Segment 2 (`4.0s – 8.0s`: dynamic ramp + gated tone bursts + calibration DI).
   - **Never** include Segment 3 (`8.0s – 15.0s`) in regression fitting!
2. **Compile `dsp-core-rust` first** (`cd dsp-core-rust && cargo build --release`) so Python uses the compiled Rust FFI bridge (`rust_bridge.py`) — 50–100× faster and bit-identical to runtime DSP.
3. **Fit the Nonlinear Readout Weights:**
   - **Amplifiers (`fit_amp_180_weights`):** Evaluates the 180 zero-slope channels (`126` 3-pole + coupled `V1 → V2`, `48` 96 kHz coupled 2-triode KONE, `6` `35/383 Hz` sag recovery) + Bounded 12AX7 Triode Bloom `sat_0`, convolves all basis columns through $H_{\text{post}}$, and solves with **Scale-Penalized Ridge Cholesky Regression** across gain brackets (e.g. `[0.15, 0.35, 0.55, 0.75, 1.00]`).
   - **Overdrive Pedals (`fit_pedal_126_weights`):** Evaluates the 126 3-pole + 96 kHz 8-state KONE features across `9` Drive brackets (`[0.00..1.00]`), setting `subtract_dry=True` for op-amp clean-bleed topologies (TS/Klon) or `False` for pure series clippers.
   - *(Tip for stacked pedals: include hot input peaks `0.40..0.70` in your calibration pass so overdrive stages stay well-behaved when boosted by another pedal!)*

---

## Stage 4 — Scalable Speaker Cabinet IR Grid (Optional)

1. Capture `1024`-tap IRs at `48 kHz` (`imp_amp = 0.5`) across spatial knots:
   - 2D: `(Position × Distance)` (e.g., `6 × 3`)
   - 3D: `(Axis × Distance × Angle)` (e.g., `3 × 2 × 2`)
2. Quantize via `quantize_cab_grid` into `<name>_cab_grid.bin` (`i16` little-endian) + `<name>_cab_scales.rs` (`f32` peak scales, `-96.3 dB` floor, 50% binary savings).
3. Wire directly into `ScalableCabEngine<N>` in `dsp-core-rust/src/cab_grid.rs`.

---

## Stage 5 — Export Pure-Rust Weights & Verify Held-Out Null Test

1. Export Rust weight tables with `export_amp_rust_weights` / `export_pedal_rust_weights`.
2. Run strictly held-out verification (`verify.py::compute_null_metrics`) on Segment 3 (`8.0s – 15.0s`):

| Metric | Target (Clean / Mid-Gain / OD) | Target (Cascading High-Gain / Sag-Heavy) | If Worse, Inspect |
| :--- | :---: | :---: | :--- |
| **Held-Out Waveform Correlation** | **`> 99.0%`** | **`> 97.5%`** | Polarity sign, lag alignment, `H_post` separation |
| **Raw ESR** | **`< -20 dB`** | **`< -15 dB`** | `sat_0` bloom knee (`cp0/cn0`), gain-bracket density |
| **Mean Spectral Error (`80 Hz–6 kHz`)** | **`< 0.5 dB`** | **`< 1.0 dB`** | Unfactored `H_post`, missing `2^K` switch combo, or cap-coupled tone stack |
| **Output RMS Delta** | **`< ±0.3 dB`** | **`< ±0.5 dB`** | `post_scale` / `level_gain` normalization or scale-collapse ridge penalty |

Consult `skill/cleanroom-dsp-profiler/references/model-weight-taxonomy.md` to map any remaining error directly to the responsible weight tensor.

---

## Stage 6 — Optional `.nam` (`v0.5.4`) + 24-Bit Cab IR Export

If you want to take a snapshot of your dialed-in Pure-Rust chain onto a hardware player (Neural Amp Modeler, Valeton, Dimehead, Mod Dwarf, etc.):
- Run `distill_nam_v054` (or pass `--export-nam` to `cleanroom-profiler profile-wav-pair`) to generate a spec-compliant `v0.5.4` WaveNet `.nam` file + optional `2048`-sample `48 kHz` **`24-bit PCM`** `.wav` cabinet IR.
- Always bypass non-stationary FX (noise gate, pitch shift, stereo doubler, LFO modulation, delays, reverbs) during static `.nam` capture!

---

## Stage 7 — Wrap Into a Cross-Platform Plugin or Standalone App

Your `dsp-core-rust` crate compiles with zero heap allocations on `aarch64-apple-darwin`, `x86_64-pc-windows-gnu`/`msvc`, and `x86_64-unknown-linux-gnu`:
- **VST3 & CLAP:** Wrap `StarterRigEngine` inside [`nih-plug`](https://github.com/robbert-vdh/nih-plug).
- **macOS AUv2:** Link `libcleanroom_dsp` via C ABI into an `AudioComponent` bundle and validate with `auval -v`.
- **Standalone App:** Drive `process_mono_to_stereo` via `cpal` (`48,000 Hz / 512` default; `cpal::BufferSize::Default` + MMCSS `"Pro Audio"` in Windows WASAPI Shared Mode).

See [`skill/cleanroom-dsp-profiler/references/daw-and-standalone-guide.md`](skill/cleanroom-dsp-profiler/references/daw-and-standalone-guide.md) for cross-platform DAW/WASAPI/AUv2 guardrails.
