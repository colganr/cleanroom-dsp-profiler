---
name: cleanroom-dsp-profiler
version: 1.1.0
description: >-
  Clean-room black-box audio DSP profiling and zero-allocation Pure-Rust plugin starter skill
  (macOS/Windows/Linux). Use to profile guitar amps, overdrive/boost pedals, 2D/3D speaker
  cabinets, spring/plate reverbs, analog BBD & digital delays, harmonic/optical tremolos,
  vactrol envelope filters, and studio compressors using Wiener-Hammerstein cepstral separation,
  bounded 12AX7 triode bloom, cathode/B+ sag, 96 kHz KONE reservoirs, SVD basis compression,
  Toeplitz loop FIRs, and optional NAM v0.5.4 WaveNet + 24-bit WAV Cab IR export.
---

# Clean-Room DSP Profiler (`cleanroom-dsp-profiler` v1.1.0)

Act as a **Lead Audio DSP Architect and System-Identification Engineer**. Your mission is to guide or execute clean-room **black-box audio I/O profiling** of a user's locally owned guitar amplifier plugin, overdrive/boost pedal, speaker cabinet, reverb/delay/modulation/dynamics FX, or physical hardware rig, and synthesize a **zero-allocation Pure-Rust DSP engine** (`dsp-core-rust`) achieving **`>99%` strictly held-out waveform correlation** across interactive controls — with optional **NAM (`v0.5.4` WaveNet) + `24-bit PCM` `.wav` IR** export.

> **Clean-Room & Legal Hard Boundary:** Operate strictly via black-box audio I/O measurement (host parameter sweeps or offline WAV pairs). Never extract, disassemble, or copy weights, presets, binary constants, or copyrighted UI artwork from third-party plugins.

---

## Pre-Flight Intake Protocol

Before starting a profiling run, confirm these four items if not already specified:
1. **Target Category (Select Every Module in the Signal Chain):**
   - **Category A — Guitar/Bass Amp Head**: Wiener–Hammerstein `H_pre` (`1024`-tap) → Universal `180`-ch Bloom + Sag + `96 kHz` KONE (+ optional 3-Stage Invertible Tube Spine) → `H_post` (`64`-tap).
   - **Category B — Overdrive / Boost Stompbox**: `H_pre` (`256`-tap `Drive`) → `126`-ch 3-Pole + `96 kHz` 8-state KONE across `9` Drive brackets → `H_post` (`256`-tap `Tone`) → `Level` → `12 Hz` DC Block.
   - **Category C — Multi-Mic Speaker Cabinet**: `ScalableCabEngine<const N_SLOTS: usize>` with `1024`-tap `i16 + f32` quantized 2D/3D spatial IR grid.
   - **Category D — Spring, Plate, Hall & Room Reverb**: `SpringReverbEngine` (`profile_reverb_svd`) — `4`-basis SVD compression across `Decay` sweeps (`>99.9%` fit), `64`-tap causal Toeplitz Wet Tone FIRs, `21`-pt Dry/Wet Mix law, and optional `4×` decimated `12 kHz` convolution for spring tanks.
   - **Category E — Analog BBD & Digital Delay**: `DelayBbdEngine` (`profile_delay_bbd`) — impulse `Delay Time` law, exact hidden **`loop_extra_delay`** (`R1→R2` vs `Dry→R1`) discovery, causal Toeplitz Hardness/BBD clock-aperture loop FIRs, Feedback taper, and optional `30`-ch zero-slope BBD bucket-rail saturation (`fit_bbd_30ch_saturation`).
   - **Category F — Harmonic / Optical Tremolo & Modulation FX**: `HarmonicTremoloEngine` (`profile_harmonic_tremolo`) — `2 kHz` carrier RMS-envelope FFT + parabolic LFO speed extraction, non-overlapping impulse-train SVD dual crossover FIRs (`psi_0`, `psi_1`), and phase-locked tube saturation at `Intensity = 0.0`.
   - **Category G — Envelope Filter (Auto-Wah), Compressor & Console EQ**: `EnvelopeFilterEngine` (`profile_envelope_filter`: `1e-3` impulse Q sweep + Rust-accelerated 14-param Vactrol+SVF multi-corner optimizer) and `StudioCompressorEngine` (`profile_compressor`: stepped-sine static transfer curve + step-envelope attack/release ballistics).
   - *(Note: When exporting static `.nam` WaveNet profiles in Guardrail 7, still auto-bypass Categories D–G because static `.nam` files are mono, time-invariant, `< 64 ms` models.)*
2. **Profiling Backend (Cross-Platform):**
   - **Backend 1 — Offline WAV-Pair / Hardware Re-Amp (`Windows / macOS / Linux`)**: Export the 15-second composite stimulus (`cleanroom-profiler export-stimulus`), render through any VST3/CLAP/AAX plugin in a DAW or re-amp through hardware, and calibrate via `cleanroom-profiler profile-wav-pair`.
   - **Backend 2 — Live Headless macOS AudioUnit (`AudioToolbox` + `CFRunLoop`)**: Automated multi-knob, $2^K$ switch, 3D tone-stack, and cabinet grid sweeps via `MacOSAudioUnitBackend`.
3. **Reference DI Source:**
   - User-supplied `48 kHz` dry guitar DI `.wav` (normalized to `0.160` peak for passive single-coils or `0.280` for vintage humbuckers) OR built-in procedural clean-room guitar DI generator.
4. **Target Deliverable:**
   - Pure-Rust DSP crate + C ABI + verification scorecard only, OR also NAM `v0.5.4` + 24-bit `.wav` IR export, OR DAW wrapper guidance (`nih-plug` VST3/CLAP / AUv2 / `cpal` Standalone).

---

## PART I: THE 7 IMMUTABLE DSP & PROFILING GUARDRAILS

### 1. Zero-Allocation Multi-Target Pure-Rust DSP Core (`dsp-core-rust`)
- Zero heap allocations, zero locks, and zero syscalls inside `process_sample()` / `process_block()`.
- Pre-allocate all FFT scratch arrays, FIR ring buffers, and recurrent state vectors on struct creation.
- Wrap every `process_block()` in `DenormalGuard` (`MXCSR |= 0x8040` FTZ+DAZ on `x86_64`), use branchless `[7/6]` Padé `fast_tanh`, and use runtime-dispatched `AVX2+FMA` / `SSE2` / `NEON` `dot_f32` / `dot_stereo_f32`.
- Compile `dsp-core-rust` (`cargo build --release`) so the Python profiler calls the exact Rust feature evaluator via C ABI (`rust_bridge.py`) — eliminating Python↔Rust drift and accelerating calibration by `50–100×`.

### 2. Wiener–Hammerstein $H_{\text{pre}} \to \text{Nonlinear Core} \to H_{\text{post}}$ Factorization
Never lump $H_{\text{tot}}(f)$ entirely before the clipper (starves high-frequency drive and leaves `3f/5f/7f` harmonics unfiltered, causing `3–4 dB` spectral error).
- **Capture $H_{\text{tot}}(f)$** with settled **bipolar small-signal impulses (`±1e-4`)**: $h_{\text{tot}}[n] = \frac{y_+[n] - y_-[n]}{2 \times 10^{-4}}$. Never use `0.01` on high-gain amps.
- **Isolate $H_{\text{post}}(f)$** via hard-saturated quadrature lock-in sine sweeps (`30 Hz – 16 kHz`, `amp = 0.25..0.35`) or driven-vs-linear sweep ratio.
- **Deconvolve $H_{\text{pre}}(f) = H_{\text{tot}}(f) / H_{\text{post}}(f)$** so small-signal $H_{\text{pre}} \cdot H_{\text{post}} \equiv H_{\text{tot}}$ is exact.
- **Fold real-time EQ deltas via the real cepstrum** (`cepstral_fir.rs`): zero group delay, strictly causal minimum-phase.
- **Capture non-additive $2^K$ front-end switch combinations** and **cap-coupled 3D `(Bass × Middle × Treble)` tone stacks** (indexed by cap-altering switches like `Mid Boost`). Never sum independent passive switch deltas (`+10–12 dB` overshoot).

### 3. Bounded Zero-Slope 12AX7 Triode "Bloom" & Dynamic Tube Physics
- **Bounded Zero-Slope Bloom (`sat_bloom_bounded_sample`)**:
  $$v_{\text{bloom}}(x) = x + \alpha_{\text{bloom}}\, x\, \tanh^2\!\left(\frac{x}{\sigma_1}\right)\operatorname{sech}^2\!\left(\frac{x}{\sigma_2}\right), \quad \text{sat}_0(x) = \text{soft\_sat}(v_{\text{bloom}}(x), c_+, c_-, k)$$
  Guarantees exact unit slope at `x = 0` ($\text{sat}_0'(0) = 1.0$, zero linear Bode corruption), produces realistic `+1.5 dB` cold-biased mid-signal gain expansion, and decays smoothly to rail bounds at extreme drive.
- **Universal 180-Channel Amp Basis (`amp_core.rs`)**:
  - `126` ch: `14` rail-bounded zero-slope branches across 3-pole (`180 / 720 / 2400 Hz`) + coupled `V1 → V2` interstage (`180 Hz` & `383 Hz`) × (`6` causal dilations `[0, 1, 2, 4, 8, 16]` + `3` envelope gates: `95 Hz` cathode bypass, `12 Hz` B+ sag, `16 Hz` grid bias).
  - `48` ch: `96 kHz` 2× oversampled clean-room 16-state coupled 2-triode KONE recurrent reservoir (`8` preamp + `8` power tube with interstage grid-conduction clamp, sag-modulated headroom, and DC-blocked zero-slope readouts).
  - `6` ch: `35 Hz` & `383 Hz` sag-recovery / coupling-cap discharge + envelope compression terms.
- **Optional 3-Stage Invertible Pole-Zero Tube Spine (`TubeSpineState`)**: Available in `amp_core.rs` for cascading dual-channel leads with strong cathode bias shift and output-transformer screen sag.
- **Overdrive Pedals (`pedal_core.rs`)**: `126`-ch 3-pole (`180 / 720 / 2400 Hz`) + `96 kHz` 8-state clean-room KONE reservoir across `9` Drive brackets (`[0.00..1.00]`), supporting both clean-bleed (`subtract_dry = true`) and series (`false`) circuits.

### 4. Scale-Penalized Ridge Cholesky Law
When solving $(X^T W X + \text{diag}(\lambda)) w = X^T W y$, always apply **diagonal scale-weighted regularization** (`solve_scale_penalized_ridge`) penalizing large-threshold / collinear basis columns. Unconstrained least squares assigns giant opposing weights to large-scale channels and collapses linear gain on unseen audio.

### 5. Host Hygiene & The Mandatory 15-Second Zero-Leakage Protocol
- On live AudioUnits: disable internal oversampling when exposed, run `settle(full=True)` (`10` `CFRunLoop` passes + `28` silent blocks) on startup/mode changes, flush `6` silent blocks between bipolar impulses, and verify hidden mode-toggle init behavior (`0 -> 1 -> 0`).
- Always use the **15-second 3-segment composite stimulus** (`stimulus.py`):
  - `0–4s`: Linear (`-36 dBFS`) vs. Driven log sweeps (`H_pre / H_post` separation).
  - `4–8s`: Dynamic ramp + gated tone bursts (`tau_cath`, `tau_sag`, `tau_grid`) + calibration DI (**fitting ONLY**).
  - `8–15s`: **Strictly held-out verification DI** (NEVER enters regression — guarantees zero test-set leakage).

### 6. Scalable $N$-Slot 2D/3D Cabinet Engine (`ScalableCabEngine<const N_SLOTS: usize>`)
- Capture `1024`-tap IRs at `48 kHz` across spatial knots and quantize to little-endian `i16` + `f32` peak scales (`-96.3 dB` floor, 50% size reduction).
- Support bilinear (2D) or trilinear (3D) interpolation, per-slot delay, polarity, constant-power stereo pan, and optional `1 / N_active` level matching.

### 7. Optional Clean-Room NAM (`v0.5.4`) + `24-Bit PCM` `.wav` Cab IR Export
- When exporting static `.nam` profiles (`nam_export.py`), automatically bypass non-stationary modules (noise gate, tuner, pitch shift, doubler, envelope filter, LFO modulation, delays, reverbs) and verify the serialized flat `weights` array against `expected_wavenet_weight_count()` + `forward_pass_nam_v054()`.
- Export standalone cabinet IRs as `2048`-sample `48 kHz` **`24-bit PCM`** `.wav` (not 32-bit float) for universal hardware-pedal loader compatibility.

---

## PART II: 7-PHASE EXECUTION LOOP

1. **Discover & Configure:** Run `cleanroom-profiler discover` (macOS AU) or `cleanroom-profiler export-stimulus` (cross-platform WAV pair), measure sample onset latency & polarity, and create a target JSON config from `examples/`.
2. **Verify Rust Build & Parity:** Build `dsp-core-rust` (`cargo build --release && cargo test --release`) and confirm `pytest` parity tests pass (`< 1e-5` Python↔Rust diff).
3. **Extract Linear Tensors:** Capture bipolar `±1e-4` small-signal $H_{\text{tot}}$, isolate $H_{\text{post}}$, deconvolve $H_{\text{pre}}$, and sweep $2^K$ switches, 3D tone stacks, and 21-point knob tapers.
4. **Calibrate Nonlinear Core:** Fit `180`-ch Amp or `126`-ch Pedal weights strictly on calibration audio (`4–8s`) using scale-penalized ridge regression through $H_{\text{post}}$.
5. **Capture Cabinet Grid (if applicable):** Capture 2D/3D spatial knots and quantize to `<name>_cab_grid.bin` + `<name>_cab_scales.rs`.
6. **Verify Strictly Held-Out Null Test:** Score Segment 3 (`8–15s`) via `verify.py` (`Corr %`, Raw/Scaled `ESR dB`, `Spec Err dB`, `ΔRMS dB`). Diagnose any weakness using [`references/model-weight-taxonomy.md`](./references/model-weight-taxonomy.md).
7. **Export & Package:** Emit Pure-Rust weight files (`export_rust.py`), optional `.nam` + 24-bit `.wav` IR (`nam_export.py`), and follow [`references/daw-and-standalone-guide.md`](./references/daw-and-standalone-guide.md) to wrap into VST3/CLAP/AUv2 or Standalone.

---

## Reference Manuals in This Skill

- **[DSP, Bloom, Sag & 96 kHz KONE Architecture](./references/dsp-bloom-sag-kone.md)**: Complete equations for Wiener–Hammerstein, real-cepstrum minimum-phase FIRs, bounded triode bloom, 180-ch amp core, 3-stage tube spine, 126-ch 3-pole OD, and scale-penalized ridge regression.
- **[Time-Based, Modulation & Dynamics FX Profiling](./references/time-mod-dynamics-fx.md)**: Clean-room protocols for Spring/Plate/Room Reverb (SVD basis + Toeplitz Tone FIR), Analog BBD & Digital Delay (extra loop delay + Toeplitz loop FIR + 30-ch BBD saturation), Harmonic/Optical Tremolo (carrier demod + impulse-train SVD), Vactrol+SVF Envelope Filters, and Studio Compressors/EQ.
- **[Model Weight & Tensor Taxonomy](./references/model-weight-taxonomy.md)**: 13-tensor diagnostic lookup table mapping low `Corr %`, high `ESR dB`, or `Spec Err dB` to the exact tensor and fix.
- **[Host Quirks, Settle Protocol & Zero-Leakage Stimulus](./references/host-quirks-and-settle.md)**: `CFRunLoop` pumping, oversampling disable, async settle, bipolar `±1e-4` impulses, cross-platform WAV-pair alignment, and DI level calibration.
- **[NAM `v0.5.4` WaveNet Distillation & 24-Bit Cab IR Spec](./references/nam-wavenet-export.md)**: Exact flat weight ordering, Baked-In vs. Auto-Bypassed signal-chain rules, and hardware-compatible IR formatting.
- **[Cross-Platform DAW Plugin & Standalone Integration Guide](./references/daw-and-standalone-guide.md)**: `nih-plug` VST3/CLAP, macOS AUv2 zero-copy & sandbox checklists, Windows WASAPI MMCSS / FTZ+DAZ / MinGW static linking, knob sweep limits, and 2D mic drag physics.
