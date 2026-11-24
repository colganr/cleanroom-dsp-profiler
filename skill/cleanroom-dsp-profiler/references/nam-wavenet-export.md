# NAM (`v0.5.4` WaveNet) Distillation & 24-Bit Cabinet IR Specification

This reference documents the clean-room `.nam` (`v0.5.4` WaveNet) distiller (`profiler/amp_profiler/nam_export.py`) and deterministic capture stimulus (`dsp-core-rust/src/nam_capture.rs`).

---

## 1. Supported NAM `v0.5.4` WaveNet Presets

Third-party hardware pedals and software loaders (Neural Amp Modeler VST3/AU, Valeton, Dimehead, Mod Dwarf, Hotone, Two Notes, etc.) parse `.nam` JSON files conforming to `"version": "0.5.4"` and `"architecture": "WaveNet"`:

| Preset | Channels | Dilated Layers (`kernel=3`) | Activation | Exact Flat `weights` Count | Target Hardware |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`standard`** | `16` | `10` (`[1, 2, 4, 8, 16, 32, 64, 128, 256, 512]`) | `Tanh` | `9,802` | Desktop DAW & high-DSP hardware |
| **`feather`** | `8` | `10` (`[1..512]`) | `Tanh` | `2,818` | Mid-range embedded NAM pedals |
| **`nano`** | `4` | `10` (`[1..512]`) | `Tanh` | `770` | Ultra-low-CPU embedded loaders |

### Exact Flat `weights` Memory Order (NeuralAmpModelerCore ABI)
For each layer block in `config.layers`:
1. `rechannel` (`channels × input_size`, no bias)
2. For each dilation `d` in `dilations`:
   - `conv.weight` (`out_ch × channels × kernel_size`) + `conv.bias` (`out_ch`)
   - `input_mixin.weight` (`out_ch × condition_size`, no bias)
   - `1x1.weight` (`channels × channels`) + `1x1.bias` (`channels`)
3. `head_rechannel.weight` (`head_size × channels`) + optional `head_rechannel.bias` (`head_size` if `head_bias == true`)
4. Final scalar `head_scale` (`1` float)

Always validate serialized `.nam` outputs with `expected_wavenet_weight_count()` and `forward_pass_nam_v054()` before saving.

---

## 2. Signal-Chain Sanitization Rules for Static `.nam` Export

A static WaveNet is a **time-invariant, mono nonlinear + short-memory model** (`receptive field ≈ 64 ms @ 48 kHz`). Always partition your rig before capturing a `.nam` trace:

| Module Type | Action During `.nam` Capture | Reason |
| :--- | :--- | :--- |
| **Overdrive / Boost Pedals** | **Baked In** | Time-invariant clipping & pre/post EQ map cleanly to dilated `Tanh` layers. |
| **Amplifier Head (Preamp + Power Stage)** | **Baked In** | Primary nonlinear target. |
| **Speaker Cabinet IR** | **Optional (`include_cab`) + Separate `.wav` IR** | Bake in for all-in-one rigs OR export separately as a **`2048`-sample `48 kHz` `24-bit PCM` `.wav`** for hardware loaders with dedicated IR slots. |
| **Static Post-EQ** | **Optional** | Minimum-phase EQ shapes the WaveNet linear readout cleanly. |
| **Noise Gate / Tuner Mute** | **Auto-Bypass** | Threshold gating chops low-level calibration sweeps and breaks small-signal linearity. |
| **Pitch Shift / Stereo Doubler** | **Auto-Bypass** | Non-harmonic / stereo-decorrelated processing corrupts mono WaveNet fitting. |
| **Envelope Filter / Auto-Wah** | **Auto-Bypass** | Dynamic filter sweeps bake into a static cocked-wah notch. |
| **LFO Tremolo / Chorus / Phaser** | **Auto-Bypass** | Free-running LFO modulation is non-stationary. |
| **Delays & Spring/Plate/Hall Reverbs** | **Auto-Bypass** | Decay tails (`> 100 ms`) exceed the `64 ms` WaveNet receptive field and cause metallic comb-filtering. |

---

## 3. Why `24-Bit PCM` `.wav` for Cabinet IRs?

Several standalone hardware IR loaders reject `32-bit IEEE float` WAV files or truncate IRs longer than `2048` samples. `save_wav_24bit_pcm()` writes strict `48,000 Hz`, mono, `24-bit signed little-endian PCM` (`2048` samples = `42.67 ms`) for universal compatibility.
