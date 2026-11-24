# Host Quirks, Settle Protocol & Zero-Leakage Stimulus

Real-world audio plugins and hardware loops have subtle runtime traps that silently ruin black-box measurements if ignored. Follow these rules every time.

---

## 1. Live Headless AudioUnit Quirks (macOS `AudioToolbox`)

1. **Disable Internal Oversampling Before Sweeping (When Exposed):**
   - Many plugins default to `2×/4×` internal oversampling, adding variable linear-phase latency and CPU overhead during thousands of sweep points. Turn oversampling off during linear/nonlinear capture (or explicitly record its exact onset latency if you intend to profile with it on).
2. **Async Background Model Loading (`CFRunLoop` + 10-Pass Settle):**
   - JUCE and neural-backed plugins frequently load internal models asynchronously on a message/worker thread over the first `~200–400 ms` after instantiation or channel switching.
   - Always call `backend.settle(full=True)` (which runs `10` passes of `CFRunLoopRunInMode(kCFRunLoopDefaultMode, 0.048, False)` + `28` silent blocks) on startup, on every multiprocessing worker process, and after channel/mode switches.
3. **Flush Between Bipolar Impulses:**
   - Render `6` silent blocks (`backend.settle(full=False)`) before and after every `+1e-4` and `-1e-4` impulse so capacitive or tank tails never contaminate the next measurement.
4. **Hidden Mode-Switch Initialization Quirk:**
   - Some plugins leave an internal charge pump or secondary gain stage uninitialized until its mode toggle is cycled (`0.0 -> 1.0 -> 0.0`). Verify small-signal impulse peak before and after toggling mode switches once during discovery.
5. **Use Bipolar Causal Onset Detection (Never Raw `argmax`):**
   - Low-pass filters (overdrive `Tone` pots, speaker cabinets, output transformers) delay the impulse peak by `1–3` samples after tap 0, while asymmetric diode/tube bias creates a non-zero startup transient before the impulse arrives.
   - `probe_onset_latency()` sends symmetric `±1e-3` impulses (`0.5 * (y_+ - y_-)` cancels bias transients completely) and finds the first sample exceeding `0.1%` of peak, recovering the exact causal onset with zero chopped leading taps.
6. **Safe Fast Exit on Process Teardown:**
   - Some AudioUnits spawn detached UI/telemetry threads that deadlock during Python interpreter shutdown. `MacOSAudioUnitBackend` registers an `atexit` hook (`os._exit(0)` after flushing stdout/stderr) to avoid hangs in automated sweeps.

---

## 2. Cross-Platform Offline WAV-Pair & Hardware Re-Amping Hygiene

When profiling VST3/CLAP/AAX plugins in a DAW (Windows/macOS/Linux) or re-amping a physical amplifier:

1. **Match Sample Rate Strictly (`48,000 Hz`):** Export and record at `48,000 Hz`, `24-bit PCM` or `32-bit float` mono WAV, with **dithering OFF**, **normalize OFF**, and **all downstream limiters/gates/reverbs bypassed**.
2. **Calibrate Guitar DI Peak Level:**
   - Real passive single-coil pickups into a `1 MΩ` Hi-Z input peak around **`0.160` (`-15.9 dBFS`)**.
   - Vintage humbuckers peak around **`0.280` (`-11.1 dBFS`)**.
   - Raw normalized `.wav` stems peaking at `0.95` (`-0.4 dBFS`) are **`+15.5 dB` too hot** and will drive clean amplifiers into artificial clipping.
3. **Automatic Latency Alignment:**
   - `WavPairBackend` runs FFT cross-correlation over the first 2 seconds (`max_lag = 4096` samples) to compensate for DAW plugin delay compensation (PDC) offsets or hardware ADC+DAC round-trip latency automatically.
4. **Check Polarity (`+1` vs `-1`):**
   - Many tube amp circuits invert polarity on odd-triode channels (`Gain` vs `FET` input, Clean vs Lead). If correlation comes back `-99%`, flip polarity (`-1.0`) on `H_pre` and re-verify!

---

## 3. Mandatory 15-Second Zero-Leakage Composite Stimulus

Never evaluate model accuracy on the same guitar phrase you trained on. `build_15s_composite_stimulus()` enforces a hard sample boundary:

| Segment | Time Range (`48 kHz` Samples) | Purpose | Used in Fitting? |
| :--- | :---: | :--- | :---: |
| **1a. Linear Log Sweep** | `0.0s – 2.0s` (`0 : 96,000`) | `-36 dBFS` (`0.01585`) `20 Hz – 20 kHz` sweep for small-signal $H_{\text{tot}}(f)$ | Linear only |
| **1b. Driven Log Sweep** | `2.0s – 4.0s` (`96,000 : 192,000`) | High-drive `20 Hz – 20 kHz` sweep for $H_{\text{post}}(f)$ separation | Linear only |
| **2. Calibration Segment** | `4.0s – 8.0s` (`192,000 : 384,000`) | Dynamic amplitude ramp + gated multi-freq bursts (`110/330/820/2000 Hz`) exciting cathode/sag/grid time constants + calibration DI phrase | **YES (Ridge Fit)** |
| **3. Held-Out Verification** | `8.0s – 15.0s` (`384,000 : 720,000`) | Unseen musical DI phrase with distinct notes, chords, palm mutes & dynamics | **NEVER (Test Only)** |
