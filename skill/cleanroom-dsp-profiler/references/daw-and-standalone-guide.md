# Cross-Platform DAW Plugin & Standalone Integration Guide

Once your profiled weights are exported into `dsp-core-rust`, you can ship the exact same zero-allocation Pure-Rust engine across macOS, Windows, and Linux as VST3, CLAP, AUv2, and Standalone apps. This checklist captures hard-won cross-platform audio traps.

---

## 1. Real-Time Audio CPU & Anti-Crackling Checklist (All Platforms)

1. **Hardware Denormal Guard (`x86_64`):**
   - Always wrap `process_block()` in `DenormalGuard::new()` (`MXCSR |= 0x8040`, FTZ + DAZ). Without it, silent note tails trigger 100-cycle x86 subnormal microcode traps and spike CPU 10× on Windows/Linux.
2. **Branchless `fast_tanh` & Hoisted Exponentials:**
   - Use `simd_ops::fast_tanh` (`[7/6]` Padé rational) instead of scalar `libm::tanhf()` inside sample loops (eliminates millions of scalar calls/sec under MinGW/MSVC).
   - Precompute all `exp(-2π f_c / sr)` filter coefficients on sample-rate or parameter changes, never per sample.
3. **Runtime-Dispatched SIMD FIRs:**
   - `simd_ops::dot_f32` and `dot_stereo_f32` automatically dispatch `AVX2+FMA` (falling back to `SSE2`) on `x86_64` and `NEON vfmaq_f32` on `aarch64` without requiring `-C target-cpu=native` (keeping release binaries portable across older CPUs).
4. **Mono-Input Fast Path (Guitar/Bass Chains):**
   - Process mono guitar input through pre-FX and amp head once, splitting to stereo only at the speaker cabinet or stereo post-FX (cuts amp CPU in half compared to redundant dual-mono processing). *(Never force mono summing for true-stereo studio FX!)*

---

## 2. VST3 & CLAP (`nih-plug` on Windows, macOS & Linux)

- Wrap a **per-instance** `StarterRigEngine` inside your `nih_plug::Plugin` struct (never use global mutable state so multiple DAW tracks never collide).
- Support `1→1`, `1→2`, and `2→2` channel layouts, report exact latency (`set_latency_samples`) and accurate tail (`ProcessStatus::Tail` vs `Normal`).
- **Windows MinGW (`x86_64-pc-windows-gnu`) Watchouts:**
  - Statically link C/C++ runtimes (avoid external `libstdc++`, `libgcc`, `libwinpthread` DLL dependencies — verify with `x86_64-w64-mingw32-objdump -p`).
  - Gate `nih-plug`'s debug `assert_process_allocs` feature off for `x86_64-pc-windows-gnu` release builds if it conflicts with MinGW TLS unwind hooks.
  - If embedding Microsoft Edge WebView2 inside `ParentWindowHandle::Win32Hwnd`, dynamically load `WebView2Loader.dll` via `LoadLibraryW` (falling back to EdgeUpdate Registry or system browser) rather than hard-linking a PE import.
- **Linux Cross-Builds on Apple Silicon:** When building `linux/amd64` inside Docker/QEMU emulation, pass `-e CARGO_BUILD_JOBS=2` to avoid `collect2` memory exhaustion.

---

## 3. macOS AUv2 (Logic Pro & GarageBand `.component`) Watchouts

1. **Zero-Copy `AudioBufferList` Handoff:**
   - In Logic Pro, `AudioUnitRender` / `inputProc` may swap `pullABL->mBuffers[i].mData` to point directly at upstream zero-copy buffers. **Always read `(const float *)pullABL->mBuffers[i].mData` *after* pulling input** — never assume the host copied samples into your pre-allocated scratch pointer.
2. **CocoaUI Export Symbols:**
   - When linking with `-Wl,-dead_strip -Wl,-exported_symbol`, explicitly export `_OBJC_CLASS_$_*CocoaViewFactory` and `_OBJC_METACLASS_$_*CocoaViewFactory` alongside `_<Product>AUFactory`, or Logic Pro will silently fall back to generic sliders.
3. **App Sandbox (`AUHostingService` / GarageBand):**
   - Avoid spawning shell subprocesses from inside an AUv2 binary; resolve user paths cleanly and declare required entitlements in `Info.plist`.
4. **Always Validate with `auval`:**
   - Run `auval -v aufx <SubType> <Mfg>` (or `aumf`) and require `AU VALIDATION SUCCEEDED` with zero warnings.
   - Bump `AudioComponents[0].version` (`(Major << 16) | (Minor << 8) | Patch`) on every update so Logic Pro refreshes cached metadata.

---

## 4. Standalone Host (`48,000 Hz / 512` Default)

- Default to **`48,000 Hz`** and **`512` samples** (`10.67 ms`).
- **Windows WASAPI Shared Mode (`cpal`):**
  - Force `cpal::BufferSize::Default` (fixed sizes cause crackling/failures on many Windows interfaces).
  - Promote capture and render audio callbacks via Win32 `SetThreadPriority(GetCurrentThread(), 15)` + MMCSS `"Pro Audio"`.
  - Bridge capture -> render through a device-level hysteresis ring buffer with a short (`32`-sample) crossfade on overrun/underrun resync, and pass parameter updates via lock-free or `try_lock` coalesced queues.
- **macOS CoreAudio:** Enumerate live non-hidden devices (`kAudioHardwarePropertyDevices`) and 1-indexed channel names (`kAudioObjectPropertyElementName` `'lchn'`), offering a `<< none >>` mute option and a live device refresh button.

---

## 5. Hardware-Accurate UI Control Physics (If Building a Custom GUI)

- **Standard Potentiometers (`270°` sweep, 7 o'clock to 5 o'clock):**
  $$\theta_{\text{std}}(\text{norm}) = -135^\circ + 270^\circ \cdot \operatorname{clamp}(\text{norm}, 0, 1)$$
- **Numbered Skirted Knobs (`1..10` around a `300°` dial with fixed 12 o'clock pointer):**
  $$\theta_{\text{skirt}}(\text{norm}) = +150^\circ - 300^\circ \cdot \operatorname{clamp}(\text{norm}, 0, 1)$$
- **2D Speaker-Cone Microphone Drag Physics:**
  - Record `(startX, startY, startPos, startDist)` on `pointerdown` + `setPointerCapture` and apply relative deltas so grabbing never jumps.
  - Dragging **down (`+dy`)** increases `Distance` (`0 → 1`) and scales the microphone **larger** (pulling back from the grille toward the viewer); dragging **outward** from the dustcap center increases `Position` (`0 → 1`).
