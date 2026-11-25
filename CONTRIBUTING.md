# Contributing to `cleanroom-dsp-profiler`

Thank you for helping push open-source, clean-room audio DSP profiling forward! Pull requests, bug fixes, new host backends, DSP accuracy improvements, SIMD optimizations, and documentation improvements are all welcome.

---

## Strict Clean-Room & IP Policy (Read First)

Pull requests **must** stay 100% clean-room. Any PR containing the following will be closed without merge:

- Commercial plugin binaries, installers, or extracted binary constants / disassembled weights
- Captured weight files, impulse responses, `.nam` models, or factory preset dumps derived from commercial plugins or copyrighted hardware captures
- Third-party UI artwork, logos, sprite sheets, or trademarked product assets
- Copyrighted audio stems / commercial DI tracks (use the procedural DI generator or CC0 / original test audio)
- Personal information (real emails, hardcoded OS home paths like `/Users/<name>` or `C:\Users\<name>`, API keys, tokens)

---

## High-Impact Contributions Wanted

We especially welcome PRs in these areas:

1. **Cross-Platform Live Plugin Host Backends (`profiler/amp_profiler/backends/`):**
   - Native VST3 / CLAP headless host driver (e.g., via `pedalboard`, `clap-host`, or a small C++/Rust bridge) for Windows, Linux, and macOS
   - Live Hardware Re-Amping driver (automatic loopback latency measurement + level calibration over ASIO / CoreAudio / PipeWire)
2. **New Nonlinear & Time-Variant DSP Topologies:**
   - Germanium / silicon fuzz circuits (input impedance loading + gating/starve sag)
   - Optical / FET program-dependent compressor profiling (photocell memory & multi-stage release)
   - BBD chorus / flanger FM depth profiling, multi-stage analog phasers, and algorithmic FDN reverbs
3. **Solver & Performance Enhancements:**
   - Accelerated multi-threaded / GPU / Apple Accelerate ridge Cholesky solvers
   - Automated hyper-parameter sweeps for bloom knee / sag time-constant adaptation
   - Additional SIMD paths (AVX-512, WASM SIMD128)
4. **AI Agent Skill Improvements (`skill/cleanroom-dsp-profiler/`):**
   - Verified troubleshooting entries for tricky DAW host quirks, multi-channel routing, or tone-stack topologies

---

## Engineering Invariants

Every PR touching DSP or profiling math must preserve these invariants:

1. **Zero Heap Allocations in Real-Time Audio (`dsp-core-rust`):** `process_sample()` and `process_block()` must perform zero heap allocations, zero locks, and zero syscalls.
2. **Exact Small-Signal Unit Slope:** Any nonlinear base saturator $\text{sat}_0(u)$ and zero-slope residual basis $\psi_k(u)$ must satisfy $\text{sat}_0'(0) = 1.0$ and $\psi_k'(0) = 0.0$ so small-signal linear frequency response ($H_{\text{pre}} \cdot H_{\text{post}}$) is never corrupted.
3. **Zero Test-Set Leakage:** Calibration and verification audio segments must be strictly disjoint. Never report training-set correlation as a benchmark.
4. **Bit-Exact Python ↔ Rust Parity:** Any change to feature generation (`amp_core`, `pedal_core`, `reservoir`, `cepstral_fir`) must update both Python and Rust implementations and pass `test_python_rust_parity.py` (`max abs diff < 1e-5`).

---

## Local Development Setup

```bash
# 1. Build and test the Pure-Rust DSP crate (including release cdylib for Python FFI tests)
cd dsp-core-rust
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test --release
cargo build --release

# 2. Install the Python profiler in editable mode with test tools
cd ../profiler
python3 -m pip install -e ".[dev]"
pytest -v

# 3. Run the public privacy & clean-room audit
cd ..
python3 scripts/audit_public_release.py
```

---

## Pull Request Process

1. Fork the repository and create a feature branch from `main`.
2. Make focused commits with clear messages.
3. Add or update unit tests (especially synthetic accuracy benchmarks if changing DSP math).
4. Run all three checks above (`cargo test`, `pytest`, `audit_public_release.py`).
5. Open a Pull Request and complete the checklist in `.github/PULL_REQUEST_TEMPLATE.md`.
