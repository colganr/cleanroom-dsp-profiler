## Summary

<!-- Describe what this PR changes and why. -->

## Type of Change

- [ ] Bug fix
- [ ] New host backend (VST3 / CLAP / Hardware / AU)
- [ ] DSP architecture or profiling accuracy improvement
- [ ] Rust SIMD / performance optimization
- [ ] AI Agent Skill (`skill/`) or documentation update

## Benchmark & Accuracy Impact (if touching DSP / profiler)

<!-- Report strictly held-out synthetic benchmark metrics before vs. after (zero test-set leakage). -->

| Target | Before Corr % / ESR dB | After Corr % / ESR dB |
| :--- | :---: | :---: |
| Synthetic Tube Amp | — | — |
| Synthetic Overdrive Pedal | — | — |

## Checklist

- [ ] **Clean-Room Attestation:** This PR contains **zero** commercial plugin binaries, extracted weights, proprietary presets, copyrighted IRs/audio stems, or third-party artwork.
- [ ] **Privacy Check:** No personal emails, secrets, tokens, or hardcoded OS home paths (`/Users/<name>`, `/home/<name>`, `C:\Users\<name>`).
- [ ] `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`, and `cargo test --release` pass.
- [ ] `pytest -v` passes (including Python ↔ Rust bit-exact feature parity tests).
- [ ] `python3 scripts/audit_public_release.py` passes with zero errors.
- [ ] Updated `README.md`, `WORKFLOW.md`, or `skill/` references where behavior changed.
