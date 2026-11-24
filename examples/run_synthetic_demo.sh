#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[1/3] Building Pure-Rust DSP starter crate (release cdylib)..."
(cd "${REPO_ROOT}/dsp-core-rust" && cargo build --release)

echo "[2/3] Installing Python profiler (editable, NumPy-only)..."
python3 -m pip install -q -e "${REPO_ROOT}/profiler"

echo "[3/3] Running clean-room synthetic Amp + Overdrive capture & held-out verification demo..."
cleanroom-profiler demo --out-dir "${REPO_ROOT}/captures/demo"
