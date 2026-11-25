#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"

echo "[1/4] Building Pure-Rust DSP starter crate (release cdylib)..."
(cd "${REPO_ROOT}/dsp-core-rust" && cargo build --release)

echo "[2/4] Setting up local virtual environment & installing Python profiler (NumPy-only)..."
if [ ! -d "${VENV_DIR}" ]; then
  python3 -m venv "${VENV_DIR}"
fi
"${VENV_DIR}/bin/python" -m pip install -q -e "${REPO_ROOT}/profiler"

echo "[3/4] Running clean-room synthetic Amp + Overdrive capture & held-out verification demo..."
"${VENV_DIR}/bin/cleanroom-profiler" demo --out-dir "${REPO_ROOT}/captures/demo"

echo ""
echo "[4/4] Running clean-room synthetic Spring Reverb, BBD Delay, Tremolo, Envelope Filter & Compressor demos..."
"${VENV_DIR}/bin/cleanroom-profiler" demo-fx
