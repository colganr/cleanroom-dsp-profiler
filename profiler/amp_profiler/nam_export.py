"""
Clean-Room NAM (`v0.5.4` WaveNet) Distiller & 24-Bit PCM `.wav` Cabinet IR Exporter.

Produces spec-compliant `.nam` (`version: "0.5.4"`, `architecture: "WaveNet"`) JSON files
matching NeuralAmpModelerCore's exact flat `weights` memory layout, plus optional 2048-sample
48 kHz 24-bit PCM `.wav` cabinet IRs compatible with hardware loaders.

Supported presets:
  - `standard` (16-ch, 10 dilated layers [1..512], kernel=3, Tanh, 9,802 weights)
  - `feather`  (8-ch,  10 dilated layers [1..512], kernel=3, Tanh, 2,818 weights)
  - `nano`     (4-ch,  10 dilated layers [1..512], kernel=3, Tanh,   770 weights)
"""

from __future__ import annotations

import datetime
import json
import math
from pathlib import Path
import numpy as np
from .stimulus import SR_DEFAULT, save_wav_24bit_pcm

DILATIONS_10 = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


def expected_wavenet_weight_count(config: dict) -> int:
    """Computes the exact flat float32 weight count required by NeuralAmpModelerCore v0.5.4."""
    total = 0
    for blk in config["layers"]:
        in_sz = int(blk["input_size"])
        cond_sz = int(blk["condition_size"])
        ch = int(blk["channels"])
        head_sz = int(blk["head_size"])
        k_sz = int(blk["kernel_size"])
        gated = bool(blk["gated"])
        head_bias = bool(blk["head_bias"])
        out_ch = ch * (2 if gated else 1)

        # 1. rechannel (channels x input_size, no bias)
        total += ch * in_sz
        # 2. per-dilation layers
        for _ in blk["dilations"]:
            total += out_ch * ch * k_sz + out_ch   # conv weight + bias
            total += out_ch * cond_sz              # input_mixin weight (no bias)
            total += ch * ch + ch                  # 1x1 weight + bias
        # 3. head_rechannel
        total += head_sz * ch + (head_sz if head_bias else 0)
    # 4. final head_scale
    total += 1
    return total


def forward_pass_nam_v054(x_in: np.ndarray, nam_doc: dict) -> np.ndarray:
    """
    Reference forward pass matching NeuralAmpModelerCore `wavenet.cpp` (v0.5.4) flat weight layout.
    Used to verify the serialized `.nam` JSON directly before saving.
    """
    x = np.asarray(x_in, dtype=np.float64)
    n = len(x)
    w = np.asarray(nam_doc["weights"], dtype=np.float64)
    cfg = nam_doc["config"]
    expected = expected_wavenet_weight_count(cfg)
    if len(w) != expected:
        raise ValueError(f"NAM weight count mismatch: got {len(w)}, expected {expected}")

    ptr = 0
    layer_in = x[None, :]  # (1, N)
    cond = x[None, :]      # (1, N)
    head_accum: np.ndarray | None = None

    for blk in cfg["layers"]:
        in_sz = int(blk["input_size"])
        cond_sz = int(blk["condition_size"])
        ch = int(blk["channels"])
        head_sz = int(blk["head_size"])
        k_sz = int(blk["kernel_size"])
        gated = bool(blk["gated"])
        head_bias = bool(blk["head_bias"])
        out_ch = ch * (2 if gated else 1)

        # rechannel: (ch, in_sz)
        w_rech = w[ptr : ptr + ch * in_sz].reshape(ch, in_sz)
        ptr += ch * in_sz
        h = w_rech @ layer_in  # (ch, N)

        if head_accum is None:
            head_accum = np.zeros((ch, n), dtype=np.float64)

        for d in blk["dilations"]:
            d = int(d)
            # conv: (out_ch, ch, k_sz) + bias (out_ch,)
            conv_w = w[ptr : ptr + out_ch * ch * k_sz].reshape(out_ch, ch, k_sz)
            ptr += out_ch * ch * k_sz
            conv_b = w[ptr : ptr + out_ch]
            ptr += out_ch

            # input_mixin: (out_ch, cond_sz)
            mix_w = w[ptr : ptr + out_ch * cond_sz].reshape(out_ch, cond_sz)
            ptr += out_ch * cond_sz

            # 1x1: (ch, ch) + bias (ch,)
            one_w = w[ptr : ptr + ch * ch].reshape(ch, ch)
            ptr += ch * ch
            one_b = w[ptr : ptr + ch]
            ptr += ch

            # Causal dilated Conv1d: tap offset (k_sz - 1 - k) * d
            z = conv_b[:, None] + (mix_w @ cond)
            for k_idx in range(k_sz):
                shift = (k_sz - 1 - k_idx) * d
                if shift == 0:
                    z += conv_w[:, :, k_idx] @ h
                elif shift < n:
                    z[:, shift:] += conv_w[:, :, k_idx] @ h[:, :-shift]

            if gated:
                z_act = np.tanh(z[:ch]) * (1.0 / (1.0 + np.exp(-z[ch:])))
            else:
                z_act = np.tanh(z)

            head_accum = head_accum + z_act
            h = h + (one_w @ z_act + one_b[:, None])

        # head_rechannel: (head_sz, ch) + optional bias (head_sz,)
        hr_w = w[ptr : ptr + head_sz * ch].reshape(head_sz, ch)
        ptr += head_sz * ch
        if head_bias:
            hr_b = w[ptr : ptr + head_sz]
            ptr += head_sz
        else:
            hr_b = np.zeros(head_sz, dtype=np.float64)

        head_accum = hr_w @ head_accum + hr_b[:, None]
        layer_in = h

    head_scale = float(w[ptr])
    ptr += 1
    assert ptr == len(w)
    assert head_accum is not None
    return (head_accum[0] * head_scale).astype(np.float32)


def distill_nam_v054(
    x_in: np.ndarray,
    y_target: np.ndarray,
    preset: str = "standard",
    sr: int = SR_DEFAULT,
    model_name: str = "CleanRoom Capture",
    ridge_lambda: float = 1e-4,
) -> tuple[dict, dict]:
    """
    Distills an `(x_in, y_target)` nonlinear trace into a spec-compliant `.nam` (`v0.5.4`)
    WaveNet JSON using deterministic multi-scale dilated reservoir projections + Cholesky ridge.

    Returns `(nam_json_dict, metrics_dict)`.
    """
    ch_map = {"standard": 16, "feather": 8, "nano": 4}
    if preset not in ch_map:
        raise ValueError(f"Unknown NAM preset {preset!r}; choose from {list(ch_map)}")
    ch = ch_map[preset]

    x = np.asarray(x_in, dtype=np.float64)
    y = np.asarray(y_target, dtype=np.float64)
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]

    config = {
        "layers": [
            {
                "input_size": 1,
                "condition_size": 1,
                "head_size": 1,
                "channels": ch,
                "kernel_size": 3,
                "dilations": list(DILATIONS_10),
                "activation": "Tanh",
                "gated": False,
                "head_bias": True,
            }
        ],
        "head": None,
        "head_scale": 1.0,
    }

    # Build deterministic multi-scale, multi-band orthogonal dilated filterbank
    scales = np.geomspace(0.45, 9.5, ch)
    biases = np.linspace(-0.12, 0.12, ch)
    w_rech = (scales * 0.25).reshape(ch, 1)

    h = w_rech @ x[None, :]
    head_sum = np.zeros((ch, n), dtype=np.float64)

    flat_weights: list[float] = w_rech.ravel().tolist()

    for l_idx, d in enumerate(DILATIONS_10):
        # Diagonal + subtle neighbor coupling across causal taps [t-2d, t-d, t]
        conv_w = np.zeros((ch, ch, 3), dtype=np.float64)
        for c in range(ch):
            # Vary low-pass / band-pass / high-pass character across channel index
            band_mode = c % 4
            if band_mode == 0:    # Broadband / fast
                taps = (0.05, 0.15, 0.45)
            elif band_mode == 1:  # Low-pass warmth
                taps = (0.20, 0.30, 0.25)
            elif band_mode == 2:  # Mid-focused difference
                taps = (-0.15, 0.35, 0.20)
            else:                 # Edge / presence derivative
                taps = (0.10, -0.25, 0.40)
            decay = 0.72 ** (l_idx * 0.35)
            conv_w[c, c, 0] = taps[0] * decay
            conv_w[c, c, 1] = taps[1] * decay
            conv_w[c, c, 2] = taps[2] * decay
            # Small stable neighbor coupling
            nbr = (c + 1) % ch
            conv_w[c, nbr, 2] = 0.06 * ((-1.0) ** (c + l_idx)) * decay

        conv_b = biases * (0.6 if l_idx % 2 == 0 else -0.4)
        mix_w = (scales * (0.65 if l_idx < 5 else 0.35) * ((-1.0) ** l_idx)).reshape(ch, 1)
        one_w = np.eye(ch, dtype=np.float64) * 0.18
        one_b = np.zeros(ch, dtype=np.float64)

        # Forward propagate layer
        z = conv_b[:, None] + (mix_w @ x[None, :])
        for k_idx in range(3):
            shift = (2 - k_idx) * d
            if shift == 0:
                z += conv_w[:, :, k_idx] @ h
            elif shift < n:
                z[:, shift:] += conv_w[:, :, k_idx] @ h[:, :-shift]
        z_act = np.tanh(z)
        head_sum += z_act
        h = h + (one_w @ z_act + one_b[:, None])

        flat_weights.extend(conv_w.ravel().tolist())
        flat_weights.extend(conv_b.ravel().tolist())
        flat_weights.extend(mix_w.ravel().tolist())
        flat_weights.extend(one_w.ravel().tolist())
        flat_weights.extend(one_b.ravel().tolist())

    # Solve head_rechannel (1 x ch) + head_bias (1,) via ridge Cholesky regression
    skip = min(1024, n // 4)
    H_design = np.vstack([head_sum[:, skip:], np.ones((1, n - skip), dtype=np.float64)]).T
    y_fit = y[skip:]
    XtX = H_design.T @ H_design
    Xty = H_design.T @ y_fit
    reg = max(float(np.trace(XtX)) / H_design.shape[1] * ridge_lambda, 1e-8)
    theta = np.linalg.solve(XtX + reg * np.eye(H_design.shape[1]), Xty)

    hr_w = theta[:ch]
    hr_b = theta[ch : ch + 1]
    flat_weights.extend(hr_w.tolist())
    flat_weights.extend(hr_b.tolist())
    flat_weights.append(1.0)  # head_scale

    nam_doc = {
        "version": "0.5.4",
        "architecture": "WaveNet",
        "config": config,
        "metadata": {
            "name": model_name,
            "modeled_by": "cleanroom-dsp-profiler",
            "date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "sample_rate": int(sr),
            "preset": preset,
        },
        "weights": [float(v) for v in flat_weights],
        "sample_rate": int(sr),
    }

    # Verify serialized JSON forward pass directly
    y_pred = forward_pass_nam_v054(x, nam_doc).astype(np.float64)
    a = y[skip:]
    b = y_pred[skip:]
    corr = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)) * 100.0
    esr = float(np.sum((a - b) ** 2) / max(float(np.sum(a ** 2)), 1e-12))
    esr_db = 10.0 * math.log10(esr + 1e-12)

    metrics = {
        "preset": preset,
        "weight_count": len(flat_weights),
        "corr_pct": round(corr, 3),
        "esr_db": round(esr_db, 3),
    }
    return nam_doc, metrics


def save_nam_and_ir(
    nam_path: str | Path,
    nam_doc: dict,
    cab_ir: np.ndarray | None = None,
    ir_path: str | Path | None = None,
    sr: int = SR_DEFAULT,
) -> tuple[Path, Path | None]:
    """Saves `.nam` JSON and optional 2048-sample 48 kHz 24-bit PCM `.wav` cabinet IR."""
    np_out = Path(nam_path)
    np_out.parent.mkdir(parents=True, exist_ok=True)
    np_out.write_text(json.dumps(nam_doc, indent=2), encoding="utf-8")

    saved_ir: Path | None = None
    if cab_ir is not None and ir_path is not None:
        ir = np.zeros(2048, dtype=np.float32)
        src = np.asarray(cab_ir, dtype=np.float32)
        take = min(2048, len(src))
        ir[:take] = src[:take]
        save_wav_24bit_pcm(ir_path, ir, sr=sr)
        saved_ir = Path(ir_path)
    return np_out, saved_ir
