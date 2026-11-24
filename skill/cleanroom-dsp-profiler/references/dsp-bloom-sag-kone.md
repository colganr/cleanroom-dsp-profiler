# DSP, Bloom, Sag & 96 kHz KONE Architecture

Mathematical reference for the clean-room `cleanroom-dsp-profiler` nonlinear modeling engines (`dsp-core-rust` & `profiler/amp_profiler`).

---

## 1. Wiener–Hammerstein $H_{\text{pre}} \to \text{Core} \to H_{\text{post}}$ Factorization

Factoring the linear dynamics around the nonlinear tube or diode core allows every switch and potentiometer to remain continuously interactive while preserving exact small-signal frequency response and post-clipping harmonic filtering:

$$x_{\text{in}}[n] \xrightarrow{H_{\text{pre}}(\theta_{\text{pre}})} u_{\text{pre}}[n] \xrightarrow{\text{sat}_0(u_{\text{pre}}) + \Phi(u_{\text{pre}}) \cdot w} y_{\text{nl}}[n] \xrightarrow{H_{\text{post}}(\theta_{\text{post}})} y_{\text{out}}[n]$$

### 1.1 Measuring $H_{\text{tot}}(f)$ & Separating $H_{\text{post}}(f)$
1. **Small-Signal Total Response $H_{\text{tot}}(f)$:** Measure with symmetric bipolar impulses of amplitude $\epsilon = 10^{-4}$ (`-80 dBFS`):
   $$h_{\text{tot}}[n] = \frac{y_+[n] - y_-[n]}{2\epsilon}$$
   Using $\pm 10^{-4}$ avoids clipping high-gain preamps (`+40..+50 dB` small-signal gain) and cancels DC offsets and asymmetric bias drift.
2. **Post-Clipping Response $H_{\text{post}}(f)$:** Drive the nonlinear target with `24–32` log-spaced hard-clipping sine tones ($30\text{ Hz} \dots 16\text{ kHz}$, amplitude $A \in [0.25, 0.35]$) so the clipper output approaches a rail-to-rail square wave whose fundamental entering $H_{\text{post}}$ is constant across frequency. Extract the fundamental via quadrature lock-in (`cos`/`sin` inner product) and normalize to $0\text{ dB}$ at `400 Hz`.
3. **Pre-Clipping Deconvolution:**
   $$H_{\text{pre}}(f) = \frac{H_{\text{tot}}(f) \cdot H_{\text{post}}^*(f)}{|H_{\text{post}}(f)|^2 + \epsilon^2}$$
   Because $H_{\text{pre}} \cdot H_{\text{post}} \equiv H_{\text{tot}}$ linearly, small-signal accuracy is preserved while high-order distortion harmonics (`3f, 5f, 7f, 9f`) are naturally sculpted by $H_{\text{post}}$.

### 1.2 Real-Cepstrum Minimum-Phase FIR Folding (`FFT_N = 2048`)
Given a 64-point log-spaced magnitude delta curve $\Delta M_{\text{dB}}(f)$:
1. Interpolate $\Delta M_{\text{dB}}(f)$ onto FFT bins $k \in [0..N/2]$ and compute log-magnitude $L[k] = \Delta M_{\text{dB}}[k] \cdot \frac{\ln 10}{20}$ (mirrored conjugate-symmetrically).
2. Inverse FFT to obtain the real cepstrum $c[n] = \operatorname{Re}(\operatorname{IFFT}(L))[n]$.
3. Apply the causal Hilbert window:
   $$W[n] = \begin{cases} 1 & n = 0 \text{ or } n = N/2 \\ 2 & 1 \le n < N/2 \\ 0 & n > N/2 \end{cases}$$
4. Forward FFT $\Phi[k] = \operatorname{FFT}(c \cdot W)[k]$, form $H_{\min}[k] = \exp(\operatorname{Re}\Phi[k])\,e^{j\operatorname{Im}\Phi[k]}$, multiply by $\operatorname{FFT}(h_{\text{base}})$, and inverse FFT with a raised-cosine tail fade.

### 1.3 Non-Additive $2^K$ Switches & Cap-Coupled 3D Tone Stacks
- Passive front-end switches (`Bright`, `Mid Boost`, `Deep`, `EQ1/EQ2`, `Tone Bypass`) bypass attenuation inside a shared passive voltage divider bounded by `0 dB`. Summing individual dB deltas overshoots gain by **`+10 to +12 dB`** when multiple switches are active. Always capture all $2^K$ switch combinations.
- When a toggle switch parallels a capacitor across `C_treble` (classic boutique Mid Boost circuits), it shifts the Treble pot corner frequency down into the midrange. Index the `5×5×5` `(Bass × Middle × Treble)` 3D tensor by that switch state too.

---

## 2. Bounded Zero-Slope 12AX7 Triode Transconductance "Bloom"

Cold-biased 12AX7 preamp stages expand in gain by `+1.4 to +1.8 dB` across mid-level playing (`-35 to -15 dBFS`) before grid-current compression sets in. Plain $\tanh(u)$ prematurely compresses mid levels, creating `-9 to -11 dB` ESR on clean/edge-of-breakup channels, while unbounded bloom terms explode when high-gain switches push $u_{\text{pre}} \gg 1$.

Our **Bounded Zero-Slope Bloom Saturator** (`sat_bloom_bounded_sample`) solves both:

$$v_{\text{bloom}}(x) = x + \alpha_{\text{bloom}}\, x\, \tanh^2\!\left(\frac{x}{\sigma_1}\right) \operatorname{sech}^2\!\left(\frac{x}{\sigma_2}\right)$$

$$\text{sat}_0(x) = \frac{v_{\text{bloom}}(x)}{\left(1 + (|v_{\text{bloom}}(x)| / c_{\pm})^{k}\right)^{1/k}}$$

Default clean-room parameters: `cp0 = 0.212`, `cn0 = 0.227`, `knee = 2.9`, `bloom_alpha = 0.18`, `sigma1 = 0.16`, `sigma2 = 0.32`.

**Key Properties:**
- **Exact unit slope at zero:** $\left.\frac{d}{dx}\text{sat}_0(x)\right|_{x=0} = 1.0$ (zero small-signal Bode leak).
- **Mid-signal bloom:** Smooth `+1.5 dB` gain lift around $|x| \approx 0.15\text{–}0.35$.
- **Strict rail asymptote:** Because $\operatorname{sech}^2(x/\sigma_2) \to 0$ as $|x| \to \infty$, bloom shuts off cleanly at high drive and $\text{sat}_0(x) \to \pm c_\pm$.

---

## 3. Universal 180-Channel Amplifier Core (`amp_core.rs`)

Wrapped inside an invertible `85 Hz / 113.9 Hz` low-shelf pre-emphasis (`x_pre[n] = u[n] - p_pole * u[n-1] + p_zero * x_pre[n-1]`) and its exact algebraic inverse post-shelf:

1. **Dynamic Tube Envelope Trackers:**
   - Cathode bypass (`95 Hz`, $\tau \approx 1.7\text{ ms}$): $g_{\text{cath}} = \tanh(0.35 \cdot \operatorname{LP}_{95}(|x_{\text{pre}}|))$
   - B+ power-supply sag (`12 Hz`, $\tau \approx 13\text{ ms}$): $g_{\text{sag}} = \tanh(0.35 \cdot \operatorname{LP}_{12}(|x_{\text{pre}}|))$
   - Grid conduction bias shift (`16 Hz`, $\tau \approx 10\text{ ms}$): $b_{\text{grid}} = \tanh(0.35 \cdot \operatorname{LP}_{16}(x_{\text{pre}}))$
2. **Channels `0..126` (`14` Rail-Bounded Zero-Slope Branches $\times$ `9` Taps):**
   - `6` wideband & asymmetric DC-blocked branches (`0.55x`, `1.45x`, asymmetric `0.80x/1.25x`, `2.80x`, `8.00x`, and rail-bounded headroom recovery `x_rail - sat0`).
   - `3` coupled `V1 → V2` interstage branches driven by `180 Hz` & `383 Hz` coupling-cap excess (`v2_inter = x_rail - 0.85 * LP(ex1)`).
   - `5` frequency-split branches across `180 Hz`, `720 Hz`, and `2400 Hz` poles.
   - Each branch emits `6` causal dilations `[0, 1, 2, 4, 8, 16]` + `3` envelope-modulated products (`g_cath * s`, `g_sag * s`, `b_grid * s`).
3. **Channels `126..174` (`16` 96 kHz 2× Oversampled Clean-Room KONE States $\times$ `3` Taps):**
   - Cubic half-sample upsampling (`0.5625*(x[n-1]+x[n]) - 0.0625*(x[n-2]+x[n])`) feeds two coupled 8-state spectral-radius-bounded recurrent cells (`h1` preamp → interstage grid clamp `v1 - 0.12 * max(0, v1 - 0.35)^2` → `h2` push-pull power stage) with sag-modulated headroom (`c1*(1-0.08*g_sag)`, `c2*(1-0.12*g_sag)`) and grid-modulated bias.
   - Every state outputs a strictly zero-slope residual:
     $$r_k = (\tanh(u_{\text{cell}} + b) - \tanh(b)) - (1 - \tanh^2(b)) \cdot c \tanh(u_{\text{cell}} / c)$$
   - Decimated back to `48 kHz` (`0.22*prev + 0.56*cur + 0.22*mid`), `28 Hz` DC-blocked, and read at `[r0, r_{t-2}, g_sag * r0]`.
4. **Channels `174..180` (`6` Sag-Recovery & Envelope Compression Channels):**
   - `35 Hz` & `383 Hz` coupling-cap discharge (`lp35_ex1_lo`, `lp383_ex1_lo`, `lp383_ex1_mid`) plus `g_sag * sat0`, `g_cathode * sat0`, and `bias_grid * sat0`.

### Optional 3-Stage Invertible Pole-Zero Coupled Tube Spine (`TubeSpineState`)
For high-gain cascading leads where strong cathode-bias envelope pumping and push-pull screen sag dominate, `amp_core.rs` also provides a modular 3-stage invertible pole-zero spine (`V1` → `V2` before $H_{\text{post}}$, and Push-Pull Output Transformer + Screen Sag after $H_{\text{post}}$).

---

## 4. 126-Channel 3-Pole Overdrive Core (`pedal_core.rs`)

For 3-knob (`Drive / Tone / Level`) or 4-knob (`+ Mix`) stompboxes:
- Rail-bound pre-filtered input $u_{\text{pre}} = 2.4 \tanh(u_{\text{raw}}/2.4)$ so hot stacked pedals never diverge.
- Define feedback signal $u_{\text{fb}} = u_{\text{pre}} - x_{\text{ref}}$ (`subtract_dry = true` for op-amp clean-bleed topologies like TS/Klon; `false` for pure series clippers).
- Across `7` wide diode scales (`[0.25, 0.55, 1.20, 2.60, 5.50, 11.50, 24.00]`), emit `18` strictly zero-slope features per scale (`126` total): `6` dilations of `p1` (`tanh`), `3` dilations of the `96 kHz` 8-state clean-room KONE reservoir, algebraic soft-knee `p2`, `2` dilations of DC-blocked asymmetric `p3`, `3` envelope-gated terms (`110 Hz`, `14 Hz`, `18 Hz`), and `3` frequency-split poles (`180 Hz`, `720 Hz`, `2400 Hz`).

---

## 5. Scale-Penalized Ridge Regularization Law

Unconstrained least squares over wide-scale zero-slope channels often assigns large opposing weights (`±50`) to high-threshold columns to approximate a linear slope shift $-k \cdot u$, causing severe scale/phase collapse on unseen playing.

Always solve:
$$\left(X^T W X + \operatorname{diag}\!\left(m_c \cdot (\lambda_{\text{rel}} \bar{d} + 0.5 \lambda_{\text{rel}} d_c + \lambda_{\text{abs}})\right)\right) w = X^T W r$$
where $d_c = (X^T W X)_{cc}$, $\bar{d} = \operatorname{mean}(d)$, and $m_c \ge 1$ applies extra penalty to high-threshold branches (e.g., $m_c = 3.0$ on wide-headroom amp branches and $m_c = 1 + (s_i / 4.0)^2$ on pedal scales).
