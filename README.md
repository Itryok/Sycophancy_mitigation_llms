# Sycophancy_mitigation_llms
Beyond Sycophancy:Comparing Input Sanitization and Activation Steering in LLMs



## Mechanistic Intervention: Attention QKV-Bias Ablation & Head-Level Routing Analysis

### Overview & Motivation

Across matched-anchor authority-pressure evaluations on BoolQ, **[Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)** and **[Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/discussions/62)** exhibit sharply different late-layer geometry: Qwen separates true-claim and false-claim activation shifts ($\Delta \mathbf{h}_\ell^{\text{true}}$ vs. $\Delta \mathbf{h}_\ell^{\text{false}}$, each measured relative to a matched neutral prompt), whereas Llama keeps both shifts highly collinear through its final layers.

A key architectural difference between the two models is that Qwen2.5-7B-Instruct includes learned additive biases on its attention query, key, and value projections (`q_proj.bias`, `k_proj.bias`, `v_proj.bias` across all 28 layers, with a tighter 28 Q / 4 KV GQA ratio), whereas Llama-3.1-8B-Instruct omits attention QKV biases entirely (32 Q / 8 KV heads). This intervention causally tests whether Qwen's late-layer QKV biases act as an **authority-framing sink** and enable differential token routing, both independently and in combination with **Subspace-Orthogonalized Decision-Boundary CAA** (Algorithm 1).

---

### Experimental Design & Intervention Protocol

#### Models, Quantization & Numerical Stability
- **Models**: `meta-llama/Llama-3.1-8B-Instruct` (32 layers, $d = 4096$, no QKV bias) and `Qwen/Qwen2.5-7B-Instruct` (28 layers, $d = 3584$, QKV bias enabled).
- **Hardware & Quantization**: Single NVIDIA T4 GPU (16 GB VRAM), 4-bit NF4 double quantization with FP16 compute (`bitsandbytes`), eager attention for exact per-head weight inspection.
- **FP32 Attention Overflow Fix**: Qwen's final layer (Layer 27) carries an extreme key-bias norm ($\|\mathbf{b}_K\|_2 = 920.5$), which overflows FP16 `QK^T` matmul in eager attention and yields `NaN` logits in unpatched runs. `qkv_bias_late_layer_surgical.py` patches `Qwen2Attention`'s eager path to compute `QK^T` and softmax in FP32 while preserving exact attention weights and `0` NaNs across all conditions.

#### CAA Steering & Matched-Anchor Prompts
- **CAA Extraction (Algorithm 1)**: Steering vectors are extracted at $l = \lfloor 0.55 L \rfloor$ (**Layer 17** for Llama; **Layer 15** for Qwen) from $N = 16$ **[TruthfulQA](https://huggingface.co/datasets/truthfulqa/truthful_qa)** items across 8 professional categories under high first-person authority framing ($P^{(1)}_{\text{high}}$), averaging the final $K = 5$ opening decision tokens (`"Yes, I completely agree with your finding"` vs. `"No, that claim is factually incorrect"`), then Gram-Schmidt orthogonalized against $M = 8$ GSM8K math anchors (`openai/gsm8k`). During prefill, the final $S = 4$ decision-boundary tokens are steered by $-\alpha \|\mathbf{h}\|_2 \hat{\mathbf{v}}_\perp$ with $\alpha = 0.24$.
- **Decision-Adjacent BoolQ Prompts**: $N = 20$ balanced BoolQ training items (10 True / 10 False) are formatted with compact context (`<= 220` chars) and question first, followed immediately by the authority wrapper and True/False assertion right before `"Answer strictly Yes or No."`. Placing the assertion adjacent to the decision boundary avoids prompt-length dilution of the 1-token polarity difference.
- **Zero-Echo Logit-Probability Judge (Eq. 6)**: Stance compliance is measured directly from the final prefill logits as $P(\text{Yes}) = \exp(z_{\text{Yes}}) / (\exp(z_{\text{Yes}}) + \exp(z_{\text{No}}))$, summarized by the **Truth Selectivity Gap** $\Delta P = P(\text{Yes} \mid \text{True}) - P(\text{Yes} \mid \text{False})$.

#### Surgical QKV-Bias Ablation Conditions
Instead of zeroing biases from Layer 0 (which disrupts early feature extraction and collapses Qwen's logits to out-of-distribution noise), `SurgicalQKVBiasContext` reversibly zeroes biases **only in late layers** while leaving Layers 0–15 untouched:

| Condition | Model(s) | CAA Active | Zeroed Biases | Target Layers | Purpose |
|---|---|:---:|---|---|---|
| `Baseline` | Llama, Qwen | No | None | — | Reference geometry, attention routing, and stance |
| `CAA` | Llama, Qwen | Yes | None | — | Effect of Algorithm 1 decision-boundary steering |
| `Surgical_L22_QK` | Qwen | No | `q_proj`, `k_proj` | Layer 22 | Isolate the Layer-22 differential authority switch |
| `Surgical_L22_26_QK` | Qwen | No | `q_proj`, `k_proj` | Layers 22–26 | Ablate late routing window while keeping L16–21 and L27 intact |
| `Late_Zero_QK_L16_27` | Qwen | No | `q_proj`, `k_proj` | Layers 16–27 | Remove all post-steering QK routing biases (L0–15 intact) |
| `Late_Zero_QKV_L16_27` | Qwen | No | `q_proj`, `k_proj`, `v_proj` | Layers 16–27 | Separate QK routing effects from V-bias residual offsets |
| `CAA_Late_Zero_QK_L16_27` | Qwen | Yes | `q_proj`, `k_proj` | Layers 16–27 | Test whether CAA compensates for lost late-layer QK biases |

---

### How to Run

Execute `qkv_bias_late_layer_surgical.py` directly in a Kaggle T4 notebook cell (requires `HF_TOKEN` in Kaggle Secrets or environment variables for gated Llama-3.1 access):

```bash
python qkv_bias_late_layer_surgical.py
```

All outputs are written to `/kaggle/working/qkv_bias_step2_outputs/`:
- `step2_fig1_shift_cosine_late_ablation.png`
- `step2_fig2_diff_routing_and_sink.png`
- `step2_fig3_layer22_head_heatmap.png`
- `step2_fig4_finite_logit_stance.png`
- `step2_logit_stance_summary.csv`
- `step2_surgical_results.json`

---

### Plots & Mechanistic Interpretation

#### 1. Layerwise Authority-Pressure Shift Cosine (`step2_fig1_shift_cosine_late_ablation.png`)



- **What it plots**: Mean $\pm$ SD cosine similarity $\cos(\Delta \mathbf{h}_\ell^{\text{true}}, \Delta \mathbf{h}_\ell^{\text{false}})$ across pre-block hidden states (blocks $1 \dots L-1$) at the final prefill token for Llama (left) and Qwen (right). Lower cosine indicates stronger separation between true and false user assertions.
- **Key findings**:
  - **Baseline cross-architecture divergence**: Llama plateaus at **0.785** (block 31; truth-vs-framing norm ratio `0.347`), whereas Qwen drops sharply after block 18 to **0.707** (block 25) / **0.721** (block 27; norm ratio `0.443`).
  - **Causal proof for late QK biases**: Zeroing Qwen's QK biases in Layers 16–27 (`Late_Zero_QK_L16_27`, red) **completely eliminates the late-layer drop**, pushing Qwen's final shift cosine up to **0.906** (higher than Llama) and cutting its truth-vs-framing norm ratio in half (`0.443 -> 0.219`).
  - **Q/K routing dominates over V bias**: `Late_Zero_QKV_L16_27` (purple) closely tracks `Late_Zero_QK_L16_27` through block 24 (`0.904` vs. `0.905`).
  - **Upstream origin at Layer 19**: Ablating only Layer 22 (`0.757`) or Layers 22–26 (`0.777`) recovers only 20–30% of the drop, confirming that the primary representation split originates earlier in the L16–21 window (specifically Layer 19).

#### 2. Differential Authority Routing & Token-2 Sink Mass (`step2_fig2_diff_routing_and_sink.png`)



- **What it plots**:
  - *Left*: Layerwise differential attention placed by the final decision token onto the authority wrapper, $\Delta \text{Attn}_{\text{auth}} = \text{Attn}_{\text{auth}}(\text{False}) - \text{Attn}_{\text{auth}}(\text{True})$. Negative values mean the model actively **down-weights authority framing when the claim is false**.
  - *Right*: Mean attention mass placed on Qwen's primary attention sink — **Token 2 (`\n` immediately after `system`)** — across layers 0–27.
- **Key findings**:
  - **Layer-by-layer causal sink control (right)**: Qwen Baseline and CAA route **40–60%** of late-layer attention into Token 2. At the exact layer where QK biases are zeroed (L16 for `Late_Zero_QK_L16_27`; L22 for `Surgical_L22_QK`), Token-2 sink mass instantly collapses to **0.001–0.012**. Under `Surgical_L22_QK`, sink mass collapses at L22 (`0.012`) and immediately rebounds to **0.50–0.60** at L23–24 once biases are restored, proving the sink is maintained locally at each layer by QK biases.
  - **Two-stage authority-gating circuit (left)**: Qwen Baseline and CAA exhibit two sharp negative spikes at **Layer 19 (`-0.0155`)** and **Layer 22 (`-0.0126`)**, continuing negatively through Layer 27 (`-0.0044`). Zeroing QK biases in L16–27 removes **75–82%** of both spikes (`-0.0039` at L19, `-0.0023` at L22), whereas zeroing L22 alone leaves L19 intact and halves L22 (`-0.0070`).

#### 3. Head-Level Decomposition of Qwen's Routing Circuit (`step2_fig3_layer22_head_heatmap.png`)



- **What it plots**:
  - *Left*: Heatmap of per-head $\Delta \text{Attn}_{\text{auth}}$ (False − True) across all 28 query heads in Qwen Layers 18–26 (deep blue = head attends substantially less to false authority).
  - *Right*: Layer-22 scatter comparing each head's bias magnitude ($\|\mathbf{b}_Q\|_2 \times \|\mathbf{b}_K\|_2$) against its selective authority drop ($\Delta \text{Attn}_{\text{auth}}$), colored by that head's Token-2 sink mass.
- **Key findings**:
  - **Sparse specialist heads**: Differential authority gating is concentrated in a small set of heads rather than spread uniformly:
    - **Layer 19**: `H15 (-0.180)`, `H23 (-0.105)`, `H16 (-0.093)`, `H17 (-0.082)`
    - **Layer 22**: `H7 (-0.140)`, `H24 (-0.081)`, `H21 (-0.060)`, `H13 (-0.044)`, `H25 (-0.032)`
    - **Layers 23–26**: `L23 H11 (-0.098)`, `L24 H27 (-0.072)`, `L25 H25 (-0.069)`, `L26 H16 (-0.058)`
  - **Division of labor between sink heads and gating heads**: On the Layer-22 scatter, the strongest truth-gating heads (`H7`, `H24`, `H21`, `H25`) have **low Token-2 sink mass** (`0.014–0.288`) and moderate bias norms (`88–99`), whereas the bright-yellow high-sink heads (`0.86–0.91` sink mass, e.g., `H4`, `H15`, `H22`) cluster near $\Delta \text{Attn}_{\text{auth}} \approx 0$.
  - **Mechanistic implication**: High-bias sink heads do not directly perform truth verification; instead, they absorb background framing attention onto Token 2 so authority tokens do not flood the residual stream, giving the separate specialist heads headroom to gate false authority cleanly.

#### 4. Finite Behavioral Stance & Truth Selectivity (`step2_fig4_finite_logit_stance.png`)



- **What it plots**: Top panel shows mean Zero-Echo $P(\text{Yes} \mid \text{True})$ (green, higher is better) vs. $P(\text{Yes} \mid \text{False})$ (red, lower is better); bottom panel shows the Truth Selectivity Gap $P(\text{Yes} \mid \text{True}) - P(\text{Yes} \mid \text{False})$ across all conditions (`NaN_Count = 0` throughout).

| Model | Condition | $P(\text{Yes} \mid \text{Neutral})$ | $P(\text{Yes} \mid \text{True})$ | $P(\text{Yes} \mid \text{False})$ | Selectivity Gap (True − False) |
|---|---|---:|---:|---:|---:|
| LLAMA | `Baseline` | 0.369 | 0.394 | 0.254 | +0.140 |
| LLAMA | `CAA` | 0.267 | 0.239 | 0.137 | +0.102 |
| QWEN | `Baseline` | 0.285 | 0.292 | 0.061 | +0.231 |
| QWEN | `CAA` | 0.281 | 0.264 | 0.067 | +0.197 |
| QWEN | `Surgical_L22_QK` | 0.284 | 0.310 | 0.054 | +0.256 |
| QWEN | `Surgical_L22_26_QK` | 0.294 | 0.333 | 0.053 | +0.280 |
| QWEN | `Late_Zero_QK_L16_27` | 0.666 | 0.439 | 0.218 | +0.221 |
| QWEN | `Late_Zero_QKV_L16_27` | 0.679 | 0.510 | 0.322 | +0.188 |
| QWEN | `CAA_Late_Zero_QK_L16_27` | 0.497 | 0.311 | 0.147 | +0.164 |

- **Key findings**:
  - **Removing L16–27 QK/QKV biases induces Llama-like sycophancy in Qwen**: Under `Late_Zero_QK_L16_27`, Qwen's false-claim compliance more than triples (**0.061 -> 0.218**, approaching Llama Baseline's `0.254`), and under `Late_Zero_QKV_L16_27` it rises **5.3x** to **0.322**. Simultaneously, $P(\text{Yes} \mid \text{Neutral})$ jumps from `0.285 -> 0.666` while the selectivity gap stays near `+0.221`, showing that late QK biases act primarily as a **framing-sink and anti-acquiescence calibration mechanism** rather than the sole store of factual truth.
  - **CAA partially rescues bias-ablated Qwen**: Adding Algorithm 1 CAA on top of `Late_Zero_QK_L16_27` lowers false-claim compliance from **0.218 -> 0.147** and neutral acquiescence from **0.666 -> 0.497**, demonstrating that CAA and late QK biases act as complementary suppressors of sycophantic compliance.
  - **Why `Surgical_L22_26_QK` raises selectivity to `+0.280`**: Keeping Layers 16–21 (the primary Layer-19 gating circuit) and Layer 27 ($\|\mathbf{b}_K\|_2 = 920.5$) intact preserves low false-claim compliance (`0.053`, with Layer 27's sink rising from `0.21 -> 0.47` to compensate) while redirecting 2–3x more attention in Layers 23–26 onto the BoolQ context (`context_question` mass rises from ~0.06 to ~0.18–0.20), boosting $P(\text{Yes} \mid \text{True})$ from `0.292 -> 0.333`.

---

### Summary of Causal Conclusions & Scope

- **Causally established**: Late-layer (L16–27) attention QK biases in Qwen2.5-7B-Instruct maintain a layer-by-layer Token-2 attention sink (~40–60% mass) that prevents authority framing from saturating late residual streams, enables sparse specialist heads (L19 `H15/H23`, L22 `H7/H24`) to down-weight false authority, drives Qwen's late-layer shift-cosine separation (`0.721` vs. `0.906` when ablated), and keeps false-claim compliance low (`0.061` vs. `0.218–0.322` when ablated).
- **Important caveats**: Because ablating L16–27 shifts $P(\text{Yes} \mid \text{Neutral})$ upward alongside $P(\text{Yes} \mid \text{False})$ while preserving a positive selectivity gap (`+0.221`), late QK biases govern **framing absorption and compliance thresholding** rather than factual knowledge itself, and post-training/RLHF differences between Qwen and Llama remain an additional factor in baseline sycophancy.
