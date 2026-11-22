# Sycophancy_mitigation_llms
Beyond Sycophancy:Comparing Input Sanitization and Activation Steering in LLMs



## Supplementary Mechanistic Analysis: The Role of Attention QKV Biases (`bias_analysis/`)

In our main paper, we observe a pronounced cross-architecture divergence under matched-anchor authority pressure: **[Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)** sharply separates true-claim and false-claim activation shifts in its late layers, whereas **[Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/discussions/62)** processes both shifts along highly collinear directions through its final block.

This add-on analysis (reproducible via [bias\_analysis/basis-analysis.ipynb](https://www.google.com/search?q=bias_analysis/basis-analysis.ipynb)) investigates a concrete architectural candidate behind that gap: **learned attention QKV biases**. Qwen2.5-7B-Instruct includes additive biases on `q_proj`, `k_proj`, and `v_proj` across all 28 layers (28 Q / 4 KV heads), whereas Llama-3.1-8B-Instruct omits attention QKV biases entirely (32 Q / 8 KV heads). Using surgical, reversible late-layer bias ablations paired with our Algorithm 1 Subspace-Orthogonalized Decision-Boundary CAA and Zero-Echo Logit-Probability Judge, we test whether Qwen's QKV biases act as an **authority-framing sink** and drive its lower baseline sycophancy.

### Intervention Setup & Repository Artifacts

To isolate late-layer routing without disrupting early feature extraction, we leave Layers 0–15 untouched and reversibly zero Qwen's attention biases only across targeted late layers on 20 balanced, decision-adjacent BoolQ prompts (compact context + question first, authority wrapper and True/False assertion immediately preceding `"Answer strictly Yes or No."`). CAA steering vectors are extracted identically to Algorithm 1 in the paper ($N = 16$ **[TruthfulQA](https://huggingface.co/datasets/truthfulqa/truthful_qa)** pairs, $M = 8$ GSM8K math anchors, $l = \lfloor 0.55 L \rfloor$, $\alpha = 0.24$, final $S = 4$ prefill tokens). To prevent FP16 eager-attention overflow caused by Qwen Layer 27's large key-bias norm ($\|\mathbf{b}_K\|_2 = 920.5$), `QK^T` matmul and softmax are computed in FP32, yielding `0` NaNs across every condition.

| Condition | Model(s) | CAA | Zeroed Biases | Target Layers | Mechanistic Role Tested |
|---|---|:---:|---|---|---|
| `Baseline` | Llama, Qwen | No | None | — | Reference geometry, attention routing, and stance |
| `CAA` | Llama, Qwen | Yes | None | — | Effect of Algorithm 1 steering with biases intact |
| `Surgical_L22_QK` | Qwen | No | `q_proj`, `k_proj` | Layer 22 | Isolate the Layer-22 differential authority switch |
| `Surgical_L22_26_QK` | Qwen | No | `q_proj`, `k_proj` | Layers 22–26 | Ablate late routing window while keeping L16–21 and L27 intact |
| `Late_Zero_QK_L16_27` | Qwen | No | `q_proj`, `k_proj` | Layers 16–27 | Remove all post-steering QK routing biases (L0–15 intact) |
| `Late_Zero_QKV_L16_27` | Qwen | No | `q_proj`, `k_proj`, `v_proj` | Layers 16–27 | Separate QK routing from V-bias value offsets |
| `CAA_Late_Zero_QK_L16_27` | Qwen | Yes | `q_proj`, `k_proj` | Layers 16–27 | Test whether CAA compensates for ablated late QK biases |

All raw logs and summary metrics for this add-on are stored in [bias\_analysis/step2\_logit\_stance\_summary.csv](https://www.google.com/search?q=bias_analysis/step2_logit_stance_summary.csv) and [bias\_analysis/step2\_surgical\_results.json](https://www.google.com/search?q=bias_analysis/step2_surgical_results.json).

### Findings & Plot-by-Plot Breakdown

#### 1. Late-Layer QK Biases Causally Drive Qwen's Representation Separation



- **What the figure shows**: Layerwise cosine similarity $\cos(\Delta \mathbf{h}_\ell^{\text{true}}, \Delta \mathbf{h}_\ell^{\text{false}})$ (mean $\pm$ SD across 20 BoolQ prompts) at the final prefill token for Llama (left) and Qwen (right). Lower values indicate stronger representational separation between true and false user assertions.
- **Interpretation**: Under decision-adjacent prompts, Llama plateaus at **0.785** (block 31; truth-vs-framing norm ratio `0.347`), whereas Qwen drops sharply after block 18 to **0.707** (block 25) / **0.721** (block 27; norm ratio `0.443`). Zeroing Qwen's QK biases in Layers 16–27 (`Late_Zero_QK_L16_27`) **completely erases this late-layer drop**: shift cosine climbs to **0.906** (exceeding Llama) and the truth-vs-framing norm ratio is halved (`0.443 -> 0.219`). Adding V-bias ablation (`Late_Zero_QKV_L16_27`) tracks QK-only ablation closely through block 24 (`0.904` vs. `0.905`), confirming that **QK routing biases—not V biases—drive the representational split**. Because ablating Layer 22 alone (`0.757`) or Layers 22–26 (`0.777`) explains only 20–30% of the gap, the primary separation originates earlier in Layers 16–21.

#### 2. QK Biases Maintain Qwen's Token-2 Attention Sink & Two-Stage Routing Switch



- **What the figure shows**:
  - *Left*: Differential attention from the final decision token to the authority wrapper, $\Delta \text{Attn}_{\text{auth}} = \text{Attn}_{\text{auth}}(\text{False}) - \text{Attn}_{\text{auth}}(\text{True})$. Negative spikes indicate that the model actively down-weights authority framing when the claim is false.
  - *Right*: Attention mass routed to Qwen's primary learned sink—**Token 2 (`\n` immediately after `system`)**—across layers 0–27.
- **Interpretation**: In Baseline and CAA, Qwen routes **40–60%** of late-layer attention into Token 2 and exhibits two sharp false-claim authority drops at **Layer 19 (`-0.0155`)** and **Layer 22 (`-0.0126`)** (persisting negatively through Layer 27 at `-0.0044`). At the exact layer where QK biases are zeroed, Token-2 sink mass immediately collapses to **0.001–0.012**; under `Surgical_L22_QK`, it collapses at Layer 22 (`0.012`) and rebounds to **0.50–0.60** at Layers 23–24 once biases are restored, demonstrating layer-by-layer causal sink control. Removing L16–27 QK biases wipes out **75–82%** of both routing spikes (`-0.0039` at L19, `-0.0023` at L22).

#### 3. Division of Labor Between Sink Heads and Specialist Truth-Gating Heads



- **What the figure shows**:
  - *Left*: Per-head $\Delta \text{Attn}_{\text{auth}}$ (False − True) across all 28 query heads in Qwen Layers 18–26 (deep blue = head attends substantially less to false authority).
  - *Right*: Layer-22 scatter of per-head bias magnitude ($\|\mathbf{b}_Q\|_2 \times \|\mathbf{b}_K\|_2$) vs. $\Delta \text{Attn}_{\text{auth}}$, colored by each head's Token-2 sink mass.
- **Interpretation**: Selective authority suppression is driven by a sparse set of specialist heads rather than uniformly across the layer—primarily **L19 `H15 (-0.180)`, `H23 (-0.105)`, `H16 (-0.093)`, `H17 (-0.082)`** and **L22 `H7 (-0.140)`, `H24 (-0.081)`, `H21 (-0.060)`, `H13 (-0.044)`** (followed by `L23 H11`, `L24 H27`, `L25 H25`, and `L26 H16`). Crucially, the Layer-22 scatter reveals a **functional division of labor**: the strongest truth-gating heads (`H7`, `H24`, `H21`, `H25`) have **low Token-2 sink mass** (`0.014–0.288`) and moderate bias norms (`88–99`), whereas the high-sink heads (`0.86–0.91` sink mass, yellow) sit near $\Delta \text{Attn}_{\text{auth}} \approx 0$. Rather than acting as truth detectors themselves, high-bias sink heads drain background framing mass onto Token 2 so authority tokens do not flood the residual stream, preserving headroom for the separate specialist heads to gate false authority.

#### 4. Behavioral Stance: Removing Late QKV Biases Induces Llama-Like Sycophancy



- **What the figure shows**: Mean Zero-Echo $P(\text{Yes} \mid \text{True})$ (green, higher is better) vs. $P(\text{Yes} \mid \text{False})$ (red, lower is better) on top, and the **Truth Selectivity Gap** $P(\text{Yes} \mid \text{True}) - P(\text{Yes} \mid \text{False})$ on the bottom (`NaN_Count = 0` across all conditions).

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

- **Interpretation**:
  - **Late QKV biases suppress false-claim compliance**: Zeroing Layers 16–27 QK biases more than **triples** Qwen's false-claim compliance (`0.061 -> 0.218`, nearly matching Llama Baseline's `0.254`), and zeroing L16–27 QKV pushes it **5.3x** higher to `0.322`. Because $P(\text{Yes} \mid \text{Neutral})$ rises simultaneously (`0.285 -> 0.666`) while the selectivity gap stays high (`+0.221`), late QK biases act primarily as a **framing-sink and anti-acquiescence calibration mechanism** rather than the sole carrier of factual truth.
  - **Complementarity with CAA**: Applying Algorithm 1 CAA on top of `Late_Zero_QK_L16_27` partially rescues compliance (`P(Yes | False)` drops from `0.218 -> 0.147`; `P(Yes | Neutral)` drops from `0.666 -> 0.497`), showing that CAA steering and late QK biases act as complementary suppressors of sycophantic agreement.
  - **Why `Surgical_L22_26_QK` improves selectivity (`+0.231 -> +0.280`)**: Preserving Layers 16–21 (the primary Layer-19 gating circuit) and Layer 27 ($\|\mathbf{b}_K\|_2 = 920.5$, whose sink rises from `0.21 -> 0.47` to compensate) keeps false-claim compliance pinned at `0.053`, while removing the L22–26 sink redirects 2–3x more attention onto the BoolQ context (`context_question` mass increases from ~0.06 to ~0.18–0.20), lifting $P(\text{Yes} \mid \text{True})$ from `0.292 -> 0.333`.

### Takeaway for the Paper

This supplementary intervention confirms that Qwen2.5-7B-Instruct's stronger late-layer separation and lower baseline sycophancy are not merely an artifact of prompt phrasing: **late-layer (L16–27) attention QK biases causally sustain a Token-2 framing sink and anti-acquiescence threshold** that protects sparse specialist heads (L19/L22) from authority-wrapper saturation. At the same time, because truth selectivity (`+0.221`) persists even after L16–27 QK ablation, QK biases explain *compliance gating and framing absorption* rather than factual knowledge alone—and post-training differences between Qwen and Llama remain an additional contributing factor.

