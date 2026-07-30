# Section 4: Benchmark Plan

## 4.1 Experimental Philosophy

All comparisons are **doubly matched**: same parameter count **and** same total training FLOPs. A CRF model and its Transformer baseline are trained from scratch under identical conditions (dataset, tokenizer, optimizer, learning rate schedule, batch size, and hardware). This rules out the common confound where a new architecture wins purely due to scale.

**FLOPs-matched baseline definition.** Given a CRF with $N_{\max}$ cells, $S$ steps, $d$ model dimension, we compute:

$$F_{\text{CRF}} = S \cdot (2 N_{\max}^2 d + 2 N_{\max} d\, d_h)$$

Then choose Transformer depth $L$ and sequence length $T$ such that:

$$F_{\text{Transformer}} = L \cdot (4 T d^2 + 2 T^2 d) \approx F_{\text{CRF}}$$

Parameter counts are matched by adjusting $d$ and $d_h$ of the CRF (or $L$ and $d$ of the Transformer) until $|\theta_{\text{CRF}}| \approx |\theta_{\text{Transformer}}|$ within 5%.

---

## 4.2 Benchmark 1 — TinyStories (Language Modeling)

**Purpose:** Measure basic language modeling quality on a controlled, small-vocabulary corpus.

**Dataset:** TinyStories (Eldan & Li 2023) — ~2M short stories written in simple English, vocab ~10K tokens.

**Metric:** Perplexity (PPL) on the held-out validation set. Lower is better.

**Model sizes:** Two scales.

| Config | $d$ | $N_{\max}$ | $S$ | $k$ | Params |
|--------|-----|-----------|-----|-----|--------|
| CRF-Small | 128 | 256 | 8 | 4 | ~2M |
| CRF-Medium | 256 | 512 | 12 | 6 | ~15M |

Matched Transformer baselines: GPT-Small-2M (3 layers, $d=256$) and GPT-Medium-15M (6 layers, $d=512$).

**Training:** AdamW, $\beta_1=0.9$, $\beta_2=0.95$, weight decay $10^{-1}$. Cosine LR decay from $3 \times 10^{-4}$ to $3 \times 10^{-5}$ over 50K steps. Batch size 256, context length 512.

**Expected output:** A table of PPL ± standard deviation over 3 seeds.

**Hypothesis:** CRF achieves within 10% of Transformer PPL at matched compute, with lower PPL on examples rated "complex" by story length or parse depth (adaptive compute advantage).

---

## 4.3 Benchmark 2 — HumanEval (Code Generation)

**Purpose:** Measure structured, rule-constrained generation requiring precise syntax and logic.

**Dataset:** HumanEval (Chen et al. 2021) — 164 Python programming problems with unit tests.

**Metric:** pass@1 and pass@10 (fraction of problems with at least 1 correct solution in 1 or 10 samples).

**Setup:** Fine-tune pre-trained CRF-Medium and GPT-Medium-15M on The Stack Python subset (~5GB), then evaluate on HumanEval with greedy decoding (pass@1) and temperature=0.8, 10 samples (pass@10).

**Protocol:**
1. Pre-train both models on Python code for 100K steps.
2. No instruction tuning — measure raw next-token prediction quality on function completion.
3. Report pass@k with 95% bootstrap confidence intervals.

**Hypothesis:** Code generation requires strong local consistency (matching brackets, variable scopes). CRF's spatial locality bias (position penalty in routing, §1.3) may help maintain local coherence, potentially improving pass@1 over the matched Transformer.

---

## 4.4 Benchmark 3 — GSM8K (Multi-Step Reasoning)

**Purpose:** Test multi-step arithmetic reasoning, where intermediate states must be maintained across many reasoning steps.

**Dataset:** GSM8K (Cobbe et al. 2021) — 8.5K grade-school math word problems with chain-of-thought solutions.

**Metric:** Exact-match accuracy on the final numeric answer. Report separately for problems requiring ≤4 steps and >4 steps.

**Setup:** Fine-tune on the GSM8K training split (7.5K examples) with full chain-of-thought targets. Evaluate on the test split (1K examples). Greedy decoding.

**Key hypothesis for CRF:** Problems with more reasoning steps should favor CRF over Transformer. CRF's cell population can redistribute energy toward "reasoning cells" as the chain-of-thought grows, while a Transformer with fixed depth has no such mechanism. We will plot accuracy vs. number-of-steps to test this directly.

**Failure mode to watch:** CRF may underperform on simple 1–2 step problems where the overhead of cell lifecycle does not pay off. This would be consistent with the adaptive computation hypothesis.

---

## 4.5 Benchmark 4 — ARC-AGI (Abstract Reasoning / Generalization)

**Purpose:** Test compositional generalization on visual grid puzzles that require discovering abstract transformation rules.

**Dataset:** ARC-AGI (Chollet 2019) — 400 training tasks, 400 evaluation tasks, each with 2–5 demonstration pairs plus one test input.

**Encoding:** Use `encode_arc_grid` (already implemented in `crf_sim.py`): flatten each grid to a vector, normalize by dividing by 10, pad/truncate to `max_dim=64` floats. Each (input_grid, output_grid) pair becomes a token-pair embedding.

**Metric:** Task-level accuracy: a task is solved iff the predicted output grid exactly matches the ground truth.

**Full model setup:** Encode all demonstration pairs as a context sequence. Feed into CRF-Medium followed by a grid reconstruction head (linear projection to grid vocab size 10). Train on the 400 training tasks with leave-one-out cross-task evaluation (few-shot adaptation).

**Why CRF may help on ARC:** ARC tasks require recognizing spatial patterns and composing transformations. CRF's explicit 2D positional layout for cells and locality-biased routing mirror the spatial structure of ARC grids more naturally than a position-agnostic Transformer.

**Simulator baseline:** Also report results using `CRFSimulator` (the bytecode DSL version in `crf_sim.py`) as an interpretability reference — though it is not expected to solve ARC, it provides a legible trace of which cells vote and with what confidence.

---

## 4.6 Experimental Controls and Rigor Requirements

For each benchmark, the following must be reported:

1. **Parameter count** of CRF and baseline (within 5%).
2. **Training FLOPs** (computed, not estimated) — use `torch.profiler` or manual FLOP counting.
3. **Inference FLOPs per example** — for CRF, report mean and standard deviation across examples (to characterize adaptive compute variance).
4. **3 random seeds** for all metrics; report mean ± std.
5. **Training curves** — validation loss vs. step for both models.
6. **Cell count over time** — for CRF, plot $N_t$ averaged over validation examples as a function of step $t$ (shows lifecycle activity).
7. **Failure modes** — examples where CRF fails but Transformer succeeds and vice versa, with qualitative analysis.

---

## 4.7 Infrastructure Requirements

- Hardware: Single A100 80GB (or equivalent) for CRF-Small and CRF-Medium.
- Estimated training time: CRF is slower per step than Transformer due to the Python-level cell loop. Before benchmarking, replace the Python loop in `CellularReasoningFabric.forward` with a batched CUDA kernel or at minimum a vectorized PyTorch operation over all cells simultaneously.
- Recommended optimization: convert `CognitiveCell.step` to operate on batched state tensors rather than a Python list of `nn.Module` objects. This is the single highest-leverage engineering change before serious benchmarking.
