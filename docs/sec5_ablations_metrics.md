# Section 5: Ablations and New Metrics

## 5.1 Ablation Study Design

Each ablation disables exactly one CRF mechanism, holding all other hyperparameters constant. The goal is to isolate which mechanisms provide signal vs. which are irrelevant or harmful. All ablations are run on TinyStories (PPL) and GSM8K (accuracy) as the two representative benchmarks.

**Naming convention:** `CRF-Full` is the complete model. Each ablation is named `CRF-No{X}`.

---

## 5.2 Ablation A — No Split/Death (Fixed Population)

**Implementation:** Set `ε_split = ∞` (never split) and `ε_die = -∞` (never die). Population is fixed at $N_0 = N_{\text{init}}$ for all steps.

**What this tests:** Whether the lifecycle dynamics (birth/death) add value beyond having a fixed population of cells running for $S$ steps.

**Expected result:** Performance degrades on hard examples (long GSM8K chains) because compute cannot concentrate via proliferation. Performance may be similar on easy examples. If `CRF-NoSplit` matches `CRF-Full` on TinyStories, the lifecycle is overhead there.

**Metric to examine:** Std deviation of $N_t$ across validation examples should collapse to 0 in this ablation vs. non-zero in `CRF-Full`.

---

## 5.3 Ablation B — No Merge

**Implementation:** Disable the cosine-similarity merge condition entirely ($\tau_{\text{merge}} = \infty$, i.e., no pair ever merges).

**What this tests:** Whether cell consolidation is necessary. Without merging, redundant near-duplicate cells accumulate, potentially wasting compute on redundant computation.

**Expected result:** Cell count grows faster (no consolidation), pushing against $N_{\max}$ ceiling sooner. Communication cost increases. If merge is helpful, `CRF-NoMerge` should show higher inference FLOPs for the same task performance. Conversely, if merge is harmful (destroys diversity), `CRF-NoMerge` outperforms `CRF-Full` — this would be a surprising finding.

---

## 5.4 Ablation C — No Routing (Uniform Aggregation)

**Implementation:** Replace the learned routing gate $w_{ij} = \sigma(W_r [\mathbf{s}_i \| \mathbf{s}_j])$ with uniform weights $w_{ij} = 1/k$ for all neighbors. This removes the 2-layer routing head $W_r$ from the parameter count (adjust baseline accordingly).

**What this tests:** Whether learned per-neighbor gating is necessary, or whether a simple mean-aggregation suffices.

**Expected result:** If routing is critical, `CRF-NoRouting` degrades significantly on tasks requiring selective attention (e.g., ARC-AGI). If routing is redundant, this is a simplification with no cost.

---

## 5.5 Ablation D — No Energy (Constant Energy)

**Implementation:** Fix $\varepsilon_i = 1.0$ for all cells at all times. The energy gate $\delta_i$ is still computed (it may influence state via backprop) but does not feed back into lifecycle decisions. Split and death conditions become:
- Split: $N_t < N_{\max}$ and random Bernoulli(0.3) at age mod 5 == 0.
- Death: never (since $\varepsilon_i = 1.0 \not< \varepsilon_{\text{die}}$).

**What this tests:** Whether the energy signal is the critical mechanism for adaptive computation, or whether stochastic lifecycle already provides sufficient adaptivity.

**Expected result:** Without energy-guided lifecycle, proliferation becomes random rather than problem-driven. This should hurt performance on reasoning tasks where the right cells need to dominate.

---

## 5.6 Ablation E — No Messaging (Independent Cells)

**Implementation:** Set $\mathbf{m}_i = \mathbf{0}$ for all cells at all steps. Each cell updates based only on its own state (no neighbor communication).

**What this tests:** Whether inter-cell communication is necessary. This reduces the CRF to a parallel ensemble of independent MLPs.

**Expected result:** This should be the most damaging ablation. Without messaging, cells cannot propagate information across the sequence, and reasoning across token positions is impossible. PPL should approach that of a single MLP (no sequential processing). This ablation serves as a lower-bound sanity check.

---

## 5.7 Ablation F — No Spatial Penalty (Pure Cosine Routing)

**Implementation:** Set $\lambda = 0$ in the similarity score:

$$A_{ij} = \frac{\mathbf{s}_i^\top \mathbf{s}_j}{\|\mathbf{s}_i\|\|\mathbf{s}_j\|}$$

Cells are routed purely by semantic similarity, with no spatial locality bias.

**What this tests:** Whether the 2D spatial layout and locality inductive bias matter for language tasks vs. spatial tasks (ARC-AGI).

**Expected hypothesis:** On ARC-AGI, spatial penalty helps significantly (grids have inherent locality). On TinyStories and GSM8K, the difference may be minimal. A significant gap here would motivate learned spatial layouts as a future direction.

---

## 5.8 Ablation Summary Table (to be filled after experiments)

| Ablation | TinyStories PPL | GSM8K Acc | Avg $N_t$ | Inference FLOPs |
|----------|----------------|-----------|-----------|-----------------|
| CRF-Full | — | — | — | — |
| CRF-NoSplit/Death | — | — | $N_0$ (fixed) | — |
| CRF-NoMerge | — | — | — | — |
| CRF-NoRouting | — | — | — | — |
| CRF-NoEnergy | — | — | — | — |
| CRF-NoMessaging | — | — | — | — |
| CRF-NoSpatial | — | — | — | — |
| Transformer (matched) | — | — | N/A | — |

---

## 5.9 New Metrics

Beyond standard accuracy and perplexity, the following metrics capture properties unique to CRF and should be reported in all experiments.

### 5.9.1 Active Cell Count ($\bar{N}$)

$$\bar{N} = \frac{1}{S} \sum_{t=1}^{S} N_t$$

Report as mean ± std across validation examples. Measures average population size and its variance across inputs (a signature of adaptive computation).

Also report: $N_{\min}$, $N_{\max}$, and the correlation of $N_t$ with per-example difficulty (measured by loss on that example).

### 5.9.2 Communication Cost ($\mathcal{K}$)

$$\mathcal{K} = \sum_{t=1}^{S} |E^{(t)}| \cdot d = \sum_{t=1}^{S} k \cdot N_t \cdot d$$

This is the total number of floating-point values transmitted as messages across all steps. For a Transformer, the analogous metric is $L \cdot T^2 \cdot d_{\text{head}}$ (attention weights times value dimension). Report $\mathcal{K}$ normalized by parameter count to get communication efficiency.

### 5.9.3 Energy Efficiency ($\eta_\varepsilon$)

$$\eta_\varepsilon = \frac{\text{task performance}}{\mathcal{F}_{\text{CRF}}}$$

where $\mathcal{F}_{\text{CRF}}$ is the total FLOPs for a given example. A model that achieves the same performance with fewer FLOPs has higher energy efficiency. Plot $\eta_\varepsilon$ vs. example difficulty to show where CRF is most efficient.

### 5.9.4 Cell Specialization ($\mathcal{S}_{\text{spec}}$)

Measure whether different cells specialize on different parts of the input. After a forward pass, compute the pairwise cosine distance matrix of anchor cell states:

$$D_{ij} = 1 - \frac{\mathbf{s}_i^\top \mathbf{s}_j}{\|\mathbf{s}_i\|\|\mathbf{s}_j\|}$$

Specialization is the mean off-diagonal distance:

$$\mathcal{S}_{\text{spec}} = \frac{2}{T(T-1)} \sum_{i < j} D_{ij}$$

High $\mathcal{S}_{\text{spec}}$ means cells have differentiated representations; low means they collapsed to the same state. Compare $\mathcal{S}_{\text{spec}}$ for CRF-Full vs. CRF-NoMessaging (should be higher with messaging, confirming that communication drives specialization).

### 5.9.5 Graph Diameter ($\text{diam}(G^{(t)})$)

The **diameter** of the communication graph at step $t$ is the length of the longest shortest path between any two cells:

$$\text{diam}(G^{(t)}) = \max_{i,j} \text{dist}_{G^{(t)}}(i, j)$$

Use BFS on the directed $k$-NN graph. Report mean diameter across steps and examples. For random $k$-regular graphs on $N$ nodes, expected diameter is $O(\log N / \log k)$ — if CRF's graph is close to this, information can propagate globally in $O(\log N)$ steps, providing a theoretical lower bound on the number of steps $S$ needed for global coherence.

### 5.9.6 Lifecycle Event Rate

Per example, count:
- $n_{\text{split}}$: total split events
- $n_{\text{death}}$: total death events
- $n_{\text{merge}}$: total merge events
- $r_{\text{turnover}} = (n_{\text{split}} + n_{\text{death}}) / (S \cdot N_0)$: fraction of cells replaced per step

High turnover indicates active exploration; low turnover indicates a stable, converged population. Correlate with task performance to understand whether instability helps or hurts.

### 5.9.7 Scaling Laws

Fit a power law to PPL vs. compute for CRF and Transformer separately:

$$\text{PPL}(C) = a \cdot C^{-b} + \text{PPL}_\infty$$

The exponent $b$ characterizes how efficiently the architecture uses additional compute. If $b_{\text{CRF}} > b_{\text{Transformer}}$, CRF scales better. Report the compute-optimal frontier (Hoffmann et al. 2022 "Chinchilla" style) for CRF by sweeping $N_{\max}$ and $S$ at fixed $C$.
