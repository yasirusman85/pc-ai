# Cellular Reasoning Fabric (CRF): A Technical Report

> **Abstract.** We introduce the Cellular Reasoning Fabric (CRF), an architecture for adaptive, decentralized computation over token sequences. CRF replaces the fixed-depth, dense-attention Transformer with a dynamic population of cognitive cells that communicate over a sparse learned graph, split and die according to an energy signal, and merge when their representations converge. We formalize the update equations, lifecycle rules, and routing algorithm; prove three theoretical properties (bounded communication, adaptive computation, and convergence under contraction); position CRF against five related architectures; specify a rigorous benchmark plan across TinyStories, HumanEval, GSM8K, and ARC-AGI using FLOPs-matched Transformer baselines; and define seven ablations and seven new metrics specific to dynamic-population architectures.

---

## Table of Contents

1. [Formalization](#1-formalization)
   - 1.1 Notation and State Space
   - 1.2 Cell Program
   - 1.3 Sparse Routing and Message Aggregation
   - 1.4 Energy Dynamics
   - 1.5 Cell Lifecycle Rules
   - 1.6 Full Forward Pass
   - 1.7 Computational Complexity
2. [Novelty](#2-novelty)
   - 2.1 vs. Transformer
   - 2.2 vs. Graph Neural Networks
   - 2.3 vs. Mixture of Experts
   - 2.4 vs. Neural Cellular Automata
   - 2.5 vs. Multi-Agent Systems
   - 2.6 Summary Table
3. [Theoretical Properties](#3-theoretical-properties)
   - 3.1 Theorem 1: Bounded Communication
   - 3.2 Theorem 2: Adaptive Computation
   - 3.3 Theorem 3: Convergence under Contraction
   - 3.4 Expressivity Bound
   - 3.5 Energy Conservation Lemma
4. [Benchmark Plan](#4-benchmark-plan)
   - 4.1 Experimental Philosophy
   - 4.2 TinyStories
   - 4.3 HumanEval
   - 4.4 GSM8K
   - 4.5 ARC-AGI
   - 4.6 Controls and Rigor Requirements
   - 4.7 Infrastructure
5. [Ablations and New Metrics](#5-ablations-and-new-metrics)
   - 5.1–5.7 Ablations
   - 5.8 Ablation Summary Table
   - 5.9 New Metrics

---

# Section 1: Formalization

## 1.1 Notation and State Space

Let $\mathcal{C}^{(t)} = \{c_1^{(t)}, c_2^{(t)}, \ldots, c_{N_t}^{(t)}\}$ denote the set of **cognitive cells** alive at discrete time step $t$, where $N_t = |\mathcal{C}^{(t)}|$ is dynamic. Each cell $c_i$ is a tuple:

$$c_i = \bigl(\mathbf{s}_i \in \mathbb{R}^d,\; \mathbf{p}_i \in \mathbb{R}^2,\; \varepsilon_i \in \mathbb{R}_{\geq 0},\; \theta_i,\; a_i \in \mathbb{N}\bigr)$$

where:
- $\mathbf{s}_i$ — state vector (the cell's latent representation)
- $\mathbf{p}_i$ — spatial position in a 2D layout grid
- $\varepsilon_i$ — energy scalar, bounded to $[0, 5]$
- $\theta_i$ — program parameters (weights of a `CellProgram` MLP)
- $a_i$ — age (steps elapsed since birth)

**Anchor cells** are a fixed subset $\mathcal{A} \subseteq \mathcal{C}^{(0)}$ seeded directly from input token embeddings; they are never removed.

---

## 1.2 Cell Program

Each cell's local computation is governed by a learned **CellProgram** $f_{\theta_i}$, a two-layer MLP with residual connection and LayerNorm:

$$\mathbf{x}_i = [\mathbf{s}_i \;\|\; \mathbf{m}_i] \in \mathbb{R}^{2d}$$

$$\tilde{\mathbf{s}}_i = \mathbf{s}_i + W_2\,\text{ReLU}(W_1 \mathbf{x}_i + \mathbf{b}_1) + \mathbf{b}_2 \in \mathbb{R}^d$$

$$\mathbf{s}_i^{(t+1)} = \text{LayerNorm}(\tilde{\mathbf{s}}_i)$$

$$\mathbf{o}_i = W_{\text{msg}}\,\mathbf{s}_i^{(t+1)} \in \mathbb{R}^d \quad \text{(outgoing message)}$$

$$\delta_i = \sigma(w_\varepsilon^\top \mathbf{s}_i^{(t+1)} + b_\varepsilon) \in (0,1) \quad \text{(energy gate)}$$

where $[\;\|\;]$ denotes vector concatenation, $\sigma$ is the sigmoid function, and $\mathbf{m}_i$ is the aggregated incoming message defined in §1.3.

All parameters $W_1 \in \mathbb{R}^{d_h \times 2d}$, $W_2 \in \mathbb{R}^{d \times d_h}$, $W_{\text{msg}} \in \mathbb{R}^{d \times d}$, $w_\varepsilon \in \mathbb{R}^d$ are specific to cell $c_i$ (each cell has its own copy, optionally shared or mutated at birth).

---

## 1.3 Sparse Routing and Message Aggregation

At each step, a **SparseFabric** constructs a directed $k$-nearest-neighbor graph over the current cell population and computes aggregated messages.

**Step 1 — Similarity score.** For each pair $(i, j)$ compute:

$$A_{ij} = \frac{\mathbf{s}_i^\top \mathbf{s}_j}{\|\mathbf{s}_i\|\|\mathbf{s}_j\|} - \lambda \cdot \|\mathbf{p}_i - \mathbf{p}_j\|_2$$

where $\lambda = 0.05$ is the spatial penalty. Self-connections are masked: $A_{ii} = -\infty$.

**Step 2 — Neighborhood selection.** Each cell $i$ selects its top-$k$ neighbors:

$$\mathcal{N}(i) = \text{top-}k_j\bigl(A_{ij}\bigr), \quad k = \min(k_{\max},\; N_t - 1)$$

**Step 3 — Learned routing weight.** For each neighbor $j \in \mathcal{N}(i)$, compute a scalar gate:

$$w_{ij} = \sigma\!\bigl(W_r\,[\mathbf{s}_i \;\|\; \mathbf{s}_j] + b_r\bigr) \in (0,1)$$

where $W_r \in \mathbb{R}^{1 \times 2d}$ is a shared routing head.

**Step 4 — Weighted message aggregation:**

$$\mathbf{m}_i = \frac{\sum_{j \in \mathcal{N}(i)} w_{ij}\,\mathbf{s}_j}{\sum_{j \in \mathcal{N}(i)} w_{ij} + \epsilon}$$

If $\sum_j w_{ij} = 0$, fall back to $\mathbf{m}_i = \mathbf{s}_i$ (self-message).

**Full routing algorithm (pseudocode):**

```
Algorithm: SparseFabric.forward(S, P)
  Input:  S ∈ R^{N×d}  (stacked cell states)
          P ∈ R^{N×2}  (stacked positions)
  Output: M ∈ R^{N×d}  (aggregated messages)

  Ŝ ← L2-normalize(S, dim=d)
  A ← Ŝ Ŝᵀ  − λ · cdist(P, P)          // N×N similarity
  A ← fill_diagonal(A, −∞)
  k ← min(k_max, N−1)
  IDX ← top-k indices per row of A       // N×k
  M ← zeros(N, d)
  for i = 1..N:
      W ← [σ(Wᵣ [sᵢ ‖ sⱼ]) for j in IDX[i]]   // k scalars
      M[i] ← Σⱼ W[j]·S[IDX[i,j]] / (Σⱼ W[j] + ε)
  return M
```

---

## 1.4 Energy Dynamics

Energy tracks how "productive" a cell is. After each step:

$$\varepsilon_i^{(t+1)} = \text{clamp}\!\bigl(0.95\,\varepsilon_i^{(t)} + 0.05\,\delta_i^{(t+1)},\; 0,\; 5\bigr)$$

The exponential moving average with $\alpha_{\text{decay}} = 0.95$ and $\alpha_{\text{gate}} = 0.05$ means energy lags behind the cell program's output. A cell that consistently produces $\delta \approx 1$ stabilizes near $\varepsilon^* = 1.0$; a cell producing $\delta \approx 0$ decays geometrically toward $0$.

**Fixed points of the energy ODE analog:**

$$\varepsilon^* = \frac{\alpha_{\text{gate}}}{\alpha_{\text{decay}} + \alpha_{\text{gate}} - 1} \cdot \delta^* \quad \Rightarrow \quad \varepsilon^* = \frac{0.05}{0.00} \;\text{(unstable unless bounded)}$$

Since $\alpha_{\text{decay}} + \alpha_{\text{gate}} = 1.0$ exactly, the system is marginally stable: $\varepsilon^{(t)} \to \delta^*$ as $t \to \infty$ when $\delta$ is constant, with exponential convergence rate $|\lambda| = 0.95$.

---

## 1.5 Cell Lifecycle Rules

The lifecycle is governed by three events applied every 3 steps (for the neural CRF) or every step (for the simulator):

### 1.5.1 Split (Birth)

**Condition:** $\lnot \text{is\_anchor}(c_i) \;\land\; \varepsilon_i > \varepsilon_{\text{split}} \;\land\; N_t < N_{\max}$

where $\varepsilon_{\text{split}} = 1.5$ in `crf.py`.

**Action:** Create child cell $c_j$:

$$\mathbf{s}_j = \mathbf{s}_i + \boldsymbol{\eta}_s, \quad \boldsymbol{\eta}_s \sim \mathcal{N}(0, 0.01^2 I_d)$$
$$\mathbf{p}_j = \mathbf{p}_i + \text{clamp}(\boldsymbol{\eta}_p,\, -0.5,\, 0.5), \quad \boldsymbol{\eta}_p \sim \mathcal{N}(0, 0.5^2 I_2)$$
$$\theta_j = \theta_i + \boldsymbol{\xi}, \quad \boldsymbol{\xi}_l \sim \mathcal{N}(0,\, (0.05\,\sigma(\theta_i^{(l)}) + 10^{-6})^2) \;\forall l$$
$$\varepsilon_j = 0.4\,\varepsilon_i, \qquad \varepsilon_i \leftarrow 0.4\,\varepsilon_i$$

Energy is split 40/40 between parent and child (60% is lost, preventing runaway proliferation).

### 1.5.2 Death (Pruning)

**Condition:** $\lnot \text{is\_anchor}(c_i) \;\land\; \varepsilon_i < \varepsilon_{\text{die}}$

where $\varepsilon_{\text{die}} = 0.01$.

**Action:** Remove $c_i$ from $\mathcal{C}^{(t)}$.

### 1.5.3 Merge (Consolidation)

**Condition (neural CRF):** Applied every 5 steps to up to $\lfloor N_t / 4 \rfloor$ randomly sampled pairs $(a,b)$:

$$\text{cos\_sim}(\mathbf{s}_a, \mathbf{s}_b) > \tau_{\text{merge}}, \quad \tau_{\text{merge}} = 0.95$$

and not both anchors.

**Action:**

$$\mathbf{s}_{\min(a,b)} \leftarrow \frac{\mathbf{s}_a + \mathbf{s}_b}{2}$$
$$\mathbf{p}_{\min(a,b)} \leftarrow \frac{\mathbf{p}_a + \mathbf{p}_b}{2}$$
$$\varepsilon_{\min(a,b)} \leftarrow \varepsilon_a + \varepsilon_b$$

Remove $c_{\max(a,b)}$ from $\mathcal{C}^{(t)}$.

**Condition (simulator):** Applied every 5 global steps using $L_1$ state distance $< 1.0$ as the criterion, on the top $\lfloor N_t / 20 \rfloor$ closest pairs.

---

## 1.6 Full Forward Pass

The complete CRF forward pass over $S$ steps is:

```
Algorithm: CellularReasoningFabric.forward(X, S)
  Input:  X ∈ R^{1×T×d}   (token embeddings for one sequence)
  Output: Y ∈ R^{1×T×d}   (output states at anchor positions)

  H ← W_in · X[0]                         // R^{T×d}, shared input projection
  Initialize C = {c₁, ..., c_{N_0}}:
    ∀ i < T:  sᵢ = Hᵢ,  pᵢ = grid(i),  anchor=True,  εᵢ = 1
    ∀ i ≥ T:  sᵢ = H_{i mod T} + η,  pᵢ = grid(i mod T) + η',  anchor=False

  for t = 1..S:
    S_mat ← stack({sᵢ : cᵢ ∈ C})           // N_t × d
    P_mat ← stack({pᵢ : cᵢ ∈ C})           // N_t × 2
    M     ← SparseFabric(S_mat, P_mat)      // N_t × d

    for each cᵢ ∈ C:
      sᵢ, oᵢ, δᵢ ← CellProgram(sᵢ, M[i])
      εᵢ ← clamp(0.95 εᵢ + 0.05 δᵢ, 0, 5)

    if t mod 3 == 0:   apply Split and Death  // §1.5.1, §1.5.2
    if t mod 5 == 0:   apply Merge            // §1.5.3

  A ← stack({sᵢ : cᵢ ∈ C, is_anchor(cᵢ)})  // ≤T × d
  pad A to T × d if needed
  return W_out · A  (unsqueeze batch dim)
```

---

## 1.7 Computational Complexity

Let $N = N_t$ be the cell count at step $t$, $d$ the state dimension, $d_h$ the hidden width of `CellProgram`, $k$ the neighborhood size, and $S$ the number of steps.

| Operation | Per-step cost | Dominant term |
|-----------|--------------|---------------|
| Cosine similarity matrix | $O(N^2 d)$ | $N^2 d$ |
| Top-$k$ selection | $O(N^2)$ | $N^2$ |
| Routing weight computation | $O(N k d)$ | $Nkd$ |
| Message aggregation | $O(N k d)$ | $Nkd$ |
| CellProgram (all cells) | $O(N d d_h)$ | $N d d_h$ |
| Split/Merge (amortized) | $O(N)$ | $N$ |

**Total per-step:** $O(N^2 d + N d d_h)$

**Total over $S$ steps:** $O(S(N^2 d + N d d_h))$

Since $N_t$ varies due to lifecycle events, we can bound it as $N_t \leq N_{\max}$ (hard cap enforced in code). Under this bound:

$$\text{CRF total FLOPs} = O(S \cdot N_{\max}^2 \cdot d)$$

**Comparison to Transformer.** A standard Transformer with sequence length $T$, $L$ layers, $H$ heads, and model dimension $d$ costs:

$$\text{Transformer total FLOPs} = O(L T^2 d)$$

For CRF to match Transformer cost: $S N_{\max}^2 \approx L T^2$. With $N_{\max} = T$ (no extra cells), $S = L$, the costs are equal. CRF offers sub-quadratic scaling in $T$ when $k \ll N$ and the similarity matrix is not recomputed densely — a direction for future sparse approximations (§4.3).

**Key difference:** CRF's $N_t$ is input-adaptive. Hard inputs can trigger more splits, allocating more compute where needed. Transformer FLOPs are fixed at graph construction time.

---

# Section 2: Novelty — Positioning CRF Among Related Architectures

## 2.1 Transformer

The standard Transformer (Vaswani et al., 2017) processes a fixed-length sequence $\mathbf{X} \in \mathbb{R}^{T \times d}$ through $L$ identical layers. Each layer applies **dense, global self-attention**:

$$\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

Every token attends to every other token at every layer. The graph is **static** (fixed $T \times T$ attention matrix), **homogeneous** (same operation everywhere), and **depth-fixed** ($L$ layers always execute).

**CRF differences:**

| Dimension | Transformer | CRF |
|-----------|-------------|-----|
| Graph topology | Dense, fixed $T \times T$ | Sparse $k$-NN, dynamically rebuilt each step |
| Node count | Fixed ($T$ tokens) | Dynamic ($N_t$ varies via split/death/merge) |
| Computation depth | Fixed $L$ layers | Adaptive $S$ steps with variable $N_t$ |
| Position encoding | Injected once (sinusoidal or RoPE) | Explicit 2D Euclidean positions on cells |
| Receptive field per step | Global ($O(T)$) | Local ($k$ neighbors) |
| Cell heterogeneity | All layers share the same weight class | Each cell can hold distinct program weights |
| Routing | Softmax over all keys | Learned sigmoid gate per neighbor pair |

The critical distinction is **adaptive topology**: the Transformer's computation graph is fully determined before training; CRF's graph is a function of the input at inference time.

---

## 2.2 Graph Neural Networks (GNNs)

GNNs (Kipf & Welling 2016; Gilmer et al. 2017) operate on a graph $G = (V, E)$ and propagate messages along edges:

$$\mathbf{h}_i^{(l+1)} = \text{UPDATE}\!\left(\mathbf{h}_i^{(l)},\; \text{AGGREGATE}\!\left(\{\mathbf{h}_j^{(l)} : j \in \mathcal{N}(i)\}\right)\right)$$

**CRF differences:**

1. **Graph is input-constructed, not pre-given.** GNNs receive a fixed graph as input; CRF builds its graph from learned cell states at each step. The topology is a latent variable that emerges from the dynamics.

2. **Node lifecycle.** GNNs have a fixed node set throughout inference. CRF nodes are created (split) and destroyed (death/merge) as computation proceeds — the node set is itself part of the computation.

3. **Heterogeneous programs.** Standard GNNs use a single shared message function. CRF cells hold per-cell program weights that mutate at split time, enabling **functional specialization** of subpopulations.

4. **Energy as compute budget signal.** GNNs have no analogue to the energy scalar. In CRF, energy is a self-regulating signal that controls where computation concentrates.

5. **Continuous-time analog.** CRF's update equations (§1.4) resemble a discretized ODE on a dynamic graph, connecting it more closely to Neural ODEs on graphs than to standard message-passing GNNs.

---

## 2.3 Mixture of Experts (MoE)

MoE models (Shazeer et al. 2017; Fedus et al. 2022) replace a single FFN with $E$ expert networks and a learned sparse router that activates top-$k$ experts per token:

$$\text{MoE}(\mathbf{x}) = \sum_{e \in \text{top-}k(\mathbf{x})} g_e(\mathbf{x})\, \text{Expert}_e(\mathbf{x})$$

**CRF differences:**

1. **Experts vs. cells.** MoE experts are static, pre-allocated modules. CRF cells are dynamically created and destroyed; the "expert" population changes during a single forward pass.

2. **Routing direction.** In MoE, tokens route *to* experts. In CRF, cells route messages *to neighboring cells* — communication is lateral, not hierarchical dispatch.

3. **Parameter sharing.** MoE experts are independent; CRF child cells inherit (mutated) parent weights, creating a genealogy of parameters rather than independently initialized experts.

4. **No fixed dispatch bottleneck.** MoE routers can suffer from load imbalance and expert collapse. CRF's energy mechanism provides a soft load-balancing signal: low-energy cells die, preventing idle compute without an auxiliary balancing loss.

5. **Spatial layout.** MoE has no notion of cell position. CRF's 2D spatial positions influence neighbor selection ($\lambda \cdot \|\mathbf{p}_i - \mathbf{p}_j\|$ term in §1.3), encoding an inductive bias about locality.

---

## 2.4 Neural Cellular Automata (NCA)

NCAs (Mordvintsev et al. 2020) define a uniform local update rule applied to every cell on a fixed grid:

$$\mathbf{s}_i^{(t+1)} = f_\theta(\mathbf{s}_i^{(t)},\; \text{perceive}(\mathcal{N}(i)^{(t)}))$$

All cells share the **same weights** $\theta$, operate on a **fixed grid**, and there is **no lifecycle** (no birth or death).

**CRF differences:**

| Dimension | NCA | CRF |
|-----------|-----|-----|
| Weight sharing | All cells share $\theta$ | Per-cell $\theta_i$, mutated at split |
| Grid topology | Fixed (e.g., 2D image grid) | Dynamic $k$-NN graph in continuous 2D space |
| Node count | Fixed | Variable via lifecycle |
| Input integration | Usually pixel-level (vision) | Arbitrary token embeddings (language/reasoning) |
| Routing | Local convolution / fixed neighborhood | Learned gated routing (§1.3 Step 3) |
| Supervision | Typically reconstruction of pattern | Language modeling / classification loss |

NCA is the closest architectural ancestor of CRF. CRF extends it by: (a) relaxing the fixed grid to a dynamic $k$-NN graph in continuous space, (b) introducing per-cell heterogeneous programs, (c) adding an energy-driven lifecycle, and (d) training end-to-end on discrete token prediction.

---

## 2.5 Multi-Agent Systems (MAS)

Multi-agent reinforcement learning (MARL; Lowe et al. 2017; Rashid et al. 2018) trains populations of agents that communicate, cooperate, or compete via environmental rewards.

**CRF differences:**

1. **Training objective.** MARL agents are trained with RL reward signals; CRF is trained end-to-end with backpropagation through all cell steps (gradient flows through `CellProgram` via the message aggregation). There is no environment-agent loop.

2. **Differentiability.** MARL communication is typically discrete or uses straight-through estimators. CRF message passing is fully differentiable (weighted sum of continuous state vectors).

3. **No separate environment.** MARL agents act in an external world. CRF cells act on each other's states — the "world" is the collective cell population itself.

4. **Scale and speed.** MARL typically involves tens to hundreds of agents trained over millions of environment steps. CRF operates with up to $N_{\max} = 1024$ cells per forward pass, completing in $S \leq 16$ steps, making it viable as an inference-time component inside a larger model.

5. **Communication protocol.** MARL often learns communication protocols with discrete messages. CRF passes continuous $d$-dimensional state vectors, making the communication channel higher-bandwidth and fully continuous.

---

## 2.6 Summary Positioning Table

| Property | Transformer | GNN | MoE | NCA | MARL | **CRF** |
|----------|-------------|-----|-----|-----|------|---------|
| Dynamic node count | ✗ | ✗ | ✗ | ✗ | ✓ | **✓** |
| Adaptive topology | ✗ | ✗ | ✗ | ✗ | Partial | **✓** |
| Per-cell heterogeneous weights | ✗ | ✗ | ✓ | ✗ | ✓ | **✓** |
| Energy-driven lifecycle | ✗ | ✗ | ✗ | ✗ | ✗ | **✓** |
| Fully differentiable | ✓ | ✓ | ✓ | ✓ | ✗ | **✓** |
| Operates on token sequences | ✓ | ✗ | ✓ | ✗ | ✗ | **✓** |
| Spatial positional layout | ✗ | ✗ | ✗ | ✓ | ✗ | **✓** |
| Learned sparse routing | ✗ | ✗ | ✓ | ✗ | ✗ | **✓** |

CRF occupies a unique intersection: it is the first architecture to combine **differentiable end-to-end training on token sequences** with **dynamic population lifecycles**, **per-cell heterogeneous programs**, and **energy-regulated adaptive computation**, all within a single forward pass.

---

# Section 3: Theoretical Properties

## 3.1 Theorem 1 — Bounded Per-Step Communication Cost

**Theorem.** At any step $t$, the total number of messages transmitted across all cells is at most $k \cdot N_t$, where $k = k_{\max}$ is the neighborhood size and $N_t \leq N_{\max}$ is the current cell count.

**Proof.**

Each cell $c_i \in \mathcal{C}^{(t)}$ selects exactly $\min(k_{\max}, N_t - 1)$ neighbors (§1.3, Step 2). Since $N_t \geq 2$ is required before routing executes (enforced by the `if len(cells) < 2: break` guard), the number of directed edges in the communication graph at step $t$ is:

$$|E^{(t)}| = \sum_{i=1}^{N_t} |\mathcal{N}(i)| \leq \sum_{i=1}^{N_t} k_{\max} = k_{\max} \cdot N_t$$

Each directed edge corresponds to exactly one message (a weighted state vector in $\mathbb{R}^d$). Therefore, the total communication volume (in units of $\mathbb{R}^d$ vectors) is at most $k_{\max} \cdot N_t \leq k_{\max} \cdot N_{\max}$.

Since $k_{\max}$ and $N_{\max}$ are hyperparameters fixed before any forward pass, **per-step communication cost is $O(k \cdot N_{\max})$, independent of input sequence length $T$**. $\square$

**Corollary.** Total communication cost over $S$ steps is $O(S \cdot k \cdot N_{\max})$. For fixed $k$ and $S$, this is linear in $N_{\max}$, contrasting with the $O(T^2)$ attention cost of the Transformer.

**Remark.** The similarity matrix computation in `SparseFabric` currently costs $O(N^2 d)$ to find top-$k$ neighbors (dense cosine similarity, then `topk`). This is a *routing overhead*, not a communication cost per se. It can be reduced to $O(N k d)$ using approximate nearest-neighbor methods (e.g., HNSW, LSH), making routing itself sub-quadratic. This is an implementation detail, not a fundamental property of the CRF model.

---

## 3.2 Theorem 2 — Adaptive Computation via Energy-Regulated Population

**Definition (Adaptive computation).** A model is adaptive if its computational cost $\mathcal{F}(\mathbf{x})$ varies across inputs $\mathbf{x}$, increasing on harder inputs and decreasing on easier ones, in expectation.

**Theorem.** Under the CRF lifecycle rules (§1.5), the expected cell population $\mathbb{E}[N_t]$ is non-decreasing in input complexity, provided complexity correlates positively with the average energy gate $\bar{\delta} = \frac{1}{N_t}\sum_i \delta_i$.

**Proof sketch.**

Consider the cell count dynamics between split/merge events. Between lifecycle steps, $N_t$ evolves only through death and split. Define:

- $B^{(t)}$ = number of cells that split at step $t$ (birth rate)
- $D^{(t)}$ = number of cells that die at step $t$ (death rate)

A cell $c_i$ splits iff $\varepsilon_i > \varepsilon_{\text{split}}$ and $N_t < N_{\max}$. From §1.4, in steady state $\varepsilon_i^* \approx \delta_i^*$ (see fixed-point analysis). Therefore:

$$B^{(t)} \approx |\{i : \delta_i^* > \varepsilon_{\text{split}}, N_t < N_{\max}\}|$$

A cell dies iff $\varepsilon_i < \varepsilon_{\text{die}} = 0.01$, which requires sustained $\delta_i \approx 0$.

If $\bar{\delta}$ increases (harder input activates more cell programs), more cells sustain $\varepsilon > \varepsilon_{\text{split}}$, so $B^{(t)}$ increases. Simultaneously, fewer cells hit $\varepsilon < \varepsilon_{\text{die}}$, so $D^{(t)}$ decreases.

$$\frac{d\,\mathbb{E}[N]}{dt} = \mathbb{E}[B^{(t)}] - \mathbb{E}[D^{(t)}]$$

Both terms move in the direction of increasing $N$ when $\bar{\delta}$ increases. Therefore $\mathbb{E}[N_t]$ is non-decreasing in $\bar{\delta}$, and since $\mathcal{F}(\mathbf{x}) = O(S N_t^2 d)$, computational cost is adaptive. $\square$

**Remark.** This is a soft adaptive computation mechanism unlike ACT (Graves 2016) or early exit (Elbayad et al. 2020), where computation depth halts at a threshold. In CRF, both depth ($S$) and width ($N_t$) are variable, with width governed by a population-level feedback loop.

---

## 3.3 Theorem 3 — Convergence of Cell States Under Fixed Population

**Theorem.** Consider a CRF with fixed population ($N_t = N$ for all $t$, i.e., no lifecycle events) and a fixed $k$-NN graph $G$ (static topology). If the CellProgram $f_\theta$ is a contraction mapping in $\mathbf{s}$ — i.e., there exists $L_f < 1$ such that $\|f_\theta(\mathbf{s}, \mathbf{m}) - f_\theta(\mathbf{s}', \mathbf{m})\|_2 \leq L_f \|\mathbf{s} - \mathbf{s}'\|_2$ for all $\mathbf{s}, \mathbf{s}', \mathbf{m}$ — then the joint state $\mathbf{S}^{(t)} = [\mathbf{s}_1^{(t)}, \ldots, \mathbf{s}_N^{(t)}]$ converges to a unique fixed point $\mathbf{S}^*$.

**Proof.**

The full update operator $\Phi: \mathbb{R}^{Nd} \to \mathbb{R}^{Nd}$ maps $\mathbf{S}^{(t)} \mapsto \mathbf{S}^{(t+1)}$ as follows:

1. Compute messages: $\mathbf{m}_i = \frac{\sum_{j \in \mathcal{N}(i)} w_{ij} \mathbf{s}_j^{(t)}}{\sum_j w_{ij}}$. Since messages are convex combinations of states, they are non-expanding: $\|\mathbf{m}_i - \mathbf{m}_i'\|_2 \leq \|\mathbf{S}^{(t)} - \mathbf{S}'^{(t)}\|_2$.

2. Apply cell program: $\mathbf{s}_i^{(t+1)} = f_\theta(\mathbf{s}_i^{(t)}, \mathbf{m}_i)$.

The Lipschitz constant of $\Phi$ satisfies:

$$\|\Phi(\mathbf{S}) - \Phi(\mathbf{S}')\|_2 \leq L_f \|\mathbf{S} - \mathbf{S}'\|_2$$

Since $L_f < 1$, $\Phi$ is a contraction on $(\mathbb{R}^{Nd}, \|\cdot\|_2)$. By the **Banach Fixed-Point Theorem**, there exists a unique $\mathbf{S}^*$ such that $\Phi(\mathbf{S}^*) = \mathbf{S}^*$, and iterating from any initial $\mathbf{S}^{(0)}$:

$$\|\mathbf{S}^{(t)} - \mathbf{S}^*\|_2 \leq L_f^t \|\mathbf{S}^{(0)} - \mathbf{S}^*\|_2 \to 0 \text{ as } t \to \infty$$

Convergence is geometric with rate $L_f$. $\square$

**Conditions for $L_f < 1$ in the implemented CellProgram.**

The CellProgram uses LayerNorm as the final normalization, which bounds outputs to a compact set around the unit hypersphere. The ReLU activations in the gate MLP are non-expansive ($L_{\text{ReLU}} = 1$). The overall contraction depends on the spectral norm of $W_1$ and $W_2$. During training, applying **spectral normalization** to these weights (a one-line change) would enforce $L_f < 1$ by construction.

**Remark on the dynamic case.** When the lifecycle is active, $\mathbf{S}^*$ shifts at each split/merge event. The system does not converge to a single point but rather tracks a slowly-moving target as the population evolves. This is analogous to a time-varying dynamical system with piecewise-stationary equilibria.

---

## 3.4 Proposition — Message Passing Expressivity Bound

**Proposition.** The class of functions computable by a CRF with $k$-NN routing and $N$ cells is strictly larger than the class computable by a $k$-NN message-passing GNN with $N$ fixed nodes after any finite number of steps, because CRF cells can dynamically subdivide the input representation via split.

**Informal argument.** Consider an input that requires distinguishing two nearly identical subgraphs. A fixed-node GNN is limited by the 1-WL isomorphism test (Xu et al. 2019) — it cannot distinguish certain non-isomorphic graphs regardless of depth. CRF's split mechanism can introduce new cells at points of high energy (model uncertainty), effectively refining the representation at ambiguous regions. This is analogous to adaptive mesh refinement in numerical PDEs, which can resolve features that uniform grids cannot. A formal proof requires defining a CRF computation class and showing it exceeds the WL hierarchy, which we leave as future work.

---

## 3.5 Energy Conservation Lemma

**Lemma.** At every split event, total system energy is non-increasing: $\varepsilon_{\text{parent}}^{\text{after}} + \varepsilon_{\text{child}}^{\text{after}} < \varepsilon_{\text{parent}}^{\text{before}}$.

**Proof.** From §1.5.1:

$$\varepsilon_{\text{parent}}^{\text{after}} = 0.4\,\varepsilon_{\text{parent}}^{\text{before}}$$
$$\varepsilon_{\text{child}}^{\text{after}} = 0.4\,\varepsilon_{\text{parent}}^{\text{before}}$$
$$\varepsilon_{\text{parent}}^{\text{after}} + \varepsilon_{\text{child}}^{\text{after}} = 0.8\,\varepsilon_{\text{parent}}^{\text{before}} < \varepsilon_{\text{parent}}^{\text{before}} \quad \square$$

**Corollary.** The total system energy $\mathcal{E}^{(t)} = \sum_i \varepsilon_i^{(t)}$ decreases by $0.2\,\varepsilon_{\text{parent}}$ at each split, acting as a natural regularizer against infinite cell proliferation. This, combined with the $N_{\max}$ hard cap, ensures $N_t$ is always finite.

---

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

---

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
