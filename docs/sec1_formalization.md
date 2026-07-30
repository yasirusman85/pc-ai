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
