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
