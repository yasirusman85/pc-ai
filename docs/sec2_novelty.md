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
