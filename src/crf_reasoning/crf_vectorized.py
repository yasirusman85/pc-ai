"""
crf_vectorized.py — Vectorized Cellular Reasoning Fabric
=========================================================
Replaces the Python-level cell loop in crf.py with fully batched tensor ops.

Key design choices:
  - All N cells stored as a single tensor S ∈ R^{N×d}, enabling batch matmul.
  - Lifecycle (split/death/merge) operates on tensor indices, not Python lists.
  - AblationConfig dataclass controls which mechanisms are active.
  - CRFMetrics collected every forward pass for empirical validation.
"""

import math
import random
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── Ablation Configuration ────────────────────────────────────────────────


@dataclass
class AblationConfig:
    """Flip any flag to False to disable that mechanism for ablation studies."""

    use_split: bool = True  # allow cell birth
    use_death: bool = True  # allow cell death
    use_merge: bool = True  # allow cell consolidation
    use_routing: bool = True  # use learned routing gate (False → uniform mean)
    use_energy: bool = True  # energy drives lifecycle (False → random lifecycle)
    use_messaging: bool = True  # cells receive neighbor messages (False → self only)
    use_spatial: bool = True  # spatial penalty in similarity (False → pure cosine)
    spatial_lambda: float = 0.05  # weight of spatial distance penalty


# ─── Per-forward-pass metrics ──────────────────────────────────────────────


@dataclass
class CRFMetrics:
    """Collected during a single forward pass."""

    n_cells_per_step: list = field(default_factory=list)  # N_t for each step
    n_splits: int = 0
    n_deaths: int = 0
    n_merges: int = 0
    comm_cost: int = 0  # total messages = sum_t k*N_t
    specialization: float = 0.0  # mean off-diagonal cosine distance of anchor states
    graph_diameters: list = field(default_factory=list)
    wall_time_s: float = 0.0


# ─── Shared CellProgram (weight-shared across all cells) ───────────────────


class SharedCellProgram(nn.Module):
    """
    A single MLP whose weights are shared across all N cells in a batch.
    Input:  [state ‖ message] ∈ R^{N×2d}
    Output: new_state ∈ R^{N×d}, out_msg ∈ R^{N×d}, energy_delta ∈ R^{N×1}

    Fix 3: energy_gate bias initialised to +1.0 so sigmoid(·) ≈ 0.73 at
    start, causing energy to accumulate to ~1.1 within a few steps and
    trigger the ε_split=1.05 threshold reliably.
    """

    def __init__(self, d_model: int, d_hidden: int = 128):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_model),
        )
        self.msg_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.energy_gate = nn.Linear(d_model, 1)
        # Fix 3: bias 2.5 → sigmoid ≈ 0.92 → E accumulates past ε_split=1.05
        nn.init.constant_(self.energy_gate.bias, 2.5)

    def forward(self, states: torch.Tensor, messages: torch.Tensor):
        """
        states:   R^{N×d}
        messages: R^{N×d}
        """
        x = torch.cat([states, messages], dim=-1)  # N×2d
        new_states = self.norm(states + self.gate(x))  # N×d
        out_msgs = self.msg_proj(new_states)  # N×d
        energy_d = torch.sigmoid(self.energy_gate(new_states))  # N×1
        return new_states, out_msgs, energy_d.squeeze(-1)  # N, N, N


# ─── Vectorized SparseFabric ────────────────────────────────────────────────


class VectorizedFabric(nn.Module):
    """
    Builds a k-NN communication graph from cell states and positions,
    then computes weighted message aggregation — fully batched.
    """

    def __init__(self, d_model: int, k: int = 4):
        super().__init__()
        self.k = k
        self.route = nn.Linear(d_model * 2, 1)  # learned gate

    def forward(
        self,
        states: torch.Tensor,  # N×d
        positions: torch.Tensor,  # N×2
        cfg: AblationConfig,
        batch_id: torch.Tensor = None,  # N — masks cross-batch edges
    ) -> torch.Tensor:  # N×d  (aggregated messages)
        N, d = states.shape
        if N <= 1:
            return states.clone()

        # Step 1: similarity matrix
        s_norm = F.normalize(states, dim=-1)  # N×d
        sim = s_norm @ s_norm.T  # N×N

        if cfg.use_spatial:
            pos_dist = torch.cdist(positions, positions)  # N×N
            sim = sim - cfg.spatial_lambda * pos_dist

        # Batch isolation: mask cross-batch edges (Fix 2)
        if batch_id is not None:
            same_batch = batch_id.unsqueeze(0) == batch_id.unsqueeze(1)
            sim = sim.masked_fill(~same_batch, float("-inf"))

        sim.fill_diagonal_(float("-inf"))

        # Step 2: top-k neighbors
        k = min(self.k, N - 1)
        _, top_idx = sim.topk(k, dim=-1)  # N×k  (indices)

        # Step 3: routing weights
        if cfg.use_routing:
            # Gather neighbor states: N×k×d
            neigh_states = states[top_idx]  # N×k×d
            si_expand = states.unsqueeze(1).expand_as(neigh_states)  # N×k×d
            pairs = torch.cat([si_expand, neigh_states], dim=-1)  # N×k×2d
            w = torch.sigmoid(self.route(pairs)).squeeze(-1)  # N×k
        else:
            w = torch.ones(N, k, device=states.device)  # uniform

        # Step 4: weighted aggregation
        # neigh_states: N×k×d, w: N×k
        if not cfg.use_messaging:
            return states.clone()  # ablation: no messages

        neigh_states = states[top_idx]  # N×k×d
        w_sum = w.sum(dim=-1, keepdim=True).clamp(min=1e-8)  # N×1
        messages = (w.unsqueeze(-1) * neigh_states).sum(dim=1) / w_sum  # N×d
        return messages


# ─── Population State ───────────────────────────────────────────────────────


class CellPopulation:
    """
    Holds the full cell population as flat tensors.
    anchor_mask: bool tensor, True for cells that cannot die.
    """

    def __init__(
        self,
        states: torch.Tensor,  # N×d
        positions: torch.Tensor,  # N×2
        energies: torch.Tensor,  # N
        anchor_mask: torch.Tensor,  # N  (bool)
        device: torch.device,
    ):
        self.states = states
        self.positions = positions
        self.energies = energies
        self.anchor_mask = anchor_mask
        self.device = device
        self.batch_id = None

    @property
    def N(self) -> int:
        return self.states.size(0)

    def apply_split(
        self,
        cfg: AblationConfig,
        program: SharedCellProgram,
        max_cells: int,
        metrics: CRFMetrics,
    ):
        if not cfg.use_split:
            return
        if self.N >= max_cells:
            return

        # Fix 3: lowered ε_split from 1.5 → 1.05 so lifecycle activates
        EPS_SPLIT = 1.05
        if cfg.use_energy:
            eligible = (~self.anchor_mask) & (self.energies > EPS_SPLIT)
        else:
            eligible = (~self.anchor_mask) & (
                torch.rand(self.N, device=self.device) < 0.3
            )

        idxs = eligible.nonzero(as_tuple=True)[0]
        if len(idxs) == 0:
            return

        n_new = min(len(idxs), max_cells - self.N)
        idxs = idxs[:n_new]

        parent_s = self.states[idxs]
        parent_p = self.positions[idxs]
        parent_e = self.energies[idxs]

        child_s = parent_s + torch.randn_like(parent_s) * 0.01
        child_p = parent_p + (torch.randn_like(parent_p) * 0.5).clamp(-0.5, 0.5)
        child_e = parent_e * 0.4

        self.energies[idxs] = parent_e * 0.4

        child_anchors = torch.zeros(len(idxs), dtype=torch.bool, device=self.device)
        self.states = torch.cat([self.states, child_s], dim=0)
        self.positions = torch.cat([self.positions, child_p], dim=0)
        self.energies = torch.cat([self.energies, child_e], dim=0)
        self.anchor_mask = torch.cat([self.anchor_mask, child_anchors], dim=0)
        # Propagate batch_id to children (Fix 2)
        if self.batch_id is not None:
            child_batch = self.batch_id[idxs]
            self.batch_id = torch.cat([self.batch_id, child_batch], dim=0)

        metrics.n_splits += len(idxs)

    def apply_death(
        self,
        cfg: AblationConfig,
        metrics: CRFMetrics,
    ):
        if not cfg.use_death:
            return

        if cfg.use_energy:
            keep = self.anchor_mask | (self.energies >= 0.01)
        else:
            keep = torch.ones(self.N, dtype=torch.bool, device=self.device)

        n_dead = (~keep).sum().item()
        if n_dead == 0:
            return

        self.states = self.states[keep]
        self.positions = self.positions[keep]
        self.energies = self.energies[keep]
        self.anchor_mask = self.anchor_mask[keep]
        if self.batch_id is not None:
            self.batch_id = self.batch_id[keep]
        metrics.n_deaths += int(n_dead)

    def apply_merge(
        self,
        cfg: AblationConfig,
        metrics: CRFMetrics,
        tau: float = 0.95,
    ):
        if not cfg.use_merge or self.N < 3:
            return

        s_norm = F.normalize(self.states, dim=-1)
        sim = s_norm @ s_norm.T  # N×N

        # zero out anchor-anchor and self pairs
        both_anchors = self.anchor_mask.unsqueeze(0) & self.anchor_mask.unsqueeze(1)
        sim = sim.masked_fill(
            torch.eye(self.N, dtype=torch.bool, device=self.device), -1
        )
        sim = sim.masked_fill(both_anchors, -1)

        n_candidates = max(1, self.N // 20)
        # find top candidates by flattening upper triangle
        upper = torch.triu(sim, diagonal=1)
        flat = upper.reshape(-1)
        top_k = min(n_candidates, (flat > tau).sum().item())
        if top_k == 0:
            return

        _, flat_idx = flat.topk(top_k)
        merged = torch.zeros(self.N, dtype=torch.bool, device=self.device)
        remove = torch.zeros(self.N, dtype=torch.bool, device=self.device)

        for fi in flat_idx:
            i = fi.item() // self.N
            j = fi.item() % self.N
            if merged[i] or merged[j]:
                continue
            if sim[i, j].item() < tau:
                continue
            # merge j into i
            self.states[i] = (self.states[i] + self.states[j]) / 2
            self.positions[i] = (self.positions[i] + self.positions[j]) / 2
            self.energies[i] = self.energies[i] + self.energies[j]
            merged[i] = True
            merged[j] = True
            remove[j] = True
            metrics.n_merges += 1

        keep = ~remove
        self.states = self.states[keep]
        self.positions = self.positions[keep]
        self.energies = self.energies[keep]
        self.anchor_mask = self.anchor_mask[keep]
        if self.batch_id is not None:
            self.batch_id = self.batch_id[keep]


# ─── Vectorized CellularReasoningFabric ────────────────────────────────────


class VectorizedCRF(nn.Module):
    """
    Drop-in replacement for CellularReasoningFabric that operates purely
    on tensors — no Python list of nn.Module cells.

    Forward pass processes one sequence at a time (batch dimension handled
    by CRFLanguageModel below).
    """

    def __init__(
        self,
        d_model: int = 256,
        d_hidden: int = 128,
        n_init: int = 64,
        max_cells: int = 512,
        k_neighbors: int = 4,
        cfg: AblationConfig = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_init = n_init
        self.max_cells = max_cells
        self.cfg = cfg or AblationConfig()

        self.program = SharedCellProgram(d_model, d_hidden)
        self.fabric = VectorizedFabric(d_model, k_neighbors)
        self.input_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        token_states: torch.Tensor,  # B×T×d  (batched!) or 1×T×d
        n_steps: int = 8,
        collect_metrics: bool = False,
    ):
        t0 = time.time()
        cfg = self.cfg
        device = token_states.device
        B, T, d = token_states.shape

        # ── Flatten batch into one population ──────────────────────────
        # Each sequence gets (n_init) cells; batch_id prevents cross-talk.
        H = self.input_proj(token_states)  # B×T×d
        H_flat = H.view(B * T, d)  # (B·T)×d

        n_grid = max(4, int(math.sqrt(self.n_init)))
        n_create = max(self.n_init, T + 8)  # cells per sequence (ideal)
        # Cap per-sequence count so total initial ≤ max_cells (Fix 2 batched)
        n_create = min(n_create, self.max_cells // max(1, B))
        # At minimum, create T anchor cells + a few extra
        n_create = max(n_create, T + 2)
        N_per_seq = n_create

        # Anchor positions (same grid for every sequence)
        anchor_idxs = torch.arange(T, device=device)
        anchor_pos = torch.stack(
            [torch.tensor([float(i % n_grid), float(i // n_grid)]) for i in range(T)],
            dim=0,
        ).to(device)
        n_extra = n_create - T
        extra_idxs = (
            torch.arange(n_extra, device=device) % T
            if n_extra > 0
            else torch.tensor([], device=device, dtype=torch.long)
        )
        extra_pos = (
            (
                anchor_pos[extra_idxs]
                + (torch.rand(n_extra, 2, device=device) * 0.6 - 0.3)
            )
            if n_extra > 0
            else torch.zeros(0, 2, device=device)
        )
        seq_positions = torch.cat([anchor_pos, extra_pos], dim=0)  # N_per_seq×2

        all_states_list = []
        all_pos_list = []
        all_energy_list = []
        all_anchor = []
        batch_ids = []
        for b in range(B):
            seq_H = H[b]  # T×d
            anchor_s = seq_H  # T×d
            extra_s = (
                (seq_H[extra_idxs] + torch.randn(n_extra, d, device=device) * 0.02)
                if n_extra > 0
                else torch.tensor([], device=device).reshape(0, d)
            )
            states_seq = torch.cat([anchor_s, extra_s], dim=0)  # N_per_seq×d
            all_states_list.append(states_seq)
            all_pos_list.append(seq_positions)
            all_energy_list.append(torch.ones(N_per_seq, device=device))
            anchor_mask_seq = torch.zeros(N_per_seq, dtype=torch.bool, device=device)
            anchor_mask_seq[:T] = True
            all_anchor.append(anchor_mask_seq)
            batch_ids.append(
                torch.full((N_per_seq,), b, device=device, dtype=torch.long)
            )

        all_states = torch.cat(all_states_list, dim=0)  # (B·N_per_seq)×d
        all_pos = torch.cat(all_pos_list, dim=0)  # (B·N_per_seq)×2
        all_energy = torch.cat(all_energy_list, dim=0)  # (B·N_per_seq)
        anchor_mask = torch.cat(all_anchor, dim=0)  # (B·N_per_seq)
        batch_id = torch.cat(batch_ids, dim=0)  # (B·N_per_seq)

        pop = CellPopulation(all_states, all_pos, all_energy, anchor_mask, device)
        pop.batch_id = batch_id

        metrics = CRFMetrics()

        # ── Step loop ──────────────────────────────────────────────────
        for step in range(n_steps):
            if pop.N < 2:
                break

            metrics.n_cells_per_step.append(pop.N)
            metrics.comm_cost += self.fabric.k * pop.N

            # 1. Routing → messages
            messages = self.fabric(pop.states, pop.positions, cfg, pop.batch_id)  # N×d

            # 2. Cell program (batched)
            new_states, _, energy_delta = self.program(pop.states, messages)
            pop.states = new_states

            # 3. Energy update
            # Fix 3: lower decay (0.95→0.90), amplify delta (×2), so E
            # climbs past ε_split=1.05 within ~5 steps even for idle cells.
            if cfg.use_energy:
                pop.energies = (0.90 * pop.energies + 0.10 * energy_delta * 2).clamp(
                    0.01, 5
                )
            else:
                pop.energies = torch.ones_like(pop.energies)

            # 4. Lifecycle (every 3 steps)
            if step % 3 == 0:
                pop.apply_split(cfg, self.program, self.max_cells, metrics)
                pop.apply_death(cfg, metrics)

            # 5. Merge (every 5 steps)
            if step % 5 == 0 and step > 0:
                pop.apply_merge(cfg, metrics)

        # ── Extract anchor outputs per batch ───────────────────────────
        # anchor_mask and batch_id partition the population.
        anchor_mask_bool = pop.anchor_mask
        anchor_states_b = pop.states[anchor_mask_bool]  # (total_anchors)×d
        anchor_batch = pop.batch_id[anchor_mask_bool]  # (total_anchors)

        # Group by batch index
        out_pieces = []
        for b in range(B):
            mask = anchor_batch == b
            piece = anchor_states_b[mask]  # ≤T × d
            if piece.size(0) < T:
                pad = torch.zeros(T - piece.size(0), d, device=device)
                piece = torch.cat([piece, pad], dim=0)
            else:
                piece = piece[:T]
            out_pieces.append(self.output_proj(piece))

        out = torch.stack(out_pieces, dim=0)  # B×T×d

        if collect_metrics:
            # specialization: mean pairwise cosine distance of anchor states
            an = F.normalize(out[0], dim=-1)
            if T > 1:
                cos_sim_mat = an @ an.T
                mask = ~torch.eye(T, dtype=torch.bool, device=device)
                metrics.specialization = (1 - cos_sim_mat[mask]).mean().item()
            metrics.wall_time_s = time.time() - t0

        return out, metrics  # B×T×d, CRFMetrics

    @torch.no_grad()
    def get_final_anchor_states(
        self,
        token_states: torch.Tensor,  # B×T×d
        n_steps: int = 8,
    ) -> torch.Tensor:
        """
        Returns anchor cell states AFTER all CRF steps have run,
        for use in graph diameter and specialization measurements.
        """
        out, _ = self.forward(token_states, n_steps, collect_metrics=False)
        return out  # B×T×d


# ─── Full Language Model ────────────────────────────────────────────────────


class CRFLanguageModel(nn.Module):
    """
    CRF-based language model. Processes batch by looping over sequence items
    (CRF is inherently per-sequence due to dynamic topology).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        d_hidden: int = 128,
        n_init_cells: int = 64,
        max_cells: int = 512,
        n_crf_steps: int = 8,
        k_neighbors: int = 4,
        max_seq_len: int = 512,
        cfg: AblationConfig = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.n_crf_steps = n_crf_steps
        self.cfg = cfg or AblationConfig()

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
        self.crf = VectorizedCRF(
            d_model, d_hidden, n_init_cells, max_cells, k_neighbors, self.cfg
        )
        self.ln = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self._init_weights()
        # Re-apply energy gate bias after _init_weights zeroed it (Fix 3)
        nn.init.constant_(self.crf.program.energy_gate.bias, 2.5)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0, 0.02)

    def forward(
        self,
        x: torch.Tensor,  # B×T
        targets: Optional[torch.Tensor] = None,  # B×T
        collect_metrics: bool = False,
    ):
        B, T = x.shape
        tok = self.token_embed(x) + self.pos_enc[:, :T]  # B×T×d

        # Fix 2: single batched CRF call — all B sequences stacked into
        # one population with batch_id masking to prevent cross-talk.
        out, all_met = self.crf(tok, self.n_crf_steps, collect_metrics)

        logits = self.lm_head(self.ln(out))  # B×T×V

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
        return logits, loss, all_met

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def generate(self, x: torch.Tensor, max_new: int = 64, temperature: float = 0.8):
        self.eval()
        for _ in range(max_new):
            xc = x[:, -self.max_seq_len :]
            logits, loss, _ = self.forward(xc)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            x = torch.cat([x, nxt], dim=1)
        return x


# ─── Sanity check ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(0)
    vocab = 1000

    for use_routing in [True, False]:
        for use_energy in [True, False]:
            cfg = AblationConfig(use_routing=use_routing, use_energy=use_energy)
            model = CRFLanguageModel(
                vocab,
                d_model=64,
                d_hidden=32,
                n_init_cells=16,
                max_cells=64,
                n_crf_steps=4,
                k_neighbors=3,
                cfg=cfg,
            )
            x = torch.randint(0, vocab, (2, 12))
            logits, loss, mets = model(x, targets=x, collect_metrics=True)
            avg_n = (
                sum(mets.n_cells_per_step) / max(1, len(mets.n_cells_per_step))
                if mets.n_cells_per_step
                else 0
            )
            print(
                f"routing={use_routing} energy={use_energy} | "
                f"params={model.n_params:,} loss={loss.item():.4f} "
                f"avg_N={avg_n:.1f} splits={mets.n_splits} merges={mets.n_merges}"
            )
