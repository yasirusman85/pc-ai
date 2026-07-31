"""
spatial_routing.py — Sub-linear k-NN neighbor lookup for CRF
============================================================
Replaces O(N²) dense similarity + topk with spatial grid hashing
plus bounded candidate scoring: O(N · m · d) where m ≪ N.

Cells already carry 2D positions; we hash them into grid bins and
only score neighbors in a local (2r+1)² window before top-k selection.
When the candidate pool is too small we expand the window once, then
fall back to brute-force for N ≤ fallback_threshold.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


def _grid_keys(positions: torch.Tensor, grid_size: float) -> torch.Tensor:
    """Map N×2 positions to integer grid cell coordinates."""
    return (positions / grid_size).floor().long()


def _build_spatial_index(
    positions: torch.Tensor,
    grid_size: float,
) -> Dict[Tuple[int, int], List[int]]:
    """Build cell_id → list of point indices mapping."""
    keys = _grid_keys(positions, grid_size)
    index: Dict[Tuple[int, int], List[int]] = {}
    for i in range(positions.size(0)):
        key = (keys[i, 0].item(), keys[i, 1].item())
        index.setdefault(key, []).append(i)
    return index


def _gather_candidates(
    index: Dict[Tuple[int, int], List[int]],
    cell_key: Tuple[int, int],
    radius: int,
) -> List[int]:
    """Collect all point indices in a (2r+1)² grid neighborhood."""
    cx, cy = cell_key
    out: List[int] = []
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            out.extend(index.get((cx + dx, cy + dy), []))
    return out


def compute_similarity_scores(
    states: torch.Tensor,
    positions: torch.Tensor,
    candidate_idx: torch.Tensor,
    query_idx: int,
    spatial_lambda: float,
    use_spatial: bool,
) -> torch.Tensor:
    """
    Score one query cell against its candidate neighbors.
    candidate_idx: 1D long tensor of neighbor indices (may include query).
    """
    qi = query_idx
    si = states[qi]
    sj = states[candidate_idx]
    s_norm_i = F.normalize(si.unsqueeze(0), dim=-1)
    s_norm_j = F.normalize(sj, dim=-1)
    sim = (s_norm_j @ s_norm_i.T).squeeze(-1)
    if use_spatial:
        pi = positions[qi]
        pj = positions[candidate_idx]
        sim = sim - spatial_lambda * torch.norm(pj - pi, dim=-1)
    return sim


def sublinear_topk_neighbors(
    states: torch.Tensor,
    positions: torch.Tensor,
    k: int,
    spatial_lambda: float = 0.05,
    use_spatial: bool = True,
    grid_size: float = 1.0,
    search_radius: int = 1,
    max_candidates: int = 32,
    fallback_threshold: int = 64,
    batch_id: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return (top_idx, top_scores) each of shape N×k for k-NN neighbors.

    Complexity: O(N · min(max_candidates, m) · d) per call vs O(N²d) dense.
    """
    N = states.size(0)
    device = states.device
    k = min(k, N - 1)
    if k <= 0:
        empty = torch.zeros(N, 0, dtype=torch.long, device=device)
        return empty, empty

    if N <= fallback_threshold:
        return _dense_topk(
            states, positions, k, spatial_lambda, use_spatial, batch_id,
        )

    index = _build_spatial_index(positions, grid_size)
    keys = _grid_keys(positions, grid_size)
    top_idx = torch.full((N, k), -1, dtype=torch.long, device=device)
    top_scores = torch.full((N, k), float('-inf'), device=device)

    for i in range(N):
        cell_key = (keys[i, 0].item(), keys[i, 1].item())
        candidates = _gather_candidates(index, cell_key, search_radius)
        # Expand window if pool is too small
        if len(candidates) < k + 2:
            candidates = _gather_candidates(index, cell_key, search_radius + 1)
        # Remove self and apply batch mask
        candidates = [c for c in candidates if c != i]
        if batch_id is not None:
            candidates = [c for c in candidates if batch_id[c] == batch_id[i]]
        if not candidates:
            continue
        if len(candidates) > max_candidates:
            # Subsample by spatial proximity to query
            pi = positions[i]
            dists = [(c, torch.norm(positions[c] - pi).item()) for c in candidates]
            dists.sort(key=lambda x: x[1])
            candidates = [c for c, _ in dists[:max_candidates]]

        cand_t = torch.tensor(candidates, dtype=torch.long, device=device)
        scores = compute_similarity_scores(
            states, positions, cand_t, i, spatial_lambda, use_spatial,
        )
        vals, order = scores.topk(min(k, scores.numel()))
        top_idx[i, : vals.numel()] = cand_t[order]
        top_scores[i, : vals.numel()] = vals

    # Fill any remaining -1 slots via dense fallback for those rows
    bad = (top_idx[:, 0] < 0).nonzero(as_tuple=True)[0]
    if bad.numel() > 0:
        dense_idx, dense_scores = _dense_topk(
            states, positions, k, spatial_lambda, use_spatial, batch_id,
        )
        top_idx[bad] = dense_idx[bad]
        top_scores[bad] = dense_scores[bad]

    return top_idx, top_scores


def _dense_topk(
    states: torch.Tensor,
    positions: torch.Tensor,
    k: int,
    spatial_lambda: float,
    use_spatial: bool,
    batch_id: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Original O(N²) routing — used as fallback for small N or sparse grids."""
    N = states.size(0)
    s_norm = F.normalize(states, dim=-1)
    sim = s_norm @ s_norm.T
    if use_spatial:
        sim = sim - spatial_lambda * torch.cdist(positions, positions)
    if batch_id is not None:
        same = batch_id.unsqueeze(0) == batch_id.unsqueeze(1)
        sim = sim.masked_fill(~same, float('-inf'))
    sim.fill_diagonal_(float('-inf'))
    scores, idx = sim.topk(k, dim=-1)
    return idx, scores


def sublinear_merge_candidates(
    states: torch.Tensor,
    positions: torch.Tensor,
    anchor_mask: torch.Tensor,
    tau: float = 0.95,
    grid_size: float = 1.0,
    search_radius: int = 1,
    max_pairs: int = 64,
    fallback_threshold: int = 64,
) -> List[Tuple[int, int, float]]:
    """
    Find merge candidate pairs with sim ≥ tau using spatial indexing.
    Returns list of (i, j, similarity) with i < j.
    """
    N = states.size(0)
    if N < 3:
        return []

    s_norm = F.normalize(states, dim=-1)

    if N <= fallback_threshold:
        return _dense_merge_candidates(states, anchor_mask, s_norm, tau, max_pairs)

    index = _build_spatial_index(positions, grid_size)
    keys = _grid_keys(positions, grid_size)
    pairs: List[Tuple[int, int, float]] = []

    for i in range(N):
        if anchor_mask[i]:
            continue
        cell_key = (keys[i, 0].item(), keys[i, 1].item())
        candidates = _gather_candidates(index, cell_key, search_radius)
        for j in candidates:
            if j <= i or anchor_mask[i] and anchor_mask[j]:
                continue
            sim = (s_norm[i] @ s_norm[j]).item()
            if sim >= tau:
                pairs.append((i, j, sim))

    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:max_pairs]


def _dense_merge_candidates(
    states: torch.Tensor,
    anchor_mask: torch.Tensor,
    s_norm: torch.Tensor,
    tau: float,
    max_pairs: int,
) -> List[Tuple[int, int, float]]:
    N = states.size(0)
    sim = s_norm @ s_norm.T
    both_anchors = anchor_mask.unsqueeze(0) & anchor_mask.unsqueeze(1)
    sim = sim.masked_fill(torch.eye(N, dtype=torch.bool, device=states.device), -1.0)
    sim = sim.masked_fill(both_anchors, -1.0)
    upper = torch.triu(sim, diagonal=1)
    flat = upper.reshape(-1)
    valid = flat > tau
    if not valid.any():
        return []
    vals = flat[valid]
    idxs = valid.nonzero(as_tuple=True)[0]
    order = vals.argsort(descending=True)
    pairs = []
    for fi in idxs[order[:max_pairs]]:
        i = fi.item() // N
        j = fi.item() % N
        pairs.append((i, j, flat[fi].item()))
    return pairs


def estimate_sublinear_routing_flops(
    N: int,
    d: int,
    k: int,
    max_candidates: int = 32,
) -> int:
    """Analytical FLOP count for one sub-linear routing step."""
    m = min(max_candidates, max(k + 1, 16))
    per_cell = (
        m * d +           # normalize + dot products for m candidates
        m +               # spatial distance
        k * 2 * d +       # routing gate pairs
        k * d             # aggregation
    )
    return N * per_cell
