"""
spatial_routing.py — Sub-linear k-NN neighbor lookup for CRF
============================================================
Replaces O(N²) dense similarity + topk with spatial grid hashing
plus bounded candidate scoring: O(N · m · d) where m ≪ N.

Cells already carry 2D positions; we hash them into grid bins and
only score neighbors in a local (2r+1)² window before top-k selection.
Implementation is fully vectorized with torch (sort + searchsorted
bin lookup, batched candidate scoring). Rows whose window contains
fewer than k candidates are refilled exactly from the dense fallback.

Fallbacks:
  - N ≤ fallback_threshold       → dense topk (original O(N²) path)
  - window pool < k candidates   → dense topk refill for those rows
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


def _grid_keys(positions: torch.Tensor, grid_size: float) -> torch.Tensor:
    """Map N×2 positions to integer grid cell coordinates."""
    return (positions / grid_size).floor().long()


def _encode_keys(keys: torch.Tensor, mult: int) -> torch.Tensor:
    """Encode N×2 grid keys as single scalars (kx*mult + ky), collision-free."""
    return keys[:, 0] * mult + keys[:, 1]


def _window_offsets(radius: int, mult: int, device: torch.device) -> torch.Tensor:
    """Encode (2r+1)² grid-window offsets as scalar deltas."""
    offs = []
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            offs.append(dx * mult + dy)
    return torch.tensor(offs, dtype=torch.long, device=device)


def _candidate_matrix(
    keys: torch.Tensor,
    radius: int,
    max_candidates: int,
    n: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build per-cell candidate index matrix (N × max_c), padded with -1.

    Vectorized via:
      - sort points by grid key, group into bins (unique_consecutive)
      - padded bin→points matrix (K × max_bin)
      - per-cell window-bin lookup with searchsorted, gather, cap, drop self
    """
    mult = max(1, int(keys[:, 1].max() - keys[:, 1].min()) + 1)
    scalars = _encode_keys(keys, mult)
    perm = torch.argsort(scalars, stable=True)
    sorted_scalars = scalars[perm]
    unique_keys, inv, counts = torch.unique_consecutive(
        sorted_scalars, return_inverse=True, return_counts=True
    )
    K = unique_keys.size(0)
    bin_start = torch.cumsum(counts, dim=0) - counts

    max_bin = min(int(counts.max().item()), max_candidates)
    max_bin = max(1, max_bin)
    bin_points = torch.full((K, max_bin), -1, dtype=torch.long, device=device)
    local = torch.arange(n, device=device) - bin_start[inv]
    keep = (local < counts[inv]) & (local < max_bin)
    bin_points[inv[keep], local[keep]] = perm[keep]

    cell_scalars = _encode_keys(keys, mult)                       # N
    offsets = _window_offsets(radius, mult, device)               # W
    win_scalars = cell_scalars.unsqueeze(1) + offsets.unsqueeze(0)  # N×W

    idx = torch.searchsorted(unique_keys, win_scalars.reshape(-1)).reshape(n, -1)
    hit = (idx < K) & (
        unique_keys[idx.clamp(max=K - 1)].reshape(n, -1) == win_scalars
    )
    idx = idx.clamp(max=K - 1)

    pool = bin_points[idx]                                         # N×W×max_bin
    pool = pool.masked_fill(~hit.unsqueeze(-1), -1)
    pool = pool.reshape(n, -1)[:, :max_candidates]

    # Remove self (query cell) from candidates.
    self_mask = pool == torch.arange(n, device=device).unsqueeze(1)
    pool = pool.masked_fill(self_mask, -1)

    counts_cell = (pool >= 0).sum(dim=1)
    return pool, counts_cell


def _score_candidates(
    states: torch.Tensor,
    positions: torch.Tensor,
    padded: torch.Tensor,
    spatial_lambda: float,
    use_spatial: bool,
    batch_id: Optional[torch.Tensor],
) -> torch.Tensor:
    """Batched similarity scoring of candidate matrix (N × max_c) → N × max_c."""
    n = states.size(0)
    max_c = padded.size(1)
    if max_c == 0:
        return padded.float()

    s_norm = F.normalize(states, dim=-1)
    src = padded.clamp(min=0)
    sim = (s_norm.unsqueeze(1) * s_norm[src]).sum(-1)             # N×max_c
    if use_spatial:
        dist = (positions[src] - positions.unsqueeze(1)).norm(dim=-1)
        sim = sim - spatial_lambda * dist

    valid = padded >= 0
    sim = sim.masked_fill(~valid, float('-inf'))

    if batch_id is not None:
        cross = batch_id[src] != batch_id.unsqueeze(1)
        sim = sim.masked_fill(cross, float('-inf'))
    return sim


def _sublinear_topk(
    states: torch.Tensor,
    positions: torch.Tensor,
    k: int,
    spatial_lambda: float,
    use_spatial: bool,
    grid_size: float,
    search_radius: int,
    max_candidates: int,
    batch_id: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Grid-routing k-NN (works only for N > fallback_threshold)."""
    N = states.size(0)
    device = states.device
    keys = _grid_keys(positions, grid_size)
    padded, _ = _candidate_matrix(keys, search_radius, max_candidates, N, device)

    if padded.size(1) == 0:
        return _dense_topk(
            states, positions, k, spatial_lambda, use_spatial, batch_id,
        )

    sim = _score_candidates(states, positions, padded, spatial_lambda,
                            use_spatial, batch_id)
    top_scores, top_idx_c = sim.topk(k, dim=-1)

    # Map padded columns back to real state indices.
    row = torch.arange(N, device=device).unsqueeze(1).expand_as(top_idx_c)
    top_idx = padded[row, top_idx_c]

    # Rows with any missing neighbor (-1 or -inf) → exact dense refill.
    missing = (top_idx < 0) | (top_scores == float('-inf'))
    rows_missing = missing.any(dim=-1)
    if rows_missing.any():
        dense_idx, dense_scores = _dense_topk(
            states, positions, k, spatial_lambda, use_spatial, batch_id,
        )
        top_idx[rows_missing] = dense_idx[rows_missing]
        top_scores[rows_missing] = dense_scores[rows_missing]

    return top_idx, top_scores


# Measured route decisions, keyed by (N-bucket, d, grid, radius, max_candidates).
# On CPU, dense matmul (MKL) beats grid routing at small N; grid routing wins
# at large N. We time both once per bucket on real data and cache the winner.
_route_cache = {}


def _pick_route(
    N: int,
    d: int,
    states: torch.Tensor,
    positions: torch.Tensor,
    k: int,
    spatial_lambda: float,
    use_spatial: bool,
    grid_size: float,
    search_radius: int,
    max_candidates: int,
    batch_id: Optional[torch.Tensor],
) -> str:
    import time

    key = (N // 64, d, grid_size, search_radius, max_candidates)
    mode = _route_cache.get(key)
    if mode is not None:
        return mode

    with torch.no_grad():
        # warmup both paths (MKL thread init skews first-timings)
        _dense_topk(states, positions, k, spatial_lambda, use_spatial, batch_id)
        _sublinear_topk(states, positions, k, spatial_lambda, use_spatial,
                        grid_size, search_radius, max_candidates, batch_id)
        td, ts = 1e9, 1e9
        for _ in range(3):
            t0 = time.perf_counter()
            _dense_topk(states, positions, k, spatial_lambda, use_spatial, batch_id)
            td = min(td, time.perf_counter() - t0)
            t0 = time.perf_counter()
            _sublinear_topk(states, positions, k, spatial_lambda, use_spatial,
                            grid_size, search_radius, max_candidates, batch_id)
            ts = min(ts, time.perf_counter() - t0)

    # bias toward exact dense: sublinear must be clearly faster to be chosen
    mode = 'sublinear' if ts < td * 0.9 else 'dense'
    _route_cache[key] = mode
    return mode


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
    For N ≤ fallback_threshold always dense; above that the faster route is
    measured once per (N-bucket, d, grid config) and cached, so the wall-clock
    winner is used on any hardware.
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

    route = _pick_route(N, states.size(1), states, positions, k,
                        spatial_lambda, use_spatial, grid_size, search_radius,
                        max_candidates, batch_id)
    if route == 'dense':
        return _dense_topk(
            states, positions, k, spatial_lambda, use_spatial, batch_id,
        )
    return _sublinear_topk(
        states, positions, k, spatial_lambda, use_spatial, grid_size,
        search_radius, max_candidates, batch_id,
    )


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

    keys = _grid_keys(positions, grid_size)
    padded, _ = _candidate_matrix(keys, search_radius, max_candidates=4096, n=N,
                                  device=states.device)
    if padded.size(1) == 0:
        return []

    src = padded.clamp(min=0)
    sim = (s_norm.unsqueeze(1) * s_norm[src]).sum(-1)

    # Valid pairs: j > i, at least one of the pair is a non-anchor,
    # and similarity above threshold.
    i_grid = torch.arange(N, device=states.device).unsqueeze(1).expand_as(padded)
    valid = (
        (padded > i_grid)
        & (sim >= tau)
        & ~(anchor_mask.unsqueeze(1) & anchor_mask[src])
    )
    vals = sim[valid]
    if vals.numel() == 0:
        return []
    ii = i_grid[valid]
    jj = padded[valid]

    order = vals.argsort(descending=True)[:max_pairs]
    return [
        (ii[t].item(), jj[t].item(), vals[t].item())
        for t in order.tolist()
    ]


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
    search_radius: int = 1,
) -> int:
    """
    Analytical FLOP count for one sub-linear routing step.

    Uses the real candidate-pool size (max_candidates) plus grid-build
    overhead (sort + searchsorted) so the number is an honest model of
    the grid path actually executed.
    """
    m = max(1, min(max_candidates, N - 1))
    W = (2 * search_radius + 1) ** 2
    logn = max(1.0, math.log2(max(2, N)))
    grid_build = N * logn + N * W * logn   # sort + window searchsorted
    per_cell = (
        m * d +           # candidate dot products
        m +               # spatial distance penalty
        k * 2 * d +       # routing gate pairs
        k * d             # aggregation
    )
    return int(N * per_cell + grid_build)
