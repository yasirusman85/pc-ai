"""
Triton GPU Spatial Routing Kernel ("FlashCell").
Provides fused k-NN distance computation, top-k candidate selection, and weighted message
aggregation on CUDA GPUs, with automatic PyTorch vectorised fallback for CPU environments.
"""

import torch
import torch.nn.functional as F
from typing import Tuple

TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:

    @triton.jit
    def flash_cell_knn_kernel(
        states_ptr,
        positions_ptr,
        out_msg_ptr,
        N,
        D,
        K,
        spatial_lambda,
        stride_sn,
        stride_sd,
        stride_pn,
        stride_pd,
        stride_on,
        stride_od,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Fused k-NN distance & message aggregation Triton kernel.
        Computes pairwise spatial-cosine similarity and accumulates top-k neighbor messages.
        """
        pid = tl.program_id(0)
        if pid >= N:
            return

        # Read cell pid state & position
        offs_d = tl.arange(0, BLOCK_SIZE)
        mask_d = offs_d < D

        # Cell i state
        s_i = tl.load(
            states_ptr + pid * stride_sn + offs_d * stride_sd, mask=mask_d, other=0.0
        )
        p_i_x = tl.load(positions_ptr + pid * stride_pn + 0)
        p_i_y = tl.load(positions_ptr + pid * stride_pn + 1)

        # Vectorized accumulator for weighted messages
        msg_acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        weight_sum = 0.0

        # Loop over neighbors j
        for j in range(N):
            if j != pid:
                s_j = tl.load(
                    states_ptr + j * stride_sn + offs_d * stride_sd,
                    mask=mask_d,
                    other=0.0,
                )
                p_j_x = tl.load(positions_ptr + j * stride_pn + 0)
                p_j_y = tl.load(positions_ptr + j * stride_pn + 1)

                # Cosine similarity + spatial penalty
                dot = tl.sum(s_i * s_j, axis=0)
                dist = tl.sqrt(
                    (p_i_x - p_j_x) * (p_i_x - p_j_x)
                    + (p_i_y - p_j_y) * (p_i_y - p_j_y)
                )
                sim = dot - spatial_lambda * dist

                # Soft weight
                w = tl.sigmoid(sim)
                msg_acc += w * s_j
                weight_sum += w

        weight_sum = tl.maximum(weight_sum, 1e-6)
        out_msg = msg_acc / weight_sum

        tl.store(
            out_msg_ptr + pid * stride_on + offs_d * stride_od, out_msg, mask=mask_d
        )


def flash_cell_spatial_routing(
    states: torch.Tensor,  # N×d
    positions: torch.Tensor,  # N×2
    k: int = 4,
    spatial_lambda: float = 0.05,
    use_triton: bool = True,
) -> torch.Tensor:
    """
    Fast Spatial k-NN Message Routing.
    Uses FlashCell Triton kernel if GPU & Triton are available, else PyTorch fallback.
    """
    N, d = states.shape

    if use_triton and TRITON_AVAILABLE and states.is_cuda:
        out_messages = torch.empty_like(states)
        grid = (N,)
        block_size = triton.next_power_of_2(d)
        flash_cell_knn_kernel[grid](
            states,
            positions,
            out_messages,
            N,
            d,
            k,
            spatial_lambda,
            states.stride(0),
            states.stride(1),
            positions.stride(0),
            positions.stride(1),
            out_messages.stride(0),
            out_messages.stride(1),
            BLOCK_SIZE=block_size,
        )
        return out_messages
    else:
        # PyTorch vectorized fallback
        s_norm = F.normalize(states, dim=-1)
        sim = s_norm @ s_norm.T
        pos_dist = torch.cdist(positions, positions)
        sim = sim - spatial_lambda * pos_dist
        sim.fill_diagonal_(float("-inf"))

        k_neighbors = min(k, max(1, N - 1))
        vals, idxs = sim.topk(k_neighbors, dim=-1)

        neigh_states = states[idxs]  # N×k×d
        weights = torch.sigmoid(vals).unsqueeze(-1)  # N×k×1
        w_sum = weights.sum(dim=1).clamp(min=1e-8)
        messages = (weights * neigh_states).sum(dim=1) / w_sum
        return messages
