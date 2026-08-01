"""
Hybrid CRF Block & Language Model Architecture.
Allows interleaving Cellular Reasoning Fabric (CRF) blocks into standard transformer architectures
(e.g., Llama, SmolLM, Qwen, GPT-2).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .crf_vectorized import VectorizedCRF, AblationConfig, CRFMetrics


class HybridCRFBlock(nn.Module):
    """
    Plug-and-play reasoning block that wraps VectorizedCRF as a residual layer.
    Can be inserted into any PyTorch transformer backbone:
        output = input + CRF(LayerNorm(input))
    """

    def __init__(
        self,
        d_model: int = 256,
        d_hidden: int = 128,
        n_init_cells: int = 32,
        max_cells: int = 128,
        n_crf_steps: int = 4,
        k_neighbors: int = 3,
        cfg: Optional[AblationConfig] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_crf_steps = n_crf_steps
        self.ln = nn.LayerNorm(d_model)
        self.crf = VectorizedCRF(
            d_model=d_model,
            d_hidden=d_hidden,
            n_init=n_init_cells,
            max_cells=max_cells,
            k_neighbors=k_neighbors,
            cfg=cfg,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,  # B×T×d
        collect_metrics: bool = False,
    ) -> Tuple[torch.Tensor, CRFMetrics]:
        normed = self.ln(hidden_states)
        crf_out, metrics = self.crf(
            normed, n_steps=self.n_crf_steps, collect_metrics=collect_metrics
        )
        # Residual connection
        output = hidden_states + crf_out
        return output, metrics


class HybridTransformerCRFLM(nn.Module):
    """
    Hybrid Language Model combining standard linear/attention embedding layers
    with interleaved CRF reasoning blocks.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 4,
        crf_interval: int = 2,  # Insert CRF block every N transformer layers
        max_seq_len: int = 512,
        cfg: Optional[AblationConfig] = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            # Standard MLP layer
            self.layers.append(
                nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Linear(d_model * 4, d_model),
                )
            )
            # Interleave CRF reasoning block every crf_interval layers
            if (i + 1) % crf_interval == 0:
                self.layers.append(
                    HybridCRFBlock(d_model=d_model, d_hidden=d_model // 2, cfg=cfg)
                )

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,  # B×T
        targets: Optional[torch.Tensor] = None,
        collect_metrics: bool = False,
    ):
        B, T = x.shape
        h = self.token_embed(x) + self.pos_enc[:, :T]

        aggregated_metrics = CRFMetrics()
        for layer in self.layers:
            if isinstance(layer, HybridCRFBlock):
                h, met = layer(h, collect_metrics=collect_metrics)
                if collect_metrics and met:
                    aggregated_metrics.n_splits += met.n_splits
                    aggregated_metrics.n_merges += met.n_merges
                    aggregated_metrics.n_deaths += met.n_deaths
                    aggregated_metrics.steps_executed += met.steps_executed
            else:
                h = h + layer(h)

        logits = self.lm_head(self.ln_f(h))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))

        return logits, loss, aggregated_metrics
