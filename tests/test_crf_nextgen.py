"""
Unit tests for Next-Gen CRF features:
1. Dynamic Halting (Adaptive Computation Time)
2. Hybrid Transformer-CRF Module
3. FlashCell Triton / PyTorch fallback kernel
4. GSM8K & HumanEval evaluators
"""

import pytest
import torch
from crf_reasoning.crf_vectorized import (
    AblationConfig,
    CRFLanguageModel,
    VectorizedCRF,
)
from crf_reasoning.hybrid_crf import HybridCRFBlock, HybridTransformerCRFLM
from crf_reasoning.flash_cell_triton import flash_cell_spatial_routing
from crf_reasoning.real_datasets import GSM8KDataset, HumanEvalDataset
from crf_reasoning.data import CharTokenizer


class TestDynamicHalting:
    """Test Adaptive Computation Time (ACT) early exit."""

    def test_dynamic_halting_saves_steps(self):
        torch.manual_seed(42)
        cfg = AblationConfig(
            use_dynamic_halting=True, halt_threshold=0.5
        )  # Loose threshold to force halt
        model = CRFLanguageModel(
            vocab_size=50,
            d_model=32,
            d_hidden=16,
            n_init_cells=8,
            max_cells=64,
            n_crf_steps=8,
            cfg=cfg,
        )
        x = torch.randint(0, 50, (1, 8))
        _, _, met = model(x, targets=x, collect_metrics=True)

        assert (
            met.steps_executed < 8
        ), f"Halting failed: executed {met.steps_executed} steps"
        assert met.flop_savings_pct > 0.0, "FLOP savings should be positive"


class TestHybridCRF:
    """Test Hybrid Transformer-CRF interleave block."""

    def test_hybrid_crf_block_forward(self):
        torch.manual_seed(0)
        block = HybridCRFBlock(d_model=32, d_hidden=16, n_init_cells=8, max_cells=32)
        h = torch.randn(2, 8, 32)
        out, met = block(h, collect_metrics=True)

        assert out.shape == (2, 8, 32)
        assert met is not None

    def test_hybrid_transformer_crf_lm(self):
        torch.manual_seed(0)
        model = HybridTransformerCRFLM(
            vocab_size=100,
            d_model=32,
            n_layers=4,
            crf_interval=2,
        )
        x = torch.randint(0, 100, (2, 8))
        logits, loss, met = model(x, targets=x, collect_metrics=True)

        assert logits.shape == (2, 8, 100)
        assert loss is not None
        assert met is not None


class TestFlashCellSpatialRouting:
    """Test FlashCell spatial k-NN routing kernel & fallback."""

    def test_spatial_routing_shape(self):
        states = torch.randn(10, 16)
        positions = torch.rand(10, 2)
        messages = flash_cell_spatial_routing(states, positions, k=3, use_triton=False)

        assert messages.shape == (10, 16)


class TestRealDatasets:
    """Test GSM8K and HumanEval dataset loaders."""

    def test_gsm8k_dataset_fallback(self):
        tokenizer = CharTokenizer()
        ds = GSM8KDataset(split="train", seq_len=16, tokenizer=tokenizer)
        assert len(ds) > 0
        x, y = ds[0]
        assert x.shape == (16,)
        assert y.shape == (16,)

    def test_humaneval_dataset_fallback(self):
        tokenizer = CharTokenizer()
        ds = HumanEvalDataset(split="test", seq_len=16, tokenizer=tokenizer)
        assert len(ds) > 0
        x, y = ds[0]
        assert x.shape == (16,)
        assert y.shape == (16,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
