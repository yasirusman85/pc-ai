"""
Unit tests for CRF components
"""

import pytest
import torch
import numpy as np
from crf_vectorized import (
    AblationConfig, CRFMetrics, SharedCellProgram,
    VectorizedFabric, CellPopulation, VectorizedCRF, CRFLanguageModel
)


class TestAblationConfig:
    """Test AblationConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration has all mechanisms enabled."""
        config = AblationConfig()
        assert config.use_split == True
        assert config.use_death == True
        assert config.use_merge == True
        assert config.use_routing == True
        assert config.use_energy == True
        assert config.use_messaging == True
        assert config.use_spatial == True
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = AblationConfig(
            use_split=False,
            use_routing=False,
            spatial_lambda=0.1
        )
        assert config.use_split == False
        assert config.use_routing == False
        assert config.spatial_lambda == 0.1


class TestCRFMetrics:
    """Test CRFMetrics dataclass."""
    
    def test_empty_metrics(self):
        """Test metrics initialization."""
        metrics = CRFMetrics()
        assert metrics.n_cells_per_step == []
        assert metrics.n_splits == 0
        assert metrics.n_deaths == 0
        assert metrics.n_merges == 0
        assert metrics.comm_cost == 0
        assert metrics.specialization == 0.0
        assert metrics.wall_time_s == 0.0
    
    def test_metrics_with_data(self):
        """Test metrics with sample data."""
        metrics = CRFMetrics(
            n_cells_per_step=[32, 35, 40],
            n_splits=5,
            n_deaths=2,
            n_merges=1,
            comm_cost=1000,
            specialization=0.5,
            wall_time_s=0.1
        )
        assert len(metrics.n_cells_per_step) == 3
        assert metrics.n_splits == 5
        assert metrics.n_deaths == 2
        assert metrics.comm_cost == 1000


class TestSharedCellProgram:
    """Test SharedCellProgram MLP."""
    
    @pytest.fixture
    def program(self):
        """Create a test program."""
        return SharedCellProgram(d_model=64, d_hidden=32)
    
    def test_forward_shape(self, program):
        """Test forward pass produces correct shapes."""
        states = torch.randn(10, 64)
        messages = torch.randn(10, 64)
        
        new_states, out_msgs, energy_d = program(states, messages)
        
        assert new_states.shape == (10, 64)
        assert out_msgs.shape == (10, 64)
        assert energy_d.shape == (10,)
    
    def test_energy_range(self, program):
        """Test energy gate outputs are in valid range."""
        states = torch.randn(10, 64)
        messages = torch.randn(10, 64)
        
        _, _, energy_d = program(states, messages)
        
        # Energy gate should be in (0, 1) due to sigmoid
        assert torch.all(energy_d > 0) and torch.all(energy_d < 1)


class TestVectorizedFabric:
    """Test VectorizedFabric routing."""
    
    @pytest.fixture
    def fabric(self):
        """Create a test fabric."""
        return VectorizedFabric(d_model=64, k=4)
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return AblationConfig()
    
    def test_single_cell_handling(self, fabric, config):
        """Test fabric handles single cell edge case."""
        states = torch.randn(1, 64)
        positions = torch.randn(1, 2)
        
        messages = fabric(states, positions, config)
        
        assert messages.shape == (1, 64)
    
    def test_routing_shape(self, fabric, config):
        """Test routing produces correct message shape."""
        states = torch.randn(10, 64)
        positions = torch.randn(10, 2)
        
        messages = fabric(states, positions, config)
        
        assert messages.shape == (10, 64)
    
    def test_no_routing_ablation(self, fabric, config):
        """Test routing ablation disables learned gates."""
        config.use_routing = False
        states = torch.randn(10, 64)
        positions = torch.randn(10, 2)
        
        messages = fabric(states, positions, config)
        
        assert messages.shape == (10, 64)


class TestCellPopulation:
    """Test CellPopulation lifecycle operations."""
    
    @pytest.fixture
    def population(self):
        """Create a test population."""
        states = torch.randn(10, 64)
        positions = torch.randn(10, 2)
        energies = torch.ones(10)
        anchor_mask = torch.tensor([True] * 5 + [False] * 5)
        device = torch.device('cpu')
        
        return CellPopulation(states, positions, energies, anchor_mask, device)
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return AblationConfig()
    
    @pytest.fixture
    def program(self):
        """Create test program."""
        return SharedCellProgram(d_model=64, d_hidden=32)
    
    def test_initial_population_size(self, population):
        """Test population initializes correctly."""
        assert population.N == 10
        assert population.states.shape == (10, 64)
        assert population.positions.shape == (10, 2)
        assert population.energies.shape == (10,)
    
    def test_split_increases_population(self, population, config, program):
        """Test split operation increases cell count."""
        initial_n = population.N
        # Set high energy to trigger split
        population.energies[:] = 2.0
        
        population.apply_split(config, program, max_cells=20, metrics=CRFMetrics())
        
        assert population.N > initial_n
    
    def test_split_respects_max_cells(self, population, config, program):
        """Test split respects max_cells limit."""
        population.energies[:] = 2.0
        
        population.apply_split(config, program, max_cells=10, metrics=CRFMetrics())
        
        assert population.N <= 10
    
    def test_death_decreases_population(self, population, config):
        """Test death operation decreases cell count."""
        initial_n = population.N
        # Set low energy to trigger death
        population.energies[5:] = 0.0
        
        population.apply_death(config, metrics=CRFMetrics())
        
        assert population.N < initial_n
    
    def test_anchors_protected_from_death(self, population, config):
        """Test anchor cells cannot die."""
        initial_anchors = population.anchor_mask.sum().item()
        # Set all energies to zero
        population.energies[:] = 0.0
        
        population.apply_death(config, metrics=CRFMetrics())
        
        # Anchors should still be present
        assert population.anchor_mask.sum().item() == initial_anchors


class TestVectorizedCRF:
    """Test VectorizedCRF forward pass."""
    
    @pytest.fixture
    def crf(self):
        """Create a test CRF."""
        return VectorizedCRF(
            d_model=64,
            d_hidden=32,
            n_init=16,
            max_cells=32,
            k_neighbors=4,
        )
    
    def test_forward_shape(self, crf):
        """Test forward pass produces correct output shape."""
        token_states = torch.randn(1, 8, 64)
        
        out, metrics = crf(token_states, n_steps=4)
        
        assert out.shape == (1, 8, 64)
        assert isinstance(metrics, CRFMetrics)
    
    def test_output_sequence_length(self, crf):
        """Test output maintains sequence length."""
        seq_len = 10
        token_states = torch.randn(1, seq_len, 64)
        
        out, _ = crf(token_states, n_steps=4)
        
        assert out.shape[1] == seq_len
    
    def test_metrics_collection(self, crf):
        """Test metrics are collected when requested."""
        token_states = torch.randn(1, 8, 64)
        
        out, metrics = crf(token_states, n_steps=4, collect_metrics=True)
        
        assert len(metrics.n_cells_per_step) > 0
        assert metrics.comm_cost > 0


class TestCRFLanguageModel:
    """Test CRFLanguageModel."""
    
    @pytest.fixture
    def model(self):
        """Create a test language model."""
        return CRFLanguageModel(
            vocab_size=99,
            d_model=64,
            d_hidden=32,
            n_init_cells=16,
            max_cells=32,
            n_crf_steps=4,
            k_neighbors=4,
        )
    
    def test_forward_shape(self, model):
        """Test forward pass produces correct shapes."""
        x = torch.randint(0, 99, (2, 10))
        
        logits, loss, metrics = model(x, targets=x, collect_metrics=True)
        
        assert logits.shape == (2, 10, 99)
        assert loss is not None
        assert isinstance(metrics, CRFMetrics)
    def test_forward_without_targets(self, model):
        """Test forward pass without targets."""
        x = torch.randint(0, 99, (2, 10))
        
        logits, loss, metrics = model(x)
        
        assert logits.shape == (2, 10, 99)
        assert loss is None
    
    def test_generation(self, model):
        """Test text generation."""
        x = torch.randint(0, 99, (1, 5))
        
        generated = model.generate(x, max_new=10, temperature=0.8)
        
        assert generated.shape == (1, 15)  # 5 + 10 new tokens
    
    def test_parameter_count(self, model):
        """Test parameter count is accessible."""
        n_params = model.n_params
        assert n_params > 0


class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_small_training_step(self):
        """Test a small training step works end-to-end."""
        model = CRFLanguageModel(
            vocab_size=99,
            d_model=32,
            d_hidden=16,
            n_init_cells=8,
            max_cells=16,
            n_crf_steps=2,
            k_neighbors=2,
        )
        
        x = torch.randint(0, 99, (4, 8))
        y = torch.randint(0, 99, (4, 8))
        
        logits, loss, metrics = model(x, targets=y)
        
        # Backward pass
        loss.backward()
        
        assert loss.item() > 0
        assert not torch.isnan(loss)
    
    def test_ablation_configurations(self):
        """Test all ablation configurations are valid."""
        from ablations import ABLATION_CONFIGS
        
        for name, config in ABLATION_CONFIGS.items():
            model = CRFLanguageModel(
                vocab_size=99,
                d_model=32,
                d_hidden=16,
                n_init_cells=8,
                max_cells=16,
                n_crf_steps=2,
                k_neighbors=2,
                cfg=config,
            )
            
            x = torch.randint(0, 99, (2, 8))
            logits, loss, _ = model(x, targets=x)
            
            assert logits.shape == (2, 8, 99)
            assert loss is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])