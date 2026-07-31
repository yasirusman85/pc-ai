"""
Unit tests for data loading and tokenization
"""

import pytest
import torch
from data import (
    CharTokenizer, SyntheticDataset, ArithmeticDataset,
    ChainOfThoughtDataset, CodeCompletionDataset, ARCProxyDataset,
    get_datasets, make_dataloader
)
from crf_vectorized import CRFLanguageModel


class TestCharTokenizer:
    """Test character-level tokenizer."""
    
    @pytest.fixture
    def tokenizer(self):
        """Create a test tokenizer."""
        return CharTokenizer()
    
    def test_vocab_size(self, tokenizer):
        """Test vocabulary size is correct."""
        assert tokenizer.vocab_size == 99  # 95 printable + 4 special
    
    def test_encode_decode_roundtrip(self, tokenizer):
        """Test encode-decode roundtrip preserves text."""
        text = "Hello, World!"
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        
        assert decoded == text
    
    def test_encode_unknown_chars(self, tokenizer):
        """Test unknown characters are handled."""
        # Use characters outside printable ASCII
        text = "Hello 世界"
        encoded = tokenizer.encode(text)
        
        # Should have some UNK tokens
        assert tokenizer.UNK_ID in encoded
    
    def test_special_tokens(self, tokenizer):
        """Test special token IDs."""
        assert tokenizer.PAD_ID == 0
        assert tokenizer.BOS_ID == 1
        assert tokenizer.EOS_ID == 2
        assert tokenizer.UNK_ID == 3


class TestSyntheticDataset:
    """Test synthetic dataset generation."""
    
    @pytest.fixture
    def dataset(self):
        """Create a test dataset."""
        return SyntheticDataset(n_stories=10, seq_len=32, seed=42)
    
    def test_dataset_length(self, dataset):
        """Test dataset has expected length."""
        assert len(dataset) > 0
    
    def test_item_shape(self, dataset):
        """Test dataset items have correct shape."""
        x, y = dataset[0]
        
        assert x.shape == (32,)
        assert y.shape == (32,)
    
    def test_dataloader_compatibility(self, dataset):
        """Test dataset works with DataLoader."""
        loader = make_dataloader(dataset, batch_size=4, shuffle=False)
        batch = next(iter(loader))
        
        x, y = batch
        assert x.shape[0] == 4
        assert x.shape == y.shape


class TestArithmeticDataset:
    """Test arithmetic dataset."""
    
    @pytest.fixture
    def dataset(self):
        """Create a test dataset."""
        return ArithmeticDataset(n_samples=50, seq_len=32, seed=42)
    
    def test_dataset_length(self, dataset):
        """Test dataset has expected length."""
        assert len(dataset) == 50
    
    def test_item_contains_arithmetic(self, dataset):
        """Test items contain arithmetic content."""
        x, y = dataset[0]
        tokenizer = CharTokenizer()
        text = tokenizer.decode(x.tolist())
        
        # Should contain arithmetic-related content
        assert "Q:" in text or "A:" in text or any(c.isdigit() for c in text)
    
    def test_difficulty_levels(self):
        """Test different difficulty levels produce different content."""
        easy_ds = ArithmeticDataset(n_samples=20, seq_len=32, difficulty='easy', seed=42)
        hard_ds = ArithmeticDataset(n_samples=20, seq_len=64, difficulty='hard', seed=42)
        
        # Hard examples should be longer on average
        easy_len = sum(len(x) for x, _ in easy_ds) / len(easy_ds)
        hard_len = sum(len(x) for x, _ in hard_ds) / len(hard_ds)
        
        assert hard_len >= easy_len


class TestChainOfThoughtDataset:
    """Test chain-of-thought dataset."""
    
    @pytest.fixture
    def dataset(self):
        """Create a test dataset."""
        return ChainOfThoughtDataset(n_samples=50, seq_len=64, seed=42)
    
    def test_dataset_length(self, dataset):
        """Test dataset has expected length."""
        assert len(dataset) == 50
    
    def test_difficulty_split(self, dataset):
        """Test difficulty split functionality."""
        easy_indices, hard_indices = dataset.difficulty_split(easy_max_steps=2)
        
        # Should have both easy and hard examples
        assert len(easy_indices) > 0
        assert len(hard_indices) > 0
        # Should be disjoint
        assert set(easy_indices).isdisjoint(set(hard_indices))
    
    def test_contains_chain_steps(self, dataset):
        """Test examples contain step-by-step reasoning."""
        x, y = dataset[0]
        tokenizer = CharTokenizer()
        text = tokenizer.decode(x.tolist())
        
        # Should contain step indicators
        assert "Step" in text or "Q:" in text


class TestCodeCompletionDataset:
    """Test code completion dataset."""
    
    @pytest.fixture
    def dataset(self):
        """Create a test dataset."""
        return CodeCompletionDataset(n_samples=50, seq_len=64, seed=42)
    
    def test_dataset_length(self, dataset):
        """Test dataset has expected length."""
        assert len(dataset) == 50
    
    def test_contains_code_structure(self, dataset):
        """Test examples contain code-like structure."""
        x, y = dataset[0]
        tokenizer = CharTokenizer()
        text = tokenizer.decode(x.tolist())
        
        # Should contain code-related keywords
        assert "def" in text or "return" in text or any(c in text for c in "()[]{}")


class TestARCProxyDataset:
    """Test ARC proxy dataset."""
    
    @pytest.fixture
    def dataset(self):
        """Create a test dataset."""
        return ARCProxyDataset(n_samples=50, grid_size=4, seq_len=64, seed=42)
    
    def test_dataset_length(self, dataset):
        """Test dataset has expected length."""
        assert len(dataset) == 50
    
    def test_contains_grid_content(self, dataset):
        """Test examples contain grid-like content."""
        x, y = dataset[0]
        tokenizer = CharTokenizer()
        text = tokenizer.decode(x.tolist())
        
        # Should contain grid-related content
        assert "Input:" in text or "Output:" in text


class TestDatasetFactory:
    """Test dataset factory function."""
    
    def test_synthetic_dataset(self):
        """Test synthetic dataset creation."""
        train_ds, val_ds = get_datasets('synthetic', seq_len=32, max_train=100, max_val=20)
        
        assert len(train_ds) > 0
        assert len(val_ds) > 0
    
    def test_arithmetic_dataset(self):
        """Test arithmetic dataset creation."""
        train_ds, val_ds = get_datasets('arithmetic', seq_len=32, max_train=100, max_val=20)
        
        assert len(train_ds) > 0
        assert len(val_ds) > 0
    
    def test_arc_dataset(self):
        """Test ARC dataset creation."""
        train_ds, val_ds = get_datasets('arc', seq_len=32, max_train=100, max_val=20)
        
        assert len(train_ds) > 0
        assert len(val_ds) > 0
    
    def test_invalid_dataset_name(self):
        """Test invalid dataset name raises error."""
        with pytest.raises(Exception):
            get_datasets('invalid_dataset', seq_len=32, max_train=100, max_val=20)


class TestDataloader:
    """Test dataloader creation and usage."""
    
    def test_batch_creation(self):
        """Test dataloader creates correct batches."""
        dataset = SyntheticDataset(n_stories=20, seq_len=32, seed=42)
        loader = make_dataloader(dataset, batch_size=4, shuffle=False)
        
        batch = next(iter(loader))
        x, y = batch
        
        assert x.shape == (4, 32)
        assert y.shape == (4, 32)
    
    def test_shuffle_functionality(self):
        """Test shuffle parameter affects order."""
        dataset = SyntheticDataset(n_stories=20, seq_len=32, seed=42)
        
        loader_shuffled = make_dataloader(dataset, batch_size=4, shuffle=True, seed=42)
        loader_not_shuffled = make_dataloader(dataset, batch_size=4, shuffle=False)
        
        # Different shuffling should produce different sequences
        batch_shuffled = next(iter(loader_shuffled))
        batch_not_shuffled = next(iter(loader_not_shuffled))
        
        # At least one batch should be different
        assert not torch.equal(batch_shuffled[0], batch_not_shuffled[0])
    
    def test_multiple_epochs(self):
        """Test dataloader can iterate multiple epochs."""
        dataset = SyntheticDataset(n_stories=10, seq_len=32, seed=42)
        loader = make_dataloader(dataset, batch_size=4, shuffle=False)
        
        # Should be able to iterate multiple times
        for epoch in range(3):
            for batch in loader:
                x, y = batch
                assert x.shape == y.shape
                assert x.shape[1] == 32 and x.shape[0] <= 4


class TestIntegration:
    """Integration tests for data pipeline."""
    
    def test_full_pipeline(self):
        """Test complete data pipeline from dataset to model."""
        from crf_vectorized import CRFLanguageModel
        
        # Create dataset
        train_ds, val_ds = get_datasets('synthetic', seq_len=32, max_train=50, max_val=10)
        train_loader = make_dataloader(train_ds, batch_size=4)
        
        # Create model
        model = CRFLanguageModel(
            vocab_size=99,
            d_model=32,
            d_hidden=16,
            n_init_cells=8,
            max_cells=16,
            n_crf_steps=2,
            k_neighbors=2,
        )
        
        # Training step
        x, y = next(iter(train_loader))
        logits, loss, metrics = model(x, targets=y)
        
        assert logits.shape == (4, 32, 99)
        assert loss is not None
        loss.backward()
    
    def test_tokenizer_model_compatibility(self):
        """Test tokenizer works correctly with model."""
        tokenizer = CharTokenizer()
        model = CRFLanguageModel(
            vocab_size=tokenizer.vocab_size,
            d_model=32,
            d_hidden=16,
            n_init_cells=8,
            max_cells=16,
            n_crf_steps=2,
            k_neighbors=2,
        )
        
        # Encode text
        text = "Hello world"
        encoded = tokenizer.encode(text)
        
        # Pad to sequence length
        if len(encoded) < 32:
            encoded = encoded + [tokenizer.PAD_ID] * (32 - len(encoded))
        
        # Create tensor
        x = torch.tensor([encoded[:32]])
        
        # Forward pass
        logits, loss, _ = model(x)
        
        assert logits.shape == (1, 32, tokenizer.vocab_size)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])