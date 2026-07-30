"""
real_datasets.py — Integration with real benchmark datasets
==========================================================
Provides access to:
  - HumanEval (code generation)
  - GSM8K (math reasoning)
  - ARC-AGI (abstract reasoning)
  - TinyStories (language modeling)
"""

import os
import json
import random
from typing import List, Tuple, Optional, Dict
from pathlib import Path

import torch
from torch.utils.data import Dataset

from data import CharTokenizer


class HumanEvalDataset(Dataset):
    """
    HumanEval dataset for code generation.
    Downloads from official HuggingFace source if available.
    """
    
    def __init__(
        self,
        split: str = 'test',
        seq_len: int = 256,
        tokenizer: Optional[CharTokenizer] = None,
        cache_dir: str = '/tmp/humaneval_cache',
    ):
        self.seq_len = seq_len
        self.tokenizer = tokenizer or CharTokenizer()
        self.vocab_size = self.tokenizer.vocab_size
        
        self._chunks = self._load(split, cache_dir)
    
    def _load(self, split: str, cache_dir: str) -> List[torch.Tensor]:
        """Load HumanEval data."""
        try:
            from datasets import load_dataset
            
            ds = load_dataset(
                'openai_humaneval',
                split=split,
                cache_dir=cache_dir,
            )
            
            chunks = []
            for item in ds:
                # Format: function signature + docstring + body
                prompt = item['prompt']
                canonical_solution = item['canonical_solution']
                
                # Create completion task
                text = prompt + canonical_solution
                
                ids = self.tokenizer.encode(text)
                for start in range(0, len(ids) - self.seq_len, self.seq_len // 2):
                    chunk = ids[start: start + self.seq_len + 1]
                    if len(chunk) == self.seq_len + 1:
                        chunks.append(torch.tensor(chunk, dtype=torch.long))
            
            if chunks:
                print(f"[data] Loaded {len(chunks)} chunks from HumanEval/{split}")
                return chunks
                
        except Exception as e:
            print(f"[data] HumanEval unavailable ({e}), using synthetic code data")
        
        # Fallback to synthetic code data
        from data import CodeCompletionDataset
        fallback = CodeCompletionDataset(
            n_samples=200,
            seq_len=self.seq_len,
            seed=42,
            tokenizer=self.tokenizer,
        )
        return fallback.chunks
    
    def __len__(self):
        return len(self._chunks)
    
    def __getitem__(self, idx):
        chunk = self._chunks[idx]
        return chunk[:-1], chunk[1:]


class GSM8KDataset(Dataset):
    """
    GSM8K dataset for mathematical reasoning.
    Uses chain-of-thought format.
    """
    
    def __init__(
        self,
        split: str = 'train',
        seq_len: int = 256,
        tokenizer: Optional[CharTokenizer] = None,
        cache_dir: str = '/tmp/gsm8k_cache',
        use_chain_of_thought: bool = True,
    ):
        self.seq_len = seq_len
        self.tokenizer = tokenizer or CharTokenizer()
        self.vocab_size = self.tokenizer.vocab_size
        self.use_cot = use_chain_of_thought
        
        self._chunks = self._load(split, cache_dir)
    
    def _load(self, split: str, cache_dir: str) -> List[torch.Tensor]:
        """Load GSM8K data."""
        try:
            from datasets import load_dataset
            
            ds = load_dataset(
                'gsm8k',
                'main',
                split=split,
                cache_dir=cache_dir,
            )
            
            chunks = []
            for item in ds:
                question = item['question']
                answer = item['answer']
                
                if self.use_cot:
                    # Use chain-of-thought format
                    text = f"Q: {question} {answer}"
                else:
                    # Extract just the final answer
                    text = f"Q: {question} A: {answer.split('####')[-1].strip()}"
                
                ids = self.tokenizer.encode(text)
                if len(ids) < self.seq_len + 1:
                    ids += [self.tokenizer.PAD_ID] * (self.seq_len + 1 - len(ids))
                ids = ids[:self.seq_len + 1]
                chunks.append(torch.tensor(ids, dtype=torch.long))
            
            if chunks:
                print(f"[data] Loaded {len(chunks)} chunks from GSM8K/{split}")
                return chunks
                
        except Exception as e:
            print(f"[data] GSM8K unavailable ({e}), using synthetic arithmetic data")
        
        # Fallback to synthetic arithmetic
        from data import ArithmeticDataset
        fallback = ArithmeticDataset(
            n_samples=200,
            seq_len=self.seq_len,
            seed=42,
            difficulty='mixed',
            tokenizer=self.tokenizer,
        )
        return fallback.chunks
    
    def __len__(self):
        return len(self._chunks)
    
    def __getitem__(self, idx):
        chunk = self._chunks[idx]
        return chunk[:-1], chunk[1:]


class ARCAGIDataset(Dataset):
    """
    ARC-AGI dataset for abstract reasoning.
    Encodes grid puzzles as sequences.
    """
    
    def __init__(
        self,
        split: str = 'train',
        seq_len: int = 512,
        tokenizer: Optional[CharTokenizer] = None,
        cache_dir: str = '/tmp/arc_cache',
        grid_encoding: str = 'flatten',  # 'flatten' or 'rle'
    ):
        self.seq_len = seq_len
        self.tokenizer = tokenizer or CharTokenizer()
        self.vocab_size = self.tokenizer.vocab_size
        self.grid_encoding = grid_encoding
        
        self._chunks = self._load(split, cache_dir)
    
    def _encode_grid(self, grid: List[List[int]]) -> str:
        """Encode a grid as text."""
        if self.grid_encoding == 'flatten':
            # Simple flatten encoding
            return ' '.join(str(cell) for row in grid for cell in row)
        elif self.grid_encoding == 'rle':
            # Run-length encoding
            flat = [str(cell) for row in grid for cell in row]
            encoded = []
            i = 0
            while i < len(flat):
                current = flat[i]
                count = 1
                while i + count < len(flat) and flat[i + count] == current:
                    count += 1
                encoded.append(f"{current}:{count}")
                i += count
            return ' '.join(encoded)
        else:
            return ' '.join(str(cell) for row in grid for cell in row)
    
    def _load(self, split: str, cache_dir: str) -> List[torch.Tensor]:
        """Load ARC-AGI data."""
        try:
            from datasets import load_dataset
            
            ds = load_dataset(
                'jaymody/arc',
                split=split,
                cache_dir=cache_dir,
            )
            
            chunks = []
            for item in ds:
                # Encode input-output pairs
                input_grid = item['input']
                output_grid = item['output']
                
                input_text = self._encode_grid(input_grid)
                output_text = self._encode_grid(output_grid)
                
                text = f"Input: {input_text} Output: {output_text}"
                
                ids = self.tokenizer.encode(text)
                if len(ids) < self.seq_len + 1:
                    ids += [self.tokenizer.PAD_ID] * (self.seq_len + 1 - len(ids))
                ids = ids[:self.seq_len + 1]
                chunks.append(torch.tensor(ids, dtype=torch.long))
            
            if chunks:
                print(f"[data] Loaded {len(chunks)} chunks from ARC-AGI/{split}")
                return chunks
                
        except Exception as e:
            print(f"[data] ARC-AGI unavailable ({e}), using synthetic ARC data")
        
        # Fallback to synthetic ARC
        from data import ARCProxyDataset
        fallback = ARCProxyDataset(
            n_samples=200,
            seq_len=self.seq_len,
            seed=42,
            tokenizer=self.tokenizer,
        )
        return fallback.chunks
    
    def __len__(self):
        return len(self._chunks)
    
    def __getitem__(self, idx):
        chunk = self._chunks[idx]
        return chunk[:-1], chunk[1:]


class RealDatasetFactory:
    """
    Factory for creating real benchmark datasets with proper fallbacks.
    """
    
    @staticmethod
    def create_dataset(
        name: str,
        split: str = 'train',
        seq_len: int = 128,
        max_samples: int = 1000,
        tokenizer: Optional[CharTokenizer] = None,
        **kwargs
    ) -> Dataset:
        """
        Create a dataset by name with automatic fallback.
        
        Args:
            name: Dataset name ('tinystories', 'humaneval', 'gsm8k', 'arc')
            split: Dataset split ('train', 'validation', 'test')
            seq_len: Sequence length
            max_samples: Maximum number of samples
            tokenizer: Custom tokenizer
            **kwargs: Additional dataset-specific arguments
            
        Returns:
            Dataset object
        """
        tokenizer = tokenizer or CharTokenizer()
        
        if name == 'tinystories':
            from data import TinyStoriesDataset
            return TinyStoriesDataset(
                split=split,
                seq_len=seq_len,
                max_rows=max_samples,
                tokenizer=tokenizer,
                **kwargs
            )
        elif name == 'humaneval':
            return HumanEvalDataset(
                split=split,
                seq_len=seq_len,
                tokenizer=tokenizer,
                **kwargs
            )
        elif name == 'gsm8k':
            return GSM8KDataset(
                split=split,
                seq_len=seq_len,
                tokenizer=tokenizer,
                **kwargs
            )
        elif name == 'arc':
            return ARCAGIDataset(
                split=split,
                seq_len=seq_len,
                tokenizer=tokenizer,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown dataset: {name}")
    
    @staticmethod
    def get_available_datasets() -> List[str]:
        """Return list of available dataset names."""
        return ['tinystories', 'humaneval', 'gsm8k', 'arc']
    
    @staticmethod
    def check_dataset_availability(name: str) -> Dict[str, bool]:
        """
        Check which datasets are available (can be downloaded).
        
        Returns:
            Dictionary mapping dataset names to availability status
        """
        availability = {}
        
        for ds_name in RealDatasetFactory.get_available_datasets():
            try:
                # Try to load a tiny sample
                ds = RealDatasetFactory.create_dataset(
                    ds_name,
                    split='train',
                    seq_len=32,
                    max_samples=10,
                )
                availability[ds_name] = len(ds) > 0
            except Exception as e:
                availability[ds_name] = False
        
        return availability


def download_all_datasets(cache_dir: str = '/tmp/datasets_cache') -> None:
    """
    Download all real datasets to cache directory.
    
    Args:
        cache_dir: Directory to cache downloaded datasets
    """
    print("Checking dataset availability...")
    availability = RealDatasetFactory.check_dataset_availability()
    
    for name, available in availability.items():
        status = "✓ Available" if available else "✗ Unavailable"
        print(f"  {name}: {status}")
    
    print(f"\nDatasets will be cached in: {cache_dir}")
    print("Note: Unavailable datasets will use synthetic fallbacks.")


if __name__ == "__main__":
    # Test dataset availability
    print("Testing real dataset integration...")
    download_all_datasets()
    
    # Test loading each dataset
    for name in RealDatasetFactory.get_available_datasets():
        print(f"\nTesting {name}...")
        try:
            ds = RealDatasetFactory.create_dataset(
                name,
                split='train',
                seq_len=64,
                max_samples=50,
            )
            print(f"  Loaded {len(ds)} samples")
            
            # Test a sample
            x, y = ds[0]
            print(f"  Sample shape: x={x.shape}, y={y.shape}")
            
        except Exception as e:
            print(f"  Error: {e}")