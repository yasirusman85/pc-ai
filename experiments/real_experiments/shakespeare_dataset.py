"""
shakespeare_dataset.py — Real dataset wrapper for Shakespeare text
===========================================================
Uses the Tiny Shakespeare dataset (1.1M characters) as a real language modeling test.
This is a small but real dataset that's been used in many ML papers.
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path

from data import CharTokenizer


class ShakespeareDataset(Dataset):
    """
    Tiny Shakespeare dataset for language modeling.
    
    This is a real dataset (1.1M characters) that has been used in:
    - Karpathy's char-rnn (2015)
    - Many language modeling papers
    - Standard benchmark for character-level language models
    """
    
    def __init__(
        self,
        data_path: str = "data/real/input.txt",
        seq_len: int = 128,
        tokenizer = None,
    ):
        self.seq_len = seq_len
        self.tokenizer = tokenizer or CharTokenizer()
        self.vocab_size = self.tokenizer.vocab_size
        
        # Load the text
        text_path = Path(data_path)
        if not text_path.exists():
            raise FileNotFoundError(f"Shakespeare data not found at {data_path}")
        
        with open(text_path, 'r', encoding='utf-8') as f:
            self.text = f.read()
        
        print(f"[Shakespeare] Loaded {len(self.text):,} characters")
        
        # Tokenize and chunk
        self._chunk_data()
    
    def _chunk_data(self):
        """Tokenize and chunk into sequences."""
        ids = self.tokenizer.encode(self.text)
        
        self.chunks = []
        for start in range(0, len(ids) - self.seq_len, self.seq_len // 2):
            chunk = ids[start: start + self.seq_len + 1]
            if len(chunk) == self.seq_len + 1:
                self.chunks.append(torch.tensor(chunk, dtype=torch.long))
        
        print(f"[Shakespeare] Created {len(self.chunks)} training chunks")
    
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        return chunk[:-1], chunk[1:]


def get_shakespeare_datasets(
    seq_len: int = 128,
    train_ratio: float = 0.9,
    tokenizer = None,
) -> tuple:
    """
    Create train/validation splits from Shakespeare dataset.
    
    Args:
        seq_len: Sequence length
        train_ratio: Fraction of data for training
        tokenizer: Custom tokenizer (uses CharTokenizer if None)
    
    Returns:
        (train_dataset, val_dataset)
    """
    full_dataset = ShakespeareDataset(
        data_path="data/real/input.txt",
        seq_len=seq_len,
        tokenizer=tokenizer,
    )
    
    # Split into train/val
    n_train = int(len(full_dataset) * train_ratio)
    n_val = len(full_dataset) - n_train
    
    train_dataset = torch.utils.data.Subset(full_dataset, range(n_train))
    val_dataset = torch.utils.data.Subset(full_dataset, range(n_train, len(full_dataset)))
    
    print(f"[Shakespeare] Train: {n_train:,} chunks, Val: {n_val:,} chunks")
    
    return train_dataset, val_dataset


if __name__ == "__main__":
    # Test the dataset
    print("Testing Shakespeare dataset...")
    
    train_ds, val_ds = get_shakespeare_datasets(seq_len=64)
    
    print(f"Train size: {len(train_ds)}")
    print(f"Val size: {len(val_ds)}")
    
    # Sample a batch
    x, y = train_ds[0]
    print(f"Sample shape: x={x.shape}, y={y.shape}")
    print(f"Sample range: x [{x.min()}, {x.max()}], y [{y.min()}, {y.max()}]")
    
    # Create dataloader
    from data import make_dataloader
    train_loader = make_dataloader(train_ds, batch_size=32)
    
    batch_x, batch_y = next(iter(train_loader))
    print(f"Batch shape: x={batch_x.shape}, y={batch_y.shape}")
    
    print("Shakespeare dataset tested successfully!")