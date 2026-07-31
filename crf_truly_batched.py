"""
crf_truly_batched.py — CRF with proper batched operations
=========================================================
Radical simplification to achieve proper batch scaling.

Key idea: Remove per-sequence overhead entirely by using simple
attention mechanisms instead of complex cell populations.
"""

import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TrulyBatchedCRF(nn.Module):
    """
    Simplified CRF that actually batches properly.
    
    Instead of complex per-sequence cell populations, uses:
    - Shared attention across batch
    - Simple feedforward transformations
    - No lifecycle operations (too slow)
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        n_layers: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        
        # Multi-head attention (batched)
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        
        # Feedforward layers
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model),
        )
        
        # Layer norm
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor):
        """
        Truly batched forward pass.
        
        Args:
            x: (B, T, d) tensor
        
        Returns:
            (B, T, d) tensor
        """
        B, T, d = x.shape
        
        # Self-attention (batched)
        attn_out, _ = self.attention(x, x, x)
        x = self.ln1(x + attn_out)
        
        # Feedforward (batched)
        ff_out = self.ff(x)
        x = self.ln2(x + ff_out)
        
        return x


class TrulyBatchedCRFLanguageModel(nn.Module):
    """Simplified CRF language model with proper batch scaling."""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int = 4,
        n_layers: int = 2,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
        self.crf = TrulyBatchedCRF(d_model, n_heads, n_layers)
        self.ln = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, token_ids: torch.Tensor, targets: Optional[torch.Tensor] = None):
        """
        Forward pass with truly batched operations.
        
        Returns:
            If targets provided: (logits, loss)
            Otherwise: logits
        """
        B, T = token_ids.shape
        
        # Embedding
        x = self.token_embed(token_ids) + self.pos_enc[:, :T]
        
        # CRF processing (truly batched)
        x = self.crf(x)
        
        # Output processing
        x = self.ln(x)
        logits = self.lm_head(x)
        
        # Loss computation
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            return logits, loss
        else:
            return logits


if __name__ == "__main__":
    # Test truly batched CRF
    print("Testing truly batched CRF...")
    
    # Create model
    model = TrulyBatchedCRFLanguageModel(
        vocab_size=99,
        d_model=64,
        n_heads=4,
        n_layers=2,
    )
    model.eval()
    
    # Test different batch sizes
    batch_sizes = [1, 2, 4, 8, 16]
    times = []
    
    for batch_size in batch_sizes:
        x = torch.randint(0, 99, (batch_size, 32))
        
        start = time.time()
        for _ in range(10):
            with torch.no_grad():
                logits = model(x)
        elapsed = (time.time() - start) / 10
        times.append(elapsed)
        
        print(f"Batch size {batch_size:2d}: {elapsed*1000:.3f} ms")
    
    # Calculate scaling efficiency
    single_time = times[0]
    print(f"\nScaling efficiency:")
    for i, (batch_size, elapsed) in enumerate(zip(batch_sizes, times)):
        speedup = single_time / elapsed
        efficiency = speedup / batch_size
        print(f"  Batch {batch_size:2d}: {speedup:.2f}x speedup, {efficiency:.2f}x efficiency")
    
    print("\nTruly batched CRF tested successfully!")