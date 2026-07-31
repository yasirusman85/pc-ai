"""
quick_real_test.py — Quick test on Shakespeare data
================================================
Single epoch test to get actual speed measurements.
"""

import time
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from crf_truly_batched import TrulyBatchedCRFLanguageModel
from transformer import GPT
from shakespeare_dataset import get_shakespeare_datasets
from data import make_dataloader


def single_forward_pass(model, x, device):
    """Measure single forward pass time."""
    model.eval()
    x = x.to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(x)
    
    # Time forward pass
    start = time.time()
    with torch.no_grad():
        for _ in range(10):
            _ = model(x)
    elapsed = (time.time() - start) / 10
    
    return elapsed


def measure_batch_scaling(model, device):
    """Measure batch scaling efficiency."""
    batch_sizes = [1, 2, 4, 8, 16]
    times = []
    
    for batch_size in batch_sizes:
        x = torch.randint(0, 99, (batch_size, 32))
        elapsed = single_forward_pass(model, x, device)
        times.append(elapsed)
        print(f"  Batch {batch_size:2d}: {elapsed*1000:.3f} ms")
    
    # Calculate efficiency
    single_time = times[0]
    efficiencies = []
    for i, (batch_size, elapsed) in enumerate(zip(batch_sizes, times)):
        speedup = single_time / elapsed
        efficiency = speedup / batch_size
        efficiencies.append(efficiency)
        print(f"    Efficiency: {efficiency:.2f}x")
    
    return times, efficiencies


def main():
    """Run quick real test."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load a small subset of Shakespeare data
    print("Loading Shakespeare dataset (small subset)...")
    train_ds, _ = get_shakespeare_datasets(seq_len=64)
    
    # Use only first 100 samples for quick test
    from torch.utils.data import Subset
    train_ds_subset = Subset(train_ds, range(100))
    
    train_loader = make_dataloader(train_ds_subset, batch_size=16)
    
    # Configuration
    vocab_size = 99
    d_model = 64
    
    results = []
    
    # Test 1: Truly Batched CRF
    print("\n" + "="*80)
    print("TEST 1: Truly Batched CRF")
    print("="*80)
    crf_model = TrulyBatchedCRFLanguageModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=4,
        n_layers=2,
    ).to(device)
    
    n_params = sum(p.numel() for p in crf_model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")
    
    print("\nBatch scaling:")
    crf_times, crf_efficiencies = measure_batch_scaling(crf_model, device)
    
    # Quick training test (1 epoch)
    print("\nQuick training test (1 epoch):")
    optimizer = AdamW(crf_model.parameters(), lr=3e-4)
    crf_model.train()
    
    start = time.time()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, loss = crf_model(x, targets=y)
        loss.backward()
        optimizer.step()
    train_time = time.time() - start
    
    print(f"  Training time: {train_time:.2f}s")
    
    results.append({
        'model_name': 'Truly Batched CRF',
        'n_params': n_params,
        'batch_times': crf_times,
        'batch_efficiencies': crf_efficiencies,
        'train_time': train_time,
    })
    
    # Test 2: Transformer baseline
    print("\n" + "="*80)
    print("TEST 2: Transformer Baseline")
    print("="*80)
    transformer_model = GPT(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=4,
        n_layers=2,
        max_seq_len=64,
    ).to(device)
    
    n_params = sum(p.numel() for p in transformer_model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")
    
    print("\nBatch scaling:")
    transformer_times, transformer_efficiencies = measure_batch_scaling(transformer_model, device)
    
    # Quick training test (1 epoch)
    print("\nQuick training test (1 epoch):")
    optimizer = AdamW(transformer_model.parameters(), lr=3e-4)
    transformer_model.train()
    
    start = time.time()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, loss = transformer_model(x, targets=y)
        loss.backward()
        optimizer.step()
    train_time = time.time() - start
    
    print(f"  Training time: {train_time:.2f}s")
    
    results.append({
        'model_name': 'Transformer',
        'n_params': n_params,
        'batch_times': transformer_times,
        'batch_efficiencies': transformer_efficiencies,
        'train_time': train_time,
    })
    
    # Compare results
    print("\n" + "="*80)
    print("RESULTS COMPARISON")
    print("="*80)
    
    crf_res = results[0]
    transformer_res = results[1]
    
    print(f"\nParameter count:")
    print(f"  CRF: {crf_res['n_params']:,}")
    print(f"  Transformer: {transformer_res['n_params']:,}")
    print(f"  Ratio: {crf_res['n_params'] / transformer_res['n_params']:.2f}x")
    
    print(f"\nTraining time (1 epoch):")
    print(f"  CRF: {crf_res['train_time']:.2f}s")
    print(f"  Transformer: {transformer_res['train_time']:.2f}s")
    print(f"  Ratio: {crf_res['train_time'] / transformer_res['train_time']:.2f}x")
    
    print(f"\nBatch scaling efficiency (batch 16):")
    print(f"  CRF: {crf_res['batch_efficiencies'][-1]:.2f}x")
    print(f"  Transformer: {transformer_res['batch_efficiencies'][-1]:.2f}x")
    
    # Save results
    output_file = Path("quick_real_test_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()