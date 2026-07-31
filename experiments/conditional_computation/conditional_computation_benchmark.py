"""
conditional_computation_benchmark.py — Benchmark conditional computation vs Transformer
===================================================================================
Test the fundamental advantage: compute efficiency through conditional activation.

Hypothesis: Conditional CRF achieves similar accuracy with 10-20x less compute than Transformer
because only a fraction of cells activate per input.
"""

import time
import json
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW

from conditional_crf import ConditionalCRFLanguageModel
from transformer import GPT
from shakespeare_dataset import get_shakespeare_datasets
from data import make_dataloader


def measure_forward_compute(model, x, device):
    """
    Measure compute time and estimate FLOPs for forward pass.
    """
    model.eval()
    x = x.to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            if hasattr(model, 'crf'):
                _ = model(x)
            else:
                _ = model(x)
    
    # Time forward pass
    start = time.time()
    with torch.no_grad():
        for _ in range(10):
            if hasattr(model, 'crf'):
                result = model(x)
                if len(result) == 3:
                    logits, _, stats = result
                else:
                    logits, stats = result
            else:
                logits = model(x)
                stats = None
    elapsed = (time.time() - start) / 10
    
    return elapsed, stats


def train_epoch_conditional(model, train_loader, optimizer, device):
    """Train one epoch with conditional CRF."""
    model.train()
    total_loss = 0.0
    total_compute_efficiency = 0.0
    correct = 0
    total = 0
    
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        
        logits, loss, stats = model(x, targets=y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_compute_efficiency += stats['compute_efficiency']
        
        predictions = logits.argmax(dim=-1)
        correct += (predictions == y).sum().item()
        total += y.numel()
    
    avg_compute_efficiency = total_compute_efficiency / len(train_loader)
    return total_loss / len(train_loader), correct / total, avg_compute_efficiency


def train_epoch_transformer(model, train_loader, optimizer, device):
    """Train one epoch with Transformer."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits, loss = model(x, targets=y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        predictions = logits.argmax(dim=-1)
        correct += (predictions == y).sum().item()
        total += y.numel()
    
    return total_loss / len(train_loader), correct / total


def validate(model, val_loader, device):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            
            if hasattr(model, 'crf'):
                result = model(x, targets=y)
                if len(result) == 3:
                    logits, loss, _ = result
                else:
                    logits, loss = result
            else:
                logits, loss = model(x, targets=y)
            
            total_loss += loss.item()
            
            predictions = logits.argmax(dim=-1)
            correct += (predictions == y).sum().item()
            total += y.numel()
    
    return total_loss / len(val_loader), correct / total


def run_benchmark():
    """
    Run benchmark comparing conditional CRF vs Transformer.
    
    Key metric: Compute efficiency (accuracy per unit compute)
    """
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Load Shakespeare dataset (small subset for quick test)
    print("Loading Shakespeare dataset...")
    train_ds, val_ds = get_shakespeare_datasets(seq_len=64)
    
    from torch.utils.data import Subset
    train_ds_subset = Subset(train_ds, range(200))
    val_ds_subset = Subset(val_ds, range(50))
    
    train_loader = make_dataloader(train_ds_subset, batch_size=16)
    val_loader = make_dataloader(val_ds_subset, batch_size=16, shuffle=False)
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Configuration
    vocab_size = 99
    d_model = 64
    n_epochs = 3
    
    results = []
    
    # Test 1: Conditional CRF (10% activation)
    print("\n" + "="*80)
    print("TEST 1: Conditional CRF (10% activation)")
    print("="*80)
    
    crf_model = ConditionalCRFLanguageModel(
        vocab_size=vocab_size,
        d_model=d_model,
        d_hidden=32,
        n_cells=100,
        activation_ratio=0.1,  # Only 10% of cells activate
    ).to(device)
    
    n_params = sum(p.numel() for p in crf_model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")
    
    # Measure compute efficiency
    x = torch.randint(0, 99, (4, 32))
    forward_time, stats = measure_forward_compute(crf_model, x, device)
    print(f"Forward time: {forward_time*1000:.3f} ms")
    print(f"Compute efficiency: {stats['compute_efficiency']:.1f}x")
    print(f"Active cells: {stats['n_active_cells']}/{stats['total_cells']}")
    
    # Training
    optimizer = AdamW(crf_model.parameters(), lr=3e-4)
    history = {'compute_efficiency': []}
    
    start_time = time.time()
    for epoch in range(n_epochs):
        train_loss, train_acc, avg_efficiency = train_epoch_conditional(crf_model, train_loader, optimizer, device)
        val_loss, val_acc = validate(crf_model, val_loader, device)
        
        history['compute_efficiency'].append(avg_efficiency)
        
        print(f"Epoch {epoch+1}/{n_epochs}: "
              f"Train Loss={train_loss:.4f}, Val Acc={val_acc:.3f}, "
              f"Compute Efficiency={avg_efficiency:.1f}x")
    
    training_time = time.time() - start_time
    
    results.append({
        'model_name': 'Conditional CRF (10%)',
        'n_params': n_params,
        'forward_time': forward_time,
        'compute_efficiency': stats['compute_efficiency'],
        'training_time': training_time,
        'final_val_accuracy': val_acc,
        'history': history,
    })
    
    # Test 2: Conditional CRF (5% activation - more aggressive)
    print("\n" + "="*80)
    print("TEST 2: Conditional CRF (5% activation)")
    print("="*80)
    
    crf_model_5 = ConditionalCRFLanguageModel(
        vocab_size=vocab_size,
        d_model=d_model,
        d_hidden=32,
        n_cells=100,
        activation_ratio=0.05,  # Only 5% of cells activate
    ).to(device)
    
    n_params = sum(p.numel() for p in crf_model_5.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")
    
    # Measure compute efficiency
    forward_time, stats = measure_forward_compute(crf_model_5, x, device)
    print(f"Forward time: {forward_time*1000:.3f} ms")
    print(f"Compute efficiency: {stats['compute_efficiency']:.1f}x")
    print(f"Active cells: {stats['n_active_cells']}/{stats['total_cells']}")
    
    # Training
    optimizer = AdamW(crf_model_5.parameters(), lr=3e-4)
    history = {'compute_efficiency': []}
    
    start_time = time.time()
    for epoch in range(n_epochs):
        train_loss, train_acc, avg_efficiency = train_epoch_conditional(crf_model_5, train_loader, optimizer, device)
        val_loss, val_acc = validate(crf_model_5, val_loader, device)
        
        history['compute_efficiency'].append(avg_efficiency)
        
        print(f"Epoch {epoch+1}/{n_epochs}: "
              f"Train Loss={train_loss:.4f}, Val Acc={val_acc:.3f}, "
              f"Compute Efficiency={avg_efficiency:.1f}x")
    
    training_time = time.time() - start_time
    
    results.append({
        'model_name': 'Conditional CRF (5%)',
        'n_params': n_params,
        'forward_time': forward_time,
        'compute_efficiency': stats['compute_efficiency'],
        'training_time': training_time,
        'final_val_accuracy': val_acc,
        'history': history,
    })
    
    # Test 3: Transformer baseline
    print("\n" + "="*80)
    print("TEST 3: Transformer Baseline")
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
    
    # Measure compute efficiency (baseline: no conditional computation)
    forward_time, _ = measure_forward_compute(transformer_model, x, device)
    print(f"Forward time: {forward_time*1000:.3f} ms")
    print(f"Compute efficiency: 1.0x (no conditional computation)")
    
    # Training
    optimizer = AdamW(transformer_model.parameters(), lr=3e-4)
    
    start_time = time.time()
    for epoch in range(n_epochs):
        train_loss, train_acc = train_epoch_transformer(transformer_model, train_loader, optimizer, device)
        val_loss, val_acc = validate(transformer_model, val_loader, device)
        
        print(f"Epoch {epoch+1}/{n_epochs}: "
              f"Train Loss={train_loss:.4f}, Val Acc={val_acc:.3f}")
    
    training_time = time.time() - start_time
    
    results.append({
        'model_name': 'Transformer',
        'n_params': n_params,
        'forward_time': forward_time,
        'compute_efficiency': 1.0,
        'training_time': training_time,
        'final_val_accuracy': val_acc,
    })
    
    # Compare results
    print("\n" + "="*80)
    print("CONDITIONAL COMPUTATION RESULTS")
    print("="*80)
    
    crf_10 = results[0]
    crf_5 = results[1]
    transformer = results[2]
    
    print(f"\n{'Model':<25} {'Params':>10} {'Comp Eff':>10} {'Time':>10} {'Accuracy':>10}")
    print("-" * 70)
    for result in results:
        print(f"{result['model_name']:<25} {result['n_params']:>10,} "
              f"{result['compute_efficiency']:>9.1f}x {result['training_time']:>9.2f}s "
              f"{result['final_val_accuracy']:>9.3f}")
    
    # Calculate compute advantage
    print(f"\nCompute Advantage vs Transformer:")
    print(f"  CRF (10%): {crf_10['compute_efficiency']:.1f}x efficiency")
    print(f"  CRF (5%): {crf_5['compute_efficiency']:.1f}x efficiency")
    
    # Calculate accuracy per compute
    print(f"\nAccuracy per Compute:")
    crf_10_efficiency = crf_10['final_val_accuracy'] / crf_10['training_time']
    crf_5_efficiency = crf_5['final_val_accuracy'] / crf_5['training_time']
    transformer_efficiency = transformer['final_val_accuracy'] / transformer['training_time']
    
    print(f"  CRF (10%): {crf_10_efficiency:.4f} acc/s")
    print(f"  CRF (5%): {crf_5_efficiency:.4f} acc/s")
    print(f"  Transformer: {transformer_efficiency:.4f} acc/s")
    print(f"  CRF (10%) advantage: {crf_10_efficiency / transformer_efficiency:.2f}x")
    print(f"  CRF (5%) advantage: {crf_5_efficiency / transformer_efficiency:.2f}x")
    
    # Save results
    output_file = Path("conditional_computation_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    return results


if __name__ == "__main__":
    results = run_benchmark()