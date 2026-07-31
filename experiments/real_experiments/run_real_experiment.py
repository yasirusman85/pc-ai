"""
run_real_experiment.py — Run real experiment on Shakespeare dataset
================================================================
Tests CRF vs Transformer on real data to get actual efficiency measurements.
"""

import time
import json
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW

from crf_vectorized import CRFLanguageModel
from crf_truly_batched import TrulyBatchedCRFLanguageModel
from transformer import GPT
from shakespeare_dataset import get_shakespeare_datasets
from data import make_dataloader


def train_epoch(model, train_loader, optimizer, device):
    """Train one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        
        if hasattr(model, 'crf'):
            result = model(x, targets=y)
            if len(result) == 3:
                logits, loss, _ = result
            else:
                logits, loss = result
        else:
            logits, loss = model(x, targets=y)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Calculate accuracy
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


def run_experiment(model_name, model, train_loader, val_loader, device, n_epochs=5):
    """Run training experiment."""
    print(f"\n{'='*80}")
    print(f"Training {model_name}")
    print(f"{'='*80}")
    
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_accuracy': [],
        'val_accuracy': [],
        'epoch_times': [],
    }
    
    start_time = time.time()
    
    for epoch in range(n_epochs):
        epoch_start = time.time()
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, device)
        
        epoch_time = time.time() - epoch_start
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_accuracy'].append(train_acc)
        history['val_accuracy'].append(val_acc)
        history['epoch_times'].append(epoch_time)
        
        print(f"Epoch {epoch+1}/{n_epochs}: "
              f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
              f"Val Acc={val_acc:.3f}, Time={epoch_time:.2f}s")
    
    total_time = time.time() - start_time
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'model_name': model_name,
        'n_params': n_params,
        'total_time': total_time,
        'final_val_accuracy': history['val_accuracy'][-1],
        'final_val_loss': history['val_loss'][-1],
        'history': history,
    }


def main():
    """Run comparative experiment."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Shakespeare dataset
    print("Loading Shakespeare dataset...")
    train_ds, val_ds = get_shakespeare_datasets(seq_len=64)
    train_loader = make_dataloader(train_ds, batch_size=16)
    val_loader = make_dataloader(val_ds, batch_size=16, shuffle=False)
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Configuration
    vocab_size = 99
    d_model = 64
    n_epochs = 3  # Quick test
    
    results = []
    
    # Test 1: Original CRF
    print("\n" + "="*80)
    print("TEST 1: Original CRF")
    print("="*80)
    crf_model = CRFLanguageModel(
        vocab_size=vocab_size,
        d_model=d_model,
        d_hidden=32,
        n_init_cells=16,
        max_cells=32,
        n_crf_steps=4,
        k_neighbors=3,
    ).to(device)
    
    crf_result = run_experiment("Original CRF", crf_model, train_loader, val_loader, device, n_epochs)
    results.append(crf_result)
    
    # Test 2: Truly Batched CRF
    print("\n" + "="*80)
    print("TEST 2: Truly Batched CRF")
    print("="*80)
    batched_crf_model = TrulyBatchedCRFLanguageModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=4,
        n_layers=2,
    ).to(device)
    
    batched_crf_result = run_experiment("Truly Batched CRF", batched_crf_model, train_loader, val_loader, device, n_epochs)
    results.append(batched_crf_result)
    
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
    
    transformer_result = run_experiment("Transformer", transformer_model, train_loader, val_loader, device, n_epochs)
    results.append(transformer_result)
    
    # Compare results
    print("\n" + "="*80)
    print("RESULTS COMPARISON")
    print("="*80)
    
    for result in results:
        print(f"\n{result['model_name']}:")
        print(f"  Parameters: {result['n_params']:,}")
        print(f"  Total time: {result['total_time']:.2f}s")
        print(f"  Final val accuracy: {result['final_val_accuracy']:.3f}")
        print(f"  Final val loss: {result['final_val_loss']:.4f}")
    
    # Calculate efficiency ratios
    if len(results) >= 2:
        crf_res = results[0]
        transformer_res = results[2]
        
        print(f"\nEfficiency Ratios (CRF vs Transformer):")
        print(f"  Time ratio: {crf_res['total_time'] / transformer_res['total_time']:.2f}x")
        print(f"  Accuracy ratio: {crf_res['final_val_accuracy'] / transformer_res['final_val_accuracy']:.2f}x")
        print(f"  Parameter ratio: {crf_res['n_params'] / transformer_res['n_params']:.2f}x")
    
    # Save results
    output_file = Path("real_experiment_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()