"""
profile_crf.py — Profile CRF implementation to find bottlenecks
===========================================================
Uses cProfile and line profiling to identify slow operations.
"""

import cProfile
import pstats
import time
from io import StringIO
import torch
from torch.utils.data import DataLoader

from crf_vectorized import CRFLanguageModel
from shakespeare_dataset import get_shakespeare_datasets
from data import make_dataloader


def profile_crf_forward():
    """Profile CRF forward pass to find bottlenecks."""
    print("Profiling CRF forward pass...")
    
    # Create small model for profiling
    model = CRFLanguageModel(
        vocab_size=99,
        d_model=64,
        d_hidden=32,
        n_init_cells=16,
        max_cells=32,
        n_crf_steps=4,
        k_neighbors=3,
        max_seq_len=64,
    )
    model.eval()
    
    # Create small batch
    x = torch.randint(0, 99, (2, 32))
    
    # Warmup
    for _ in range(3):
        with torch.no_grad():
            _, _, _ = model(x)
    
    # Profile forward pass
    pr = cProfile.Profile()
    with torch.no_grad():
        pr.enable()
        for _ in range(10):
            _, _, _ = model(x)
        pr.disable()
    
    # Analyze results
    s = StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)  # Top 20 functions
    
    print("\n" + "="*80)
    print("TOP 20 PROFILING RESULTS")
    print("="*80)
    print(s.getvalue())
    
    # Also print by time
    s2 = StringIO()
    ps2 = pstats.Stats(pr, stream=s2).sort_stats('time')
    ps2.print_stats(15)
    
    print("\n" + "="*80)
    print("TOP 15 BY TIME")
    print("="*80)
    print(s2.getvalue())


def profile_components():
    """Profile individual components of CRF."""
    print("\nProfiling individual CRF components...")
    
    model = CRFLanguageModel(
        vocab_size=99,
        d_model=64,
        d_hidden=32,
        n_init_cells=16,
        max_cells=32,
        n_crf_steps=4,
        k_neighbors=3,
        max_seq_len=64,
    )
    model.eval()
    
    x = torch.randint(0, 99, (2, 32))
    
    # Profile each component
    times = {}
    
    # Embedding
    start = time.time()
    for _ in range(10):
        with torch.no_grad():
            emb = model.token_embed(x) + model.pos_enc[:, :x.size(1)]
    times['embedding'] = (time.time() - start) / 10
    
    # CRF forward
    start = time.time()
    for _ in range(10):
        with torch.no_grad():
            emb = model.token_embed(x) + model.pos_enc[:, :x.size(1)]
            _, _ = model.crf(emb)
    times['crf_forward'] = (time.time() - start) / 10
    
    # Full forward
    start = time.time()
    for _ in range(10):
        with torch.no_grad():
            result = model(x)
            if len(result) == 3:
                _, _, _ = result
            else:
                _, _ = result
    times['full_forward'] = (time.time() - start) / 10
    
    print("\nComponent timings (ms):")
    for component, timing in times.items():
        print(f"  {component}: {timing*1000:.3f} ms")


def profile_cell_operations():
    """Profile the cell operations specifically."""
    print("\nProfiling cell operations...")
    
    model = CRFLanguageModel(
        vocab_size=99,
        d_model=64,
        d_hidden=32,
        n_init_cells=16,
        max_cells=32,
        n_crf_steps=4,
        k_neighbors=3,
        max_seq_len=64,
    )
    model.eval()
    
    x = torch.randint(0, 99, (2, 32))
    
    # Get CRF input
    with torch.no_grad():
        emb = model.token_embed(x) + model.pos_enc[:, :x.size(1)]
    
    # Profile CRF internals
    crf_times = {}
    
    # Overall CRF
    start = time.time()
    for _ in range(10):
        with torch.no_grad():
            result = model.crf(emb, n_steps=4, collect_metrics=True)
            if len(result) == 2:
                out, metrics = result
            else:
                out = result[0]
                metrics = result[1] if len(result) > 1 else None
    crf_times['total_crf'] = (time.time() - start) / 10
    
    # Break down by step
    step_times = []
    for step in range(4):
        start = time.time()
        for _ in range(10):
            with torch.no_grad():
                result = model.crf(emb, n_steps=1, collect_metrics=False)
                if len(result) == 2:
                    out, _ = result
                else:
                    out = result[0]
        step_times.append((time.time() - start) / 10)
    
    crf_times['per_step_times'] = step_times
    
    print(f"Total CRF time: {crf_times['total_crf']*1000:.3f} ms")
    step_times_formatted = [f"{t*1000:.3f}" for t in step_times]
    print(f"Per-step times: {step_times_formatted} ms")
    print(f"Average per step: {sum(step_times)/len(step_times)*1000:.3f} ms")
    
    # Analyze metrics
    if metrics and hasattr(metrics, 'n_cells_per_step') and metrics.n_cells_per_step:
        print(f"Cell counts per step: {metrics.n_cells_per_step}")
        print(f"Communication cost: {metrics.comm_cost}")


def profile_batch_processing():
    """Profile batch processing efficiency."""
    print("\nProfiling batch processing...")
    
    model = CRFLanguageModel(
        vocab_size=99,
        d_model=64,
        d_hidden=32,
        n_init_cells=16,
        max_cells=32,
        n_crf_steps=4,
        k_neighbors=3,
        max_seq_len=64,
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
                result = model(x)
                if len(result) == 3:
                    _, _, _ = result
                else:
                    _, _ = result
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


if __name__ == "__main__":
    print("="*80)
    print("CRF PROFILING - Finding Bottlenecks")
    print("="*80)
    
    profile_crf_forward()
    profile_components()
    profile_cell_operations()
    profile_batch_processing()
    
    print("\n" + "="*80)
    print("Profiling complete")
    print("="*80)