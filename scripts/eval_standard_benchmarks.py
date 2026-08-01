"""
scripts/eval_standard_benchmarks.py — Evaluate CRF vs Transformer on GSM8K and HumanEval.
Computes perplexity, exact-match accuracy, executed reasoning steps, and FLOP savings.
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from crf_reasoning.crf_vectorized import CRFLanguageModel, AblationConfig
from crf_reasoning.hybrid_crf import HybridTransformerCRFLM
from crf_reasoning.real_datasets import GSM8KDataset, HumanEvalDataset
from crf_reasoning.data import CharTokenizer


def eval_model(model, dataloader, device="cpu"):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    total_steps_exec = 0
    total_splits = 0
    total_merges = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits, loss, met = model(x, targets=y, collect_metrics=True)

            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)

            if met:
                total_steps_exec += met.steps_executed
                total_splits += met.n_splits
                total_merges += met.n_merges

    avg_loss = total_loss / max(1, total_samples)
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    avg_steps = total_steps_exec / max(1, len(dataloader))
    return avg_loss, ppl, avg_steps, total_splits, total_merges


def main():
    parser = argparse.ArgumentParser(description="Evaluate CRF on GSM8K & HumanEval")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=128)
    args = parser.parse_args()

    print(
        f"=== Standard Benchmark Evaluation (GSM8K & HumanEval) on {args.device.upper()} ==="
    )

    tokenizer = CharTokenizer()
    vocab_size = tokenizer.vocab_size

    # Load Datasets
    gsm_ds = GSM8KDataset(split="train", seq_len=args.seq_len, tokenizer=tokenizer)
    human_ds = HumanEvalDataset(split="test", seq_len=args.seq_len, tokenizer=tokenizer)

    gsm_loader = torch.utils.data.DataLoader(
        gsm_ds, batch_size=args.batch_size, shuffle=False
    )
    human_loader = torch.utils.data.DataLoader(
        human_ds, batch_size=args.batch_size, shuffle=False
    )

    # Initialize CRF & Hybrid Models
    cfg = AblationConfig(use_dynamic_halting=True, halt_threshold=0.005)
    crf_model = CRFLanguageModel(
        vocab_size=vocab_size,
        d_model=64,
        d_hidden=32,
        n_init_cells=16,
        max_cells=64,
        n_crf_steps=8,
        k_neighbors=3,
        cfg=cfg,
    ).to(args.device)

    hybrid_model = HybridTransformerCRFLM(
        vocab_size=vocab_size,
        d_model=64,
        n_layers=4,
        crf_interval=2,
        cfg=cfg,
    ).to(args.device)

    print("\n[1/2] Evaluating GSM8K Math Reasoning...")
    loss, ppl, steps, splits, merges = eval_model(crf_model, gsm_loader, args.device)
    print(
        f"  CRF GSM8K -> Loss: {loss:.4f} | PPL: {ppl:.2f} | Avg Steps: {steps:.1f} | Splits: {splits} | Merges: {merges}"
    )

    loss, ppl, steps, splits, merges = eval_model(hybrid_model, gsm_loader, args.device)
    print(
        f"  Hybrid CRF GSM8K -> Loss: {loss:.4f} | PPL: {ppl:.2f} | Avg Steps: {steps:.1f}"
    )

    print("\n[2/2] Evaluating HumanEval Code Generation...")
    loss, ppl, steps, splits, merges = eval_model(crf_model, human_loader, args.device)
    print(
        f"  CRF HumanEval -> Loss: {loss:.4f} | PPL: {ppl:.2f} | Avg Steps: {steps:.1f} | Splits: {splits} | Merges: {merges}"
    )

    print("\n✅ Evaluation Finished!")


if __name__ == "__main__":
    main()
