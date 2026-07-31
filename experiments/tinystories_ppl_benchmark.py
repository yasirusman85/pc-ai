"""
tinystories_ppl_benchmark.py — Phase 3: TinyStories perplexity vs FLOP-matched baseline
=======================================================================================
Compares CRF and FLOP-matched Transformer on TinyStories validation perplexity.
Reference GPT-2-small target (~29 PPL on TinyStories with BPE tokenizer) is
recorded for context; char-level PPL will differ numerically.

Usage:
  python experiments/tinystories_ppl_benchmark.py --dry-run
  python experiments/tinystories_ppl_benchmark.py --d-model=128 --budget=100000
  python experiments/tinystories_ppl_benchmark.py --eval-only --crf-checkpoint=path.pt
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src' / 'crf_reasoning'))

from data import get_datasets, make_dataloader, CharTokenizer
from crf_vectorized import CRFLanguageModel, AblationConfig
from ablations import ModelConfig, make_crf
from train import make_flop_matched_transformer, train
from metrics import estimate_crf_flops, estimate_transformer_flops, perplexity
from hypothesis import SIZE_PRESETS, train_budget


# Published reference (BPE tokenizer, not directly comparable to char-level)
GPT2_SMALL_TINYSTORIES_PPL_REFERENCE = 29.0


@dataclass
class TinyStoriesPPLResult:
    model_type: str
    d_model: int
    n_params: int
    per_forward_flops: int
    val_ppl: float
    val_loss: float
    val_acc: float
    tokens_trained: int
    wall_time_s: float
    tr_match: str

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class TinyStoriesBenchmarkResult:
    crf: TinyStoriesPPLResult
    transformer: TinyStoriesPPLResult
    ppl_ratio_crf_over_tr: float
    flop_ratio: float
    reference_gpt2_small_ppl: float
    crf_beats_tr: bool
    notes: str

    def to_dict(self) -> Dict:
        return {
            'crf': self.crf.to_dict(),
            'transformer': self.transformer.to_dict(),
            'ppl_ratio_crf_over_tr': self.ppl_ratio_crf_over_tr,
            'flop_ratio': self.flop_ratio,
            'reference_gpt2_small_ppl': self.reference_gpt2_small_ppl,
            'crf_beats_tr': self.crf_beats_tr,
            'notes': self.notes,
        }


@torch.no_grad()
def evaluate_ppl(model, val_loader, model_type: str, device: torch.device) -> Dict:
    model.eval()
    total_loss = 0.0
    n_correct = 0
    n_total = 0
    n_batches = 0
    for xb, yb in val_loader:
        xb, yb = xb.to(device), yb.to(device)
        if model_type == 'crf':
            logits, loss, _ = model(xb, yb)
        else:
            logits, loss = model(xb, targets=yb)
        total_loss += loss.item()
        n_correct += (logits.argmax(-1) == yb).sum().item()
        n_total += yb.numel()
        n_batches += 1
    avg_loss = total_loss / max(1, n_batches)
    return {
        'val_loss': avg_loss,
        'val_ppl': math.exp(avg_loss),
        'val_acc': n_correct / max(1, n_total),
    }


def build_models(
    d_model: int,
    seq_len: int,
    tr_match: str,
    vocab_size: int,
    device: torch.device,
):
    presets = SIZE_PRESETS.get(d_model, SIZE_PRESETS[64])
    mcfg = ModelConfig(
        vocab_size=vocab_size,
        d_model=presets['d_model'],
        d_hidden=presets['d_model'] // 2,
        n_init_cells=presets['n_init'],
        max_cells=presets['max_cells'],
        n_crf_steps=presets['n_steps'],
        k_neighbors=presets['k'],
        max_seq_len=seq_len,
    )
    crf = make_crf(mcfg, 'full')
    crf_fwd = estimate_crf_flops(
        crf.crf.max_cells, crf.crf.d_model,
        crf.crf.program.gate[0].out_features,
        crf.crf.fabric.k, crf.n_crf_steps,
        use_sublinear=True,
    )
    if tr_match == 'flop':
        tr = make_flop_matched_transformer(
            vocab_size=vocab_size,
            target_flops=crf_fwd,
            seq_len=seq_len,
            max_seq_len=seq_len,
            device=device,
        )
    else:
        from ablations import make_transformer
        tr = make_transformer(mcfg, target_params=crf.n_params)
    tr_fwd = estimate_transformer_flops(seq_len, tr.d_model, len(tr.blocks))
    return crf, tr, crf_fwd, tr_fwd


def run_benchmark(
    d_model: int = 128,
    budget: int = 100_000,
    seq_len: int = 128,
    max_train: int = 5000,
    max_val: int = 500,
    batch_size: int = 8,
    seed: int = 0,
    tr_match: str = 'flop',
    device: str = 'cpu',
    crf_checkpoint: Optional[str] = None,
    tr_checkpoint: Optional[str] = None,
    skip_training: bool = False,
) -> TinyStoriesBenchmarkResult:
    torch.manual_seed(seed)
    dev = torch.device(device)
    tok = CharTokenizer()
    train_ds, val_ds = get_datasets(
        'tinystories', seq_len=seq_len, max_train=max_train,
        max_val=max_val, tokenizer=tok, use_real=True,
    )
    train_dl = make_dataloader(train_ds, batch_size=batch_size)
    val_dl = make_dataloader(val_ds, batch_size=batch_size, shuffle=False)

    crf, tr, crf_fwd, tr_fwd = build_models(d_model, seq_len, tr_match, tok.vocab_size, dev)

    if crf_checkpoint:
        crf.load_state_dict(torch.load(crf_checkpoint, map_location=dev))
    if tr_checkpoint:
        tr.load_state_dict(torch.load(tr_checkpoint, map_location=dev))

    if not skip_training:
        crf_run = train_budget(crf, 'crf', train_dl, val_dl, budget_tokens=budget,
                               run_name=f'CRF-TinyStories-d{d_model}', verbose=True)
        tr_run = train_budget(tr, 'transformer', train_dl, val_dl, budget_tokens=budget,
                              run_name=f'TR-TinyStories-d{d_model}', verbose=True)
        crf_metrics = evaluate_ppl(crf, val_dl, 'crf', dev)
        tr_metrics = evaluate_ppl(tr, val_dl, 'transformer', dev)
        crf_tokens = crf_run.total_tokens
        tr_tokens = tr_run.total_tokens
        crf_time = crf_run.total_time_s
        tr_time = tr_run.total_time_s
    else:
        crf_metrics = evaluate_ppl(crf, val_dl, 'crf', dev)
        tr_metrics = evaluate_ppl(tr, val_dl, 'transformer', dev)
        crf_tokens = tr_tokens = 0
        crf_time = tr_time = 0.0

    crf_result = TinyStoriesPPLResult(
        model_type='crf', d_model=d_model,
        n_params=sum(p.numel() for p in crf.parameters()),
        per_forward_flops=crf_fwd,
        val_ppl=crf_metrics['val_ppl'],
        val_loss=crf_metrics['val_loss'],
        val_acc=crf_metrics['val_acc'],
        tokens_trained=crf_tokens,
        wall_time_s=crf_time,
        tr_match=tr_match,
    )
    tr_result = TinyStoriesPPLResult(
        model_type='transformer', d_model=d_model,
        n_params=sum(p.numel() for p in tr.parameters()),
        per_forward_flops=tr_fwd,
        val_ppl=tr_metrics['val_ppl'],
        val_loss=tr_metrics['val_loss'],
        val_acc=tr_metrics['val_acc'],
        tokens_trained=tr_tokens,
        wall_time_s=tr_time,
        tr_match=tr_match,
    )

    return TinyStoriesBenchmarkResult(
        crf=crf_result,
        transformer=tr_result,
        ppl_ratio_crf_over_tr=crf_result.val_ppl / max(1e-8, tr_result.val_ppl),
        flop_ratio=crf_fwd / max(1, tr_fwd),
        reference_gpt2_small_ppl=GPT2_SMALL_TINYSTORIES_PPL_REFERENCE,
        crf_beats_tr=crf_result.val_ppl < tr_result.val_ppl,
        notes=(
            'Char-level PPL; GPT-2-small reference uses BPE and is not directly '
            'comparable. Lower PPL is better. tr_match=' + tr_match
        ),
    )


def parse_args():
    p = argparse.ArgumentParser(description='Phase 3: TinyStories PPL benchmark')
    p.add_argument('--d-model', type=int, default=128)
    p.add_argument('--budget', type=int, default=100_000)
    p.add_argument('--seq-len', type=int, default=128)
    p.add_argument('--max-train', type=int, default=5000)
    p.add_argument('--max-val', type=int, default=500)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--tr-match', choices=['param', 'flop'], default='flop')
    p.add_argument('--device', default='cpu')
    p.add_argument('--crf-checkpoint', type=str, default=None)
    p.add_argument('--tr-checkpoint', type=str, default=None)
    p.add_argument('--eval-only', action='store_true',
                   help='Skip training; evaluate checkpoints or random init')
    p.add_argument('--dry-run', action='store_true',
                   help='Print config only; no train or eval')
    p.add_argument('--out', type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out) if args.out else ROOT / 'results' / 'tinystories_ppl_benchmark.json'

    print('=' * 80)
    print('PHASE 3: TinyStories Perplexity Benchmark')
    print('=' * 80)
    print(f'  d_model={args.d_model}  budget={args.budget:,}  tr_match={args.tr_match}')
    print(f'  seq_len={args.seq_len}  GPT-2-small ref PPL≈{GPT2_SMALL_TINYSTORIES_PPL_REFERENCE} (BPE)')

    if args.dry_run:
        presets = SIZE_PRESETS.get(args.d_model, SIZE_PRESETS[64])
        crf_fwd = estimate_crf_flops(
            presets['max_cells'], presets['d_model'],
            presets['d_model'] // 2, presets['k'], presets['n_steps'],
            use_sublinear=True,
        )
        print(f'  [dry-run] CRF est FLOPs/forward (sublinear): {crf_fwd:,}')
        print('  Exiting without train/eval.')
        return

    result = run_benchmark(
        d_model=args.d_model,
        budget=args.budget,
        seq_len=args.seq_len,
        max_train=args.max_train,
        max_val=args.max_val,
        batch_size=args.batch_size,
        seed=args.seed,
        tr_match=args.tr_match,
        device=args.device,
        crf_checkpoint=args.crf_checkpoint,
        tr_checkpoint=args.tr_checkpoint,
        skip_training=args.eval_only,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(result.to_dict(), f, indent=2)

    print('\n--- Results ---')
    print(f"  CRF PPL:          {result.crf.val_ppl:.2f}  ({result.crf.n_params:,} params)")
    print(f"  Transformer PPL:  {result.transformer.val_ppl:.2f}  ({result.transformer.n_params:,} params)")
    print(f"  CRF beats TR:     {result.crf_beats_tr}")
    print(f"  PPL ratio CRF/TR: {result.ppl_ratio_crf_over_tr:.3f}")
    print(f"  Saved to {out}")


if __name__ == '__main__':
    main()
