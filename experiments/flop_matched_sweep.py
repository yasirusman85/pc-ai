"""
flop_matched_sweep.py — Phase 2: FLOP-matched CRF vs Transformer at d=128/256
==============================================================================
Runs hypothesis sweep with --tr-match=flop at scale sizes where the
acc/FLOP collapse was observed under param matching.

Usage (do not run without GPU/time budget):
  python experiments/flop_matched_sweep.py
  python experiments/flop_matched_sweep.py --d=128 --budgets=50000,100000
  python experiments/flop_matched_sweep.py --d=128,256 --datasets=synthetic,arithmetic,gsm8k
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from repo root or experiments/
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src' / 'crf_reasoning'))

from hypothesis import (
    SweepConfig,
    run_sweep,
    print_summary,
    RESULTS_DIR,
    SIZE_PRESETS,
)
from metrics import estimate_crf_flops, evaluate_crf_flops


FLOP_SWEEP_PRESETS = {
    'd128': SweepConfig(
        d_models=[128],
        budgets=[50_000, 100_000, 250_000],
        seeds=[0, 1, 2],
        datasets=['synthetic', 'arithmetic', 'gsm8k'],
        tr_match='flop',
        seq_len=32,
        batch_size=8,
        max_train=2000,
        max_val=200,
    ),
    'd256': SweepConfig(
        d_models=[256],
        budgets=[100_000, 250_000, 500_000],
        seeds=[0, 1, 2],
        datasets=['synthetic', 'arithmetic', 'gsm8k'],
        tr_match='flop',
        seq_len=64,
        batch_size=4,
        max_train=2000,
        max_val=200,
    ),
    'both': SweepConfig(
        d_models=[128, 256],
        budgets=[50_000, 100_000, 250_000],
        seeds=[0, 1, 2],
        datasets=['synthetic', 'arithmetic', 'gsm8k'],
        tr_match='flop',
        seq_len=32,
        batch_size=8,
        max_train=2000,
        max_val=200,
    ),
}


def print_flop_estimates(d_models: list):
    """Show analytical FLOP comparison: sub-linear CRF vs dense (old) routing."""
    print('\n--- Analytical FLOP estimates (sub-linear routing) ---')
    for d in d_models:
        p = SIZE_PRESETS.get(d, SIZE_PRESETS[64])
        N, k, S = p['max_cells'], p['k'], p['n_steps']
        d_h = d // 2
        sub = estimate_crf_flops(N, d, d_h, k, S, use_sublinear=True)
        dense = estimate_crf_flops(N, d, d_h, k, S, use_sublinear=False)
        ratio = dense / max(1, sub)
        print(f"  d={d} N={N} k={k} S={S}: sublinear={sub:,} dense={dense:,} "
              f"speedup={ratio:.1f}x")


def parse_args():
    p = argparse.ArgumentParser(description='Phase 2: FLOP-matched sweep at d=128/256')
    p.add_argument('--preset', choices=['d128', 'd256', 'both'], default='both')
    p.add_argument('--d', type=str, help='Override d_models (comma-separated)')
    p.add_argument('--budgets', type=str)
    p.add_argument('--seeds', type=str)
    p.add_argument('--datasets', type=str)
    p.add_argument('--out', type=str, default=None,
                   help='Results JSON (default: results/flop_matched_sweep_results.json)')
    p.add_argument('--dry-run', action='store_true',
                   help='Print config and FLOP estimates only; do not train')
    return p.parse_args()


def main():
    args = parse_args()
    cfg = FLOP_SWEEP_PRESETS[args.preset]
    if args.d:
        cfg.d_models = [int(x) for x in args.d.split(',')]
    if args.budgets:
        cfg.budgets = [int(x) for x in args.budgets.split(',')]
    if args.seeds:
        cfg.seeds = [int(x) for x in args.seeds.split(',')]
    if args.datasets:
        cfg.datasets = [x.strip() for x in args.datasets.split(',')]

    out_path = Path(args.out) if args.out else RESULTS_DIR / 'flop_matched_sweep_results.json'

    print('=' * 80)
    print('PHASE 2: FLOP-MATCHED SWEEP (d=128/256)')
    print('=' * 80)
    print(f'  d_models:  {cfg.d_models}')
    print(f'  budgets:   {cfg.budgets}')
    print(f'  seeds:     {cfg.seeds}')
    print(f'  datasets:  {cfg.datasets}')
    print(f'  tr_match:  {cfg.tr_match}')
    print(f'  output:    {out_path}')
    print_flop_estimates(cfg.d_models)

    if args.dry_run:
        n = len(cfg.d_models) * len(cfg.budgets) * len(cfg.seeds) * len(cfg.datasets)
        print(f'\n[dry-run] Would run {n} experiments. Exiting without training.')
        return

    # Override results path for this sweep
    import hypothesis as hyp_mod
    hyp_mod.RESULTS_PATH = out_path
    hyp_mod.SUMMARY_PATH = out_path.with_name(out_path.stem + '_summary.json')

    results = run_sweep(cfg)
    summary = print_summary(results)
    with open(hyp_mod.SUMMARY_PATH, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f'\nSummary saved to {hyp_mod.SUMMARY_PATH}')


if __name__ == '__main__':
    main()
