"""
ablations_sweep.py — mechanism isolation at scale
==================================================
Runs every CRF ablation variant against the full CRF and a param-matched
Transformer at d=128 (largest size in the hypothesis sweep). Answers:

  "Which mechanism does the work — sparsity, dynamism, or specialization?"

Variants (from ablations.py):
  full           — all mechanisms
  no_split_death — fixed population (no birth/death)
  no_merge       — no consolidation
  no_routing     — uniform message aggregation (no learned gate)
  no_energy      — constant energy, random lifecycle
  no_messaging   — independent cells (no communication)
  no_spatial     — pure cosine routing

Usage:
  python experiments/ablations_sweep.py --d=128 --budget=50000 --datasets=synthetic,arithmetic
"""

import sys, os, json, argparse, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'crf_reasoning'))

import torch
from ablations import ModelConfig, make_crf, make_transformer
from hypothesis import SIZE_PRESETS, DEVICE, train_budget
from data import get_datasets, make_dataloader, CharTokenizer

ABLATION_ORDER = [
    'full', 'no_split_death', 'no_merge', 'no_routing',
    'no_energy', 'no_messaging', 'no_spatial',
]


def run_one(mcfg, ablation, dataset, budget, seed, seq_len=32, batch_size=8,
            max_train=2000, max_val=200):
    torch.manual_seed(seed)
    tok = CharTokenizer()
    tr_ds, va_ds = get_datasets(dataset, seq_len=seq_len, max_train=max_train,
                                max_val=max_val, tokenizer=tok)
    tr_dl = make_dataloader(tr_ds, batch_size=batch_size)
    va_dl = make_dataloader(va_ds, batch_size=batch_size, shuffle=False)
    if ablation == 'transformer':
        model = make_transformer(mcfg, target_params=None)
        mtype = 'transformer'
    else:
        model = make_crf(mcfg, ablation)
        mtype = 'crf'
    name = f'{ablation}-{dataset[:3]}-s{seed}'
    run = train_budget(model, mtype, tr_dl, va_dl, budget_tokens=budget,
                       run_name=name, verbose=False)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    best_acc = max(s.val_acc for s in run.snapshots)
    final_ppl = run.snapshots[-1].val_ppl
    return {'ablation': ablation, 'dataset': dataset, 'seed': seed,
            'budget': budget, 'n_params': n_params,
            'best_acc': round(best_acc, 4), 'final_ppl': round(final_ppl, 4),
            'time_s': round(run.total_time_s, 1)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--d', type=int, default=128)
    p.add_argument('--budget', type=int, default=50000)
    p.add_argument('--seeds', type=str, default='0')
    p.add_argument('--datasets', type=str, default='synthetic,arithmetic')
    p.add_argument('--out', type=str,
                   default=str(Path(__file__).parent.parent / 'results' / 'ablations_scale_results.json'))
    args = p.parse_args()

    presets = SIZE_PRESETS[args.d]
    mcfg = ModelConfig(vocab_size=99, d_model=presets['d_model'],
                       d_hidden=presets['d_model'] // 2,
                       n_init_cells=presets['n_init'], max_cells=presets['max_cells'],
                       n_crf_steps=presets['n_steps'], k_neighbors=presets['k'],
                       max_seq_len=32)
    seeds = [int(x) for x in args.seeds.split(',')]
    datasets = [x for x in args.datasets.split(',')]
    variants = ABLATION_ORDER + ['transformer']

    out_path = Path(args.out)
    done = {}
    if out_path.exists():
        done = {(r['ablation'], r['dataset'], r['seed']): r
                for r in json.load(open(out_path))}
        print(f'[resume] {len(done)} completed')

    results = []
    for ds in datasets:
        for seed in seeds:
            for variant in variants:
                key = (variant, ds, seed)
                if key in done:
                    results.append(done[key])
                    continue
                t0 = time.time()
                r = run_one(mcfg, variant, ds, args.budget, seed)
                r['time_s'] = round(time.time() - t0, 1)
                results.append(r)
                done[key] = r
                print(f"  [{variant:14s} {ds:10s} s{seed}] "
                      f"acc={r['best_acc']:.4f} ppl={r['final_ppl']:.2f} "
                      f"params={r['n_params']:,} ({r['time_s']:.0f}s)")
                json.dump(results, open(out_path, 'w'), indent=2, default=str)

    # Console table
    print('\n' + '=' * 78)
    print(f'ABLATION SCALE SWEEP — d={args.d}, budget={args.budget:,}')
    print('=' * 78)
    print(f"{'variant':<16} {'dataset':<12} {'best_acc':>9} {'final_ppl':>9} {'params':>9}")
    for ds in datasets:
        base = next(r for r in results if r['ablation'] == 'full' and r['dataset'] == ds
                    and r['seed'] == seeds[0])
        tr = next(r for r in results if r['ablation'] == 'transformer' and r['dataset'] == ds
                  and r['seed'] == seeds[0])
        print(f"{'--':<16} {ds:<12} {'---':>9} {'---':>9} {'---':>9}")
        for r in [x for x in results if x['dataset'] == ds and x['seed'] == seeds[0]]:
            delta = ''
            if r['ablation'] != 'transformer':
                d = (r['best_acc'] - base['best_acc']) * 100
                delta = f" ({d:+.2f})"
            print(f"{r['ablation']:<16} {ds:<12} {r['best_acc']:>9.4f} "
                  f"{r['final_ppl']:>9.2f} {r['n_params']:>9,}{delta}")
        print(f"{'transformer':<16} {ds:<12} {tr['best_acc']:>9.4f} "
              f"{tr['final_ppl']:>9.2f} {tr['n_params']:>9,}")
    print('=' * 78)
    print(f'Saved {out_path}')


if __name__ == '__main__':
    main()
