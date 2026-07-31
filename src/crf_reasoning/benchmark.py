"""
benchmark.py — Full experimental evaluation pipeline
=====================================================
Runs all experiments and saves results to results/results.json.

Experiments:
  1. Language modeling  (synthetic, arithmetic, arc proxy)
  2. FLOPs-matched Transformer baseline
  3. 7 ablation studies
  4. Profiling: latency, memory, FLOPs
  5. Scaling: sweep N_init and S
  6. Theoretical validation: bounded comm, adaptive compute, energy conservation
"""

import os
import sys
import json
import time
import copy
import math
import random
import argparse
from typing import Dict, List, Any

import torch
import torch.nn as nn

# local imports
from data import get_datasets, make_dataloader, CharTokenizer
from crf_vectorized import CRFLanguageModel, AblationConfig
from ablations import ABLATION_CONFIGS, ABLATION_DESCRIPTIONS, ModelConfig, make_crf, make_transformer
from train import train, cosine_lr
from metrics import (
    perplexity, aggregate_metrics, estimate_crf_flops,
    estimate_transformer_flops, measure_inference_latency,
    measure_memory_mb, validate_bounded_communication,
    validate_adaptive_computation, validate_energy_conservation,
    fit_scaling_law, compute_graph_diameter_from_states,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = torch.device('cpu')
torch.manual_seed(42)
random.seed(42)


# ─── Experiment config ───────────────────────────────────────────────────────

def get_exp_cfg(fast: bool = False) -> Dict:
    """Returns experiment hyperparameters. fast=True uses tiny sizes."""
    if fast:
        return dict(
            seq_len      = 32,
            n_train      = 400,
            n_val        = 100,
            batch_size   = 16,
            n_epochs     = 3,
            lr           = 3e-4,
            warmup_steps = 20,
            # model
            d_model      = 64,
            d_hidden     = 32,
            n_init_cells = 16,
            max_cells    = 64,
            n_crf_steps  = 4,
            k_neighbors  = 3,
            # scaling sweep
            scale_cells  = [8, 16, 32],
            scale_steps  = [2, 4, 6],
            n_seeds      = 1,
        )
    else:
        return dict(
            seq_len      = 64,
            n_train      = 1500,
            n_val        = 300,
            batch_size   = 32,
            n_epochs     = 8,
            lr           = 3e-4,
            warmup_steps = 100,
            # model
            d_model      = 128,
            d_hidden     = 64,
            n_init_cells = 32,
            max_cells    = 128,
            n_crf_steps  = 6,
            k_neighbors  = 4,
            # scaling sweep
            scale_cells  = [8, 16, 32, 64],
            scale_steps  = [2, 4, 6, 8],
            n_seeds      = 2,
        )


def model_cfg_from_exp(ec: Dict, vocab_size: int) -> ModelConfig:
    return ModelConfig(
        vocab_size   = vocab_size,
        d_model      = ec['d_model'],
        d_hidden     = ec['d_hidden'],
        n_init_cells = ec['n_init_cells'],
        max_cells    = ec['max_cells'],
        n_crf_steps  = ec['n_crf_steps'],
        k_neighbors  = ec['k_neighbors'],
        max_seq_len  = ec['seq_len'],
    )


# ─── Helpers ─────────────────────────────────────────────────────────────────

def run_one(
    model,
    model_type: str,
    tr_dl, va_dl,
    ec: Dict,
    run_name: str,
    seed: int = 42,
) -> Dict:
    """Train one model and return its history."""
    torch.manual_seed(seed)
    hist = train(
        model        = model,
        train_loader = tr_dl,
        val_loader   = va_dl,
        model_type   = model_type,
        n_epochs     = ec['n_epochs'],
        lr           = ec['lr'],
        lr_min       = ec['lr'] / 10,
        warmup_steps = ec['warmup_steps'],
        device       = DEVICE,
        run_name     = run_name,
        collect_crf_metrics = (model_type == 'crf'),
        verbose      = True,
    )
    return hist


def collect_inference_profile(model, model_type: str, va_dl, n_batches: int = 5) -> Dict:
    """Collect latency + memory stats on a few val batches."""
    model.eval()
    batches = []
    for xb, yb in va_dl:
        batches.append((xb.to(DEVICE), yb.to(DEVICE)))
        if len(batches) >= n_batches:
            break

    lat = measure_inference_latency(model, batches[0], n_warmup=1, n_trials=5)
    mem = measure_memory_mb()

    # Collect CRF metrics on a few batches
    all_mets = []
    if model_type == 'crf':
        with torch.no_grad():
            for xb, yb in batches:
                _, _, met = model(xb, yb, collect_metrics=True)
                all_mets.append(met)

    result = {**lat, 'memory_mb': mem}
    if all_mets:
        result['crf_inference_metrics'] = aggregate_metrics(all_mets)
    return result


# ─── Experiment 1: CRF vs Transformer on all datasets ────────────────────────

def exp_main_comparison(ec: Dict, results: Dict) -> None:
    print("\n" + "="*60)
    print("EXPERIMENT 1: CRF vs Transformer (main comparison)")
    print("="*60)

    tok       = CharTokenizer()
    vocab     = tok.vocab_size
    mcfg      = model_cfg_from_exp(ec, vocab)
    datasets  = ['synthetic', 'arithmetic', 'arc', 'chain_of_thought', 'code']
    results['main_comparison'] = {}

    for ds_name in datasets:
        print(f"\n  Dataset: {ds_name}")
        tr_ds, va_ds = get_datasets(ds_name, ec['seq_len'], ec['n_train'], ec['n_val'], tok)
        tr_dl = make_dataloader(tr_ds, ec['batch_size'])
        va_dl = make_dataloader(va_ds, ec['batch_size'], shuffle=False)

        ds_results = {}

        # ── CRF ──
        crf = make_crf(mcfg, 'full').to(DEVICE)
        crf_params = crf.n_params
        print(f"    CRF params: {crf_params:,}")
        hist = run_one(crf, 'crf', tr_dl, va_dl, ec, f'crf_{ds_name}')
        prof = collect_inference_profile(crf, 'crf', va_dl)
        ds_results['crf'] = {**hist, 'profile': prof}

        # ── Transformer (param-matched) ──
        tr_model = make_transformer(mcfg, target_params=crf_params).to(DEVICE)
        tr_params = sum(p.numel() for p in tr_model.parameters() if p.requires_grad)
        print(f"    Transformer params: {tr_params:,}")
        hist_t = run_one(tr_model, 'transformer', tr_dl, va_dl, ec, f'transformer_{ds_name}')
        prof_t = collect_inference_profile(tr_model, 'transformer', va_dl)
        ds_results['transformer'] = {**hist_t, 'profile': prof_t}

        ds_results['param_match_ratio'] = crf_params / max(1, tr_params)
        results['main_comparison'][ds_name] = ds_results

    # Save checkpoint
    _save(results)


# ─── Experiment 2: Ablation studies ──────────────────────────────────────────

def exp_ablations(ec: Dict, results: Dict) -> None:
    print("\n" + "="*60)
    print("EXPERIMENT 2: Ablation studies")
    print("="*60)

    tok  = CharTokenizer()
    vocab = tok.vocab_size
    mcfg  = model_cfg_from_exp(ec, vocab)

    # Run ablations on synthetic (fast) and arithmetic (reasoning)
    abl_datasets = ['synthetic', 'arithmetic']
    results['ablations'] = {}

    for ds_name in abl_datasets:
        print(f"\n  Ablation dataset: {ds_name}")
        tr_ds, va_ds = get_datasets(ds_name, ec['seq_len'], ec['n_train'], ec['n_val'], tok)
        tr_dl = make_dataloader(tr_ds, ec['batch_size'])
        va_dl = make_dataloader(va_ds, ec['batch_size'], shuffle=False)

        results['ablations'][ds_name] = {}

        for abl_name in ABLATION_CONFIGS:
            print(f"    [{abl_name}]")
            model = make_crf(mcfg, abl_name).to(DEVICE)
            hist  = run_one(model, 'crf', tr_dl, va_dl, ec,
                            f'abl_{abl_name}_{ds_name}')
            prof  = collect_inference_profile(model, 'crf', va_dl)
            results['ablations'][ds_name][abl_name] = {
                'best_val_ppl':  hist['best_val_ppl'],
                'best_val_loss': hist['best_val_loss'],
                'val_ppl':       hist['val_ppl'],
                'crf_metrics':   hist['crf_metrics'],
                'profile':       prof,
                'description':   ABLATION_DESCRIPTIONS[abl_name],
            }

    _save(results)


# ─── Experiment 3: Scaling behavior ──────────────────────────────────────────

def exp_scaling(ec: Dict, results: Dict) -> None:
    print("\n" + "="*60)
    print("EXPERIMENT 3: Scaling behavior")
    print("="*60)

    tok   = CharTokenizer()
    vocab = tok.vocab_size
    tr_ds, va_ds = get_datasets('synthetic', ec['seq_len'],
                                ec['n_train'], ec['n_val'], tok)
    tr_dl = make_dataloader(tr_ds, ec['batch_size'])
    va_dl = make_dataloader(va_ds, ec['batch_size'], shuffle=False)

    results['scaling'] = {'cells': {}, 'steps': {}, 'd_model': {}}

    # Sweep N_init (number of cells)
    print("\n  Sweeping N_init (cell count):")
    compute_vals, ppl_vals = [], []
    for n_cells in ec['scale_cells']:
        mcfg = ModelConfig(
            vocab_size=vocab, d_model=ec['d_model'], d_hidden=ec['d_hidden'],
            n_init_cells=n_cells, max_cells=n_cells*4,
            n_crf_steps=ec['n_crf_steps'], k_neighbors=ec['k_neighbors'],
            max_seq_len=ec['seq_len'],
        )
        model = make_crf(mcfg, 'full').to(DEVICE)
        hist  = run_one(model, 'crf', tr_dl, va_dl, ec, f'scale_cells_{n_cells}')
        flops = estimate_crf_flops(n_cells, ec['d_model'], ec['d_hidden'],
                                   ec['k_neighbors'], ec['n_crf_steps'])
        compute_vals.append(flops)
        ppl_vals.append(hist['best_val_ppl'])
        results['scaling']['cells'][str(n_cells)] = {
            'best_val_ppl': hist['best_val_ppl'],
            'n_params':     hist['n_params'],
            'flops':        flops,
        }
        print(f"    N={n_cells:3d} ppl={hist['best_val_ppl']:.2f} flops={flops:,}")

    results['scaling']['cells_law'] = fit_scaling_law(compute_vals, ppl_vals)

    # Sweep S (number of steps)
    print("\n  Sweeping n_crf_steps:")
    compute_vals2, ppl_vals2 = [], []
    for s in ec['scale_steps']:
        mcfg = ModelConfig(
            vocab_size=vocab, d_model=ec['d_model'], d_hidden=ec['d_hidden'],
            n_init_cells=ec['n_init_cells'], max_cells=ec['max_cells'],
            n_crf_steps=s, k_neighbors=ec['k_neighbors'],
            max_seq_len=ec['seq_len'],
        )
        model = make_crf(mcfg, 'full').to(DEVICE)
        hist  = run_one(model, 'crf', tr_dl, va_dl, ec, f'scale_steps_{s}')
        flops = estimate_crf_flops(ec['n_init_cells'], ec['d_model'],
                                   ec['d_hidden'], ec['k_neighbors'], s)
        compute_vals2.append(flops)
        ppl_vals2.append(hist['best_val_ppl'])
        results['scaling']['steps'][str(s)] = {
            'best_val_ppl': hist['best_val_ppl'],
            'n_params':     hist['n_params'],
            'flops':        flops,
        }
        print(f"    S={s:2d} ppl={hist['best_val_ppl']:.2f} flops={flops:,}")

    results['scaling']['steps_law'] = fit_scaling_law(compute_vals2, ppl_vals2)

    # Sweep d_model (Fix 5)
    print("\n  Sweeping d_model:")
    compute_vals3, ppl_vals3 = [], []
    for d in [32, 48, 64, 96, 128, 160, 192]:
        mcfg = ModelConfig(
            vocab_size=vocab, d_model=d, d_hidden=max(16, d // 2),
            n_init_cells=ec['n_init_cells'], max_cells=ec['max_cells'],
            n_crf_steps=ec['n_crf_steps'], k_neighbors=ec['k_neighbors'],
            max_seq_len=ec['seq_len'],
        )
        model = make_crf(mcfg, 'full').to(DEVICE)
        hist  = run_one(model, 'crf', tr_dl, va_dl, ec, f'scale_d_{d}')
        flops = estimate_crf_flops(ec['n_init_cells'], d, max(16, d // 2),
                                   ec['k_neighbors'], ec['n_crf_steps'])
        compute_vals3.append(flops)
        ppl_vals3.append(hist['best_val_ppl'])
        results['scaling']['d_model'][str(d)] = {
            'best_val_ppl': hist['best_val_ppl'],
            'n_params':     hist['n_params'],
            'flops':        flops,
        }
        print(f"    d={d:4d} ppl={hist['best_val_ppl']:.2f} flops={flops:,}")

    results['scaling']['d_model_law'] = fit_scaling_law(compute_vals3, ppl_vals3)
    _save(results)


# ─── Experiment 4: Theoretical property validation ───────────────────────────

def exp_theory_validation(ec: Dict, results: Dict) -> None:
    print("\n" + "="*60)
    print("EXPERIMENT 4: Theoretical property validation")
    print("="*60)

    tok  = CharTokenizer()
    vocab = tok.vocab_size
    mcfg  = model_cfg_from_exp(ec, vocab)
    model = make_crf(mcfg, 'full').to(DEVICE)

    # Load best checkpoint if available
    ckpt = os.path.join(RESULTS_DIR, 'crf_synthetic_best.pt')
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location='cpu')['state_dict'])
        print("  Loaded pre-trained checkpoint")

    tr_ds, va_ds = get_datasets('synthetic', ec['seq_len'],
                                ec['n_train'], ec['n_val'], tok)
    va_dl = make_dataloader(va_ds, 8, shuffle=False)

    # Collect metrics on entire val set
    all_mets = []
    model.eval()
    with torch.no_grad():
        for xb, yb in va_dl:
            _, _, met = model(xb.to(DEVICE), yb.to(DEVICE), collect_metrics=True)
            all_mets.append(met)

    print(f"  Collected metrics on {len(all_mets)} examples")

    # Theorem 1: Bounded communication
    t1 = validate_bounded_communication(
        all_mets,
        k_max = ec['k_neighbors'],
        N_max = ec['max_cells'],
    )
    print(f"  Theorem 1 (bounded comm): holds={t1['theorem_holds']} "
          f"violations={t1['violations']}/{t1['n_checked']}")

    # Theorem 2: Adaptive computation
    # Split val set into easy (short story) and hard (arithmetic with more ops)
    easy_ds  = get_datasets('synthetic',   ec['seq_len'], 100, 100, tok)[1]
    hard_ds  = get_datasets('arithmetic',  ec['seq_len'], 100, 100, tok)[1]
    easy_dl  = make_dataloader(easy_ds, 8, shuffle=False)
    hard_dl  = make_dataloader(hard_ds, 8, shuffle=False)

    def collect(dl):
        mets = []
        with torch.no_grad():
            for xb, yb in dl:
                _, _, m = model(xb.to(DEVICE), collect_metrics=True)
                mets.append(m)
        return mets

    easy_mets = collect(easy_dl)
    hard_mets = collect(hard_dl)
    t2 = validate_adaptive_computation(easy_mets, hard_mets)
    print(f"  Theorem 2 (adaptive compute): adaptive={t2['adaptive']} "
          f"easy_N={t2['easy_mean_N']:.1f} hard_N={t2['hard_mean_N']:.1f} "
          f"ratio={t2['ratio']:.2f}")

    # Energy conservation lemma
    t3 = validate_energy_conservation(all_mets)
    print(f"  Lemma (energy conservation): {t3['energy_lemma']}, "
          f"splits/fwd={t3['splits_per_fwd']:.2f}")

    # Graph diameter on a sample batch — use real CRF states (Fix 6)
    xb, _ = next(iter(va_dl))
    tok_in = model.token_embed(xb[:1].to(DEVICE)) + model.pos_enc[:, :xb.size(1)]
    crf_states = model.crf.get_final_anchor_states(tok_in, model.n_crf_steps)
    states = crf_states[0]  # T×d — actual CRF output
    diam   = compute_graph_diameter_from_states(states, ec['k_neighbors'])
    log_bound = math.log(states.size(0)) / math.log(max(2, ec['k_neighbors']))
    print(f"  Graph diameter: {diam} (log_k(N) ≈ {log_bound:.1f})")

    results['theory_validation'] = {
        'theorem1_bounded_comm':    t1,
        'theorem2_adaptive_compute': t2,
        'lemma_energy_conservation': t3,
        'graph_diameter':           diam,
        'log_k_n_bound':            round(log_bound, 2),
    }
    _save(results)


# ─── Experiment 5: Wall-clock and memory profiling ───────────────────────────

def exp_profiling(ec: Dict, results: Dict) -> None:
    print("\n" + "="*60)
    print("EXPERIMENT 5: Wall-clock and memory profiling")
    print("="*60)

    tok  = CharTokenizer()
    vocab = tok.vocab_size
    mcfg  = model_cfg_from_exp(ec, vocab)
    _, va_ds = get_datasets('synthetic', ec['seq_len'], 100, 100, tok)
    va_dl    = make_dataloader(va_ds, ec['batch_size'], shuffle=False)

    results['profiling'] = {}

    crf_model = make_crf(mcfg, 'full').to(DEVICE)
    tr_model  = make_transformer(mcfg, target_params=crf_model.n_params).to(DEVICE)

    for name, model, mtype in [
        ('crf',         crf_model, 'crf'),
        ('transformer', tr_model,  'transformer'),
    ]:
        print(f"\n  Profiling {name}...")
        prof = collect_inference_profile(model, mtype, va_dl, n_batches=8)

        # Also measure training step time
        xb, yb = next(iter(va_dl))
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            if mtype == 'crf':
                _, loss, _ = model(xb, yb)
            else:
                _, loss = model(xb, targets=yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            times.append((time.perf_counter() - t0) * 1000)

        train_t = torch.tensor(times)
        prof['train_step_ms_mean'] = train_t.mean().item()
        prof['train_step_ms_std']  = train_t.std().item()
        prof['n_params']           = sum(p.numel() for p in model.parameters()
                                         if p.requires_grad)

        # FLOPs estimate
        if mtype == 'crf':
            prof['estimated_flops'] = estimate_crf_flops(
                ec['n_init_cells'], ec['d_model'], ec['d_hidden'],
                ec['k_neighbors'], ec['n_crf_steps'])
        else:
            try:
                L = len(tr_model.blocks)
                prof['estimated_flops'] = estimate_transformer_flops(
                    ec['seq_len'], ec['d_model'], L)
            except Exception:
                prof['estimated_flops'] = 0

        print(f"    infer={prof['latency_ms_mean']:.1f}ms  "
              f"train_step={prof['train_step_ms_mean']:.1f}ms  "
              f"mem={prof['memory_mb']:.0f}MB  "
              f"params={prof['n_params']:,}")
        results['profiling'][name] = prof

    _save(results)


# ─── Save helper ─────────────────────────────────────────────────────────────

def _save(results: Dict) -> None:
    path = os.path.join(RESULTS_DIR, 'results.json')
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=str)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fast', action='store_true',
                        help='Use tiny model/data for quick testing')
    parser.add_argument('--exp', type=str, default='all',
                        help='Which experiment to run: all|main|ablations|scaling|theory|profiling')
    args = parser.parse_args()

    ec      = get_exp_cfg(fast=args.fast)
    results = {'config': ec, 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')}

    print(f"\nCRF Benchmark  [fast={args.fast}]")
    print(f"Config: d={ec['d_model']} N={ec['n_init_cells']} S={ec['n_crf_steps']} "
          f"k={ec['k_neighbors']} seq={ec['seq_len']} epochs={ec['n_epochs']}")

    run_all = args.exp == 'all'

    if run_all or args.exp == 'main':
        exp_main_comparison(ec, results)

    if run_all or args.exp == 'ablations':
        exp_ablations(ec, results)

    if run_all or args.exp == 'scaling':
        exp_scaling(ec, results)

    if run_all or args.exp == 'theory':
        exp_theory_validation(ec, results)

    if run_all or args.exp == 'profiling':
        exp_profiling(ec, results)

    _save(results)
    print(f"\nResults saved to {os.path.join(RESULTS_DIR, 'results.json')}")


if __name__ == '__main__':
    main()
