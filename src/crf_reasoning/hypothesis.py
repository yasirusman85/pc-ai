"""
hypothesis.py — Multi-Dimensional Hypothesis Evaluation Framework
==================================================================
Tests: CRF achieves Transformer-level performance with substantially
fewer training tokens across model sizes, datasets, and random seeds.

Methodology:
  1. Full-factorial sweep over d_model, seed, dataset, token budget
  2. Param-matched CRF & Transformer at each size
  3. Token, FLOP, and wall-clock efficiency ratios
  4. Incremental save/load (survives partial runs)
  5. Aggregate table + hypothesis verdict per setting

Usage:
  python hypothesis.py                    # fast sweep (d=32,64, synth+arithmetic)
  python hypothesis.py --mode=full        # d=32,64,128; 3 seeds; 4 datasets
  python hypothesis.py --d=32,64,128,256  # custom sweep
"""

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from .data import get_datasets, make_dataloader, CharTokenizer
from .crf_vectorized import CRFLanguageModel, AblationConfig
from .ablations import ModelConfig, make_crf, make_transformer
from .metrics import estimate_crf_flops, estimate_transformer_flops, perplexity
from scripts.train import count_gpt_params

# Repo-root results dir (survives src/ package move)
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
DEVICE = torch.device("cpu")


# ─── CRF size presets ─────────────────────────────────────────────────────

SIZE_PRESETS = {
    32: dict(d_model=32, n_init=8, max_cells=32, n_steps=4, k=3),
    64: dict(d_model=64, n_init=16, max_cells=64, n_steps=4, k=3),
    128: dict(d_model=128, n_init=32, max_cells=128, n_steps=6, k=4),
    256: dict(d_model=256, n_init=48, max_cells=192, n_steps=8, k=4),
}


# ─── Sweep configuration ─────────────────────────────────────────────────


@dataclass
class SweepConfig:
    d_models: List[int] = field(default_factory=lambda: [32, 64])
    budgets: List[int] = field(default_factory=lambda: [25_000, 50_000, 100_000])
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    datasets: List[str] = field(default_factory=lambda: ["synthetic", "arithmetic"])
    tr_match: str = "param"
    seq_len: int = 32
    batch_size: int = 8
    max_train: int = 2000
    max_val: int = 200
    verbose: bool = True


# ─── Efficiency tracking ─────────────────────────────────────────────────


@dataclass
class EfficiencySnapshot:
    tokens_seen: int
    wall_time_s: float
    flops: int
    val_loss: float
    val_ppl: float
    val_acc: float
    epoch: int


@dataclass
class EfficiencyRun:
    model_type: str
    n_params: int
    config: dict
    snapshots: List[EfficiencySnapshot] = field(default_factory=list)
    total_tokens: int = 0
    total_time_s: float = 0.0
    total_flops: int = 0

    def add(self, s: EfficiencySnapshot):
        self.snapshots.append(s)
        self.total_tokens = max(self.total_tokens, s.tokens_seen)
        self.total_time_s = max(self.total_time_s, s.wall_time_s)
        self.total_flops = max(self.total_flops, s.flops)


# ─── Per-experiment result ───────────────────────────────────────────────


@dataclass
class ExperimentResult:
    d_model: int
    budget: int
    seed: int
    dataset: str
    crf_params: int
    tr_params: int
    crf_best_acc: float
    tr_best_acc: float
    crf_final_acc: float
    tr_final_acc: float
    crf_final_ppl: float
    tr_final_ppl: float
    crf_time_s: float
    tr_time_s: float
    crf_flops: int
    tr_flops: int
    # Normalized metrics (accuracy per unit of resource)
    crf_acc_per_param: float
    tr_acc_per_param: float
    crf_acc_per_flop: float
    tr_acc_per_flop: float
    # Efficiency ratios (how many x more TR needs)
    token_efficiency: float  # TR_tokens / CRF_tokens to reach TR_final*0.9
    flops_efficiency: float
    time_efficiency: float
    # Hypothesis: CRF reaches >=90% TR final acc with <=50% tokens
    hypothesis_supported: bool
    # Matching mode used for the Transformer baseline ('param' | 'flop')
    tr_match: str = "param"
    # Learning curves (tokens, acc) for plotting
    crf_curve: list = field(default_factory=list)
    tr_curve: list = field(default_factory=list)


# ─── Training with budget tracking ───────────────────────────────────────


def train_budget(
    model,
    model_type: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    budget_tokens: int,
    lr: float = 3e-4,
    warmup_steps: int = 50,
    eval_every_tokens: Optional[int] = None,
    run_name: str = "run",
    verbose: bool = True,
) -> EfficiencyRun:
    model = model.to(DEVICE)
    model.train()
    opt = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    tokens_seen = 0
    step = 0
    epoch = 0
    t_start = time.time()
    try:
        T = next(iter(train_loader))[0].size(1)
    except (StopIteration, RuntimeError):
        T = 32
    total_steps = max(1, budget_tokens // (train_loader.batch_size * T))
    warmup = min(warmup_steps, total_steps // 10)
    eval_interval = eval_every_tokens or max(1, budget_tokens // 10)
    run = EfficiencyRun(
        model_type=model_type,
        n_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
        config={},
    )

    def evaluate():
        model.eval()
        total_loss = 0.0
        n_correct = 0
        n_total = 0
        n_batches = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                if model_type == "crf":
                    logits, loss, _ = model(xb, yb)
                else:
                    logits, loss = model(xb, targets=yb)
                total_loss += loss.item()
                n_correct += (logits.argmax(-1) == yb).sum().item()
                n_total += yb.numel()
                n_batches += 1
        model.train()
        return (total_loss / n_batches, n_correct / n_total)

    while tokens_seen < budget_tokens:
        epoch += 1
        for xb, yb in train_loader:
            if tokens_seen >= budget_tokens:
                break
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            B, T = xb.shape
            tokens_here = B * T
            if model_type == "crf":
                _, loss, _ = model(xb, yb)
            else:
                _, loss = model(xb, targets=yb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step < warmup:
                lr_curr = lr * (step + 1) / warmup
            else:
                p = (step - warmup) / max(1, total_steps - warmup)
                lr_curr = lr * 0.1 + 0.5 * (lr - lr * 0.1) * (1 + math.cos(math.pi * p))
            for pg in opt.param_groups:
                pg["lr"] = lr_curr
            tokens_seen += tokens_here
            step += 1
            prev_eval = (tokens_seen // eval_interval) * eval_interval
            if tokens_seen >= eval_interval and (tokens_seen - tokens_here) < prev_eval:
                val_loss, val_acc = evaluate()
                est_flops = (
                    estimate_crf_flops(
                        model.crf.max_cells,
                        model.crf.d_model,
                        model.crf.program.gate[0].out_features,
                        model.crf.fabric.k,
                        model.n_crf_steps,
                        max_candidates=model.cfg.routing_max_candidates,
                        search_radius=model.cfg.routing_search_radius,
                    )
                    if model_type == "crf"
                    else estimate_transformer_flops(T, model.d_model, len(model.blocks))
                )
                snap = EfficiencySnapshot(
                    tokens_seen=tokens_seen,
                    wall_time_s=time.time() - t_start,
                    flops=est_flops * step,
                    val_loss=val_loss,
                    val_ppl=math.exp(val_loss),
                    val_acc=val_acc,
                    epoch=epoch,
                )
                run.add(snap)
                if verbose:
                    print(
                        f"  [{run_name}] tok={tokens_seen:>8,} val_loss={val_loss:.4f} "
                        f"val_acc={val_acc:.4f} t={snap.wall_time_s:.1f}s"
                    )
    val_loss, val_acc = evaluate()
    if model_type == "crf":
        est_flops = estimate_crf_flops(
            model.crf.max_cells,
            model.crf.d_model,
            model.crf.program.gate[0].out_features,
            model.crf.fabric.k,
            model.n_crf_steps,
            max_candidates=model.cfg.routing_max_candidates,
            search_radius=model.cfg.routing_search_radius,
        )
    else:
        est_flops = estimate_transformer_flops(T, model.d_model, len(model.blocks))
    final = EfficiencySnapshot(
        tokens_seen=tokens_seen,
        wall_time_s=time.time() - t_start,
        flops=est_flops * step,
        val_loss=val_loss,
        val_ppl=math.exp(val_loss),
        val_acc=val_acc,
        epoch=epoch,
    )
    run.add(final)
    if verbose:
        print(
            f"  [{run_name}] FINAL tok={tokens_seen:>8,} val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f} t={final.wall_time_s:.1f}s"
        )
    return run


# ─── Single experiment ───────────────────────────────────────────────────


def run_experiment(
    d_model: int,
    budget: int,
    seed: int,
    dataset: str,
    seq_len: int = 32,
    batch_size: int = 8,
    max_train: int = 2000,
    max_val: int = 200,
    tr_match: str = "param",
    verbose: bool = True,
) -> ExperimentResult:
    """Train CRF & Transformer (param- or FLOP-matched); return result."""
    presets = SIZE_PRESETS.get(d_model, SIZE_PRESETS[64])
    torch.manual_seed(seed)
    tok = CharTokenizer()
    if dataset in ("gsm8k", "humaneval", "chain_of_thought", "code"):
        seq_len = max(seq_len, 64)
    use_real = dataset in ("gsm8k", "humaneval", "arc") and not dataset == "arc"
    tr_ds, va_ds = get_datasets(
        dataset,
        seq_len=seq_len,
        max_train=max_train,
        max_val=max_val,
        tokenizer=tok,
        use_real=use_real,
    )
    tr_dl = make_dataloader(tr_ds, batch_size=batch_size)
    va_dl = make_dataloader(va_ds, batch_size=batch_size, shuffle=False)
    mcfg = ModelConfig(
        vocab_size=tok.vocab_size,
        d_model=presets["d_model"],
        d_hidden=presets["d_model"] // 2,
        n_init_cells=presets["n_init"],
        max_cells=presets["max_cells"],
        n_crf_steps=presets["n_steps"],
        k_neighbors=presets["k"],
        max_seq_len=seq_len,
    )
    crf = make_crf(mcfg, "full")
    if tr_match == "flop":
        from scripts.train import make_flop_matched_transformer

        crf_fwd = estimate_crf_flops(
            crf.crf.max_cells,
            crf.crf.d_model,
            crf.crf.program.gate[0].out_features,
            crf.crf.fabric.k,
            crf.n_crf_steps,
            max_candidates=crf.cfg.routing_max_candidates,
            search_radius=crf.cfg.routing_search_radius,
        )
        tr = make_flop_matched_transformer(
            vocab_size=tok.vocab_size,
            target_flops=crf_fwd,
            seq_len=seq_len,
            max_seq_len=seq_len,
        )
    else:
        tr = make_transformer(mcfg, target_params=crf.n_params)
    if verbose:
        print(
            f"\n  [d={d_model}, seed={seed}, dataset={dataset}, budget={budget:,}, "
            f"tr_match={tr_match}] "
            f"CRF params={crf.n_params:,} TR params={sum(p.numel() for p in tr.parameters()):,}"
        )
    name = f"CRF-d{d_model}-{dataset[:3]}-s{seed}"
    crf_run = train_budget(
        crf, "crf", tr_dl, va_dl, budget_tokens=budget, run_name=name, verbose=verbose
    )
    name = f"TR-d{d_model}-{dataset[:3]}-s{seed}"
    tr_run = train_budget(
        tr,
        "transformer",
        tr_dl,
        va_dl,
        budget_tokens=budget,
        run_name=name,
        verbose=verbose,
    )
    eff = compute_efficiency(crf_run, tr_run)
    s = eff["summary"]
    return ExperimentResult(
        d_model=d_model,
        budget=budget,
        seed=seed,
        dataset=dataset,
        crf_params=crf_run.n_params,
        tr_params=tr_run.n_params,
        crf_best_acc=s["crf_best_acc"],
        tr_best_acc=s["tr_final_acc"],
        crf_final_acc=crf_run.snapshots[-1].val_acc,
        tr_final_acc=tr_run.snapshots[-1].val_acc,
        crf_final_ppl=crf_run.snapshots[-1].val_ppl,
        tr_final_ppl=tr_run.snapshots[-1].val_ppl,
        crf_time_s=crf_run.total_time_s,
        tr_time_s=tr_run.total_time_s,
        crf_flops=crf_run.total_flops,
        tr_flops=tr_run.total_flops,
        tr_match=tr_match,
        token_efficiency=s.get("token_efficiency_ratio") or 0.0,
        flops_efficiency=s.get("flops_efficiency_ratio") or 0.0,
        time_efficiency=s.get("time_efficiency_ratio") or 0.0,
        hypothesis_supported=s.get("hypothesis_supported", False),
        crf_acc_per_param=crf_run.snapshots[-1].val_acc / max(1e-8, crf_run.n_params),
        tr_acc_per_param=tr_run.snapshots[-1].val_acc / max(1e-8, tr_run.n_params),
        crf_acc_per_flop=crf_run.snapshots[-1].val_acc
        / max(1e-8, crf_run.total_flops),
        tr_acc_per_flop=tr_run.snapshots[-1].val_acc / max(1e-8, tr_run.total_flops),
        crf_curve=[
            {
                "tokens": snap.tokens_seen,
                "acc": snap.val_acc,
                "flops": snap.flops,
                "time": snap.wall_time_s,
            }
            for snap in crf_run.snapshots
        ],
        tr_curve=[
            {
                "tokens": snap.tokens_seen,
                "acc": snap.val_acc,
                "flops": snap.flops,
                "time": snap.wall_time_s,
            }
            for snap in tr_run.snapshots
        ],
    )


# ─── Efficiency computation ──────────────────────────────────────────────


def compute_efficiency(run_crf: EfficiencyRun, run_tr: EfficiencyRun) -> Dict:
    rows = []
    for sc in run_crf.snapshots:
        nearest = min(
            run_tr.snapshots, key=lambda s: abs(s.tokens_seen - sc.tokens_seen)
        )
        ratio_tokens = nearest.tokens_seen / max(1, sc.tokens_seen)
        ratio_time = nearest.wall_time_s / max(1e-6, sc.wall_time_s)
        ratio_flops = nearest.flops / max(1, sc.flops)
        acc_gap = nearest.val_acc - sc.val_acc
        rows.append(
            {
                "tokens_seen": sc.tokens_seen,
                "crf_val_acc": round(sc.val_acc, 4),
                "crf_val_ppl": round(sc.val_ppl, 2),
                "crf_time_s": round(sc.wall_time_s, 1),
                "crf_flops": sc.flops,
                "tr_val_acc": round(nearest.val_acc, 4),
                "tr_val_ppl": round(nearest.val_ppl, 2),
                "tr_time_s": round(nearest.wall_time_s, 1),
                "tr_flops": nearest.flops,
                "acc_gap": round(acc_gap, 4),
                "tokens_efficiency": round(ratio_tokens, 1),
                "time_efficiency": round(ratio_time, 1),
                "flops_efficiency": round(ratio_flops, 1),
            }
        )
    tr_final = run_tr.snapshots[-1].val_acc
    crf_best = max(s.val_acc for s in run_crf.snapshots)
    threshold = tr_final * 0.9
    crf_at_90 = next((s for s in run_crf.snapshots if s.val_acc >= threshold), None)
    tr_at_90 = next((s for s in run_tr.snapshots if s.val_acc >= threshold), None)
    tok_eff = None
    flops_eff = None
    time_eff = None
    if crf_at_90 and tr_at_90 and crf_at_90.tokens_seen > 0:
        tok_eff = tr_at_90.tokens_seen / crf_at_90.tokens_seen
        flops_eff = tr_at_90.flops / max(1, crf_at_90.flops)
        time_eff = tr_at_90.wall_time_s / max(1e-6, crf_at_90.wall_time_s)
    hyp = (
        crf_at_90 is not None
        and crf_best >= 0.9 * tr_final
        and (tok_eff is not None and tok_eff >= 2.0)
    )
    summary = {
        "tr_final_acc": round(tr_final, 4),
        "crf_best_acc": round(crf_best, 4),
        "crf_pct_of_tr": round(crf_best / max(1e-8, tr_final) * 100, 1),
        "crf_tokens_to_90pct": crf_at_90.tokens_seen if crf_at_90 else None,
        "tr_tokens_to_90pct": tr_at_90.tokens_seen if tr_at_90 else None,
        "token_efficiency_ratio": tok_eff,
        "flops_efficiency_ratio": flops_eff,
        "time_efficiency_ratio": time_eff,
        "hypothesis_supported": hyp,
    }
    return {"profile": rows, "summary": summary}


# ─── Sweep runner with incremental save/load ─────────────────────────────

RESULTS_PATH = RESULTS_DIR / "hypothesis_sweep_results.json"
SUMMARY_PATH = RESULTS_DIR / "hypothesis_sweep_summary.json"


def experiment_key(r: ExperimentResult) -> tuple:
    return (r.d_model, r.budget, r.seed, r.dataset, r.tr_match)


def load_results() -> List[ExperimentResult]:
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            data = json.load(f)
        out = []
        for item in data:
            for k in (
                "crf_acc_per_param",
                "tr_acc_per_param",
                "crf_acc_per_flop",
                "tr_acc_per_flop",
            ):
                item.setdefault(k, 0.0)
            item.setdefault("tr_match", "param")
            item.setdefault("crf_curve", [])
            item.setdefault("tr_curve", [])
            out.append(ExperimentResult(**item))
        return out
    return []


def save_results(results: List[ExperimentResult]):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2, default=str)


def run_sweep(cfg: SweepConfig) -> List[ExperimentResult]:
    """Full factorial sweep. Skips already-computed experiments."""
    completed = load_results()
    completed_keys = {experiment_key(r) for r in completed}
    total = len(cfg.d_models) * len(cfg.budgets) * len(cfg.seeds) * len(cfg.datasets)
    run_idx = 0
    print(
        f"\nSweep: {len(cfg.d_models)} sizes x {len(cfg.budgets)} budgets x "
        f"{len(cfg.seeds)} seeds x {len(cfg.datasets)} datasets = {total} experiments"
    )
    print(f"Already completed: {len(completed)}")
    for dataset in cfg.datasets:
        for seed in cfg.seeds:
            for d_model in cfg.d_models:
                for budget in cfg.budgets:
                    key = (d_model, budget, seed, dataset, cfg.tr_match)
                    if key in completed_keys:
                        continue
                    run_idx += 1
                    print(
                        f"\n--- Experiment {run_idx}/{total - len(completed)}: "
                        f"d={d_model} budget={budget:,} seed={seed} dataset={dataset} "
                        f"tr_match={cfg.tr_match} ---"
                    )
                    try:
                        result = run_experiment(
                            d_model=d_model,
                            budget=budget,
                            seed=seed,
                            dataset=dataset,
                            seq_len=cfg.seq_len,
                            batch_size=cfg.batch_size,
                            max_train=cfg.max_train,
                            max_val=cfg.max_val,
                            tr_match=cfg.tr_match,
                            verbose=cfg.verbose,
                        )
                        completed.append(result)
                        completed_keys.add(key)
                        save_results(completed)
                        print(
                            f"  [saved] d={d_model} budget={budget:,} seed={seed} dataset={dataset}"
                        )
                    except Exception as e:
                        print(f"  [FAILED] {e}")
                        import traceback

                        traceback.print_exc()
    return completed


# ─── Summary table ───────────────────────────────────────────────────────


def print_summary(results: List[ExperimentResult]):
    if not results:
        print("No results.")
        return
    print("\n" + "=" * 130)
    print("HYPOTHESIS SWEEP - AGGREGATE RESULTS")
    print("=" * 130)
    header = (
        f"{'d':>4} {'seed':>4} {'dataset':<14} {'budget':>8} {'TR_acc':>8} {'CRF_acc':>8} "
        f"{'CRF/TR':>7} {'tok_eff':>7} {'acc/FLOP':>9} {'acc/param':>9} {'match':>5} {'hyp?':>5}"
    )
    print(header)
    print("-" * 130)
    for r in sorted(results, key=lambda x: (x.dataset, x.d_model, x.seed, x.budget)):
        ratio = r.crf_best_acc / max(1e-8, r.tr_best_acc)
        tok = f"{r.token_efficiency:.1f}x" if r.token_efficiency > 0 else "N/A"
        hyp = "Y" if r.hypothesis_supported else "N"
        af = (
            (r.crf_acc_per_flop / max(1e-12, r.tr_acc_per_flop))
            if r.tr_acc_per_flop > 0
            else 0
        )
        ap = (
            (r.crf_acc_per_param / max(1e-12, r.tr_acc_per_param))
            if r.tr_acc_per_param > 0
            else 0
        )
        print(
            f"{r.d_model:>4} {r.seed:>4} {r.dataset:<14} {r.budget:>8,} "
            f"{r.tr_best_acc:>8.4f} {r.crf_best_acc:>8.4f} {ratio:>6.2f}x "
            f"{tok:>7} {af:>8.1f}x {ap:>8.1f}x {r.tr_match:>5} {hyp:>5}"
        )
    print("-" * 130)
    n_hyp = sum(1 for r in results if r.hypothesis_supported)
    tok_effs = [r.token_efficiency for r in results if r.token_efficiency > 0]
    flop_effs = [r.flops_efficiency for r in results if r.flops_efficiency > 0]
    time_effs = [r.time_efficiency for r in results if r.time_efficiency > 0]
    afs = [
        r.crf_acc_per_flop / max(1e-12, r.tr_acc_per_flop)
        for r in results
        if r.tr_acc_per_flop > 0
    ]
    aps = [
        r.crf_acc_per_param / max(1e-12, r.tr_acc_per_param)
        for r in results
        if r.tr_acc_per_param > 0
    ]
    avg_tok = sum(tok_effs) / max(1, len(tok_effs))
    avg_flop = sum(flop_effs) / max(1, len(flop_effs))
    avg_time = sum(time_effs) / max(1, len(time_effs))
    print(
        f"\nHypothesis supported: {n_hyp}/{len(results)} settings ({n_hyp/max(1,len(results))*100:.0f}%)"
    )
    if tok_effs:
        print(
            f"Avg token efficiency:  {avg_tok:.2f}x (CRF needs 1 token vs TR's {avg_tok:.1f}x)"
        )
    if flop_effs:
        print(f"Avg FLOP efficiency:   {avg_flop:.2f}x")
    if time_effs:
        print(f"Avg time efficiency:   {avg_time:.2f}x")
    if afs:
        print(
            f"Avg acc/FLOP ratio:    {sum(afs)/len(afs):.2f}x (CRF accuracy per FLOP vs TR)"
        )
    if aps:
        print(
            f"Avg acc/param ratio:   {sum(aps)/len(aps):.2f}x (CRF accuracy per param vs TR)"
        )
    print("=" * 130)
    return {
        "n_hyp_supported": n_hyp,
        "total": len(results),
        "avg_token_efficiency": round(avg_tok, 2) if tok_effs else None,
        "avg_flops_efficiency": round(avg_flop, 2) if flop_effs else None,
        "avg_time_efficiency": round(avg_time, 2) if time_effs else None,
        "avg_acc_per_flop_ratio": round(sum(afs) / len(afs), 2) if afs else None,
        "avg_acc_per_param_ratio": round(sum(aps) / len(aps), 2) if aps else None,
        "results": [asdict(r) for r in results],
    }


# ─── Rapid adaptation test ───────────────────────────────────────────────


def test_rapid_adaptation(
    crf_model,
    tr_model,
    dataset_name="arithmetic",
    n_adapt_tokens=500,
    seq_len=24,
    batch_size=8,
):
    """Fine-tune pre-trained models on a new task with very few tokens."""
    tok = CharTokenizer()
    train_ds, val_ds = get_datasets(
        dataset_name, seq_len=seq_len, max_train=200, max_val=50, tokenizer=tok
    )
    adapt_ds = Subset(train_ds, list(range(min(10, len(train_ds)))))
    adapt_loader = make_dataloader(adapt_ds, batch_size=batch_size, shuffle=True)
    val_loader = make_dataloader(val_ds, batch_size=batch_size, shuffle=False)
    results = {}
    for name, model, mtype in [
        ("CRF", crf_model, "crf"),
        ("Transformer", tr_model, "transformer"),
    ]:
        model = model.to(DEVICE)
        model.train()
        opt = optim.AdamW(model.parameters(), lr=1e-4)
        tokens_seen = 0
        acc_before = _eval_acc(model, val_loader, mtype)
        while tokens_seen < n_adapt_tokens:
            for xb, yb in adapt_loader:
                if tokens_seen >= n_adapt_tokens:
                    break
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                if mtype == "crf":
                    _, loss, _ = model(xb, yb)
                else:
                    _, loss = model(xb, targets=yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                tokens_seen += xb.numel()
        acc_after = _eval_acc(model, val_loader, mtype)
        results[name] = {
            "acc_before": acc_before,
            "acc_after": acc_after,
            "improvement": acc_after - acc_before,
            "tokens_seen": tokens_seen,
        }
    print("\n" + "=" * 80)
    print("RAPID ADAPTATION TEST")
    print("=" * 80)
    for name, r in results.items():
        print(
            f'{name:15s}: {r["acc_before"]:.4f} -> {r["acc_after"]:.4f} '
            f'(+{r["improvement"]:.4f}) after {r["tokens_seen"]} tokens'
        )
    print("=" * 80)
    return results


def _eval_acc(model, loader, mtype):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            if mtype == "crf":
                logits, _, _ = model(xb, yb)
            else:
                logits, _ = model(xb, targets=yb)
            correct += (logits.argmax(-1) == yb).sum().item()
            total += yb.numel()
    return correct / max(1, total)


# ─── Efficiency curves ───────────────────────────────────────────────────


def plot_efficiency_curves(
    results: List[ExperimentResult], path: Optional[Path] = None
):
    """4-panel figure: acc vs tokens, acc vs FLOPs, acc vs wall-time, acc-per-FLOP."""
    path = path or (RESULTS_DIR / "fig_efficiency_curves.png")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not available")
        return
    if not results:
        return
    # One representative run per (d, dataset): use largest budget, seed 0
    rep = {}
    for r in sorted(results, key=lambda x: x.budget):
        key = (r.d_model, r.dataset)
        rep[key] = r
    rows = sorted(rep.items(), key=lambda kv: (kv[1].dataset, kv[1].d_model))
    n = len(rows)
    cols = 4
    rows_n = math.ceil(n / cols)
    fig, axes = plt.subplots(
        rows_n, cols, figsize=(5 * cols, 4 * rows_n), squeeze=False
    )
    colors = {"crf": "#1f77b4", "tr": "#d62728"}
    for i, ((d, ds), r) in enumerate(rows):
        ax_tok = axes[i // cols][i % cols]
        ax_tok.plot(
            [p["tokens"] for p in r.crf_curve],
            [p["acc"] for p in r.crf_curve],
            "-o",
            color=colors["crf"],
            label="CRF",
            ms=3,
        )
        ax_tok.plot(
            [p["tokens"] for p in r.tr_curve],
            [p["acc"] for p in r.tr_curve],
            "-o",
            color=colors["tr"],
            label="Transformer",
            ms=3,
        )
        ax_tok.set_title(f"{ds} d={d} (CRF {r.crf_params:,}p / TR {r.tr_params:,}p)")
        ax_tok.set_xlabel("tokens seen")
        ax_tok.set_ylabel("val accuracy")
        ax_tok.legend(fontsize=8)
        ax_tok.grid(alpha=0.3)
    # Hide unused axes
    for i in range(n, rows_n * cols):
        axes[i // cols][i % cols].axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"\n[plot] saved {path}")


# ─── CLI ─────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description="Hypothesis sweep: CRF vs Transformer efficiency"
    )
    p.add_argument("--mode", choices=["fast", "full"], default="fast")
    p.add_argument("--d", type=str, help="Comma-separated d_model values")
    p.add_argument("--budgets", type=str, help="Comma-separated token budgets")
    p.add_argument("--seeds", type=str, help="Comma-separated seeds")
    p.add_argument("--datasets", type=str, help="Comma-separated dataset names")
    p.add_argument(
        "--tr-match",
        choices=["param", "flop"],
        default="param",
        help="How to size the Transformer baseline (param=matched params, "
        "flop=matched per-forward FLOPs)",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output results JSON path (default: results/hypothesis_sweep_results.json)",
    )
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-train", type=int, default=2000)
    p.add_argument("--max-val", type=int, default=200)
    p.add_argument("--verbose", action="store_true", default=True)
    p.add_argument("--no-verbose", dest="verbose", action="false")
    p.add_argument("--plot", action="store_true", help="Generate efficiency curves")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.out:
        RESULTS_PATH = Path(args.out)
        SUMMARY_PATH = RESULTS_PATH.with_name(RESULTS_PATH.stem + "_summary.json")
    if args.mode == "full":
        cfg = SweepConfig(
            d_models=[32, 64, 128],
            budgets=[50_000, 100_000, 250_000],
            seeds=[0, 1, 2],
            datasets=["synthetic", "arithmetic", "arc"],
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            max_train=args.max_train,
            max_val=args.max_val,
            verbose=args.verbose,
        )
    else:
        cfg = SweepConfig(
            d_models=[32, 64],
            budgets=[25_000, 50_000],
            seeds=[0, 1],
            datasets=["synthetic", "arithmetic"],
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            max_train=args.max_train,
            max_val=args.max_val,
            verbose=args.verbose,
        )
    # Override from CLI
    if args.d:
        cfg.d_models = [int(x) for x in args.d.split(",")]
    if args.budgets:
        cfg.budgets = [int(x) for x in args.budgets.split(",")]
    if args.seeds:
        cfg.seeds = [int(x) for x in args.seeds.split(",")]
    if args.datasets:
        cfg.datasets = [x for x in args.datasets.split(",")]
    if args.tr_match:
        cfg.tr_match = args.tr_match
    results = run_sweep(cfg)
    summary = print_summary(results)
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nFull summary saved to {SUMMARY_PATH}")
    if args.plot:
        plot_efficiency_curves(results)
