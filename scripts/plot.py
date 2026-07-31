"""
plot.py — Generate all result tables and charts from results/results.json
=========================================================================
Outputs:
  results/tables.txt        — ASCII result tables (always generated)
  results/fig_*.png         — Matplotlib figures (if matplotlib available)
"""

import os
import sys
import json
import math
from typing import Dict, Any, List
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = str(ROOT_DIR / "results")


# ─── Load ────────────────────────────────────────────────────────────────────


def load_results(path: str = None) -> Dict:
    path = path or os.path.join(RESULTS_DIR, "results.json")
    if not os.path.exists(path):
        print(f"[plot] Results file not found: {path}")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


# ─── ASCII tables ────────────────────────────────────────────────────────────


def fmt(val, decimals=2, pct=False):
    if val is None:
        return "N/A"
    if isinstance(val, float):
        if pct:
            return f"{val*100:.1f}%"
        return f"{val:.{decimals}f}"
    return str(val)


def table_main_comparison(results: Dict) -> str:
    mc = results.get("main_comparison", {})
    if not mc:
        return "  [no main_comparison data]\n"

    lines = []
    lines.append(f"\n{'─'*72}")
    lines.append(f"  TABLE 1 — CRF vs Transformer (FLOPs/param-matched)")
    lines.append(f"{'─'*72}")
    hdr = f"  {'Dataset':<14} {'Model':<14} {'Val PPL':>8} {'Params':>10} {'Infer ms':>10} {'Comm Cost':>12}"
    lines.append(hdr)
    lines.append(f"  {'-'*68}")

    for ds, ds_res in mc.items():
        for mtype in ["crf", "transformer"]:
            r = ds_res.get(mtype, {})
            ppl = fmt(r.get("best_val_ppl"))
            prms = f"{r.get('n_params', 0):,}"
            lat = fmt(r.get("profile", {}).get("latency_ms_mean"), 1)
            comm = "N/A"
            if mtype == "crf":
                ci = r.get("profile", {}).get("crf_inference_metrics", {})
                comm = fmt(ci.get("comm_cost_mean", 0), 0)
            lines.append(
                f"  {ds:<14} {mtype:<14} {ppl:>8} {prms:>10} {lat:>10} {comm:>12}"
            )
        lines.append(f"  {'-'*68}")

    return "\n".join(lines)


def table_ablations(results: Dict) -> str:
    abl = results.get("ablations", {})
    if not abl:
        return "  [no ablation data]\n"

    lines = []
    lines.append(f"\n{'─'*80}")
    lines.append(f"  TABLE 2 — Ablation Study Results")
    lines.append(f"{'─'*80}")
    hdr = f"  {'Ablation':<22} {'Description':<34} {'Synth PPL':>10} {'Arith PPL':>10}"
    lines.append(hdr)
    lines.append(f"  {'-'*76}")

    ablation_names = list(abl.get("synthetic", {}).keys())
    for name in ablation_names:
        desc = abl.get("synthetic", {}).get(name, {}).get("description", name)[:33]
        s_ppl = fmt(abl.get("synthetic", {}).get(name, {}).get("best_val_ppl"))
        a_ppl = fmt(abl.get("arithmetic", {}).get(name, {}).get("best_val_ppl"))
        lines.append(f"  {name:<22} {desc:<34} {s_ppl:>10} {a_ppl:>10}")

    lines.append(f"{'─'*80}")
    return "\n".join(lines)


def table_ablation_metrics(results: Dict) -> str:
    abl = results.get("ablations", {})
    syn = abl.get("synthetic", {})
    if not syn:
        return "  [no ablation metric data]\n"

    lines = []
    lines.append(f"\n{'─'*80}")
    lines.append(f"  TABLE 3 — Ablation CRF Metrics (synthetic dataset)")
    lines.append(f"{'─'*80}")
    hdr = (
        f"  {'Ablation':<22} {'Avg N':>7} {'Splits/ex':>10} "
        f"{'Deaths/ex':>10} {'Spec':>8} {'Comm':>10}"
    )
    lines.append(hdr)
    lines.append(f"  {'-'*76}")

    for name, data in syn.items():
        cm_list = data.get("crf_metrics", [])
        if not cm_list:
            lines.append(
                f"  {name:<22} {'—':>7} {'—':>10} {'—':>10} {'—':>8} {'—':>10}"
            )
            continue
        # average across epochs
        avg_n = sum(c.get("n_cells_mean", 0) for c in cm_list) / len(cm_list)
        avg_spl = sum(c.get("splits_per_ex", 0) for c in cm_list) / len(cm_list)
        avg_dea = sum(c.get("deaths_per_ex", 0) for c in cm_list) / len(cm_list)
        avg_spec = sum(c.get("specialization", 0) for c in cm_list) / len(cm_list)
        avg_comm = sum(c.get("comm_cost_mean", 0) for c in cm_list) / len(cm_list)
        lines.append(
            f"  {name:<22} {avg_n:>7.1f} {avg_spl:>10.2f} "
            f"{avg_dea:>10.2f} {avg_spec:>8.3f} {avg_comm:>10.0f}"
        )

    lines.append(f"{'─'*80}")
    return "\n".join(lines)


def table_scaling(results: Dict) -> str:
    sc = results.get("scaling", {})
    if not sc:
        return "  [no scaling data]\n"

    lines = []
    lines.append(f"\n{'─'*60}")
    lines.append(f"  TABLE 4 — Scaling: N_init sweep")
    lines.append(f"{'─'*60}")
    lines.append(f"  {'N_init':>8} {'Val PPL':>10} {'FLOPs':>14} {'Params':>10}")
    lines.append(f"  {'-'*56}")
    for k, v in sc.get("cells", {}).items():
        lines.append(
            f"  {k:>8} {fmt(v.get('best_val_ppl')):>10} "
            f"{v.get('flops', 0):>14,} {v.get('n_params', 0):>10,}"
        )
    law = sc.get("cells_law", {})
    lines.append(
        f"\n  Scaling law fit (cells): b={fmt(law.get('b'))} a={fmt(law.get('a'))}"
    )

    lines.append(f"\n{'─'*60}")
    lines.append(f"  TABLE 5 — Scaling: n_steps sweep")
    lines.append(f"{'─'*60}")
    lines.append(f"  {'Steps':>8} {'Val PPL':>10} {'FLOPs':>14} {'Params':>10}")
    lines.append(f"  {'-'*56}")
    for k, v in sc.get("steps", {}).items():
        lines.append(
            f"  {k:>8} {fmt(v.get('best_val_ppl')):>10} "
            f"{v.get('flops', 0):>14,} {v.get('n_params', 0):>10,}"
        )
    law2 = sc.get("steps_law", {})
    lines.append(
        f"\n  Scaling law fit (steps): b={fmt(law2.get('b'))} a={fmt(law2.get('a'))}"
    )

    lines.append(f"\n{'─'*60}")
    lines.append(f"  TABLE 5b — Scaling: d_model sweep")
    lines.append(f"{'─'*60}")
    lines.append(f"  {'d_model':>8} {'Val PPL':>10} {'FLOPs':>14} {'Params':>10}")
    lines.append(f"  {'-'*56}")
    for k, v in sc.get("d_model", {}).items():
        lines.append(
            f"  {k:>8} {fmt(v.get('best_val_ppl')):>10} "
            f"{v.get('flops', 0):>14,} {v.get('n_params', 0):>10,}"
        )
    law3 = sc.get("d_model_law", {})
    lines.append(
        f"\n  Scaling law fit (d_model): b={fmt(law3.get('b'))} a={fmt(law3.get('a'))}"
    )

    return "\n".join(lines)


def table_theory(results: Dict) -> str:
    tv = results.get("theory_validation", {})
    if not tv:
        return "  [no theory validation data]\n"

    t1 = tv.get("theorem1_bounded_comm", {})
    t2 = tv.get("theorem2_adaptive_compute", {})
    t3 = tv.get("lemma_energy_conservation", {})

    lines = []
    lines.append(f"\n{'─'*70}")
    lines.append(f"  TABLE 6 — Theoretical Property Validation")
    lines.append(f"{'─'*70}")

    lines.append(f"\n  Theorem 1 — Bounded Communication Cost")
    lines.append(f"    Bound:           k * N_max = {t1.get('bound', '?')}")
    lines.append(
        f"    Violations:      {t1.get('violations', '?')} / {t1.get('n_checked', '?')}"
    )
    lines.append(f"    Theorem holds:   {t1.get('theorem_holds', '?')}")

    lines.append(f"\n  Theorem 2 — Adaptive Computation")
    lines.append(f"    Easy input avg N:  {fmt(t2.get('easy_mean_N'), 2)}")
    lines.append(f"    Hard input avg N:  {fmt(t2.get('hard_mean_N'), 2)}")
    lines.append(f"    Adaptive:          {t2.get('adaptive', '?')}")
    lines.append(f"    Ratio hard/easy:   {fmt(t2.get('ratio'), 3)}")

    lines.append(f"\n  Energy Conservation Lemma")
    lines.append(f"    Status:            {t3.get('energy_lemma', '?')}")
    lines.append(f"    Splits/forward:    {fmt(t3.get('splits_per_fwd'), 3)}")

    lines.append(f"\n  Graph Diameter")
    lines.append(f"    Observed diameter: {tv.get('graph_diameter', '?')}")
    lines.append(f"    log_k(N) bound:    {tv.get('log_k_n_bound', '?')}")

    lines.append(f"{'─'*70}")
    return "\n".join(lines)


def table_profiling(results: Dict) -> str:
    pr = results.get("profiling", {})
    if not pr:
        return "  [no profiling data]\n"

    lines = []
    lines.append(f"\n{'─'*70}")
    lines.append(f"  TABLE 7 — Wall-clock and Memory Profiling")
    lines.append(f"{'─'*70}")
    hdr = (
        f"  {'Model':<14} {'Params':>10} {'Infer ms':>10} "
        f"{'Train ms':>10} {'Mem MB':>8} {'FLOPs':>14}"
    )
    lines.append(hdr)
    lines.append(f"  {'-'*66}")
    for name, p in pr.items():
        lines.append(
            f"  {name:<14} {p.get('n_params', 0):>10,} "
            f"{fmt(p.get('latency_ms_mean'), 1):>10} "
            f"{fmt(p.get('train_step_ms_mean'), 1):>10} "
            f"{fmt(p.get('memory_mb'), 0):>8} "
            f"{p.get('estimated_flops', 0):>14,}"
        )
    lines.append(f"{'─'*70}")
    return "\n".join(lines)


# ─── Matplotlib figures ───────────────────────────────────────────────────────


def make_figures(results: Dict) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("[plot] matplotlib not available, skipping figures")
        return

    # Figure 1: Training curves per dataset
    mc = results.get("main_comparison", {})
    if mc:
        datasets = list(mc.keys())
        fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4))
        if len(datasets) == 1:
            axes = [axes]
        for ax, ds in zip(axes, datasets):
            for mtype, color, label in [
                ("crf", "#2196F3", "CRF"),
                ("transformer", "#FF5722", "Transformer"),
            ]:
                hist = mc[ds].get(mtype, {})
                vp = hist.get("val_ppl", [])
                if vp:
                    ax.plot(vp, color=color, label=label, linewidth=2)
            ax.set_title(f"Val PPL — {ds}", fontsize=11)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Perplexity")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_yscale("log")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "fig_training_curves.png"), dpi=120)
        plt.close()
        print("  Saved fig_training_curves.png")

    # Figure 2: Ablation bar chart
    abl = results.get("ablations", {})
    if abl:
        datasets_abl = list(abl.keys())
        fig, axes = plt.subplots(
            1, len(datasets_abl), figsize=(6 * len(datasets_abl), 5)
        )
        if len(datasets_abl) == 1:
            axes = [axes]
        colors = plt.cm.tab10.colors
        for ax, ds in zip(axes, datasets_abl):
            names = list(abl[ds].keys())
            ppls = [abl[ds][n].get("best_val_ppl", 0) for n in names]
            short = [n.replace("no_", "−").replace("_", "/") for n in names]
            bars = ax.bar(range(len(names)), ppls, color=colors[: len(names)])
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(short, rotation=30, ha="right", fontsize=8)
            ax.set_title(f"Ablation PPL — {ds}", fontsize=11)
            ax.set_ylabel("Val Perplexity")
            ax.grid(True, axis="y", alpha=0.3)
            # Highlight full model
            if "full" in names:
                fi = names.index("full")
                bars[fi].set_edgecolor("black")
                bars[fi].set_linewidth(2)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "fig_ablations.png"), dpi=120)
        plt.close()
        print("  Saved fig_ablations.png")

    # Figure 3: Scaling laws
    sc = results.get("scaling", {})
    if sc:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))
        for ax, key, xlabel in [
            (ax1, "cells", "N_init (cells)"),
            (ax2, "steps", "n_crf_steps"),
        ]:
            data = sc.get(key, {})
            if not data:
                continue
            xs = sorted(data.keys(), key=lambda x: int(x))
            ppls = [data[x]["best_val_ppl"] for x in xs]
            flops = [data[x]["flops"] for x in xs]
            ax.plot(
                [int(x) for x in xs],
                ppls,
                "o-",
                color="#2196F3",
                linewidth=2,
                markersize=7,
            )
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Val Perplexity")
            ax.set_title(f"Scaling: {xlabel}", fontsize=11)
            ax.grid(True, alpha=0.3)

        dm_data = sc.get("d_model", {})
        if dm_data:
            xs = sorted(dm_data.keys(), key=lambda x: int(x))
            ppls = [dm_data[x]["best_val_ppl"] for x in xs]
            ax3.plot(
                [int(x) for x in xs],
                ppls,
                "s-",
                color="#4CAF50",
                linewidth=2,
                markersize=7,
            )
            ax3.set_xlabel("d_model")
            ax3.set_ylabel("Val Perplexity")
            ax3.set_title("Scaling: d_model", fontsize=11)
            ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "fig_scaling.png"), dpi=120)
        plt.close()
        print("  Saved fig_scaling.png")

    # Figure 4: CRF metrics over training (specialization, cell count)
    abl_full = {}
    for ds, ds_data in (results.get("ablations", {}) or {}).items():
        if "full" in ds_data and ds_data["full"].get("crf_metrics"):
            abl_full[ds] = ds_data["full"]["crf_metrics"]

    if abl_full:
        fig, axes = plt.subplots(1, len(abl_full), figsize=(5 * len(abl_full), 4))
        if len(abl_full) == 1:
            axes = [axes]
        for ax, (ds, metrics_list) in zip(axes, abl_full.items()):
            epochs = list(range(1, len(metrics_list) + 1))
            n_cells = [m.get("n_cells_mean", 0) for m in metrics_list]
            specs = [m.get("specialization", 0) for m in metrics_list]
            ax2 = ax.twinx()
            (l1,) = ax.plot(epochs, n_cells, "b-o", label="Avg cells", linewidth=2)
            (l2,) = ax2.plot(epochs, specs, "r--s", label="Specialization", linewidth=2)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Avg Cell Count", color="b")
            ax2.set_ylabel("Specialization", color="r")
            ax.set_title(f"Cell Dynamics — {ds}", fontsize=11)
            ax.legend(handles=[l1, l2], loc="upper left", fontsize=8)
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "fig_cell_dynamics.png"), dpi=120)
        plt.close()
        print("  Saved fig_cell_dynamics.png")


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    results = load_results()

    print("\n" + "=" * 80)
    print("  CRF EXPERIMENTAL RESULTS")
    print("=" * 80)
    print(f"  Run timestamp: {results.get('timestamp', 'unknown')}")
    cfg = results.get("config", {})
    if cfg:
        print(
            f"  Config: d={cfg.get('d_model')} N={cfg.get('n_init_cells')} "
            f"S={cfg.get('n_crf_steps')} k={cfg.get('k_neighbors')} "
            f"seq={cfg.get('seq_len')} epochs={cfg.get('n_epochs')}"
        )

    # Print all tables
    print(table_main_comparison(results))
    print(table_ablations(results))
    print(table_ablation_metrics(results))
    print(table_scaling(results))
    print(table_theory(results))
    print(table_profiling(results))

    # Save to file
    tables_path = os.path.join(RESULTS_DIR, "tables.txt")
    with open(tables_path, "w") as f:
        f.write("CRF EXPERIMENTAL RESULTS\n")
        f.write(f"Timestamp: {results.get('timestamp', 'unknown')}\n\n")
        for fn in [
            table_main_comparison,
            table_ablations,
            table_ablation_metrics,
            table_scaling,
            table_theory,
            table_profiling,
        ]:
            f.write(fn(results) + "\n")
    print(f"\n  Tables saved to {tables_path}")

    # Generate figures
    print("\nGenerating figures...")
    make_figures(results)


if __name__ == "__main__":
    main()
