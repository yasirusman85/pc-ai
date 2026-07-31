"""
train.py — Unified training engine for CRF and Transformer
===========================================================
Features:
  - Single train() function handles both model types
  - Cosine LR schedule with warmup
  - FLOPs tracking per batch
  - Gradient clipping
  - Validation loop with perplexity + CRF-specific metrics
  - Checkpointing (best model by val loss)
  - Returns full training history for benchmark.py
  - Config integration for reproducibility
  - Mixed precision training support
  - Experiment tracking (TensorBoard/W&B)
"""

import os
import math
import time
import copy
import sys
from typing import Optional, Dict, List, Tuple
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from crf_reasoning.metrics import (
    perplexity,
    aggregate_metrics,
    estimate_crf_flops,
    estimate_transformer_flops,
    measure_memory_mb,
)

# Experiment tracking (optional)
try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ─── LR schedule ────────────────────────────────────────────────────────────


def cosine_lr(
    optimizer: optim.Optimizer,
    step: int,
    warmup: int,
    total: int,
    lr_max: float,
    lr_min: float,
):
    if step < warmup:
        lr = lr_max * (step + 1) / warmup
    else:
        progress = (step - warmup) / max(1, total - warmup)
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


# ─── Single epoch ────────────────────────────────────────────────────────────


def run_epoch(
    model,
    loader: DataLoader,
    optimizer: Optional[optim.Optimizer],
    device: torch.device,
    is_train: bool,
    model_type: str,  # 'crf' | 'transformer'
    collect_metrics: bool = False,
    grad_clip: float = 1.0,
    step_counter: List[int] = None,  # mutable ref
    schedule_kwargs: Optional[Dict] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> Dict:
    """
    Runs one epoch. Returns dict with loss, ppl, optional CRF metrics.
    """
    model.train(is_train)
    total_loss = 0.0
    n_batches = 0
    all_metrics = []

    ctx = torch.no_grad() if not is_train else torch.enable_grad()

    with ctx:
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            if model_type == "crf":
                logits, loss, batch_metrics = model(
                    xb, yb, collect_metrics=collect_metrics
                )
                if collect_metrics:
                    all_metrics.append(batch_metrics)
            else:
                logits, loss = model(xb, targets=yb)

            if is_train and loss is not None:
                optimizer.zero_grad()

                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

                if step_counter is not None and schedule_kwargs is not None:
                    cosine_lr(optimizer, step_counter[0], **schedule_kwargs)
                    step_counter[0] += 1

            total_loss += loss.item()
            n_batches += 1

    avg_loss = total_loss / max(1, n_batches)
    result = {
        "loss": avg_loss,
        "ppl": perplexity([avg_loss]),
        "n_batches": n_batches,
    }

    if collect_metrics and all_metrics:
        result["crf_metrics"] = aggregate_metrics(all_metrics)

    return result


# ─── Main training function ──────────────────────────────────────────────────


def train(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_type: str = "crf",  # 'crf' | 'transformer'
    n_epochs: int = 10,
    lr: float = 3e-4,
    lr_min: float = 3e-5,
    warmup_steps: int = 100,
    weight_decay: float = 0.1,
    grad_clip: float = 1.0,
    device: torch.device = torch.device("cpu"),
    checkpoint_dir: Optional[str] = None,
    run_name: str = "run",
    collect_crf_metrics: bool = True,
    verbose: bool = True,
    use_amp: bool = False,
    tensorboard_dir: Optional[str] = None,
    wandb_project: Optional[str] = None,
) -> Dict:
    """
    Trains model for n_epochs. Returns training history dict.
    """
    model = model.to(device)

    # Initialize experiment tracking
    writer = None
    if tensorboard_dir and TENSORBOARD_AVAILABLE:
        writer = SummaryWriter(tensorboard_dir)

    if wandb_project and WANDB_AVAILABLE:
        wandb.init(
            project=wandb_project,
            name=run_name,
            config={
                "model_type": model_type,
                "n_epochs": n_epochs,
                "lr": lr,
                "weight_decay": weight_decay,
            },
        )

    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler() if use_amp and device.type == "cuda" else None

    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.95),
        weight_decay=weight_decay,
    )

    total_steps = n_epochs * len(train_loader)
    step_counter = [0]
    sched_kwargs = {
        "warmup": warmup_steps,
        "total": total_steps,
        "lr_max": lr,
        "lr_min": lr_min,
    }

    history = {
        "run_name": run_name,
        "model_type": model_type,
        "train_loss": [],
        "train_ppl": [],
        "val_loss": [],
        "val_ppl": [],
        "epoch_time": [],
        "lr_trace": [],
        "crf_metrics": [],  # one entry per val epoch
        "n_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }

    # Estimate FLOPs
    if model_type == "crf" and hasattr(model, "crf"):
        crf = model.crf
        flops = estimate_crf_flops(
            N=crf.n_init,
            d=crf.d_model,
            d_h=crf.program.gate[0].out_features,
            k=crf.fabric.k,
            S=model.n_crf_steps,
        )
    elif model_type == "transformer":
        # Try to extract from model attributes
        try:
            L = len(model.blocks)
            d = model.d_model
            T = model.max_seq_len
            flops = estimate_transformer_flops(T, d, L)
        except Exception:
            flops = 0
    else:
        flops = 0
    history["estimated_flops_per_fwd"] = flops

    best_val_loss = float("inf")
    best_state = None

    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    t_total_start = time.time()

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()

        # ── Train ──
        train_res = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            is_train=True,
            model_type=model_type,
            collect_metrics=False,
            grad_clip=grad_clip,
            step_counter=step_counter,
            schedule_kwargs=sched_kwargs,
            scaler=scaler,
        )

        # ── Validate ──
        do_collect = collect_crf_metrics and model_type == "crf"
        val_res = run_epoch(
            model,
            val_loader,
            None,
            device,
            is_train=False,
            model_type=model_type,
            collect_metrics=do_collect,
            scaler=None,  # No AMP for validation
        )

        elapsed = time.time() - t0

        history["train_loss"].append(train_res["loss"])
        history["train_ppl"].append(train_res["ppl"])
        history["val_loss"].append(val_res["loss"])
        history["val_ppl"].append(val_res["ppl"])
        history["epoch_time"].append(elapsed)
        history["lr_trace"].append(optimizer.param_groups[0]["lr"])

        if "crf_metrics" in val_res:
            history["crf_metrics"].append(val_res["crf_metrics"])

        if verbose:
            crf_str = ""
            if "crf_metrics" in val_res and val_res["crf_metrics"]:
                cm = val_res["crf_metrics"]
                crf_str = (
                    f" | cells={cm['n_cells_mean']:.1f}"
                    f" splits={cm['splits_per_ex']:.1f}"
                    f" spec={cm['specialization']:.3f}"
                )
            print(
                f"[{run_name}] ep {epoch:3d}/{n_epochs} "
                f"train_ppl={train_res['ppl']:6.2f} "
                f"val_ppl={val_res['ppl']:6.2f} "
                f"lr={optimizer.param_groups[0]['lr']:.2e} "
                f"t={elapsed:.1f}s{crf_str}"
            )

        # Log to TensorBoard
        if writer is not None:
            writer.add_scalar("Loss/train", train_res["loss"], epoch)
            writer.add_scalar("Loss/val", val_res["loss"], epoch)
            writer.add_scalar("Perplexity/train", train_res["ppl"], epoch)
            writer.add_scalar("Perplexity/val", val_res["ppl"], epoch)
            writer.add_scalar("Learning_rate", optimizer.param_groups[0]["lr"], epoch)

            if "crf_metrics" in val_res and val_res["crf_metrics"]:
                cm = val_res["crf_metrics"]
                writer.add_scalar("CRF/n_cells_mean", cm["n_cells_mean"], epoch)
                writer.add_scalar("CRF/splits_per_ex", cm["splits_per_ex"], epoch)
                writer.add_scalar("CRF/specialization", cm["specialization"], epoch)

        # Log to W&B
        if wandb_project and WANDB_AVAILABLE:
            log_dict = {
                "epoch": epoch,
                "train_loss": train_res["loss"],
                "val_loss": val_res["loss"],
                "train_ppl": train_res["ppl"],
                "val_ppl": val_res["ppl"],
                "learning_rate": optimizer.param_groups[0]["lr"],
                "epoch_time": elapsed,
            }
            if "crf_metrics" in val_res and val_res["crf_metrics"]:
                cm = val_res["crf_metrics"]
                log_dict.update(
                    {
                        "crf_n_cells_mean": cm["n_cells_mean"],
                        "crf_splits_per_ex": cm["splits_per_ex"],
                        "crf_specialization": cm["specialization"],
                    }
                )
            wandb.log(log_dict)

        # Checkpoint
        if val_res["loss"] < best_val_loss:
            best_val_loss = val_res["loss"]
            best_state = copy.deepcopy(model.state_dict())
            if checkpoint_dir:
                path = os.path.join(checkpoint_dir, f"{run_name}_best.pt")
                torch.save(
                    {
                        "state_dict": best_state,
                        "epoch": epoch,
                        "val_loss": best_val_loss,
                    },
                    path,
                )

    history["total_train_time_s"] = time.time() - t_total_start
    history["best_val_loss"] = best_val_loss
    history["best_val_ppl"] = perplexity([best_val_loss])

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # Close experiment tracking
    if writer is not None:
        writer.close()

    if wandb_project and WANDB_AVAILABLE:
        wandb.finish()

    return history


# ─── FLOPs-matched model factory ─────────────────────────────────────────────


def count_gpt_params(
    vocab_size, d_model, n_layers, tie_weights=True, activation="swiglu"
):
    """
    Exact GPT parameter count:
      token_embed:  vocab_size * d_model
      pos_enc:      max_seq_len * d_model  (ignored — same for both)
      per layer:
        attn QKV+O: 4 * d_model^2
        FFN:
          SwiGLU:   w1 (d→2*d_ff) + w2 (d_ff→d) = 3*d_model*d_ff
          GELU:     fc1 (d→d_ff) + fc2 (d_ff→d) = 2*d_model*d_ff
        norms:      2 * d_model  (gamma+beta per norm, 2 norms)
      lm_head:      0 if tie_weights else vocab_size * d_model
      final norm:   2 * d_model
    """
    embed = vocab_size * d_model
    d_ff = d_model * 4
    ffn = 3 * d_model * d_ff if activation == "swiglu" else 2 * d_model * d_ff
    layer = 4 * d_model * d_model + ffn + 2 * d_model
    head = 0 if tie_weights else vocab_size * d_model
    norm = 2 * d_model
    return embed + n_layers * layer + head + norm


def make_matched_transformer(
    vocab_size: int,
    target_params: int,
    max_seq_len: int = 128,
    device: torch.device = torch.device("cpu"),
):
    """
    Searches a dense (d_model, n_layers) grid to find transformer whose
    exact param count is closest to target_params. Uses same vocab,
    tie_weights=True, and SwiGLU activation.
    """
    from crf_reasoning.transformer import GPT

    best_model = None
    best_delta = float("inf")
    best_cfg = None

    # Dense grid: every 8 in small range, every 16 in mid, every 32 in large
    d_values = (
        list(range(32, 128, 8)) + list(range(128, 256, 16)) + list(range(256, 512, 32))
    )
    l_values = list(range(1, 9))

    for d in d_values:
        for L in l_values:
            n_heads = next(h for h in [8, 4, 2, 1] if d % h == 0)
            p = count_gpt_params(
                vocab_size, d, L, tie_weights=True, activation="swiglu"
            )
            delta = abs(p - target_params)
            if delta < best_delta:
                best_delta = delta
                best_cfg = (d, n_heads, L)

    d, n_heads, L = best_cfg
    model = GPT(
        vocab_size=vocab_size,
        d_model=d,
        n_heads=n_heads,
        n_layers=L,
        max_seq_len=max_seq_len,
        dropout=0.0,
        tie_weights=True,
        activation="swiglu",
    )
    actual = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"    [Transformer match] d={d} L={L} h={n_heads} "
        f"params={actual:,} target={target_params:,} ratio={actual/max(1,target_params):.3f}"
    )

    return model.to(device)


# ─── Quick smoke test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    from crf_reasoning.data import get_datasets, make_dataloader
    from crf_reasoning.crf_vectorized import CRFLanguageModel, AblationConfig

    tok_size = 99
    cfg = AblationConfig()
    model = CRFLanguageModel(
        vocab_size=tok_size,
        d_model=64,
        d_hidden=32,
        n_init_cells=16,
        max_cells=64,
        n_crf_steps=4,
        k_neighbors=3,
        cfg=cfg,
    )

    tr, va = get_datasets("synthetic", seq_len=32, max_train=200, max_val=50)
    tr_dl = make_dataloader(tr, batch_size=16)
    va_dl = make_dataloader(va, batch_size=16, shuffle=False)

    hist = train(
        model,
        tr_dl,
        va_dl,
        model_type="crf",
        n_epochs=3,
        lr=3e-4,
        run_name="smoke_test",
        verbose=True,
    )

    print(f"\nBest val PPL: {hist['best_val_ppl']:.2f}")
    print(f"Params: {hist['n_params']:,}")
    print(f"Est FLOPs/fwd: {hist['estimated_flops_per_fwd']:,}")
