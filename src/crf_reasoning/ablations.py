"""
ablations.py — Ablation configuration registry
===============================================
Defines all 6 ablation variants as named AblationConfig objects,
plus a factory to build the corresponding CRFLanguageModel.

Ablation registry:
  full          — all mechanisms enabled (baseline CRF)
  no_split_death — disable cell birth and death
  no_merge       — disable cell consolidation
  no_routing     — uniform message aggregation (no learned gate)
  no_energy      — constant energy, random lifecycle
  no_messaging   — cells receive zero messages (independent MLPs)
  no_spatial     — pure cosine routing (no spatial penalty)
"""

from dataclasses import dataclass, asdict
from typing import Dict, Optional

from .crf_vectorized import AblationConfig, CRFLanguageModel


# ─── Named ablation registry ─────────────────────────────────────────────────

ABLATION_CONFIGS: Dict[str, AblationConfig] = {
    "full": AblationConfig(
        use_split=True,
        use_death=True,
        use_merge=True,
        use_routing=True,
        use_energy=True,
        use_messaging=True,
        use_spatial=True,
    ),
    "no_split_death": AblationConfig(
        use_split=False,
        use_death=False,
        use_merge=True,
        use_routing=True,
        use_energy=True,
        use_messaging=True,
        use_spatial=True,
    ),
    "no_merge": AblationConfig(
        use_split=True,
        use_death=True,
        use_merge=False,
        use_routing=True,
        use_energy=True,
        use_messaging=True,
        use_spatial=True,
    ),
    "no_routing": AblationConfig(
        use_split=True,
        use_death=True,
        use_merge=True,
        use_routing=False,
        use_energy=True,
        use_messaging=True,
        use_spatial=True,
    ),
    "no_energy": AblationConfig(
        use_split=True,
        use_death=True,
        use_merge=True,
        use_routing=True,
        use_energy=False,
        use_messaging=True,
        use_spatial=True,
    ),
    "no_messaging": AblationConfig(
        use_split=False,
        use_death=False,
        use_merge=False,
        use_routing=False,
        use_energy=True,
        use_messaging=False,
        use_spatial=True,
    ),
    "no_spatial": AblationConfig(
        use_split=True,
        use_death=True,
        use_merge=True,
        use_routing=True,
        use_energy=True,
        use_messaging=True,
        use_spatial=False,
    ),
}

# Human-readable descriptions for the report table
ABLATION_DESCRIPTIONS: Dict[str, str] = {
    "full": "Full CRF (all mechanisms)",
    "no_split_death": "No split/death (fixed population)",
    "no_merge": "No merge (no consolidation)",
    "no_routing": "No routing gate (uniform mean)",
    "no_energy": "No energy (constant ε=1, random lifecycle)",
    "no_messaging": "No messaging (independent cells)",
    "no_spatial": "No spatial penalty (pure cosine)",
}


# ─── Model factory ───────────────────────────────────────────────────────────


@dataclass
class ModelConfig:
    """Hyperparameters for CRFLanguageModel."""

    vocab_size: int = 99
    d_model: int = 128
    d_hidden: int = 64
    n_init_cells: int = 32
    max_cells: int = 128
    n_crf_steps: int = 6
    k_neighbors: int = 4
    max_seq_len: int = 128


def make_crf(
    model_cfg: ModelConfig,
    ablation: str = "full",
) -> CRFLanguageModel:
    """
    Creates a CRFLanguageModel with the given ablation config.
    """
    if ablation not in ABLATION_CONFIGS:
        raise ValueError(
            f"Unknown ablation '{ablation}'. "
            f"Choose from: {list(ABLATION_CONFIGS.keys())}"
        )

    cfg = ABLATION_CONFIGS[ablation]
    return CRFLanguageModel(
        vocab_size=model_cfg.vocab_size,
        d_model=model_cfg.d_model,
        d_hidden=model_cfg.d_hidden,
        n_init_cells=model_cfg.n_init_cells,
        max_cells=model_cfg.max_cells,
        n_crf_steps=model_cfg.n_crf_steps,
        k_neighbors=model_cfg.k_neighbors,
        max_seq_len=model_cfg.max_seq_len,
        cfg=cfg,
    )


def make_transformer(
    model_cfg: ModelConfig,
    target_params: Optional[int] = None,
) -> "GPT":
    """
    Creates a FLOPs-/param-matched Transformer.
    If target_params is None, uses a fixed small GPT that roughly matches
    a small CRF.
    """
    from .transformer import GPT

    if target_params is None:
        # Use a comparable fixed config
        d = model_cfg.d_model
        # n_heads: power of 2, divides d
        n_heads = 1
        for h in [8, 4, 2, 1]:
            if d % h == 0:
                n_heads = h
                break
        return GPT(
            vocab_size=model_cfg.vocab_size,
            d_model=d,
            n_heads=n_heads,
            n_layers=4,
            max_seq_len=model_cfg.max_seq_len,
            dropout=0.0,
            tie_weights=True,
        )
    else:
        from scripts.train import make_matched_transformer

        return make_matched_transformer(
            vocab_size=model_cfg.vocab_size,
            target_params=target_params,
            max_seq_len=model_cfg.max_seq_len,
        )


def list_ablations() -> None:
    """Prints a summary of all ablations and their configs."""
    print(f"\n{'='*60}")
    print(f"{'Ablation':<20} {'Description'}")
    print(f"{'='*60}")
    for name, desc in ABLATION_DESCRIPTIONS.items():
        cfg = ABLATION_CONFIGS[name]
        flags = "  ".join(
            f"{k[4:]}={'Y' if v else 'N'}"
            for k, v in asdict(cfg).items()
            if k.startswith("use_")
        )
        print(f"  {name:<18} {desc}")
        print(f"  {'':18} [{flags}]")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    list_ablations()

    cfg = ModelConfig(
        vocab_size=99,
        d_model=64,
        d_hidden=32,
        n_init_cells=16,
        max_cells=64,
        n_crf_steps=4,
    )
    for name in ABLATION_CONFIGS:
        m = make_crf(cfg, name)
        p = sum(x.numel() for x in m.parameters() if x.requires_grad)
        print(f"  [{name:20s}] params={p:,}")

    tr = make_transformer(cfg)
    tp = sum(x.numel() for x in tr.parameters() if x.requires_grad)
    print(f"  [{'transformer':20s}] params={tp:,}")
