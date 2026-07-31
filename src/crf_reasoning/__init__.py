"""
CRF Reasoning Package
Cellular Reasoning Fabric implementation
"""

import sys
from pathlib import Path

# Project convention: modules import each other flat (e.g. `from data import`).
# Make that work when imported as a package too.
_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from .crf_vectorized import CRFLanguageModel, VectorizedCRF, AblationConfig, CRFMetrics
from .transformer import GPT
from .data import get_datasets, make_dataloader
from .train import train
from .hypothesis import run_sweep
from .metrics import evaluate_crf_flops, evaluate_transformer_flops
from .real_datasets import RealDatasetFactory

__version__ = "0.1.0"