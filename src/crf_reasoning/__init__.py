"""
CRF Reasoning Package
Cellular Reasoning Fabric implementation
"""

from .crf_vectorized import CRFLanguageModel, VectorizedCRF, AblationConfig, CRFMetrics
from .transformer import GPT
from .data import get_datasets, make_dataloader
from .train import train
from .benchmark import run_sweep
from .metrics import evaluate_crf_flops, evaluate_transformer_flops
from .real_datasets import RealDatasetFactory

__version__ = "0.1.0"