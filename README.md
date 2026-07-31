# Cellular Reasoning Fabric (CRF)

A novel neural architecture that replaces fixed-depth Transformer attention with dynamic populations of cognitive cells that communicate over sparse graphs and adaptively split, die, and merge based on energy signals.

## Overview

CRF (Cellular Reasoning Fabric) is a biologically-inspired approach to sequence modeling that features:

- **Dynamic Population**: Cells are created (split), destroyed (death), and consolidated (merge) during forward passes
- **Sparse Communication**: k-nearest-neighbor routing instead of dense attention
- **Energy-Based Lifecycle**: Cells track productivity via energy signals that determine where computation concentrates
- **Per-Cell Specialization**: Each cell can develop distinct weights through mutation at split time
- **Adaptive Computation**: Hard inputs automatically trigger more cells and computation

## Key Innovation

To our knowledge, CRF is the first architecture to combine all of the following:
- Differentiable end-to-end training on token sequences
- Dynamic population lifecycles (cells created/destroyed during forward pass)
- Per-cell heterogeneous programs (functional specialization emerges)
- Energy-regulated adaptive computation (hard inputs get more resources)

Each individual component has prior art (adaptive computation time, neural
cellular automata, mixture-of-experts, message passing). The claimed novelty
is the specific combination, and the claim is empirical, not proven.

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/pc-ai.git
cd pc-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Quick Start

### Basic Usage

```python
from src.crf_reasoning import CRFLanguageModel, AblationConfig, get_datasets, make_dataloader, train

# Load synthetic dataset
train_ds, val_ds = get_datasets('synthetic', seq_len=64, max_train=1000, max_val=200)
train_loader = make_dataloader(train_ds, batch_size=32)
val_loader = make_dataloader(val_ds, batch_size=32, shuffle=False)

# Create CRF model
model = CRFLanguageModel(
    vocab_size=99,
    d_model=128,
    d_hidden=64,
    n_init_cells=32,
    max_cells=128,
    n_crf_steps=6,
    k_neighbors=4,
)

# Train
history = train(
    model, train_loader, val_loader,
    model_type='crf',
    n_epochs=5,
    lr=3e-4,
    run_name='my_experiment'
)
```

### Efficiency-Optimized Training

Working hypothesis: "CRF can reach Transformer-level accuracy with fewer
training tokens, by replacing weight memorization with dynamic computation."

**Current measured results** (see `results/doc/README.md`): across 24+ settings
(d=32/64, 3 datasets incl. real GSM8K, 2 seeds), CRF consistently exceeds this
bar. Average token efficiency ~5.9× (not yet 10–100× — that is the stretch
target, not the measured number). Average accuracy-per-FLOP ~10×, accuracy-per-
parameter ~4×. Results are CPU-scale and must not be extrapolated to GPU-scale
LLM training without new evidence.

```python
from experiments.efficiency_benchmark import EfficiencyBenchmark

# Run comparative efficiency benchmark
benchmark = EfficiencyBenchmark()
results = benchmark.run_comparative_benchmark(config={
    'vocab_size': 99,
    'd_model': 128,
    'n_init_cells': 32,
    'max_cells': 128,
    'n_crf_steps': 6,
    'k_neighbors': 4,
    'seq_len': 64,
    'dataset': 'arithmetic',
    'max_train': 2000,
    'max_val': 500,
    'batch_size': 32,
    'n_epochs': 15,
    'lr': 3e-4,
})

# Results include efficiency ratios and hypothesis validation
print(results['comparative_analysis'])
```

### Running Benchmarks

```bash
# Fast mode (tiny models for testing)
python benchmark.py --fast --exp main

# Full benchmark suite
python benchmark.py --exp all

# Specific experiments
python benchmark.py --exp ablations
python benchmark.py --exp scaling
python benchmark.py --exp theory

# Efficiency hypothesis testing
python efficiency_benchmark.py --fast                    # Quick test
python efficiency_benchmark.py                           # Full benchmark
python efficiency_benchmark.py --crf-only               # CRF only
python efficiency_benchmark.py --transformer-only        # Transformer only
```

### Generating Results

```bash
# Generate tables and figures from results
python plot.py
```

## Project Structure

```
pc-ai/
├── crf_vectorized.py      # Vectorized CRF implementation
├── transformer.py         # Transformer baseline
├── train.py               # Training pipeline
├── benchmark.py           # Experimental evaluation
├── data.py                # Dataset loaders
├── metrics.py             # Evaluation metrics
├── ablations.py           # Ablation configurations
├── plot.py                # Result visualization
├── config.yaml            # Configuration files
├── tests/                 # Unit tests
├── docs/                  # Technical documentation
├── results/               # Experiment outputs
└── CRF_TECHNICAL.md       # Full technical report
```

## Configuration

The project uses YAML configuration files for experiment management. See `config/default.yaml` for default settings:

```yaml
model:
  d_model: 128
  d_hidden: 64
  n_init_cells: 32
  max_cells: 128
  n_crf_steps: 6
  k_neighbors: 4

training:
  n_epochs: 10
  batch_size: 32
  lr: 3e-4
  weight_decay: 0.1
```

## Datasets

The project supports multiple datasets:

- **synthetic**: Generated stories with patterns (default, no download)
- **tinystories**: HuggingFace TinyStories dataset
- **arithmetic**: Math word problems (GSM8K proxy)
- **arc**: Pattern completion tasks (ARC-AGI proxy)
- **humaneval**: Code generation tasks
- **gsm8k**: Full GSM8K math reasoning dataset

## Ablation Studies

Seven ablation configurations are available:

- `full`: All mechanisms enabled (baseline)
- `no_split_death`: Fixed population
- `no_merge`: No consolidation
- `no_routing`: Uniform message aggregation
- `no_energy`: Constant energy, random lifecycle
- `no_messaging`: Independent cells
- `no_spatial`: Pure cosine similarity

## Theoretical Properties

The architecture is formally characterized with three theorems:

1. **Bounded Communication**: Per-step communication ≤ k·N_max
2. **Adaptive Computation**: Harder inputs → larger expected N_t
3. **Convergence under Contraction**: Convergence guarantees under certain conditions

See `CRF_TECHNICAL.md` for complete formalization.

## Limitations (read before citing)

- **Scale**: all experiments are small (d ≤ 128, ≤ 100k tokens, CPU). No
  evidence yet at LLM scale. The sample-efficiency advantage may not survive
  scaling; this is untested, not predicted.
- **Speed**: CRF per-token cost is currently slower than an equal-size
  Transformer on CPU. Wall-clock advantage comes only from needing fewer
  tokens. GPU parity is unimplemented.
- **FLOP estimates**: `estimate_crf_flops` is an analytical upper-bound, not a
  profiler measurement. Accuracy-per-FLOP ratios are approximate.
- **Lifecycle decisions are thresholded, not learned**: cell energy is produced
  by a learned gate, but the split/death/merge *decisions* apply fixed
  thresholds (ε_split = 1.05, death < 0.01, merge sim > 0.95). The energy
  signal itself is differentiable; making the discrete lifecycle choices
  learned end-to-end (e.g. straight-through estimators) is open work and may
  be required to avoid plateauing at larger scale.
- **Baseline matching**: parameter-matched Transformers are found by grid
  search and can be up to 1.6× larger than the CRF they are matched to.
  Normalized metrics (acc/param, acc/FLOP) mitigate but do not fully remove
  this bias.
- **Data**: GSM8K/Shakespeare/TinyStories are real; synthetic and arithmetic
  are proxy tasks. None of these are frontier reasoning benchmarks.
- **Theorems** in `docs/sec3_theory.md` are elementary bounds and convergence
  sketches, not deep results. They do not prove practical superiority.

## Experiment Tracking

### TensorBoard

```python
from tensorboardX import SummaryWriter
writer = SummaryWriter('runs/experiment_1')
# Training automatically logs to TensorBoard
```

### Weights & Biases

```bash
wandb login
wandb init crf-research
python benchmark.py  # Automatically logs to W&B
```

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=crf --cov-report=html

# Run specific test
pytest tests/test_crf.py::test_cell_population
```

## Performance Optimization

### Mixed Precision Training

```python
from src.crf_reasoning import train

history = train(
    model, train_loader, val_loader,
    use_amp=True,  # Enable mixed precision
    ...
)
```

### Distributed Training

```bash
torchrun --nproc_per_node=4 benchmark.py --exp main
```

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{crf2024,
  title={Cellular Reasoning Fabric: Adaptive Computation via Dynamic Cell Populations},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/pc-ai}
}
```

## License

MIT License - see LICENSE file for details.

## Contributing

We welcome contributions! Please see CONTRIBUTING.md for guidelines.

## Acknowledgments

- Inspired by Neural Cellular Automata and Mixture of Experts
- Built on PyTorch and HuggingFace datasets
- Technical documentation follows academic paper format

## Contact

For questions and issues, please open a GitHub issue or contact [your-email@example.com].

## Roadmap

- [ ] CUDA kernel optimization for cell operations
- [ ] Real dataset integration (TinyStories, HumanEval, GSM8K, ARC-AGI)
- [ ] Distributed training support
- [ ] Advanced visualization tools
- [ ] Pre-trained model checkpoints
- [ ] API for inference serving
