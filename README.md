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

CRF is the first architecture to combine:
- Differentiable end-to-end training on token sequences
- Dynamic population lifecycles (cells created/destroyed during forward pass)
- Per-cell heterogeneous programs (functional specialization emerges)
- Energy-regulated adaptive computation (hard inputs get more resources)

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
from crf_reasoning.crf_vectorized import CRFLanguageModel, AblationConfig
from crf_reasoning.data import get_datasets, make_dataloader
from scripts.train import train

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

### Running Benchmarks

```bash
# Fast mode (tiny models for testing)
python -m scripts.benchmark --fast --exp main

# Full benchmark suite
python -m scripts.benchmark --exp all

# Specific experiments
python -m scripts.benchmark --exp ablations
python -m scripts.benchmark --exp scaling
python -m scripts.benchmark --exp theory
```

### Generating Results

```bash
# Generate tables and figures from results
python -m scripts.plot
```

## Project Structure

```
pc-ai/
├── src/
│   └── crf_reasoning/     # Core package (models, data, metrics, configs)
├── scripts/               # Train/benchmark/plot entry points
├── config/                # Configuration files
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
python -m scripts.benchmark  # Automatically logs to W&B
```

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=crf_reasoning --cov-report=html

# Run specific test
pytest tests/test_crf.py::test_cell_population
```

## Performance Optimization

### Mixed Precision Training

```python
from scripts.train import train

history = train(
    model, train_loader, val_loader,
    use_amp=True,  # Enable mixed precision
    ...
)
```

### Distributed Training

```bash
torchrun --nproc_per_node=4 -m scripts.benchmark --exp main
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
