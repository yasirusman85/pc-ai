# Project Structure

## Reorganized Folder Structure

```
pc-ai/
├── src/
│   └── crf_reasoning/          # Main package
│       ├── __init__.py        # Package initialization
│       ├── crf_vectorized.py  # Core CRF implementation
│       ├── crf.py             # Original CRF
│       ├── crf_sim.py         # Simulation
│       ├── transformer.py     # Transformer baseline
│       ├── data.py            # Dataset loaders
│       ├── train.py           # Training pipeline
│       ├── benchmark.py       # Benchmark suite
│       ├── metrics.py         # Evaluation metrics
│       ├── ablations.py       # Ablation configurations
│       ├── real_datasets.py   # Real dataset integration
│       ├── hypothesis.py      # Hypothesis evaluation
│       ├── plot.py            # Visualization
│       └── config_loader.py   # Configuration management
├── experiments/               # Experimental code
│   ├── conditional_computation/  # Conditional computation experiments
│   │   ├── conditional_crf.py
│   │   ├── conditional_computation_benchmark.py
│   │   ├── simple_conditional_test.py
│   │   ├── crf_optimized.py
│   │   ├── crf_truly_batched.py
│   │   └── conditional_computation_results.json
│   ├── continual_learning/     # Continual learning experiments
│   │   ├── continual_learning_benchmark.py
│   │   └── continual_learning_results.json
│   ├── real_experiments/       # Real dataset experiments
│   │   ├── quick_real_test.py
│   │   ├── run_real_experiment.py
│   │   ├── shakespeare_dataset.py
│   │   └── quick_real_test_results.json
│   ├── efficiency_benchmark.py # General efficiency benchmark
│   ├── efficiency_optimizations.py
│   ├── token_efficiency_tracker.py
│   └── profile_crf.py
├── docs/                      # Documentation
│   ├── ADDITIONS_SUMMARY.md
│   ├── CHANGES.md
│   ├── CODE_OF_CONDUCT.md
│   ├── CONTRIBUTING.md
│   ├── CONDITIONAL_COMPUTATION_RESULTS.md
│   ├── CONTINUAL_LEARNING_RESULTS.md
│   ├── REAL_EXPERIMENT_RESULTS.md
│   ├── HOW_TO_BEAT_TRANSFORMERS.md
│   └── EFFICIENCY_HYPOTHESIS.md
├── config/                    # Configuration files
│   ├── default.yaml
│   └── fast.yaml
├── data/                      # Data storage
│   └── real/
│       └── input.txt         # Shakespeare dataset
├── results/                   # Experiment results
├── tests/                     # Unit tests
├── README.md                  # Main documentation
├── CRF_TECHNICAL.md          # Technical documentation
├── LICENSE                    # MIT License
├── requirements.txt           # Dependencies
└── setup.py                   # Package installation
```

## Usage with New Structure

### Install as Package
```bash
pip install -e .
```

### Import from Package
```python
from src.crf_reasoning import CRFLanguageModel, train, get_datasets
```

### Run Experiments
```bash
# Conditional computation experiments
python experiments/conditional_computation/simple_conditional_test.py

# Continual learning experiments
python experiments/continual_learning/continual_learning_benchmark.py

# Real experiments
python experiments/real_experiments/quick_real_test.py
```

## Notes

- Core CRF code is now in `src/crf_reasoning/` for proper package structure
- Experimental code is organized by type in `experiments/`
- Documentation is consolidated in `docs/`
- This structure makes the codebase more maintainable and installable