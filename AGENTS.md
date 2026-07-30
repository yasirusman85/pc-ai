# AGENTS.md

## Purpose
This file provides instructions for automated coding agents working in this repository.

## Repository Summary
- Project: **Cellular Reasoning Fabric (CRF)** research codebase
- Language: **Python**
- Main areas:
  - `crf_vectorized.py`, `crf.py`, `crf_sim.py`: CRF model implementations
  - `train.py`, `benchmark.py`, `ablations.py`: training and experiment workflows
  - `data.py`, `real_datasets.py`: dataset loading and preparation
  - `tests/`: unit tests
  - `docs/`: technical writeups

## Environment Setup
1. Create and activate a virtual environment.
2. Install dependencies:
   - `pip install -r requirements.txt`
   - `pip install -e ".[dev]"`

## Validation Commands
Run these before finalizing code changes:
- Tests: `pytest tests/ -v`
- Formatting: `black .`
- Linting: `flake8 .`
- Type checks: `mypy .`

If your changes are documentation-only, skip runtime tests unless docs are coupled to code examples that were updated.

## Coding Expectations
- Follow PEP 8 and keep code compatible with Python 3.8+.
- Prefer small, focused changes over broad refactors.
- Add or update tests when behavior changes.
- Keep public function/class docstrings clear and consistent.
- Avoid introducing new dependencies unless necessary.

## Documentation Expectations
- Update `README.md` for user-facing workflow changes.
- Update files in `docs/` or `CRF_TECHNICAL.md` when architecture or theory details change.
- Keep examples and command snippets synchronized with actual behavior.

## Safety and Operational Rules
- Do not commit secrets, credentials, or tokens.
- Do not modify generated outputs in `results/` unless the task explicitly requires regenerating artifacts.
- Preserve backwards compatibility unless a breaking change is explicitly requested.

## Change Checklist for Agents
- Understand the target module and related tests.
- Implement minimal, task-scoped edits.
- Run validation commands relevant to the modified code.
- Ensure no unrelated files are changed.
- Summarize what changed and how it was validated.
