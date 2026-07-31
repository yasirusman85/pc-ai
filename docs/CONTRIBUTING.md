# Contributing to CRF (Cellular Reasoning Fabric)

Thank you for your interest in contributing to CRF! This document provides guidelines and instructions for contributing to the project.

## How to Contribute

### Reporting Bugs

Before creating bug reports, please check the existing issues to avoid duplicates. When creating a bug report, include:

- **Description**: A clear description of the problem
- **Reproduction steps**: Steps to reproduce the issue
- **Expected behavior**: What you expected to happen
- **Actual behavior**: What actually happened
- **Environment**: Python version, OS, PyTorch version, etc.
- **Logs**: Relevant error messages or logs

### Suggesting Enhancements

Enhancement suggestions are welcome! Please include:

- **Description**: A clear description of the enhancement
- **Motivation**: Why this enhancement would be useful
- **Alternatives**: Any alternative solutions or features you've considered
- **Additional context**: Any other context about the feature request

### Pull Requests

#### Setup Development Environment

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/yourusername/pc-ai.git
   cd pc-ai
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e ".[dev]"
   ```

#### Making Changes

1. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes following our coding standards

3. Write tests for your changes

4. Run tests to ensure everything works:
   ```bash
   pytest tests/ -v
   ```

5. Run linting:
   ```bash
   black .
   flake8 .
   mypy .
   ```

6. Commit your changes with clear messages:
   ```bash
   git commit -m "Add feature: brief description of changes"
   ```

7. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

8. Create a pull request to the main repository

#### Pull Request Guidelines

- **Title**: Use a clear title describing the changes
- **Description**: Provide a detailed description of your changes
- **Testing**: Describe how you tested your changes
- **Documentation**: Update relevant documentation if needed
- **Breaking Changes**: Clearly note any breaking changes

## Coding Standards

### Python Style

- Follow PEP 8 style guidelines
- Use Black for code formatting
- Keep functions focused and modular
- Add docstrings to all functions and classes
- Use type hints where appropriate

### Documentation

- Update README.md if user-facing changes are made
- Update technical documentation in docs/ if architecture changes
- Add comments for complex logic
- Keep docstrings consistent with Google style

### Testing

- Write unit tests for new functionality
- Ensure all tests pass before submitting PR
- Aim for >80% code coverage on new code
- Test edge cases and error conditions

## Project Structure

```
pc-ai/
├── src/
│   └── crf_reasoning/     # Core package implementation
├── scripts/               # Train/benchmark/plot entry points
├── config/                # Configuration files
├── tests/                 # Test suite
├── docs/                  # Documentation
└── results/               # Experiment outputs
```

## Development Workflow

1. **Issue Discussion**: Discuss major changes in an issue first
2. **Branching**: Create a branch for each feature/fix
3. **Testing**: Write and run tests
4. **Documentation**: Update relevant documentation
5. **Code Review**: Submit PR for review
6. **Integration**: Maintainers will integrate approved changes

## Release Process

Releases are managed by project maintainers:

1. Update version number in setup.py
2. Update CHANGELOG.md
3. Create git tag
4. Build and publish to PyPI
5. Create GitHub release

## Questions?

- Open an issue for questions
- Check existing documentation
- Contact maintainers via GitHub issues

## License

By contributing, you agree that your contributions will be licensed under the MIT License.