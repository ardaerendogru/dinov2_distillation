# Contributing to DINOv2 Knowledge Distillation Framework

Thank you for your interest in contributing to the DINOv2 Knowledge Distillation Framework! This document provides guidelines and instructions for contributing to the project.

## Table of Contents
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contribution Guidelines](#contribution-guidelines)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Adding New Features](#adding-new-features)

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- CUDA-compatible GPU (recommended for testing)
- Familiarity with PyTorch and PyTorch Lightning

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
```bash
git clone https://github.com/YOUR_USERNAME/dinov2_distillation.git
cd dinov2_distillation
```

3. Add the upstream repository:
```bash
git remote add upstream https://github.com/ardaerendogru/dinov2_distillation.git
```

## Development Setup

### 1. Create Development Environment

```bash
# Create virtual environment
python -m venv dev_env
source dev_env/bin/activate  # On Windows: dev_env\Scripts\activate

# Install in development mode
pip install -e .
pip install -r requirements-dev.txt
```

### 2. Install Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
```

This will run code formatting and linting checks before each commit.

### 3. Development Dependencies

Create `requirements-dev.txt`:
```txt
# Include all from requirements.txt
-r requirements.txt

# Development tools
pytest>=7.0.0
pytest-cov>=4.0.0
black>=22.0.0
flake8>=5.0.0
isort>=5.10.0
mypy>=0.990
pre-commit>=2.20.0
sphinx>=5.0.0
sphinx-rtd-theme>=1.0.0
```

## Contribution Guidelines

### Types of Contributions

We welcome various types of contributions:

1. **Bug Fixes**: Fix issues reported in GitHub Issues
2. **New Features**: Add new models, loss functions, or capabilities
3. **Performance Improvements**: Optimize existing code
4. **Documentation**: Improve or add documentation
5. **Tests**: Add unit tests or integration tests
6. **Examples**: Add new examples or tutorials

### Before Contributing

1. Check existing issues and pull requests
2. Open an issue to discuss major changes
3. Ensure your contribution aligns with the project's goals

## Code Style

### Python Style Guide

We follow PEP 8 with some modifications:

```python
# Good: Clear variable names
student_features = model(images)
teacher_features = teacher_model(images)

# Bad: Unclear abbreviations
s_feat = m(imgs)
t_feat = t_m(imgs)
```

### Code Formatting

Use `black` for automatic formatting:
```bash
black . --line-length 88
```

Use `isort` for import sorting:
```bash
isort . --profile black
```

### Type Hints

Add type hints to all new functions:
```python
from typing import Dict, List, Optional, Tuple

def compute_loss(
    student_features: Dict[str, torch.Tensor],
    teacher_features: Dict[str, torch.Tensor],
    weights: Optional[List[float]] = None
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute distillation loss.
    
    Args:
        student_features: Dictionary of student feature maps
        teacher_features: Dictionary of teacher feature maps
        weights: Optional loss weights
        
    Returns:
        Tuple of (total_loss, loss_dict)
    """
    pass
```

### Docstrings

Use Google-style docstrings:
```python
class NewModel(nn.Module):
    """
    Brief description of the model.
    
    Longer description explaining the model architecture,
    its purpose, and any important details.
    
    Args:
        num_classes (int): Number of output classes
        pretrained (bool): Whether to load pretrained weights
        
    Attributes:
        backbone (nn.Module): The backbone network
        head (nn.Module): The classification head
        
    Example:
        >>> model = NewModel(num_classes=1000)
        >>> output = model(torch.randn(1, 3, 224, 224))
    """
```

## Testing

### Writing Tests

Create tests in the `tests/` directory:

```python
# tests/test_models.py
import pytest
import torch
from models import ModelWrapper

class TestModelWrapper:
    def test_forward_shape(self):
        """Test output shape of model forward pass."""
        model = ModelWrapper(
            model_name="convnext_tiny",
            n_patches=(16, 16),
            target_feature=["res5"]
        )
        
        # Test input
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        
        # Assertions
        assert "res5" in output
        assert output["res5"].shape == (2, 768, 16, 16)
    
    def test_invalid_model_name(self):
        """Test error handling for invalid model names."""
        with pytest.raises(KeyError):
            ModelWrapper(model_name="invalid_model")
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=.

# Run specific test file
pytest tests/test_models.py

# Run with verbose output
pytest -v
```

### Integration Tests

```python
# tests/test_integration.py
def test_training_pipeline():
    """Test complete training pipeline."""
    # Create minimal config
    cfg = {
        "student": {"model_name": "convnext_atto"},
        "teacher": {"model_name": "dinov2_vits14"},
        "train": {"max_epochs": 1, "fast_dev_run": True},
        # ... minimal config
    }
    
    # Run training
    trainer = DistillationTrainer(cfg)
    trainer.train()
    
    # Verify checkpoint exists
    assert os.path.exists("checkpoints/last.ckpt")
```

## Documentation

### Adding Documentation

1. **Code Documentation**: Add docstrings to all public functions and classes
2. **User Documentation**: Update relevant `.md` files in `docs/`
3. **Examples**: Add examples to `docs/examples.md`

### Building Documentation

```bash
cd docs
make html
# View at _build/html/index.html
```

### Documentation Standards

- Use clear, concise language
- Include code examples
- Add links to related sections
- Keep documentation up-to-date with code changes

## Pull Request Process

### 1. Create Feature Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-number
```

### 2. Make Changes

- Write clean, documented code
- Add tests for new functionality
- Update documentation as needed

### 3. Commit Changes

```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "feat: add new loss function for X

- Implement ScaleKD variant for specific use case
- Add tests and documentation
- Update configuration schema"
```

### Commit Message Format

Follow conventional commits:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes
- `refactor:` Code refactoring
- `test:` Test additions/changes
- `perf:` Performance improvements

### 4. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub with:
- Clear title and description
- Reference to related issues
- List of changes made
- Screenshots/examples if applicable

### 5. PR Review Process

- Maintainers will review your PR
- Address any feedback
- Ensure all tests pass
- Merge once approved

## Adding New Features

### Adding a New Model

1. **Create backbone implementation**:
```python
# models/backbones/newmodel.py
import torch.nn as nn

class NewModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        # Implementation
```

2. **Create wrapper**:
```python
# models/wrappers/newmodel_wrapper.py
from .base import BaseWrapper

class NewModelWrapper(BaseWrapper):
    def __init__(self, **kwargs):
        super().__init__()
        # Implementation
```

3. **Register model**:
```python
# models/model_zoo.py
ModelWrapper.MODEL_MAP['new_model'] = NewModelWrapper
```

4. **Add tests**:
```python
# tests/test_new_model.py
def test_new_model():
    # Test implementation
```

5. **Update documentation**:
- Add to `docs/models.md`
- Add configuration example
- Update API reference

### Adding a New Loss Function

1. **Implement loss**:
```python
# losses/new_loss.py
import torch.nn as nn

class NewLoss(nn.Module):
    def forward(self, student_features, teacher_features):
        # Implementation
        return loss_dict
```

2. **Register loss**:
```python
# losses/__init__.py
LOSS_REGISTRY['new_loss'] = NewLoss
```

3. **Add configuration schema**:
```yaml
# Example configuration
loss:
  losses:
    - type: "new_loss"
      weight: 1.0
      kwargs:
        param1: value1
```

### Adding a New Dataset

1. **Create dataset class**:
```python
# datasets/new_dataset.py
from torch.utils.data import Dataset

class NewDataset(Dataset):
    def __init__(self, **kwargs):
        # Implementation
    
    def __getitem__(self, idx):
        # Return image, label
```

2. **Create DataModule**:
```python
# datasets/new_datamodule.py
import lightning as L

class NewDataModule(L.LightningDataModule):
    def setup(self, stage=None):
        # Setup datasets
```

## Code Review Checklist

Before submitting a PR, ensure:

- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] No commented-out code
- [ ] No print statements (use logging)
- [ ] Type hints added
- [ ] Docstrings complete
- [ ] Configuration examples provided
- [ ] Backwards compatibility maintained

## Getting Help

- **Discord**: Join our Discord server
- **Issues**: Open a GitHub issue
- **Discussions**: Use GitHub Discussions
- **Email**: contact@example.com

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Given credit in documentation

Thank you for contributing to make this project better!