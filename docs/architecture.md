# Architecture Overview

This document provides a comprehensive overview of the DINOv2 Knowledge Distillation Framework's architecture, design principles, and component interactions.

## Table of Contents
- [High-Level Architecture](#high-level-architecture)
- [Core Components](#core-components)
- [Design Principles](#design-principles)
- [Data Flow](#data-flow)
- [Module Interactions](#module-interactions)
- [Extension Points](#extension-points)

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Configuration System                          │
│                    (YAML + OmegaConf)                            │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│                    Training Pipeline                              │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │   Data      │  │    Model     │  │      Training          │ │
│  │  Pipeline   │  │   Factory    │  │      Module            │ │
│  │             │  │              │  │  (Lightning Module)    │ │
│  └──────┬──────┘  └──────┬───────┘  └───────────┬────────────┘ │
│         │                 │                       │              │
│  ┌──────▼──────┐  ┌──────▼───────┐  ┌───────────▼────────────┐ │
│  │  Dataset    │  │   Teacher    │  │    Loss Functions      │ │
│  │  & Augment  │  │   & Student  │  │    (Scale-KD)          │ │
│  └─────────────┘  └──────────────┘  └────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│                    Monitoring & Logging                           │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │ TensorBoard │  │   Weights &  │  │    Checkpointing       │ │
│  │             │  │    Biases    │  │                        │ │
│  └─────────────┘  └──────────────┘  └────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Configuration System

The framework uses a hierarchical configuration system built on OmegaConf:

```python
# config/config.yaml structure
config/
├── wandb:          # Experiment tracking
├── student:        # Student model configuration
├── teacher:        # Teacher model configuration
├── data_transform: # Augmentation settings
├── optimizer:      # Optimization configuration
├── loss:           # Loss function settings
├── train:          # Training parameters
├── data_loader:    # Dataset configuration
└── checkpoints:    # Checkpoint settings
```

**Key Features:**
- YAML-based configuration for readability
- Command-line override support
- Type safety and validation
- Nested configuration merging

### 2. Model Architecture

#### Teacher Models (DINOv2)

The framework supports various DINOv2 Vision Transformer variants:

```python
# models/backbones/dinov2.py
class DINOv2ViT:
    - dinov2_vits14  # Small: 21M parameters
    - dinov2_vitb14  # Base: 86M parameters
    - dinov2_vitl14  # Large: 300M parameters
    - dinov2_vitg14  # Giant: 1.1B parameters
```

#### Student Models

Student models are organized by architecture family:

```
models/
├── backbones/
│   ├── convnext.py     # ConvNeXt variants
│   ├── swin.py         # Swin Transformer
│   ├── stdc.py         # STDC networks
│   ├── resnet.py       # ResNet family
│   ├── mobilenet_v2.py # MobileNetV2
│   ├── mobilenet_v3.py # MobileNetV3
│   └── ...
└── wrappers/
    ├── convnext_wrapper.py
    ├── swin_wrapper.py
    └── ...
```

#### Model Wrapper Pattern

Each student model is wrapped to provide a consistent interface:

```python
class ModelWrapper(nn.Module):
    def __init__(self, model_name, n_patches, target_feature, checkpoint_path):
        # Initialize backbone
        # Setup feature extraction
        # Load checkpoint if provided
    
    def forward(self, x):
        # Extract features at specified levels
        # Interpolate to target patch size
        # Return feature dictionary
```

### 3. Distillation Module

The `DistillationModule` is the core Lightning module that orchestrates training:

```python
# train/distillation_module.py
class DistillationModule(L.LightningModule):
    def __init__(self, teacher, student, loss_fn, optimizer_cfg):
        self.teacher = teacher  # Frozen
        self.student = student  # Trainable
        self.loss_fn = loss_fn
        
    def training_step(self, batch, batch_idx):
        # Forward pass through both models
        # Compute distillation loss
        # Log metrics
        
    def validation_step(self, batch, batch_idx):
        # Evaluate without gradients
        # Compute validation metrics
```

### 4. Loss Functions

#### Scale-KD Loss

The primary distillation loss that transfers multi-scale attention patterns:

```python
# losses/scalekd.py
class ScaleKD(nn.Module):
    def __init__(self, alpha, student_dims, teacher_dims, ...):
        # Initialize attention projectors
        # Setup learnable parameters
        
    def forward(self, preds_S, preds_T):
        # Project student features
        # Compute spatial and frequency losses
        # Return weighted combination
```

**Key Components:**
- **Attention Projector**: Maps student features to teacher space
- **Spatial Loss**: Preserves spatial attention patterns
- **Frequency Loss**: Maintains frequency domain characteristics
- **Multi-scale Support**: Different configurations per feature level

### 5. Data Pipeline

#### Dataset Management

```python
# datasets/CustomDataset.py
class CustomDataset(Dataset):
    - Flexible directory-based loading
    - Support for multiple image formats
    - Efficient caching mechanisms
```

#### Augmentation Pipeline

```python
# datasets/augmentations.py
class DataAugmentationDINO:
    - Global and local crop strategies
    - Color jittering and Gaussian blur
    - Solarization and random erasing
    - Consistent with DINOv2 training
```

### 6. Training Pipeline

The training pipeline integrates all components:

```python
# train.py
class DistillationTrainer:
    def __init__(self, cfg):
        # Load configuration
        # Initialize components
        
    def setup_models(self):
        # Create teacher (frozen)
        # Create student (trainable)
        # Setup model wrappers
        
    def setup_training(self):
        # Create data loaders
        # Initialize loss functions
        # Setup optimizers
        # Configure callbacks
        
    def train(self):
        # Create Lightning trainer
        # Start training loop
```

## Design Principles

### 1. Modularity

Each component is self-contained and can be:
- Tested independently
- Replaced without affecting others
- Extended for new use cases

### 2. Flexibility

The framework supports:
- Multiple teacher architectures
- Diverse student models
- Custom loss functions
- Various training strategies

### 3. Scalability

Built-in support for:
- Multi-GPU training (DDP)
- Mixed precision training
- Gradient accumulation
- Distributed data loading

### 4. Reproducibility

Ensures consistent results through:
- Fixed random seeds
- Deterministic operations
- Configuration versioning
- Comprehensive logging

## Data Flow

### Training Flow

```
1. Image Batch
   ↓
2. Data Augmentation
   ↓
3. Teacher Forward Pass (no gradients)
   ↓
4. Student Forward Pass (with gradients)
   ↓
5. Feature Extraction at Multiple Levels
   ↓
6. Loss Computation (Scale-KD)
   ↓
7. Backward Pass (student only)
   ↓
8. Optimizer Step
   ↓
9. Metric Logging
```

### Feature Extraction Flow

```
Input Image (B, 3, H, W)
   ↓
Backbone Network
   ├─→ res2 features (B, C1, H/4, W/4)
   ├─→ res3 features (B, C2, H/8, W/8)
   ├─→ res4 features (B, C3, H/16, W/16)
   └─→ res5 features (B, C4, H/32, W/32)
        ↓
Feature Interpolation
   ↓
Aligned Features (B, C, P, P)
```

## Module Interactions

### Configuration → Training

```python
# Configuration drives component initialization
config = OmegaConf.load("config.yaml")
   ↓
trainer = DistillationTrainer(config)
   ↓
# Components created based on config
teacher = create_teacher(config.teacher)
student = create_student(config.student)
loss_fn = create_loss(config.loss)
```

### Model Factory Pattern

```python
# models/model_zoo.py
MODEL_MAP = {
    'convnext_tiny': ConvNextWrapper,
    'swin_tiny': SwinWrapper,
    'stdc_1': STDCWrapper,
    ...
}

def create_model(config):
    wrapper_class = MODEL_MAP[config.model_name]
    return wrapper_class(**config.kwargs)
```

### Loss Function Registry

```python
# losses/__init__.py
LOSS_REGISTRY = {
    'scalekd': ScaleKD,
    'mse': MSELoss,
    'cosine': CosineSimilarityLoss,
}

def create_loss(loss_configs):
    losses = []
    for cfg in loss_configs:
        loss_class = LOSS_REGISTRY[cfg.type]
        losses.append(loss_class(**cfg.kwargs))
    return CombinedLoss(losses, weights)
```

## Extension Points

### Adding New Student Models

1. Implement backbone in `models/backbones/`
2. Create wrapper in `models/wrappers/`
3. Register in `MODEL_MAP`
4. Update configuration schema

### Adding New Loss Functions

1. Implement loss in `losses/`
2. Register in `LOSS_REGISTRY`
3. Add configuration parameters
4. Update documentation

### Custom Data Augmentations

1. Extend `DataAugmentationDINO`
2. Add new transformation methods
3. Update configuration options
4. Ensure compatibility with existing pipeline

### Custom Training Strategies

1. Extend `DistillationModule`
2. Override training/validation steps
3. Add custom callbacks
4. Update trainer configuration

## Performance Considerations

### Memory Optimization

- Teacher model in eval mode (no gradients)
- Gradient checkpointing for large students
- Mixed precision training support
- Efficient data loading with multiple workers

### Computational Efficiency

- Parallel data augmentation
- Cached feature extraction
- Optimized loss computation
- Distributed training support

### Monitoring and Debugging

- Comprehensive metric logging
- Gradient flow visualization
- Memory usage tracking
- Performance profiling hooks

## Best Practices

### Configuration Management

- Use version control for configs
- Document all parameters
- Provide sensible defaults
- Validate configurations early

### Model Development

- Follow consistent naming conventions
- Implement comprehensive tests
- Document expected inputs/outputs
- Provide usage examples

### Training Workflow

- Start with small experiments
- Monitor metrics closely
- Use validation for hyperparameter tuning
- Save checkpoints regularly

### Debugging

- Enable verbose logging
- Use smaller datasets for testing
- Verify data augmentation visually
- Check gradient flow regularly