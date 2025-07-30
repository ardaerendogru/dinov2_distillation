# Configuration Guide

This guide provides a comprehensive reference for all configuration options in the DINOv2 Knowledge Distillation Framework.

## Table of Contents
- [Configuration Overview](#configuration-overview)
- [Configuration Sections](#configuration-sections)
- [Advanced Configuration](#advanced-configuration)
- [Configuration Examples](#configuration-examples)
- [Best Practices](#best-practices)

## Configuration Overview

The framework uses YAML files for configuration, powered by OmegaConf. Configurations can be:
- Loaded from files
- Overridden via command line
- Merged from multiple sources
- Validated at runtime

### Basic Structure

```yaml
# config/config.yaml
wandb:        # Experiment tracking
student:      # Student model settings
teacher:      # Teacher model settings
optimizer:    # Optimization parameters
loss:         # Loss function configuration
train:        # Training settings
data_loader:  # Data loading configuration
checkpoints:  # Checkpoint management
```

### Command Line Overrides

Override any configuration parameter from the command line:

```bash
python train.py --config config/config.yaml \
    --student.model_name swin_tiny \
    --optimizer.kwargs.lr 5e-4 \
    --train.max_epochs 100
```

## Configuration Sections

### 1. Weights & Biases (wandb)

Configure experiment tracking and logging:

```yaml
wandb:
  project: "dinov2_distillation"      # W&B project name
  entity: null                        # W&B team/entity (optional)
  name: null                          # Run name (auto-generated if null)
  tags: ["distillation", "convnext"]  # Searchable tags
  notes: "Distilling DINOv2 to ConvNeXt-Tiny"  # Run description
  mode: "online"                      # "online", "offline", or "disabled"
  log_model: true                     # Log model checkpoints to W&B
  log_code: true                      # Save code to W&B
```

### 2. Student Model Configuration

Define the student model architecture and settings:

```yaml
student:
  model_name: "convnext_tiny"         # Model architecture
  student_keys: ["res5", "res4"]      # Feature levels to extract
  checkpoint_path: null               # Pre-trained weights (optional)
  
  # Model-specific parameters (passed to model constructor)
  model_kwargs:
    drop_path_rate: 0.1               # Stochastic depth rate
    layer_scale_init_value: 1e-6      # Layer scale initialization
```

#### Available Student Models

**ConvNeXt Family:**
- `convnext_atto`: Ultra-small (3.5M params)
- `convnext_pico`: Very small (9M params)
- `convnext_nano`: Small (15M params)
- `convnext_tiny`: Standard (28M params)
- `convnext_base`: Large (88M params)

**Swin Transformer:**
- `swin_tiny`: 28M parameters
- `swin_small`: 50M parameters

**STDC Networks:**
- `stdc_1`: Lightweight (8M params)
- `stdc_2`: Standard (12M params)

**ResNet Family:**
- `resnet_18`, `resnet_34`, `resnet_50`, `resnet_101`

**MobileNet:**
- `mobilenet_v2`, `mobilenet_v3_small`, `mobilenet_v3_large`

**Other Architectures:**
- `mit_b0` to `mit_b5`: MIT (SegFormer) backbones
- `darknet_n` to `darknet_x`: DarkNet variants
- Various TIMM models (EfficientNet, EdgeNext, etc.)

### 3. Teacher Model Configuration

Configure the DINOv2 teacher model:

```yaml
teacher:
  model_name: "dinov2_vits14"         # DINOv2 variant
  patch_size: 14                      # Patch size (14 or 16)
  freeze: true                        # Keep teacher frozen
  
  # Feature extraction settings
  feature_dims:
    res2: 384                         # Dimension at each level
    res3: 384
    res4: 384
    res5: 384
```

#### Available Teacher Models

- `dinov2_vits14`: Small, 21M params, 384 dims
- `dinov2_vitb14`: Base, 86M params, 768 dims
- `dinov2_vitl14`: Large, 300M params, 1024 dims
- `dinov2_vitg14`: Giant, 1.1B params, 1536 dims

### 4. Data Transformation Configuration

Configure data augmentation pipeline:

```yaml
data_transform:
  # Global crop settings
  global_crops_scale: [0.4, 1.0]      # Scale range for global crops
  global_crops_size: 224              # Size of global crops
  global_crops_number: 2              # Number of global crops
  
  # Local crop settings
  local_crops_scale: [0.05, 0.4]      # Scale range for local crops
  local_crops_size: 96                # Size of local crops
  local_crops_number: 8               # Number of local crops
  
  # Augmentation parameters
  color_jitter: 0.4                   # Color jitter strength
  blur_prob: 0.5                      # Gaussian blur probability
  solarize_prob: 0.1                  # Solarization probability
  normalize_mean: [0.485, 0.456, 0.406]  # ImageNet normalization
  normalize_std: [0.229, 0.224, 0.225]
```

### 5. Optimizer Configuration

Configure optimization and learning rate scheduling:

```yaml
optimizer:
  type: "AdamW"                       # Optimizer type
  kwargs:
    lr: 1e-3                          # Learning rate
    betas: [0.9, 0.999]               # Adam betas
    weight_decay: 0.01                # Weight decay
    eps: 1e-8                         # Epsilon for numerical stability
  
  # Learning rate scheduler
  scheduler:
    type: "CosineAnnealingLR"         # Scheduler type
    kwargs:
      T_max: 240                      # Total epochs
      eta_min: 1e-5                   # Minimum learning rate
    monitor: "val_loss"               # Metric to monitor
    interval: "epoch"                 # Update frequency
    frequency: 1                      # Update every N intervals
    
  # Alternative scheduler: OneCycleLR
  # scheduler:
  #   type: "OneCycleLR"
  #   kwargs:
  #     max_lr: 1e-3
  #     epochs: 240
  #     steps_per_epoch: 1000
  #     pct_start: 0.1
```

### 6. Loss Function Configuration

Configure distillation losses:

```yaml
loss:
  losses:
    # Scale-KD loss for res4 features
    - type: "scalekd"
      weight: 1.0                     # Loss weight
      kwargs:
        alpha: [0.08, 0.06]           # Spatial/frequency weights
        window_shapes: [1, 1]         # Attention window size
        self_query: false             # Use learnable queries
        softmax_scale: [5.0, 5.0]     # Temperature scaling
        num_heads: 16                 # Number of attention heads
        name: "scalekd_res4"          # Must follow pattern
        
    # Scale-KD loss for res5 features
    - type: "scalekd"
      weight: 1.0
      kwargs:
        alpha: [0.08, 0.06]
        window_shapes: [1, 1]
        self_query: true              # Use self-attention
        softmax_scale: [5.0, 5.0]
        num_heads: 24
        name: "scalekd_res5"
```

#### Scale-KD Parameters Explained

- **alpha**: Controls spatial vs frequency component weights
- **window_shapes**: Attention window size ([1,1] = global attention)
- **self_query**: Whether to use feature maps as queries
- **softmax_scale**: Temperature for attention softmax
- **num_heads**: Should match teacher's attention heads at that level

### 7. Training Configuration

Configure training parameters:

```yaml
train:
  max_epochs: 240                     # Maximum training epochs
  accelerator: "gpu"                  # Training accelerator
  devices: [0, 1]                     # GPU device IDs
  num_nodes: 1                        # Number of nodes
  strategy: "ddp_find_unused_parameters_true"  # DDP strategy
  
  # Advanced settings
  accumulate_grad_batches: 1          # Gradient accumulation steps
  gradient_clip_val: 1.0              # Gradient clipping value
  gradient_clip_algorithm: "norm"     # Clipping algorithm
  
  # Mixed precision training
  precision: 16                       # 16 for mixed precision, 32 for full
  
  # Early stopping
  early_stopping:
    monitor: "val_loss"
    patience: 20
    mode: "min"
    
  # Resume training
  resume_from_checkpoint: null        # Path to checkpoint
```

#### Available Training Strategies

- `"ddp"`: Standard distributed data parallel
- `"ddp_find_unused_parameters_true"`: DDP with unused parameter detection
- `"dp"`: Data parallel (single machine)
- `"deepspeed"`: DeepSpeed integration
- `"fsdp"`: Fully sharded data parallel

### 8. Data Loader Configuration

Configure dataset and data loading:

```yaml
data_loader:
  # Training data
  data_dir: ["/path/to/train1", "/path/to/train2"]  # Training directories
  
  # Validation data (optional)
  val_dir: ["/path/to/val"]           # Validation directories
  val_split: 0.1                      # Use 10% of train data for val if val_dir not provided
  
  # Loader settings
  batch_size: 32                      # Batch size per GPU
  num_workers: 8                      # Data loading workers
  pin_memory: true                    # Pin memory for faster GPU transfer
  persistent_workers: true            # Keep workers alive between epochs
  prefetch_factor: 2                  # Number of batches to prefetch
  
  # Data sampling
  shuffle: true                       # Shuffle training data
  drop_last: true                     # Drop incomplete batches
  
  # Distributed sampling
  replace_sampler_ddp: true           # Use distributed sampler
```

### 9. Checkpoint Configuration

Configure model checkpointing:

```yaml
checkpoints:
  dirpath: "checkpoints"              # Directory for checkpoints
  filename: "{epoch}-{val_loss:.3f}"  # Checkpoint naming pattern
  monitor: "val_scalekd_res5_spatial_similarity"  # Metric to monitor
  mode: "max"                         # "max" or "min"
  save_top_k: 3                       # Keep best K checkpoints
  save_last: true                     # Always save last checkpoint
  every_n_epochs: 10                  # Save every N epochs
  save_on_train_epoch_end: false      # Save at epoch end
  
  # Auto-resume
  auto_resume: true                   # Resume from last checkpoint
  
  # Model export
  export_onnx: false                  # Export to ONNX format
  export_torchscript: false           # Export to TorchScript
```

## Advanced Configuration

### Multi-Stage Training

Configure different training stages:

```yaml
# Stage 1: Warm-up with frozen backbone
stage1:
  student:
    freeze_backbone: true
  optimizer:
    kwargs:
      lr: 1e-4
  train:
    max_epochs: 50

# Stage 2: Full training
stage2:
  student:
    freeze_backbone: false
  optimizer:
    kwargs:
      lr: 1e-3
  train:
    max_epochs: 200
```

### Custom Loss Combinations

Combine multiple loss functions:

```yaml
loss:
  losses:
    # Scale-KD losses
    - type: "scalekd"
      weight: 1.0
      kwargs:
        name: "scalekd_res5"
        
    # MSE loss for feature matching
    - type: "mse"
      weight: 0.1
      kwargs:
        reduction: "mean"
        
    # Cosine similarity loss
    - type: "cosine"
      weight: 0.5
      kwargs:
        dim: 1
```

### Distributed Training Configuration

For multi-node training:

```yaml
train:
  num_nodes: 4                        # Number of nodes
  devices: 8                          # GPUs per node
  strategy: "ddp"
  
  # SLURM configuration
  plugins:
    - class_path: "lightning.pytorch.plugins.environments.SLURMEnvironment"
      init_args:
        auto_requeue: true
        
  # Network settings
  ddp_kwargs:
    find_unused_parameters: false
    broadcast_buffers: false
    bucket_cap_mb: 25
```

## Configuration Examples

### Example 1: Lightweight Mobile Deployment

```yaml
# Distill to MobileNetV3 for mobile deployment
student:
  model_name: "mobilenet_v3_small"
  student_keys: ["res4", "res3"]

teacher:
  model_name: "dinov2_vits14"  # Use smaller teacher

optimizer:
  type: "AdamW"
  kwargs:
    lr: 5e-4
    weight_decay: 0.05

loss:
  losses:
    - type: "scalekd"
      weight: 1.0
      kwargs:
        alpha: [0.1, 0.05]
        num_heads: 8
        name: "scalekd_res4"

train:
  max_epochs: 300
  precision: 16  # Mixed precision for faster training

data_loader:
  batch_size: 64  # Larger batch for small model
```

### Example 2: High-Accuracy Server Deployment

```yaml
# Distill to Swin-Small for high accuracy
student:
  model_name: "swin_small"
  student_keys: ["res5", "res4", "res3", "res2"]

teacher:
  model_name: "dinov2_vitl14"  # Large teacher

optimizer:
  type: "AdamW"
  kwargs:
    lr: 1e-4
    weight_decay: 0.01
  scheduler:
    type: "OneCycleLR"
    kwargs:
      max_lr: 1e-3
      epochs: 400
      pct_start: 0.05

loss:
  losses:
    - type: "scalekd"
      weight: 1.0
      kwargs:
        name: "scalekd_res5"
        num_heads: 16
    - type: "scalekd"
      weight: 0.8
      kwargs:
        name: "scalekd_res4"
        num_heads: 16
    - type: "scalekd"
      weight: 0.6
      kwargs:
        name: "scalekd_res3"
        num_heads: 8
    - type: "scalekd"
      weight: 0.4
      kwargs:
        name: "scalekd_res2"
        num_heads: 8

train:
  max_epochs: 400
  devices: [0, 1, 2, 3]
  accumulate_grad_batches: 4
```

### Example 3: Quick Experimentation

```yaml
# Fast iteration for testing
student:
  model_name: "convnext_atto"
  student_keys: ["res5"]

teacher:
  model_name: "dinov2_vits14"

optimizer:
  type: "Adam"
  kwargs:
    lr: 1e-3

loss:
  losses:
    - type: "scalekd"
      weight: 1.0
      kwargs:
        name: "scalekd_res5"
        num_heads: 12

train:
  max_epochs: 50
  devices: [0]
  fast_dev_run: false  # Set to true for debugging

data_loader:
  batch_size: 128
  num_workers: 4
```

## Best Practices

### 1. Configuration Management

- **Version Control**: Always version control your configuration files
- **Naming Convention**: Use descriptive names: `convnext_tiny_imagenet_300ep.yaml`
- **Documentation**: Add comments explaining non-obvious parameters
- **Validation**: Test configurations with small runs first

### 2. Hyperparameter Selection

- **Learning Rate**: Start with 1e-3 for small models, 1e-4 for large
- **Batch Size**: Larger batches generally work better (within memory limits)
- **Weight Decay**: 0.01-0.05 for ConvNets, 0.01 for Transformers
- **Epochs**: 200-300 for full training, 50-100 for fine-tuning

### 3. Performance Optimization

- **Multi-GPU**: Use all available GPUs with DDP
- **Mixed Precision**: Enable for 2x speedup with minimal accuracy loss
- **Gradient Accumulation**: Simulate larger batches on limited memory
- **Data Loading**: Set num_workers = 2-4 × num_GPUs

### 4. Debugging Configuration

```yaml
# Debug configuration
debug:
  enabled: true
  
train:
  fast_dev_run: 5  # Run only 5 batches
  limit_train_batches: 0.1  # Use 10% of data
  limit_val_batches: 0.1
  num_sanity_val_steps: 2
  
data_loader:
  batch_size: 4  # Small batch for debugging
  num_workers: 0  # Disable multiprocessing
```

### 5. Configuration Templates

Create base configurations to extend:

```yaml
# config/base.yaml
defaults:
  - data: imagenet
  - model: convnext
  - training: standard

# config/experiments/my_experiment.yaml  
defaults:
  - /base

student:
  model_name: convnext_tiny
  
# Override specific parameters
optimizer:
  kwargs:
    lr: 5e-4
```