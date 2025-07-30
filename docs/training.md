# Training Guide

This guide covers the complete training pipeline, from setup to deployment, including best practices, monitoring, and optimization strategies.

## Table of Contents
- [Training Overview](#training-overview)
- [Training Pipeline](#training-pipeline)
- [Monitoring Training](#monitoring-training)
- [Advanced Training Strategies](#advanced-training-strategies)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting Training Issues](#troubleshooting-training-issues)
- [Best Practices](#best-practices)

## Training Overview

The DINOv2 Knowledge Distillation Framework uses PyTorch Lightning for training, providing:
- Automatic mixed precision training
- Distributed training support
- Comprehensive logging and monitoring
- Checkpoint management
- Learning rate scheduling

### Training Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Configuration                       │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│              DistillationTrainer                     │
│  ┌────────────┐  ┌────────────┐  ┌───────────────┐ │
│  │   Models   │  │    Data    │  │   Optimizer   │ │
│  │  (T & S)   │  │   Module   │  │   & Loss      │ │
│  └──────┬─────┘  └──────┬─────┘  └───────┬───────┘ │
└─────────┼───────────────┼────────────────┼─────────┘
          │               │                 │
┌─────────▼───────────────▼────────────────▼─────────┐
│           DistillationModule (Lightning)             │
│  ┌────────────┐  ┌────────────┐  ┌───────────────┐ │
│  │  Training  │  │ Validation │  │   Callbacks   │ │
│  │   Step     │  │    Step    │  │  & Logging    │ │
│  └────────────┘  └────────────┘  └───────────────┘ │
└─────────────────────────────────────────────────────┘
```

## Training Pipeline

### 1. Data Preparation

The framework uses DINOv2-style augmentations for optimal distillation:

```python
# datasets/augmentations.py
class DataAugmentationDINO:
    def __init__(self, global_crops_scale=(0.4, 1.0), 
                 local_crops_scale=(0.05, 0.4),
                 global_crops_number=2,
                 local_crops_number=8):
        # Global crops for main features
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.GaussianBlur(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        
        # Local crops for detailed features
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.GaussianBlur(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
```

### 2. Model Initialization

```python
# train.py
class DistillationTrainer:
    def setup_models(self):
        # Initialize teacher (frozen)
        self.teacher = DINOv2ViT(self.cfg.teacher.model_name)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
            
        # Initialize student (trainable)
        self.student = ModelWrapper(
            model_name=self.cfg.student.model_name,
            n_patches=(16, 16),
            target_feature=self.cfg.student.student_keys,
            checkpoint_path=self.cfg.student.checkpoint_path
        )
```

### 3. Loss Function Setup

```python
def setup_loss_functions(self):
    losses = []
    for loss_cfg in self.cfg.loss.losses:
        if loss_cfg.type == "scalekd":
            loss = ScaleKD(
                name=loss_cfg.kwargs.name,
                alpha=loss_cfg.kwargs.alpha,
                student_dims=student_dims,
                teacher_dims=teacher_dims,
                **loss_cfg.kwargs
            )
        losses.append((loss, loss_cfg.weight))
    return CombinedLoss(losses)
```

### 4. Training Loop

The Lightning module handles the training loop:

```python
# train/distillation_module.py
class DistillationModule(L.LightningModule):
    def training_step(self, batch, batch_idx):
        images = batch
        
        # Teacher forward (no gradients)
        with torch.no_grad():
            teacher_features = self.teacher(images)
            
        # Student forward
        student_features = self.student(images)
        
        # Compute losses
        loss_dict = self.loss_fn(student_features, teacher_features)
        total_loss = sum(loss_dict.values())
        
        # Log metrics
        self.log_dict({
            'train_loss': total_loss,
            **{f'train_{k}': v for k, v in loss_dict.items()}
        })
        
        return total_loss
```

### 5. Validation

```python
def validation_step(self, batch, batch_idx):
    images = batch
    
    with torch.no_grad():
        teacher_features = self.teacher(images)
        student_features = self.student(images)
        
        # Compute validation losses
        loss_dict = self.loss_fn(student_features, teacher_features)
        
        # Compute similarity metrics
        similarities = self.compute_similarities(
            student_features, teacher_features
        )
        
    # Log validation metrics
    self.log_dict({
        'val_loss': sum(loss_dict.values()),
        **{f'val_{k}': v for k, v in loss_dict.items()},
        **{f'val_{k}': v for k, v in similarities.items()}
    })
```

## Monitoring Training

### 1. Weights & Biases Integration

The framework integrates seamlessly with W&B for experiment tracking:

```yaml
wandb:
  project: "dinov2_distillation"
  tags: ["experiment", "convnext"]
  notes: "Testing new augmentations"
  log_model: true
  log_code: true
```

**Key Metrics Tracked:**
- Training/validation losses
- Spatial similarity scores
- Learning rate schedules
- GPU memory usage
- Training speed (iterations/second)

### 2. TensorBoard Logging

```bash
# Launch TensorBoard
tensorboard --logdir logs/

# View specific run
tensorboard --logdir logs/experiment_name/version_0
```

**TensorBoard Panels:**
- **Scalars**: Loss curves, metrics
- **Images**: Sample augmentations
- **Graphs**: Model architecture
- **Histograms**: Weight distributions

### 3. Custom Metrics

The framework tracks distillation-specific metrics:

```python
# Spatial Similarity
def compute_spatial_similarity(student_feat, teacher_feat):
    """Measure how well student mimics teacher's spatial attention"""
    student_attn = compute_attention_map(student_feat)
    teacher_attn = compute_attention_map(teacher_feat)
    return F.cosine_similarity(student_attn, teacher_attn).mean()

# Feature Alignment
def compute_feature_alignment(student_feat, teacher_feat):
    """Measure feature space alignment"""
    return 1 - torch.cdist(student_feat, teacher_feat).mean()
```

### 4. Checkpoint Monitoring

```yaml
checkpoints:
  monitor: "val_scalekd_res5_spatial_similarity"
  mode: "max"  # Higher similarity is better
  save_top_k: 3
  filename: "{epoch}-{val_loss:.3f}-{val_similarity:.3f}"
```

## Advanced Training Strategies

### 1. Progressive Distillation

Train from higher to lower feature levels:

```python
# Stage 1: Distill only res5 features
stage1_config = {
    "student_keys": ["res5"],
    "losses": [{"type": "scalekd", "name": "scalekd_res5"}],
    "max_epochs": 100
}

# Stage 2: Add res4 features
stage2_config = {
    "student_keys": ["res5", "res4"],
    "losses": [
        {"type": "scalekd", "name": "scalekd_res5"},
        {"type": "scalekd", "name": "scalekd_res4"}
    ],
    "max_epochs": 200
}
```

### 2. Knowledge Distillation with Self-Supervision

Combine distillation with self-supervised objectives:

```yaml
loss:
  losses:
    # Distillation loss
    - type: "scalekd"
      weight: 1.0
      kwargs:
        name: "scalekd_res5"
    
    # Self-supervised loss
    - type: "dino_loss"
      weight: 0.1
      kwargs:
        temperature: 0.04
```

### 3. Multi-Teacher Distillation

Distill from multiple teachers:

```python
class MultiTeacherDistillation(DistillationModule):
    def __init__(self, teachers, student, weights):
        self.teachers = nn.ModuleList(teachers)
        self.teacher_weights = weights
        
    def training_step(self, batch):
        # Get features from all teachers
        teacher_features = []
        for teacher in self.teachers:
            with torch.no_grad():
                teacher_features.append(teacher(batch))
                
        # Weighted combination
        combined_features = self.combine_teacher_features(
            teacher_features, self.teacher_weights
        )
        
        # Standard distillation
        student_features = self.student(batch)
        loss = self.loss_fn(student_features, combined_features)
```

### 4. Curriculum Learning

Gradually increase training difficulty:

```python
class CurriculumScheduler:
    def __init__(self, initial_crop_scale=(0.8, 1.0), 
                 final_crop_scale=(0.4, 1.0),
                 warmup_epochs=50):
        self.initial_scale = initial_crop_scale
        self.final_scale = final_crop_scale
        self.warmup_epochs = warmup_epochs
        
    def get_scale(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear interpolation
            alpha = epoch / self.warmup_epochs
            return (
                self.initial_scale[0] + alpha * (self.final_scale[0] - self.initial_scale[0]),
                self.initial_scale[1] + alpha * (self.final_scale[1] - self.initial_scale[1])
            )
        return self.final_scale
```

## Performance Optimization

### 1. Mixed Precision Training

Enable automatic mixed precision for 2x speedup:

```yaml
train:
  precision: 16  # Use 16-bit precision
  
# Or in code
trainer = Trainer(precision=16)
```

### 2. Gradient Accumulation

Simulate larger batches on limited memory:

```yaml
train:
  accumulate_grad_batches: 4  # Effective batch = batch_size * 4
```

### 3. Distributed Training

#### Multi-GPU (Single Node)

```yaml
train:
  accelerator: "gpu"
  devices: [0, 1, 2, 3]  # Use 4 GPUs
  strategy: "ddp"
```

#### Multi-Node Training

```yaml
train:
  num_nodes: 4
  devices: 8  # 8 GPUs per node
  strategy: "ddp"
  
# Launch with
# python train.py --config config.yaml --num_nodes 4
```

### 4. Data Loading Optimization

```yaml
data_loader:
  num_workers: 8  # Parallel data loading
  pin_memory: true  # Faster GPU transfer
  persistent_workers: true  # Keep workers alive
  prefetch_factor: 2  # Prefetch batches
```

### 5. Memory Optimization

```python
# Gradient checkpointing for large models
class MemoryEfficientStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = create_model()
        self.backbone.gradient_checkpointing = True
        
# Clear cache periodically
def on_train_batch_end(self):
    if self.global_step % 100 == 0:
        torch.cuda.empty_cache()
```

## Troubleshooting Training Issues

### 1. Loss Not Decreasing

**Symptoms**: Training loss plateaus or increases

**Solutions**:
```yaml
# Reduce learning rate
optimizer:
  kwargs:
    lr: 1e-4  # Try 10x smaller

# Check loss weights
loss:
  losses:
    - type: "scalekd"
      weight: 0.1  # Reduce if dominating

# Verify data augmentation
data_transform:
  global_crops_scale: [0.8, 1.0]  # Less aggressive
```

### 2. Out of Memory Errors

**Solutions**:
```yaml
# Reduce batch size
data_loader:
  batch_size: 8  # Smaller batches

# Enable gradient accumulation
train:
  accumulate_grad_batches: 8

# Use gradient checkpointing
student:
  model_kwargs:
    use_checkpoint: true
```

### 3. Slow Training

**Solutions**:
```yaml
# Increase workers
data_loader:
  num_workers: 16

# Enable mixed precision
train:
  precision: 16

# Use faster augmentations
data_transform:
  use_aa: false  # Disable AutoAugment
```

### 4. Poor Validation Performance

**Symptoms**: Training improves but validation doesn't

**Solutions**:
```python
# Add dropout
student:
  model_kwargs:
    drop_rate: 0.1
    drop_path_rate: 0.1

# Reduce model capacity
student:
  model_name: "convnext_tiny"  # Smaller model

# Add regularization
optimizer:
  kwargs:
    weight_decay: 0.05  # Increase
```

## Best Practices

### 1. Training Schedule

```yaml
# Warm-up phase (10% of training)
warmup:
  epochs: 30
  lr: 1e-5

# Main training
main:
  epochs: 270
  lr: 1e-3
  
# Fine-tuning
finetune:
  epochs: 50
  lr: 1e-5
```

### 2. Learning Rate Scheduling

```yaml
# Cosine annealing (recommended)
scheduler:
  type: "CosineAnnealingLR"
  kwargs:
    T_max: 300
    eta_min: 1e-6

# OneCycleLR (alternative)
scheduler:
  type: "OneCycleLR"
  kwargs:
    max_lr: 1e-3
    epochs: 300
    pct_start: 0.1
```

### 3. Monitoring Guidelines

- **Check every 10 epochs**: Overall trends
- **Check every epoch**: Learning rate, loss
- **Check every 100 steps**: GPU memory, speed
- **Save checkpoints**: Every 10 epochs + best models

### 4. Experiment Management

```bash
# Naming convention
experiments/
├── baseline_convnext_tiny_300ep/
├── ablation_no_local_crops/
├── scaling_convnext_base_vitl/
└── production_final_config/

# Track configurations
git add config/experiment_*.yaml
git commit -m "Add experiment: description"
```

### 5. Production Training

```yaml
# Production configuration
train:
  max_epochs: 300
  devices: [0,1,2,3,4,5,6,7]
  precision: 16
  gradient_clip_val: 1.0
  
optimizer:
  type: "AdamW"
  kwargs:
    lr: 1e-4
    weight_decay: 0.05
    
checkpoints:
  save_top_k: 5
  save_last: true
  
# Enable all logging
wandb:
  mode: "online"
  log_model: true
```

## Training Examples

### Example 1: Quick Experimentation

```yaml
# config/quick_test.yaml
train:
  max_epochs: 50
  fast_dev_run: false
  limit_train_batches: 0.1
  
data_loader:
  batch_size: 32
  
student:
  model_name: "convnext_atto"
```

### Example 2: High-Performance Training

```yaml
# config/high_performance.yaml
train:
  max_epochs: 400
  devices: [0,1,2,3,4,5,6,7]
  precision: 16
  accumulate_grad_batches: 4
  
student:
  model_name: "swin_small"
  
teacher:
  model_name: "dinov2_vitl14"
  
optimizer:
  scheduler:
    type: "OneCycleLR"
```

### Example 3: Memory-Constrained Training

```yaml
# config/memory_efficient.yaml
train:
  accumulate_grad_batches: 16
  
data_loader:
  batch_size: 4
  
student:
  model_name: "mobilenet_v3_small"
  model_kwargs:
    use_checkpoint: true
```