# Troubleshooting Guide

This guide helps you diagnose and resolve common issues when using the DINOv2 Knowledge Distillation Framework.

## Table of Contents
- [Installation Issues](#installation-issues)
- [Training Issues](#training-issues)
- [Memory Issues](#memory-issues)
- [Performance Issues](#performance-issues)
- [Model Issues](#model-issues)
- [Data Pipeline Issues](#data-pipeline-issues)
- [Distributed Training Issues](#distributed-training-issues)
- [Debugging Tools](#debugging-tools)

## Installation Issues

### Issue: CUDA Not Available

**Symptoms:**
```python
RuntimeError: CUDA out of memory
torch.cuda.is_available() returns False
```

**Solutions:**

1. Check CUDA installation:
```bash
nvidia-smi  # Should show GPU info
nvcc --version  # Should show CUDA version
```

2. Install correct PyTorch version:
```bash
# For CUDA 11.7
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117

# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

3. Verify PyTorch CUDA:
```python
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.device_count())  # Number of GPUs
print(torch.version.cuda)  # CUDA version
```

### Issue: Missing Dependencies

**Symptoms:**
```
ModuleNotFoundError: No module named 'timm'
ImportError: cannot import name 'DINOv2ViT'
```

**Solutions:**

1. Install all requirements:
```bash
pip install -r requirements.txt
```

2. For specific missing packages:
```bash
pip install timm>=0.9.0
pip install lightning>=2.0.0
pip install omegaconf>=2.3.0
```

3. Check Python path:
```python
import sys
print(sys.path)  # Should include project directory
```

## Training Issues

### Issue: Loss Not Decreasing

**Symptoms:**
- Training loss plateaus early
- Loss oscillates without improvement
- Validation loss increases while training loss decreases

**Solutions:**

1. **Check learning rate:**
```yaml
# Try reducing learning rate
optimizer:
  kwargs:
    lr: 1e-4  # Was 1e-3
```

2. **Verify data augmentation:**
```yaml
# Reduce augmentation strength
data_transform:
  global_crops_scale: [0.8, 1.0]  # Was [0.4, 1.0]
  color_jitter_strength: 0.2  # Was 0.4
```

3. **Check loss weights:**
```yaml
# Balance loss components
loss:
  losses:
    - type: "scalekd"
      weight: 0.5  # Was 1.0
      kwargs:
        alpha: [0.05, 0.03]  # Reduce weights
```

4. **Debug with smaller dataset:**
```python
# Overfit on small dataset first
train:
  limit_train_batches: 10  # Use only 10 batches
  limit_val_batches: 5
```

### Issue: NaN or Inf Loss

**Symptoms:**
```
Loss: nan
RuntimeError: Loss is inf or nan
```

**Solutions:**

1. **Enable gradient clipping:**
```yaml
train:
  gradient_clip_val: 1.0
  gradient_clip_algorithm: "norm"
```

2. **Add numerical stability:**
```python
# In loss computation
def stable_loss(pred, target):
    pred = torch.clamp(pred, min=-10, max=10)
    eps = 1e-8
    return F.mse_loss(pred + eps, target + eps)
```

3. **Check for division by zero:**
```python
# In Scale-KD loss
attention_weights = F.softmax(scores / (temperature + 1e-8), dim=-1)
```

4. **Use mixed precision carefully:**
```yaml
train:
  precision: 32  # Try full precision first
```

### Issue: Poor Validation Performance

**Symptoms:**
- Large train/val gap
- Validation metrics not improving
- Overfitting

**Solutions:**

1. **Add regularization:**
```yaml
student:
  model_kwargs:
    drop_rate: 0.1
    drop_path_rate: 0.1

optimizer:
  kwargs:
    weight_decay: 0.05  # Increase
```

2. **Use data augmentation:**
```yaml
data_transform:
  # Add more augmentation
  solarize_prob: 0.2
  random_erasing_prob: 0.1
```

3. **Early stopping:**
```yaml
callbacks:
  early_stopping:
    monitor: "val_loss"
    patience: 20
    mode: "min"
```

## Memory Issues

### Issue: CUDA Out of Memory

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate X MiB
```

**Solutions:**

1. **Reduce batch size:**
```yaml
data_loader:
  batch_size: 16  # Was 32
```

2. **Use gradient accumulation:**
```yaml
train:
  accumulate_grad_batches: 4  # Effective batch = 16 * 4 = 64
```

3. **Enable gradient checkpointing:**
```python
# For compatible models
student:
  model_kwargs:
    use_checkpoint: true
```

4. **Clear cache periodically:**
```python
def on_train_batch_end(self):
    if self.global_step % 100 == 0:
        torch.cuda.empty_cache()
```

5. **Use smaller models:**
```yaml
# Switch to smaller variants
student:
  model_name: "convnext_atto"  # Was convnext_tiny
teacher:
  model_name: "dinov2_vits14"  # Was dinov2_vitb14
```

### Issue: CPU Memory Overflow

**Symptoms:**
- System becomes unresponsive
- Process killed by OOM killer

**Solutions:**

1. **Reduce number of workers:**
```yaml
data_loader:
  num_workers: 4  # Was 8
  persistent_workers: false  # Free memory between epochs
```

2. **Limit dataset size in memory:**
```python
class MemoryEfficientDataset(Dataset):
    def __init__(self, image_paths):
        # Store paths, not images
        self.paths = image_paths
        
    def __getitem__(self, idx):
        # Load on demand
        return Image.open(self.paths[idx])
```

## Performance Issues

### Issue: Slow Training

**Symptoms:**
- Low GPU utilization
- Long epoch times
- Data loading bottleneck

**Solutions:**

1. **Optimize data loading:**
```yaml
data_loader:
  num_workers: 8  # Increase if CPU allows
  pin_memory: true
  persistent_workers: true
  prefetch_factor: 2
```

2. **Enable mixed precision:**
```yaml
train:
  precision: 16  # Automatic mixed precision
```

3. **Profile training:**
```python
# Enable profiler
trainer = Trainer(
    profiler="simple",  # or "advanced"
    max_epochs=1
)
```

4. **Check data pipeline:**
```python
# Time data loading
import time

start = time.time()
for batch in train_loader:
    load_time = time.time() - start
    print(f"Batch load time: {load_time:.3f}s")
    break
```

### Issue: Low GPU Utilization

**Symptoms:**
- nvidia-smi shows < 90% GPU usage
- Training is CPU-bound

**Solutions:**

1. **Increase batch size:**
```yaml
data_loader:
  batch_size: 64  # Maximize GPU memory usage
```

2. **Reduce data augmentation on CPU:**
```python
# Move some augmentations to GPU
class GPUAugmentation(nn.Module):
    def forward(self, x):
        # Apply augmentations on GPU
        return F.dropout2d(x, p=0.1)
```

3. **Use faster data formats:**
```python
# Convert dataset to efficient format
# Use WebDataset, LMDB, or HDF5
```

## Model Issues

### Issue: Model Not Converging

**Symptoms:**
- Loss stuck at high value
- Metrics don't improve
- Model outputs constant values

**Solutions:**

1. **Check initialization:**
```python
# Verify model initialization
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"{name}: mean={param.mean():.3f}, std={param.std():.3f}")
```

2. **Use pretrained weights:**
```yaml
student:
  checkpoint_path: "path/to/imagenet_pretrained.pth"
```

3. **Gradual unfreezing:**
```python
# Freeze backbone initially
for param in model.backbone.parameters():
    param.requires_grad = False

# Unfreeze after N epochs
def on_train_epoch_start(self):
    if self.current_epoch == 10:
        for param in self.model.backbone.parameters():
            param.requires_grad = True
```

### Issue: Feature Dimension Mismatch

**Symptoms:**
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied
ValueError: Expected features of size X, got Y
```

**Solutions:**

1. **Check model compatibility:**
```python
# Verify dimensions
print(f"Student dims: {student_model.feature_dims}")
print(f"Teacher dims: {teacher_model.embed_dim}")
```

2. **Update configuration:**
```yaml
# Ensure correct feature levels
student:
  student_keys: ["res5", "res4"]  # Must match model architecture
```

## Data Pipeline Issues

### Issue: Dataset Not Loading

**Symptoms:**
```
FileNotFoundError: No such file or directory
ValueError: num_samples should be a positive integer
```

**Solutions:**

1. **Verify paths:**
```python
import os
data_dir = "/path/to/data"
print(f"Directory exists: {os.path.exists(data_dir)}")
print(f"Files: {len(os.listdir(data_dir))}")
```

2. **Check image formats:**
```python
# Supported formats
valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
```

3. **Validate dataset:**
```python
# Test dataset loading
dataset = CustomDataset(data_dirs=[data_dir])
print(f"Dataset size: {len(dataset)}")
sample = dataset[0]
print(f"Sample shape: {sample.shape}")
```

### Issue: Augmentation Errors

**Symptoms:**
```
ValueError: empty range for randrange()
RuntimeError: stack expects each tensor to be equal size
```

**Solutions:**

1. **Check image sizes:**
```python
# Ensure minimum size
min_size = 224  # For 224x224 crops
transform = transforms.Compose([
    transforms.Resize(min_size + 32),  # Add margin
    transforms.RandomCrop(min_size),
])
```

2. **Handle edge cases:**
```python
def safe_transform(image):
    try:
        return transform(image)
    except Exception as e:
        print(f"Transform failed: {e}")
        # Return center crop as fallback
        return fallback_transform(image)
```

## Distributed Training Issues

### Issue: DDP Hangs

**Symptoms:**
- Training freezes at initialization
- Processes wait indefinitely

**Solutions:**

1. **Check NCCL environment:**
```bash
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1  # If using different GPU types
```

2. **Verify network connectivity:**
```python
# Test distributed init
import torch.distributed as dist
dist.init_process_group(backend="nccl", timeout=timedelta(minutes=5))
```

3. **Use proper strategy:**
```yaml
train:
  strategy: "ddp_find_unused_parameters_true"  # For models with unused params
```

### Issue: Uneven GPU Memory Usage

**Symptoms:**
- One GPU has much higher memory usage
- OOM on single GPU in multi-GPU setup

**Solutions:**

1. **Balance batch distribution:**
```python
# Ensure even distribution
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
```

2. **Use DistributedDataParallel:**
```yaml
train:
  strategy: "ddp"  # Instead of dp
  sync_batchnorm: true
```

## Debugging Tools

### 1. Enable Verbose Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or in config
logging_level: "DEBUG"
```

### 2. Use Debugging Hooks

```python
def debug_forward_hook(module, input, output):
    print(f"{module.__class__.__name__}:")
    print(f"  Input shape: {input[0].shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")

# Register hooks
for name, module in model.named_modules():
    module.register_forward_hook(debug_forward_hook)
```

### 3. Gradient Checking

```python
def check_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"{name}: grad_norm={grad_norm:.6f}")
            if grad_norm == 0:
                print(f"  WARNING: Zero gradient!")
            if not torch.isfinite(param.grad).all():
                print(f"  WARNING: Non-finite gradient!")
```

### 4. Visualize Training

```python
# Save sample predictions
def visualize_predictions(model, dataloader, save_path):
    model.eval()
    images, features = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            images.append(batch)
            features.append(model(batch))
            if len(images) >= 4:
                break
    
    # Plot images and features
    plot_features(images, features, save_path)
```

### 5. Common Debug Commands

```bash
# Check GPU status
watch -n 1 nvidia-smi

# Monitor training logs
tail -f logs/training.log

# Profile memory usage
py-spy top -- python train.py

# Check disk I/O
iotop -o

# Monitor CPU usage
htop
```

## Getting Help

If you're still experiencing issues:

1. **Check existing issues**: [GitHub Issues](https://github.com/ardaerendogru/dinov2_distillation/issues)
2. **Enable debug mode**: Set `logging_level: "DEBUG"`
3. **Collect information**:
   - Full error traceback
   - Configuration file
   - Environment details (PyTorch version, CUDA version, etc.)
4. **Create minimal reproduction**: Isolate the issue with minimal code
5. **Ask for help**: Create a new issue with all relevant information