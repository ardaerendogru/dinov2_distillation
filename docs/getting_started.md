# Getting Started Guide

This guide will help you set up the DINOv2 Knowledge Distillation Framework and run your first distillation experiment.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Understanding the Output](#understanding-the-output)
- [Next Steps](#next-steps)

## Prerequisites

Before installing the framework, ensure you have:

- **Python 3.8 or higher**
- **CUDA-compatible GPU(s)** (recommended for training)
- **CUDA toolkit** (version 11.7 or higher recommended)
- **Git** for cloning the repository
- **At least 16GB of GPU memory** for training with default settings

### System Requirements

- **Operating System**: Linux (Ubuntu 18.04+), macOS, or Windows with WSL2
- **RAM**: Minimum 16GB, 32GB+ recommended
- **Storage**: At least 50GB free space for datasets and checkpoints

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/ardaerendogru/dinov2_distillation.git
cd dinov2_distillation
```

### Step 2: Create a Virtual Environment

We recommend using `uv` for fast package management:

```bash
# Install uv if you haven't already
pip install uv

# Create virtual environment
uv venv .venv

# Activate the environment
source .venv/bin/activate  # On Linux/macOS
# or
.venv\Scripts\activate  # On Windows
```

Alternatively, you can use standard Python venv:

```bash
python -m venv .venv
source .venv/bin/activate  # On Linux/macOS
```

### Step 3: Install Dependencies

Using uv (recommended):
```bash
uv pip install -r requirements.txt
```

Using standard pip:
```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import lightning; print(f'Lightning version: {lightning.__version__}')"
```

## Quick Start

### Step 1: Prepare Your Dataset

The framework expects image datasets organized in directories. Create a directory structure like:

```
data/
├── train/
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
└── val/
    ├── image1.jpg
    ├── image2.png
    └── ...
```

### Step 2: Create a Configuration File

Create a configuration file `config/my_first_distillation.yaml`:

```yaml
# Basic configuration for ConvNeXt-Tiny distillation
wandb:
  project: "my_first_distillation"
  tags: ["test", "convnext", "dinov2"]
  notes: "First distillation experiment"

student:
  model_name: convnext_tiny
  student_keys: [res5, res4]  # Feature levels to distill

teacher:
  model_name: dinov2_vits14  # Small DINOv2 model for quick testing

optimizer:
  type: AdamW
  kwargs:
    lr: 1e-3
    weight_decay: 0.01
  scheduler:
    type: CosineAnnealingLR
    kwargs:
      T_max: 50
      eta_min: 1e-5

loss:
  losses:
    - type: scalekd
      weight: 1.0
      kwargs:
        alpha: [0.08, 0.06]
        window_shapes: [1, 1]
        self_query: False
        softmax_scale: [5.0, 5.0]
        num_heads: 16
        name: scalekd_res4

train:
  max_epochs: 50
  accelerator: gpu
  devices: [0]  # Use first GPU
  num_nodes: 1

data_loader:
  data_dir: [/path/to/your/data/train]
  val_dir: [/path/to/your/data/val]
  batch_size: 16
  num_workers: 4

checkpoints:
  dirpath: checkpoints/my_first_run
  monitor: val_scalekd_res4_spatial_similarity
  mode: max
  save_top_k: 3
```

### Step 3: Start Training

```bash
python train.py --config config/my_first_distillation.yaml
```

### Step 4: Monitor Training

The framework provides multiple ways to monitor training:

#### TensorBoard
```bash
tensorboard --logdir logs/
```
Then open http://localhost:6006 in your browser.

#### Weights & Biases
If configured, training metrics will be automatically logged to W&B. Check your project dashboard at https://wandb.ai/

## Understanding the Output

### Directory Structure After Training

```
dinov2_distillation/
├── checkpoints/
│   └── my_first_run/
│       ├── epoch=10-val_loss=0.123.ckpt
│       └── ...
├── logs/
│   └── my_first_distillation/
│       └── version_0/
│           ├── events.out.tfevents...
│           └── hparams.yaml
└── wandb/
    └── run-20240115_120000-abcd1234/
        └── ...
```

### Key Metrics to Monitor

1. **Training Loss**: Should decrease over time
2. **Validation Loss**: Should decrease but may plateau
3. **Spatial Similarity**: Measures how well student mimics teacher (higher is better)
4. **Learning Rate**: Should follow the scheduled decay

### Sample Training Output

```
Epoch 0:  100%|████████| 1000/1000 [05:23<00:00, 3.09it/s, loss=0.456]
Validation: 100%|███████| 100/100 [00:32<00:00, 3.08it/s]
Epoch 0 - val_loss: 0.423, val_scalekd_res4_spatial_similarity: 0.812
```

## Next Steps

### 1. Experiment with Different Students

Try different student architectures:
```yaml
student:
  model_name: swin_tiny  # or stdc_1, mobilenet_v3_large, etc.
```

### 2. Add More Feature Levels

Distill multiple feature levels:
```yaml
student:
  student_keys: [res5, res4, res3]

loss:
  losses:
    - type: scalekd
      weight: 1.0
      kwargs:
        name: scalekd_res5
        num_heads: 24
    - type: scalekd
      weight: 1.0
      kwargs:
        name: scalekd_res4
        num_heads: 16
    - type: scalekd
      weight: 0.5
      kwargs:
        name: scalekd_res3
        num_heads: 8
```

### 3. Use Pre-trained Students

Load pre-trained weights:
```yaml
student:
  model_name: convnext_tiny
  checkpoint_path: path/to/pretrained_convnext.pth
```

### 4. Scale Up Training

For production training:
```yaml
train:
  devices: [0, 1, 2, 3]  # Multi-GPU
  strategy: ddp_find_unused_parameters_true
  accumulate_grad_batches: 4  # Gradient accumulation
```

### 5. Convert Models for Deployment

After training, convert to Anyma format:
```bash
python scripts/convert_to_anyma.py checkpoints/best_model.ckpt models/student_anyma.pkl
```

## Troubleshooting Quick Tips

1. **CUDA Out of Memory**: Reduce batch size or use smaller models
2. **Slow Training**: Check num_workers, use more GPUs, or enable mixed precision
3. **Poor Performance**: Try different learning rates, longer training, or more data
4. **Import Errors**: Ensure virtual environment is activated and dependencies installed

For more detailed troubleshooting, see the [Troubleshooting Guide](troubleshooting.md).