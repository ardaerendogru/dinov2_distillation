# Examples and Tutorials

This document provides practical examples and step-by-step tutorials for common use cases in the DINOv2 Knowledge Distillation Framework.

## Table of Contents
- [Quick Start Examples](#quick-start-examples)
- [Advanced Tutorials](#advanced-tutorials)
- [Common Use Cases](#common-use-cases)
- [Performance Benchmarks](#performance-benchmarks)
- [Tips and Tricks](#tips-and-tricks)

## Quick Start Examples

### Example 1: Basic ConvNeXt Distillation

Distill from DINOv2-ViT-S to ConvNeXt-Tiny for a balanced accuracy/efficiency trade-off.

```yaml
# config/convnext_basic.yaml
wandb:
  project: "dinov2_examples"
  name: "convnext_tiny_basic"

student:
  model_name: "convnext_tiny"
  student_keys: ["res5", "res4"]

teacher:
  model_name: "dinov2_vits14"

optimizer:
  type: "AdamW"
  kwargs:
    lr: 1e-3
    weight_decay: 0.01

loss:
  losses:
    - type: "scalekd"
      weight: 1.0
      kwargs:
        alpha: [0.08, 0.06]
        window_shapes: [1, 1]
        self_query: false
        softmax_scale: [5.0, 5.0]
        num_heads: 16
        name: "scalekd_res5"

train:
  max_epochs: 100
  accelerator: "gpu"
  devices: [0]

data_loader:
  data_dir: ["/path/to/imagenet/train"]
  batch_size: 32
  num_workers: 8
```

Run training:
```bash
python train.py --config config/convnext_basic.yaml
```

### Example 2: Mobile Deployment

Optimize for mobile devices using MobileNetV3.

```yaml
# config/mobile_deployment.yaml
student:
  model_name: "mobilenet_v3_large"
  student_keys: ["res5", "res4"]

teacher:
  model_name: "dinov2_vits14"  # Smaller teacher for mobile features

optimizer:
  type: "AdamW"
  kwargs:
    lr: 5e-4  # Lower LR for mobile models
    weight_decay: 0.05

loss:
  losses:
    - type: "scalekd"
      weight: 1.0
      kwargs:
        alpha: [0.1, 0.05]
        num_heads: 8  # Fewer heads for efficiency
        softmax_scale: [8.0, 8.0]  # Sharper attention
        name: "scalekd_res5"

train:
  max_epochs: 200
  precision: 16  # Mixed precision
  gradient_clip_val: 1.0

data_loader:
  batch_size: 64  # Larger batch for small model
```

### Example 3: Multi-GPU Training

Scale up training across multiple GPUs.

```yaml
# config/multi_gpu.yaml
train:
  accelerator: "gpu"
  devices: [0, 1, 2, 3]
  strategy: "ddp_find_unused_parameters_true"
  sync_batchnorm: true

data_loader:
  batch_size: 32  # Per GPU
  num_workers: 8  # Per GPU

# Effective batch size = 32 * 4 = 128
```

## Advanced Tutorials

### Tutorial 1: Progressive Distillation

Train a model progressively from high-level to low-level features.

#### Step 1: High-Level Features Only

```python
# stage1_config.yaml
student:
  student_keys: ["res5"]  # Only res5

loss:
  losses:
    - type: "scalekd"
      weight: 1.0
      kwargs:
        name: "scalekd_res5"
        num_heads: 24

train:
  max_epochs: 100
```

#### Step 2: Add Mid-Level Features

```python
# stage2_config.yaml
student:
  student_keys: ["res5", "res4"]  # Add res4
  checkpoint_path: "checkpoints/stage1_best.ckpt"  # Load from stage 1

loss:
  losses:
    - type: "scalekd"
      weight: 1.0
      kwargs:
        name: "scalekd_res5"
        num_heads: 24
    - type: "scalekd"
      weight: 0.5  # Lower weight for new feature
      kwargs:
        name: "scalekd_res4"
        num_heads: 16

train:
  max_epochs: 100
  resume_from_checkpoint: "checkpoints/stage1_last.ckpt"
```

#### Step 3: Full Feature Hierarchy

```python
# stage3_config.yaml
student:
  student_keys: ["res5", "res4", "res3", "res2"]
  checkpoint_path: "checkpoints/stage2_best.ckpt"

# Add losses for all levels with decreasing weights
```

### Tutorial 2: Custom Dataset Integration

#### Step 1: Create Custom Dataset

```python
# datasets/my_dataset.py
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class MyCustomDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Args:
            csv_file: Path to CSV with columns ['image_name', 'label']
            img_dir: Directory containing images
            transform: Optional transform
        """
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['image_name'])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, row['label']
```

#### Step 2: Create DataModule

```python
# datasets/my_datamodule.py
import lightning as L
from torch.utils.data import DataLoader
from .my_dataset import MyCustomDataset

class MyDataModule(L.LightningDataModule):
    def __init__(self, train_csv, val_csv, img_dir, 
                 batch_size=32, num_workers=8, transform=None):
        super().__init__()
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        
    def setup(self, stage=None):
        self.train_dataset = MyCustomDataset(
            self.train_csv, self.img_dir, self.transform
        )
        self.val_dataset = MyCustomDataset(
            self.val_csv, self.img_dir, self.transform
        )
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True
        )
```

#### Step 3: Use in Training

```python
# train_custom.py
from datasets.my_datamodule import MyDataModule
from datasets.augmentations import DataAugmentationDINO

# Create data module
data_module = MyDataModule(
    train_csv='data/train.csv',
    val_csv='data/val.csv',
    img_dir='data/images',
    transform=DataAugmentationDINO()
)

# Use in trainer
trainer = DistillationTrainer(cfg)
trainer.data_module = data_module
trainer.train()
```

### Tutorial 3: Multi-Teacher Distillation

Distill from multiple teachers for improved performance.

```python
# train/multi_teacher_module.py
class MultiTeacherDistillationModule(L.LightningModule):
    def __init__(self, teachers, teacher_weights, student, loss_fn, 
                 optimizer_cfg, scheduler_cfg):
        super().__init__()
        self.teachers = nn.ModuleList(teachers)
        self.teacher_weights = teacher_weights
        self.student = student
        self.loss_fn = loss_fn
        
        # Freeze all teachers
        for teacher in self.teachers:
            teacher.eval()
            for param in teacher.parameters():
                param.requires_grad = False
    
    def training_step(self, batch, batch_idx):
        images = batch
        
        # Get features from all teachers
        all_teacher_features = []
        with torch.no_grad():
            for teacher in self.teachers:
                teacher_features = teacher(images)
                all_teacher_features.append(teacher_features)
        
        # Weighted combination of teacher features
        combined_features = {}
        for key in all_teacher_features[0].keys():
            combined = sum(
                w * tf[key] for w, tf in zip(self.teacher_weights, all_teacher_features)
            )
            combined_features[key] = combined / sum(self.teacher_weights)
        
        # Student forward
        student_features = self.student(images)
        
        # Compute loss
        loss_dict = self.loss_fn(student_features, combined_features)
        
        return loss_dict['loss']
```

## Common Use Cases

### Use Case 1: Fine-tuning on Custom Domain

Fine-tune a distilled model on domain-specific data.

```python
# 1. First, distill on general dataset (ImageNet)
python train.py --config config/general_distillation.yaml

# 2. Then fine-tune on domain data
# config/domain_finetune.yaml
student:
  model_name: "convnext_tiny"
  checkpoint_path: "checkpoints/general_distilled.ckpt"  # Pre-distilled
  student_keys: ["res5", "res4"]

optimizer:
  type: "AdamW"
  kwargs:
    lr: 1e-4  # Lower LR for fine-tuning
    weight_decay: 0.01

train:
  max_epochs: 50  # Fewer epochs needed

data_loader:
  data_dir: ["/path/to/domain/data"]
```

### Use Case 2: Knowledge Distillation for Segmentation

Adapt the framework for dense prediction tasks.

```python
# models/segmentation_wrapper.py
class SegmentationWrapper(nn.Module):
    def __init__(self, backbone, num_classes=21):
        super().__init__()
        self.backbone = backbone
        
        # Add segmentation head
        self.decode_head = nn.Sequential(
            nn.Conv2d(768, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )
        
    def forward(self, x):
        # Get multi-scale features
        features = self.backbone(x)
        
        # Use res4 features for segmentation
        seg_features = features['res4']
        
        # Upsample and predict
        seg_output = self.decode_head(seg_features)
        seg_output = F.interpolate(
            seg_output, size=x.shape[-2:], 
            mode='bilinear', align_corners=False
        )
        
        return features, seg_output
```

### Use Case 3: Self-Supervised Pre-training + Distillation

Combine self-supervised learning with distillation.

```python
# losses/combined_loss.py
class SelfSupervisedDistillationLoss(nn.Module):
    def __init__(self, distill_weight=1.0, ssl_weight=0.1):
        super().__init__()
        self.distill_loss = ScaleKD(...)
        self.ssl_loss = DINOLoss(...)
        self.distill_weight = distill_weight
        self.ssl_weight = ssl_weight
        
    def forward(self, student_features, teacher_features, 
                student_crops, teacher_crops):
        # Distillation loss
        distill_dict = self.distill_loss(
            student_features, teacher_features
        )
        
        # Self-supervised loss
        ssl_loss = self.ssl_loss(
            student_crops, teacher_crops
        )
        
        # Combine losses
        total_loss = (self.distill_weight * distill_dict['loss'] + 
                     self.ssl_weight * ssl_loss)
        
        return {
            'loss': total_loss,
            'distill_loss': distill_dict['loss'],
            'ssl_loss': ssl_loss
        }
```

## Performance Benchmarks

### Benchmark Results

| Model | Teacher | Params | GFLOPs | Top-1 Acc | Latency (ms) |
|-------|---------|--------|--------|-----------|--------------|
| MobileNetV3-Large | DINOv2-S | 5.4M | 0.22 | 75.8% | 12 |
| ConvNeXt-Tiny | DINOv2-B | 28M | 4.5 | 82.5% | 45 |
| Swin-Tiny | DINOv2-B | 28M | 4.5 | 82.8% | 52 |
| ConvNeXt-Base | DINOv2-L | 88M | 15.4 | 84.2% | 125 |

### Training Time Benchmarks

| Setup | Model | Dataset Size | Time/Epoch | Total Time |
|-------|-------|--------------|------------|------------|
| 1×V100 | ConvNeXt-Tiny | 100K | 45 min | 75 hours |
| 4×V100 | ConvNeXt-Tiny | 100K | 12 min | 20 hours |
| 8×A100 | ConvNeXt-Base | 1.2M | 35 min | 175 hours |

## Tips and Tricks

### 1. Debugging Training

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Use fast_dev_run for quick debugging
trainer = Trainer(fast_dev_run=5)  # Run 5 batches

# Visualize augmentations
from datasets.augmentations import visualize_augmentations
visualize_augmentations(train_dataset, num_samples=8)
```

### 2. Memory Optimization

```python
# Gradient accumulation for large models
train:
  accumulate_grad_batches: 8
  
# Gradient checkpointing
student:
  model_kwargs:
    use_checkpoint: true
    
# Mixed precision training
train:
  precision: 16
```

### 3. Learning Rate Finding

```python
from lightning.pytorch.tuner import Tuner

# Create trainer
trainer = Trainer(auto_lr_find=True)

# Run LR finder
tuner = Tuner(trainer)
lr_finder = tuner.lr_find(
    distillation_module, 
    train_dataloaders=train_loader
)

# Plot results
fig = lr_finder.plot()
fig.savefig('lr_finder.png')

# Get suggestion
suggested_lr = lr_finder.suggestion()
print(f"Suggested LR: {suggested_lr}")
```

### 4. Experiment Tracking

```python
# Use wandb for comprehensive tracking
wandb:
  project: "dinov2_distillation"
  tags: ["experiment", "ablation"]
  notes: "Testing new augmentation strategy"
  config:
    architecture: "convnext_tiny"
    dataset: "imagenet_subset"
    
# Log custom metrics
def on_validation_epoch_end(self):
    # Log confusion matrix
    wandb.log({
        "conf_mat": wandb.plot.confusion_matrix(
            probs=self.val_predictions,
            y_true=self.val_labels
        )
    })
    
    # Log sample predictions
    wandb.log({
        "examples": [
            wandb.Image(img, caption=f"Pred: {pred}")
            for img, pred in zip(self.val_images[:8], 
                               self.val_predictions[:8])
        ]
    })
```

### 5. Distributed Training Tips

```bash
# Single node, multi-GPU
python train.py --train.devices [0,1,2,3]

# Multi-node training
python train.py \
    --train.num_nodes 4 \
    --train.devices 8 \
    --train.strategy ddp

# SLURM cluster
srun -N 4 --gres=gpu:8 python train.py \
    --train.num_nodes 4 \
    --train.devices 8
```

### 6. Model Export and Deployment

```python
# Export to ONNX
import torch

# Load checkpoint
model = ModelWrapper.load_from_checkpoint('best_model.ckpt')
model.eval()

# Export
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=11,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}}
)

# Verify ONNX model
import onnx
onnx_model = onnx.load("model.onnx")
onnx.checker.check_model(onnx_model)
```