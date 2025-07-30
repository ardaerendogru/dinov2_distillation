# Models Documentation

This document provides comprehensive information about all available models in the DINOv2 Knowledge Distillation Framework, including teacher models, student architectures, and how to integrate custom models.

## Table of Contents
- [Overview](#overview)
- [Teacher Models (DINOv2)](#teacher-models-dinov2)
- [Student Models](#student-models)
- [Model Wrappers](#model-wrappers)
- [Feature Extraction](#feature-extraction)
- [Custom Models](#custom-models)
- [Model Selection Guide](#model-selection-guide)

## Overview

The framework supports a wide variety of model architectures, organized into:
- **Teacher Models**: DINOv2 Vision Transformers that provide target representations
- **Student Models**: Efficient architectures that learn from teachers
- **Model Wrappers**: Unified interface for feature extraction

### Model Architecture Hierarchy

```
models/
├── backbones/          # Core model implementations
│   ├── dinov2.py       # Teacher models
│   ├── convnext.py     # ConvNeXt variants
│   ├── swin.py         # Swin Transformers
│   ├── stdc.py         # STDC networks
│   └── ...
├── wrappers/           # Feature extraction wrappers
│   ├── base.py         # Base wrapper class
│   └── *_wrapper.py    # Architecture-specific wrappers
└── model_zoo.py        # Model factory and registry
```

## Teacher Models (DINOv2)

DINOv2 models are self-supervised Vision Transformers that serve as teachers in the distillation process.

### Available DINOv2 Models

| Model Name | Parameters | Embedding Dim | Patch Size | Use Case |
|------------|------------|---------------|------------|----------|
| `dinov2_vits14` | 21M | 384 | 14 | Fast experimentation |
| `dinov2_vitb14` | 86M | 768 | 14 | Balanced performance |
| `dinov2_vitl14` | 300M | 1024 | 14 | High accuracy |
| `dinov2_vitg14` | 1.1B | 1536 | 14 | Maximum performance |

### Teacher Model Configuration

```yaml
teacher:
  model_name: "dinov2_vitb14"
  patch_size: 14
  freeze: true  # Always frozen during distillation
```

### Feature Dimensions by Model

```python
# Feature dimensions at different levels
DINOV2_FEATURE_DIMS = {
    'dinov2_vits14': {'res2': 384, 'res3': 384, 'res4': 384, 'res5': 384},
    'dinov2_vitb14': {'res2': 768, 'res3': 768, 'res4': 768, 'res5': 768},
    'dinov2_vitl14': {'res2': 1024, 'res3': 1024, 'res4': 1024, 'res5': 1024},
    'dinov2_vitg14': {'res2': 1536, 'res3': 1536, 'res4': 1536, 'res5': 1536},
}
```

## Student Models

The framework supports a diverse range of student architectures optimized for different deployment scenarios.

### ConvNeXt Family

Modern ConvNet architecture with excellent performance-efficiency trade-offs.

| Model | Parameters | GFLOPs | Top-1 Acc | Use Case |
|-------|------------|--------|-----------|----------|
| `convnext_atto` | 3.5M | 0.5 | 75.7% | Edge devices |
| `convnext_pico` | 9M | 1.3 | 79.5% | Mobile |
| `convnext_nano` | 15M | 2.4 | 81.0% | Efficient server |
| `convnext_tiny` | 28M | 4.5 | 82.1% | Balanced |
| `convnext_base` | 88M | 15.4 | 83.8% | High accuracy |

**Configuration Example:**
```yaml
student:
  model_name: "convnext_tiny"
  student_keys: ["res5", "res4", "res3"]
  model_kwargs:
    drop_path_rate: 0.1
    layer_scale_init_value: 1e-6
```

### Swin Transformer

Hierarchical Vision Transformer with shifted windows.

| Model | Parameters | GFLOPs | Top-1 Acc | Use Case |
|-------|------------|--------|-----------|----------|
| `swin_tiny` | 28M | 4.5 | 81.3% | Efficient transformer |
| `swin_small` | 50M | 8.7 | 83.0% | High accuracy |

**Configuration Example:**
```yaml
student:
  model_name: "swin_tiny"
  student_keys: ["res5", "res4", "res3", "res2"]
  model_kwargs:
    window_size: 7
    drop_path_rate: 0.2
```

### STDC Networks

Short-Term Dense Concatenate networks for real-time applications.

| Model | Parameters | GFLOPs | FPS (V100) | Use Case |
|-------|------------|--------|------------|----------|
| `stdc_1` | 8M | 0.8 | 250 | Real-time |
| `stdc_2` | 12M | 1.8 | 188 | Balanced real-time |

**Configuration Example:**
```yaml
student:
  model_name: "stdc_1"
  student_keys: ["res5", "res4"]
  model_kwargs:
    pretrain: false
    use_conv_last: true
```

### ResNet Family

Classic convolutional architectures.

| Model | Parameters | GFLOPs | Top-1 Acc | Use Case |
|-------|------------|--------|-----------|----------|
| `resnet_18` | 11M | 1.8 | 69.8% | Lightweight |
| `resnet_34` | 21M | 3.7 | 73.3% | Efficient |
| `resnet_50` | 25M | 4.1 | 76.1% | Standard |
| `resnet_101` | 44M | 7.8 | 77.4% | High capacity |

### MobileNet Family

Optimized for mobile and embedded devices.

| Model | Parameters | GFLOPs | Top-1 Acc | Use Case |
|-------|------------|--------|-----------|----------|
| `mobilenet_v2` | 3.4M | 0.3 | 71.8% | Mobile |
| `mobilenet_v3_small` | 2.5M | 0.06 | 67.4% | Ultra-light |
| `mobilenet_v3_large` | 5.4M | 0.22 | 75.2% | Mobile balanced |

### MIT (SegFormer) Backbones

Mix Transformer architectures designed for dense prediction.

| Model | Parameters | GFLOPs | mIoU | Use Case |
|-------|------------|--------|------|----------|
| `mit_b0` | 3.7M | 0.8 | 37.4 | Lightweight |
| `mit_b1` | 14M | 2.1 | 42.2 | Efficient |
| `mit_b2` | 25M | 3.9 | 45.6 | Balanced |
| `mit_b3` | 45M | 7.9 | 47.8 | High accuracy |
| `mit_b4` | 62M | 11.6 | 48.6 | Maximum accuracy |
| `mit_b5` | 82M | 15.1 | 49.1 | State-of-the-art |

### DarkNet Family

YOLO backbone architectures.

| Model | Parameters | GFLOPs | mAP | Use Case |
|-------|------------|--------|-----|----------|
| `darknet_n` | 3M | 0.5 | 28.4 | Nano detection |
| `darknet_s` | 9M | 1.5 | 37.4 | Small detection |
| `darknet_m` | 25M | 5.0 | 45.4 | Medium detection |
| `darknet_l` | 53M | 11.0 | 50.1 | Large detection |
| `darknet_x` | 99M | 21.0 | 53.0 | Extra large |

### TIMM Models

Models from the timm library.

| Model | Parameters | Top-1 Acc | Use Case |
|-------|------------|-----------|----------|
| `efficientnet_b0` | 5.3M | 77.1% | Efficient baseline |
| `efficientnet_b1` | 7.8M | 79.1% | Slightly larger |
| `efficientnet_b2` | 9.2M | 80.1% | Balanced |
| `efficientnet_b3` | 12M | 81.6% | High accuracy |
| `efficientnet_b4` | 19M | 82.9% | Very high accuracy |
| `edgenext_xx_small` | 1.3M | 71.2% | Edge deployment |
| `edgenext_x_small` | 2.3M | 74.9% | Small edge |
| `edgenext_small` | 5.6M | 79.4% | Efficient edge |
| `edgenext_base` | 18.5M | 82.5% | High-perf edge |

## Model Wrappers

Model wrappers provide a unified interface for feature extraction across different architectures.

### Base Wrapper Interface

```python
class BaseWrapper(nn.Module):
    """Base class for all model wrappers"""
    
    def __init__(self, model_name, n_patches, target_feature, checkpoint_path=None):
        super().__init__()
        self.model = self._build_model(model_name)
        self.n_patches = n_patches
        self.target_feature = target_feature
        
    def forward(self, x):
        """Extract features at specified levels"""
        features = self._extract_features(x)
        return self._format_output(features)
```

### Wrapper Functionality

Each wrapper provides:
1. **Feature Extraction**: Extract features at multiple resolutions
2. **Feature Alignment**: Interpolate features to consistent sizes
3. **Output Formatting**: Return features in standardized format

### Example: ConvNeXt Wrapper

```python
class ConvNextWrapper(BaseWrapper):
    def __init__(self, model_name='convnext_tiny', n_patches=(16, 16), 
                 target_feature=['res5', 'res4'], checkpoint_path=None):
        # Initialize ConvNeXt backbone
        self.model = self._build_convnext(model_name)
        
        # Define feature dimensions
        self.feature_dims = {
            'res2': dims[0], 'res3': dims[1], 
            'res4': dims[2], 'res5': dims[3]
        }
        
    def _extract_features(self, x):
        features = {}
        # Extract hierarchical features
        x = self.model.stem(x)
        features['res2'] = self.model.stages[0](x)
        features['res3'] = self.model.stages[1](features['res2'])
        features['res4'] = self.model.stages[2](features['res3'])
        features['res5'] = self.model.stages[3](features['res4'])
        return features
```

## Feature Extraction

### Feature Levels

The framework uses a standardized naming convention for feature levels:

- **res2**: 1/4 resolution features (early/low-level)
- **res3**: 1/8 resolution features (mid-level)
- **res4**: 1/16 resolution features (high-level)
- **res5**: 1/32 resolution features (semantic/final)

### Feature Dimensions by Architecture

```python
FEATURE_DIMENSIONS = {
    'convnext_tiny': {'res2': 96, 'res3': 192, 'res4': 384, 'res5': 768},
    'swin_tiny': {'res2': 96, 'res3': 192, 'res4': 384, 'res5': 768},
    'resnet_50': {'res2': 256, 'res3': 512, 'res4': 1024, 'res5': 2048},
    'mobilenet_v3_large': {'res2': 24, 'res3': 40, 'res4': 112, 'res5': 960},
    # ... more models
}
```

### Feature Interpolation

Features are interpolated to match teacher dimensions:

```python
def interpolate_features(features, target_size):
    """Interpolate features to target spatial size"""
    return F.interpolate(
        features, 
        size=target_size, 
        mode='bilinear', 
        align_corners=False
    )
```

## Custom Models

### Adding a New Student Model

#### Step 1: Implement the Backbone

Create `models/backbones/mymodel.py`:

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, num_classes=1000, **kwargs):
        super().__init__()
        # Define your architecture
        self.features = nn.Sequential(
            # ... your layers
        )
        
    def forward(self, x):
        return self.features(x)
```

#### Step 2: Create the Wrapper

Create `models/wrappers/mymodel_wrapper.py`:

```python
from .base import BaseWrapper
from ..backbones.mymodel import MyModel

class MyModelWrapper(BaseWrapper):
    def __init__(self, model_name='mymodel', n_patches=(16, 16), 
                 target_feature=['res5', 'res4'], checkpoint_path=None):
        super().__init__()
        
        # Initialize your model
        self.model = MyModel()
        
        # Define feature extraction points
        self.feature_extractors = {
            'res2': self.model.layer1,
            'res3': self.model.layer2,
            'res4': self.model.layer3,
            'res5': self.model.layer4,
        }
        
    def forward(self, x):
        features = {}
        
        # Extract features at each level
        for name, extractor in self.feature_extractors.items():
            x = extractor(x)
            if name in self.target_feature:
                # Interpolate to target size
                feat = F.interpolate(x, size=self.n_patches, 
                                   mode='bilinear', align_corners=False)
                features[name] = feat
                
        return features
```

#### Step 3: Register in Model Zoo

Update `models/model_zoo.py`:

```python
from .wrappers.mymodel_wrapper import MyModelWrapper

# Add to MODEL_MAP
ModelWrapper.MODEL_MAP['mymodel'] = MyModelWrapper
```

#### Step 4: Update Configuration

Use your model in configuration:

```yaml
student:
  model_name: "mymodel"
  student_keys: ["res5", "res4"]
  model_kwargs:
    custom_param: value
```

### Custom Teacher Models

While the framework is designed for DINOv2 teachers, you can add custom teachers:

```python
class CustomTeacher(nn.Module):
    def __init__(self, model_name, **kwargs):
        super().__init__()
        # Load your pre-trained model
        self.model = load_pretrained_model(model_name)
        
    def forward(self, x):
        # Return features in DINOv2 format
        return self.model.forward_features(x)
```

## Model Selection Guide

### For Mobile Deployment

**Requirements**: Low latency, small model size, battery efficiency

**Recommended Models**:
1. `mobilenet_v3_small` - Ultra lightweight
2. `mobilenet_v3_large` - Better accuracy, still mobile-friendly
3. `convnext_atto` - Modern architecture, very efficient
4. `efficientnet_b0` - Good accuracy/efficiency balance

**Example Configuration**:
```yaml
student:
  model_name: "mobilenet_v3_large"
  student_keys: ["res5", "res4"]
teacher:
  model_name: "dinov2_vits14"  # Smaller teacher for mobile features
```

### For Edge Devices

**Requirements**: Real-time processing, limited compute

**Recommended Models**:
1. `stdc_1` - Designed for real-time
2. `edgenext_small` - Optimized for edge
3. `convnext_pico` - Small but powerful
4. `darknet_s` - Good for detection tasks

### For Server Deployment

**Requirements**: High accuracy, reasonable latency

**Recommended Models**:
1. `swin_small` - Excellent accuracy
2. `convnext_base` - State-of-the-art CNN
3. `mit_b3` - Great for dense prediction
4. `resnet_101` - Classic, well-supported

### For Maximum Accuracy

**Requirements**: Best possible performance

**Recommended Models**:
1. `swin_small` with `dinov2_vitl14` teacher
2. `convnext_base` with `dinov2_vitg14` teacher
3. `mit_b5` for segmentation tasks

**Example Configuration**:
```yaml
student:
  model_name: "swin_small"
  student_keys: ["res5", "res4", "res3", "res2"]
teacher:
  model_name: "dinov2_vitl14"
loss:
  losses:
    - type: "scalekd"
      weight: 1.0
      kwargs:
        name: "scalekd_res5"
        num_heads: 16
    # ... more losses for other levels
```

### Performance Comparison

| Deployment | Model | Params | Latency (ms) | Accuracy |
|------------|-------|--------|--------------|----------|
| Mobile | mobilenet_v3_small | 2.5M | 5 | 67.4% |
| Mobile | convnext_atto | 3.5M | 8 | 75.7% |
| Edge | stdc_1 | 8M | 4 | 74.5% |
| Edge | edgenext_small | 5.6M | 12 | 79.4% |
| Server | convnext_tiny | 28M | 25 | 82.1% |
| Server | swin_small | 50M | 35 | 83.0% |
| Cloud | convnext_base | 88M | 60 | 83.8% |

*Latency measured on NVIDIA V100 with batch size 1

## Best Practices

### 1. Model Selection

- Start with smaller models for experimentation
- Match student capacity to deployment constraints
- Consider teacher-student capacity gap

### 2. Feature Level Selection

- Use multiple feature levels for better performance
- Higher levels (res4, res5) capture semantic information
- Lower levels (res2, res3) preserve spatial details

### 3. Pre-training

- Use ImageNet pre-trained weights when available
- Fine-tune from related tasks for better initialization
- Consider self-supervised pre-training for custom domains

### 4. Architecture-Specific Tips

**ConvNeXt**:
- Benefits from longer training
- Use drop_path for regularization
- Layer scale helps training stability

**Swin Transformer**:
- Sensitive to learning rate
- Benefits from stronger augmentation
- Window size affects accuracy/speed trade-off

**MobileNet**:
- Careful with batch normalization
- May need different learning rates for depthwise layers
- Consider quantization-aware training

**STDC**:
- Designed for speed, don't over-parameterize
- Works well with smaller teacher models
- Good for real-time applications