# Loss Functions Documentation

This document provides detailed information about the loss functions available in the DINOv2 Knowledge Distillation Framework, with a focus on the Scale-KD loss function.

## Table of Contents
- [Overview](#overview)
- [Scale-KD Loss](#scale-kd-loss)
- [Loss Configuration](#loss-configuration)
- [Implementation Details](#implementation-details)
- [Custom Loss Functions](#custom-loss-functions)
- [Loss Function Selection Guide](#loss-function-selection-guide)
- [Debugging Loss Functions](#debugging-loss-functions)

## Overview

The framework implements state-of-the-art distillation losses designed to transfer knowledge from DINOv2 teachers to student models effectively.

### Available Loss Functions

1. **Scale-KD** (Primary): Scale-aware Knowledge Distillation
2. **MSE Loss**: Mean Squared Error for feature matching
3. **Cosine Similarity Loss**: Angular alignment of features
4. **Combined Loss**: Weighted combination of multiple losses

### Loss Function Architecture

```
Loss Computation Pipeline
├── Student Features ──┐
├── Teacher Features ──┤
│                      ↓
├── Feature Projection (if needed)
├── Attention Computation
├── Spatial & Frequency Components
├── Loss Aggregation
└── Weighted Combination
```

## Scale-KD Loss

Scale-KD (Scale-aware Knowledge Distillation) is the primary loss function that enables effective knowledge transfer by matching attention patterns at multiple scales.

### Key Concepts

1. **Multi-Scale Attention**: Captures both global and local patterns
2. **Spatial-Frequency Decomposition**: Separates spatial and frequency information
3. **Learnable Projections**: Adapts student features to teacher space
4. **Window-based Attention**: Flexible attention computation

### Mathematical Formulation

The Scale-KD loss consists of two components:

**Spatial Component:**
```
L_spatial = MSE(Attention_S(F_S), Attention_T(F_T))
```

**Frequency Component:**
```
L_freq = MSE(FFT(Attention_S(F_S)), FFT(Attention_T(F_T)))
```

**Total Loss:**
```
L_ScaleKD = α[0] * L_spatial + α[1] * L_freq
```

Where:
- `F_S`, `F_T`: Student and teacher features
- `Attention`: Attention computation function
- `FFT`: Fast Fourier Transform
- `α`: Weighting factors

### Architecture Components

```python
class ScaleKD(nn.Module):
    def __init__(self, name, alpha, student_dims, teacher_dims, 
                 query_hw, pos_hw, pos_dims, window_shapes=(1,1),
                 self_query=True, softmax_scale=[1,1], num_heads=8):
        """
        Args:
            name: Loss identifier (e.g., 'scalekd_res5')
            alpha: [spatial_weight, frequency_weight]
            student_dims: Student feature dimensions
            teacher_dims: Teacher feature dimensions
            query_hw: Query spatial dimensions
            pos_hw: Position encoding dimensions
            pos_dims: Position embedding dimensions
            window_shapes: Attention window size
            self_query: Use features as queries
            softmax_scale: Temperature scaling
            num_heads: Number of attention heads
        """
```

### Attention Projector

The attention projector maps student features to match teacher attention space:

```python
class AttentionProjector(nn.Module):
    def __init__(self, student_dims, teacher_dims, query_hw, 
                 pos_dims, window_shapes, self_query, 
                 softmax_scale, num_heads):
        # Query projection
        if not self_query:
            self.queries = nn.Parameter(
                torch.randn(1, query_hw[0]*query_hw[1], teacher_dims)
            )
        
        # Feature projections
        self.project_kv = nn.Linear(student_dims, 2*teacher_dims)
        
        # Multi-head attention
        self.num_heads = num_heads
        self.head_dim = teacher_dims // num_heads
```

## Loss Configuration

### Basic Configuration

```yaml
loss:
  losses:
    - type: "scalekd"
      weight: 1.0
      kwargs:
        alpha: [0.08, 0.06]  # Spatial and frequency weights
        window_shapes: [1, 1]  # Global attention
        self_query: false
        softmax_scale: [5.0, 5.0]
        num_heads: 16
        name: "scalekd_res4"  # Must follow naming convention
```

### Multi-Level Configuration

For distilling multiple feature levels:

```yaml
loss:
  losses:
    # High-level features (res5)
    - type: "scalekd"
      weight: 1.0
      kwargs:
        alpha: [0.08, 0.06]
        window_shapes: [1, 1]
        self_query: true
        softmax_scale: [5.0, 5.0]
        num_heads: 24  # More heads for deeper features
        name: "scalekd_res5"
    
    # Mid-level features (res4)
    - type: "scalekd"
      weight: 0.8
      kwargs:
        alpha: [0.1, 0.05]
        window_shapes: [2, 2]  # Local attention
        self_query: false
        softmax_scale: [4.0, 4.0]
        num_heads: 16
        name: "scalekd_res4"
    
    # Low-level features (res3)
    - type: "scalekd"
      weight: 0.5
      kwargs:
        alpha: [0.12, 0.04]
        window_shapes: [4, 4]  # More local attention
        self_query: false
        softmax_scale: [3.0, 3.0]
        num_heads: 8
        name: "scalekd_res3"
```

### Parameter Guidelines

#### Alpha (α) Values
- **Typical Range**: [0.05, 0.15]
- **Spatial Weight** (α[0]): Higher for preserving spatial structure
- **Frequency Weight** (α[1]): Higher for texture preservation

#### Window Shapes
- **[1, 1]**: Global attention (entire feature map)
- **[2, 2]**: 2×2 local windows
- **[4, 4]**: 4×4 local windows
- **Larger windows**: More global context but higher memory

#### Self Query
- **True**: Use feature maps as queries (self-attention)
- **False**: Use learnable query embeddings

#### Softmax Scale
- **Higher values** (5-10): Sharper attention
- **Lower values** (1-3): Softer attention
- **Typical**: 5.0 for most cases

#### Number of Heads
- **res2/res3**: 8 heads
- **res4**: 16 heads
- **res5**: 24-32 heads
- Match teacher's attention heads when possible

## Implementation Details

### Forward Pass

```python
def forward(self, preds_S, preds_T, query_s=None, query_f=None):
    """
    Args:
        preds_S: Student features (B, C, H, W)
        preds_T: Teacher features (B, C, H, W)
        query_s: Optional spatial queries
        query_f: Optional frequency queries
    
    Returns:
        Dictionary with losses and metrics
    """
    # Project student features
    preds_S_spat = self.project_feat_spat(preds_S, query=query_s)
    preds_S_freq = self.project_feat_freq(preds_S, query=query_f)
    
    # Compute losses
    spat_loss, spatial_similarity = self.get_spat_loss(
        preds_S_spat, preds_T
    )
    freq_loss, frequency_similarity = self.get_freq_loss(
        preds_S_freq, preds_T
    )
    
    # Combine losses
    total_loss = self.alpha[0] * spat_loss + self.alpha[1] * freq_loss
    
    return {
        'loss': total_loss,
        'spatial_loss': spat_loss,
        'frequency_loss': freq_loss,
        'spatial_similarity': spatial_similarity,
        'frequency_similarity': frequency_similarity
    }
```

### Attention Computation

```python
def compute_attention(self, features, queries=None):
    """Compute multi-head attention"""
    B, C, H, W = features.shape
    
    # Reshape for attention
    features = features.flatten(2).transpose(1, 2)  # B, HW, C
    
    # Project to K, V
    kv = self.project_kv(features)
    k, v = kv.chunk(2, dim=-1)
    
    # Use queries or self-attention
    if queries is None:
        q = k
    else:
        q = queries
    
    # Multi-head attention
    q = q.reshape(B, -1, self.num_heads, self.head_dim)
    k = k.reshape(B, -1, self.num_heads, self.head_dim)
    v = v.reshape(B, -1, self.num_heads, self.head_dim)
    
    # Compute attention scores
    attn = torch.matmul(q, k.transpose(-2, -1))
    attn = attn * self.softmax_scale / math.sqrt(self.head_dim)
    attn = F.softmax(attn, dim=-1)
    
    # Apply attention
    out = torch.matmul(attn, v)
    return out, attn
```

### Frequency Component

```python
def compute_frequency_attention(self, spatial_attention):
    """Convert spatial attention to frequency domain"""
    # Apply 2D FFT
    freq_attn = torch.fft.fft2(spatial_attention, norm='ortho')
    
    # Take magnitude (ignore phase)
    freq_attn_mag = torch.abs(freq_attn)
    
    # Optional: Apply frequency weighting
    freq_weights = self.get_frequency_weights(freq_attn_mag.shape)
    freq_attn_weighted = freq_attn_mag * freq_weights
    
    return freq_attn_weighted
```

## Custom Loss Functions

### Adding a New Loss Function

#### Step 1: Implement the Loss

Create `losses/my_loss.py`:

```python
import torch
import torch.nn as nn

class MyDistillationLoss(nn.Module):
    def __init__(self, temperature=4.0, **kwargs):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, student_features, teacher_features):
        """
        Args:
            student_features: Dict of student features
            teacher_features: Dict of teacher features
        
        Returns:
            Dict with loss values and metrics
        """
        total_loss = 0
        losses = {}
        
        for key in student_features:
            if key in teacher_features:
                # Your loss computation
                loss = self.compute_loss(
                    student_features[key],
                    teacher_features[key]
                )
                losses[f'{key}_loss'] = loss
                total_loss += loss
        
        losses['loss'] = total_loss
        return losses
    
    def compute_loss(self, s_feat, t_feat):
        # Implement your loss logic
        pass
```

#### Step 2: Register the Loss

Update `losses/__init__.py`:

```python
from .my_loss import MyDistillationLoss

LOSS_REGISTRY = {
    'scalekd': ScaleKD,
    'my_loss': MyDistillationLoss,
}
```

#### Step 3: Use in Configuration

```yaml
loss:
  losses:
    - type: "my_loss"
      weight: 1.0
      kwargs:
        temperature: 4.0
```

### Example: Feature Matching Loss

```python
class FeatureMatchingLoss(nn.Module):
    def __init__(self, normalize=True, distance='l2'):
        super().__init__()
        self.normalize = normalize
        self.distance = distance
        
    def forward(self, student_features, teacher_features):
        losses = {}
        
        for key in student_features:
            s_feat = student_features[key]
            t_feat = teacher_features[key]
            
            # Normalize if needed
            if self.normalize:
                s_feat = F.normalize(s_feat, dim=1)
                t_feat = F.normalize(t_feat, dim=1)
            
            # Compute distance
            if self.distance == 'l2':
                loss = F.mse_loss(s_feat, t_feat)
            elif self.distance == 'l1':
                loss = F.l1_loss(s_feat, t_feat)
            elif self.distance == 'cosine':
                loss = 1 - F.cosine_similarity(
                    s_feat.flatten(2), 
                    t_feat.flatten(2)
                ).mean()
            
            losses[f'{key}_{self.distance}_loss'] = loss
        
        losses['loss'] = sum(losses.values())
        return losses
```

## Loss Function Selection Guide

### For Different Tasks

#### Image Classification
```yaml
loss:
  losses:
    - type: "scalekd"
      weight: 1.0
      kwargs:
        name: "scalekd_res5"  # Focus on high-level features
        alpha: [0.1, 0.05]
        num_heads: 24
```

#### Dense Prediction (Segmentation/Detection)
```yaml
loss:
  losses:
    - type: "scalekd"
      weight: 1.0
      kwargs:
        name: "scalekd_res4"  # Mid-level features important
        alpha: [0.12, 0.08]
        window_shapes: [2, 2]  # Local attention
    - type: "scalekd"
      weight: 0.5
      kwargs:
        name: "scalekd_res3"  # Low-level details
        alpha: [0.15, 0.05]
        window_shapes: [4, 4]
```

#### Feature Learning
```yaml
loss:
  losses:
    - type: "scalekd"
      weight: 1.0
      kwargs:
        name: "scalekd_res5"
        self_query: true  # Self-attention for features
    - type: "cosine"
      weight: 0.2
      kwargs:
        dim: 1
```

### For Different Model Sizes

#### Small Students (MobileNet, STDC)
- Use fewer attention heads
- Higher softmax scale for sharper attention
- Focus on key feature levels

```yaml
loss:
  losses:
    - type: "scalekd"
      weight: 1.0
      kwargs:
        num_heads: 8  # Fewer heads
        softmax_scale: [8.0, 8.0]  # Sharper
        window_shapes: [1, 1]  # Global only
```

#### Large Students (Swin, ConvNeXt-Base)
- More attention heads
- Multiple feature levels
- Balanced spatial-frequency weights

```yaml
loss:
  losses:
    - type: "scalekd"
      weight: 1.0
      kwargs:
        num_heads: 32
        alpha: [0.08, 0.08]  # Balanced
```

## Debugging Loss Functions

### Common Issues and Solutions

#### 1. Loss Explosion

**Symptoms**: Loss suddenly increases to inf/nan

**Solutions**:
```python
# Add gradient clipping
train:
  gradient_clip_val: 1.0

# Reduce learning rate
optimizer:
  kwargs:
    lr: 1e-5

# Check for numerical stability
class StableLoss(nn.Module):
    def forward(self, s_feat, t_feat):
        # Add small epsilon
        s_feat = s_feat + 1e-8
        t_feat = t_feat + 1e-8
        
        # Clamp values
        s_feat = torch.clamp(s_feat, -10, 10)
```

#### 2. Loss Not Decreasing

**Check**:
```python
# Verify feature dimensions match
print(f"Student dims: {student_features['res5'].shape}")
print(f"Teacher dims: {teacher_features['res5'].shape}")

# Check loss computation
with torch.no_grad():
    loss_dict = loss_fn(student_features, teacher_features)
    print(f"Loss components: {loss_dict}")
```

#### 3. Poor Convergence

**Solutions**:
```yaml
# Adjust loss weights
loss:
  losses:
    - type: "scalekd"
      weight: 0.5  # Reduce weight
      
# Modify alpha values
kwargs:
  alpha: [0.05, 0.03]  # Smaller values
  
# Change window shapes
kwargs:
  window_shapes: [2, 2]  # Try local attention
```

### Visualization Tools

```python
def visualize_attention(attention_map, save_path):
    """Visualize attention patterns"""
    import matplotlib.pyplot as plt
    
    # Average over heads
    attn = attention_map.mean(dim=1)
    
    # Reshape to 2D
    H = W = int(math.sqrt(attn.shape[-1]))
    attn = attn.reshape(-1, H, W)
    
    # Plot
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        if i < len(attn):
            im = ax.imshow(attn[i].cpu().numpy(), cmap='hot')
            ax.axis('off')
    
    plt.colorbar(im, ax=axes.ravel().tolist())
    plt.savefig(save_path)
    plt.close()
```

### Loss Monitoring

```python
# Add to training loop
def on_train_batch_end(self, outputs, batch, batch_idx):
    # Log individual loss components
    if batch_idx % 100 == 0:
        for name, value in outputs['loss_dict'].items():
            self.logger.experiment.add_scalar(
                f'loss_components/{name}', 
                value, 
                self.global_step
            )
        
        # Log attention statistics
        if hasattr(self.loss_fn, 'last_attention'):
            attn = self.loss_fn.last_attention
            self.logger.experiment.add_histogram(
                'attention_weights', 
                attn, 
                self.global_step
            )
```

## Best Practices

1. **Start Simple**: Begin with single-level distillation
2. **Validate Losses**: Ensure losses decrease on small dataset
3. **Monitor Components**: Track spatial and frequency separately
4. **Adjust Gradually**: Make small changes to parameters
5. **Use Visualizations**: Inspect attention patterns regularly
6. **Balance Losses**: Ensure no single component dominates
7. **Match Teacher**: Align parameters with teacher architecture