# API Reference

This document provides a comprehensive API reference for all major classes and functions in the DINOv2 Knowledge Distillation Framework.

## Table of Contents
- [Core Classes](#core-classes)
- [Models](#models)
- [Loss Functions](#loss-functions)
- [Data Pipeline](#data-pipeline)
- [Training Components](#training-components)
- [Utilities](#utilities)

## Core Classes

### DistillationTrainer

Main class orchestrating the distillation training process.

```python
class DistillationTrainer:
    """
    Orchestrates the knowledge distillation training pipeline.
    
    Args:
        cfg (Dict[str, Any]): Configuration dictionary from YAML file
    
    Attributes:
        cfg: Configuration dictionary
        transform: Data augmentation pipeline
        data_module: PyTorch Lightning DataModule
        teacher: Pre-trained teacher model (frozen)
        student: Student model being trained
        distillation_module: Lightning module for training
        trainer: PyTorch Lightning Trainer
    """
    
    def __init__(self, cfg: Dict[str, Any])
    
    def setup_data_module(self) -> L.LightningDataModule:
        """Initialize data module with augmentations."""
        
    def setup_models(self) -> Tuple[nn.Module, nn.Module]:
        """Initialize teacher and student models."""
        
    def setup_loss_functions(self) -> nn.Module:
        """Create and configure loss functions."""
        
    def setup_callbacks(self) -> List[Callback]:
        """Configure training callbacks."""
        
    def train(self) -> None:
        """Execute the training process."""
```

### DistillationModule

PyTorch Lightning module handling the training logic.

```python
class DistillationModule(L.LightningModule):
    """
    Lightning module for knowledge distillation.
    
    Args:
        teacher (nn.Module): Teacher model (frozen)
        student (nn.Module): Student model to train
        loss_fn (nn.Module): Loss function(s)
        optimizer_cfg (Dict): Optimizer configuration
        scheduler_cfg (Dict): LR scheduler configuration
    """
    
    def __init__(self, teacher, student, loss_fn, optimizer_cfg, scheduler_cfg)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through student model."""
        
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> Dict:
        """Single training step."""
        
    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> Dict:
        """Single validation step."""
        
    def configure_optimizers(self) -> Dict:
        """Configure optimizers and schedulers."""
        
    def on_train_epoch_end(self) -> None:
        """Called at the end of training epoch."""
```

## Models

### ModelWrapper

Base class for all model wrappers providing unified feature extraction.

```python
class ModelWrapper(nn.Module):
    """
    Versatile wrapper for backbone models.
    
    Args:
        model_name (str): Name of backbone architecture
        n_patches (Tuple[int, int]): Output spatial size (H, W)
        target_feature (List[str]): Feature levels to extract
        checkpoint_path (Optional[str]): Path to pretrained weights
        **model_kwargs: Additional model-specific arguments
    
    Class Attributes:
        MODEL_MAP (Dict[str, Type]): Registry of available models
    """
    
    def __init__(self, model_name, n_patches, target_feature, 
                 checkpoint_path=None, **model_kwargs)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract features at specified levels.
        
        Returns:
            Dict mapping feature names to tensors
        """
        
    @classmethod
    def register_model(cls, name: str, model_class: Type) -> None:
        """Register a new model in the MODEL_MAP."""
```

### DINOv2ViT

Teacher model implementation for DINOv2 Vision Transformers.

```python
class DINOv2ViT(nn.Module):
    """
    DINOv2 Vision Transformer models.
    
    Args:
        model_name (str): One of ['dinov2_vits14', 'dinov2_vitb14', 
                         'dinov2_vitl14', 'dinov2_vitg14']
        pretrained (bool): Load pretrained weights
        freeze (bool): Freeze model parameters
    
    Attributes:
        model: Loaded DINOv2 model
        embed_dim: Feature dimension
        patch_size: Patch size (14 or 16)
    """
    
    def __init__(self, model_name='dinov2_vitb14', 
                 pretrained=True, freeze=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning patch tokens."""
        
    def forward_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract hierarchical features."""
```

### Architecture-Specific Wrappers

Each architecture has a specialized wrapper:

```python
class ConvNextWrapper(BaseWrapper):
    """Wrapper for ConvNeXt models."""
    
class SwinWrapper(BaseWrapper):
    """Wrapper for Swin Transformer models."""
    
class STDCWrapper(BaseWrapper):
    """Wrapper for STDC models."""
    
class ResNetWrapper(BaseWrapper):
    """Wrapper for ResNet models."""
    
class MobileNetWrapper(BaseWrapper):
    """Wrapper for MobileNet models."""
```

## Loss Functions

### ScaleKD

Main distillation loss function implementing Scale-aware Knowledge Distillation.

```python
class ScaleKD(nn.Module):
    """
    Scale-aware Knowledge Distillation loss.
    
    Args:
        name (str): Loss identifier (e.g., 'scalekd_res5')
        alpha (List[float]): [spatial_weight, frequency_weight]
        student_dims (int): Student feature dimensions
        teacher_dims (int): Teacher feature dimensions
        query_hw (Tuple[int, int]): Query spatial dimensions
        pos_hw (Tuple[int, int]): Position encoding dimensions
        pos_dims (int): Position embedding dimensions
        window_shapes (Tuple[int, int]): Attention window size
        self_query (bool): Use features as queries
        softmax_scale (List[float]): Temperature scaling
        num_heads (int): Number of attention heads
    """
    
    def __init__(self, name, alpha, student_dims, teacher_dims,
                 query_hw, pos_hw, pos_dims, window_shapes=(1,1),
                 self_query=True, softmax_scale=[1,1], num_heads=8)
    
    def forward(self, preds_S: torch.Tensor, preds_T: torch.Tensor,
                query_s: Optional[torch.Tensor] = None,
                query_f: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute Scale-KD loss.
        
        Returns:
            Dictionary containing:
            - 'loss': Total loss
            - 'spatial_loss': Spatial component
            - 'frequency_loss': Frequency component
            - 'spatial_similarity': Spatial similarity metric
            - 'frequency_similarity': Frequency similarity metric
        """
```

### AttentionProjector

Projects student features to match teacher attention space.

```python
class AttentionProjector(nn.Module):
    """
    Projects features for attention computation.
    
    Args:
        student_dims (int): Student feature dimensions
        teacher_dims (int): Teacher feature dimensions
        query_hw (Tuple[int, int]): Query spatial size
        pos_dims (int): Position embedding dimensions
        window_shapes (Tuple[int, int]): Window size
        self_query (bool): Use self-attention
        softmax_scale (float): Temperature scaling
        num_heads (int): Number of attention heads
    """
    
    def forward(self, x: torch.Tensor, 
                query: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Project features and compute attention."""
```

## Data Pipeline

### CustomDataset

Flexible dataset for loading images from directories.

```python
class CustomDataset(Dataset):
    """
    Custom Dataset for Image Loading.
    
    Args:
        img_dirs (List[str]): List of directory paths
        transform (Optional[Callable]): Image transformations
    
    Attributes:
        images (List[str]): List of all image paths
    """
    
    def __init__(self, img_dirs: List[str] = ["path/to/dir"], 
                 transform: Optional[Callable] = None)
    
    def __len__(self) -> int:
        """Return total number of images."""
        
    def __getitem__(self, idx: int) -> torch.Tensor:
        """Load and transform image at index."""
```

### CustomDataModule

PyTorch Lightning DataModule for data management.

```python
class CustomDataModule(L.LightningDataModule):
    """
    Lightning DataModule for dataset management.
    
    Args:
        data_dir (List[str]): Training data directories
        val_dir (Optional[List[str]]): Validation directories
        batch_size (int): Batch size per GPU
        num_workers (int): Data loading workers
        transform (Optional[Callable]): Data transformations
        val_split (float): Validation split ratio
    """
    
    def __init__(self, data_dir, val_dir=None, batch_size=32,
                 num_workers=8, transform=None, val_split=0.1)
    
    def setup(self, stage: Optional[str] = None) -> None:
        """Setup datasets for train/val/test."""
        
    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        
    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
```

### DataAugmentationDINO

DINOv2-style data augmentation pipeline.

```python
class DataAugmentationDINO:
    """
    DINOv2-style multi-crop augmentation.
    
    Args:
        global_crops_scale (Tuple[float, float]): Scale range for global crops
        local_crops_scale (Tuple[float, float]): Scale range for local crops
        global_crops_number (int): Number of global crops
        local_crops_number (int): Number of local crops
        global_crops_size (int): Size of global crops
        local_crops_size (int): Size of local crops
    
    Attributes:
        global_transfo1: First global crop transformation
        global_transfo2: Second global crop transformation
        local_transfo: Local crop transformation
    """
    
    def __init__(self, global_crops_scale=(0.4, 1.0),
                 local_crops_scale=(0.05, 0.4),
                 global_crops_number=2, local_crops_number=8,
                 global_crops_size=224, local_crops_size=96)
    
    def __call__(self, image: PIL.Image) -> List[torch.Tensor]:
        """Apply augmentations and return multiple crops."""
```

## Training Components

### Callbacks

```python
class ModelCheckpoint(Callback):
    """
    Save model checkpoints during training.
    
    Args:
        dirpath (str): Directory to save checkpoints
        monitor (str): Metric to monitor
        mode (str): 'min' or 'max'
        save_top_k (int): Number of best models to keep
    """

class EarlyStopping(Callback):
    """
    Stop training when metric stops improving.
    
    Args:
        monitor (str): Metric to monitor
        patience (int): Epochs to wait before stopping
        mode (str): 'min' or 'max'
    """

class LearningRateMonitor(Callback):
    """Log learning rate during training."""
```

### Optimizers and Schedulers

```python
def create_optimizer(model: nn.Module, 
                    cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    """
    Create optimizer from configuration.
    
    Args:
        model: Model to optimize
        cfg: Optimizer configuration
        
    Returns:
        Configured optimizer
    """

def create_scheduler(optimizer: torch.optim.Optimizer,
                    cfg: Dict[str, Any]) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer to schedule
        cfg: Scheduler configuration
        
    Returns:
        Configured scheduler
    """
```

## Utilities

### Logger

Custom logging utilities.

```python
def get_logger(name: Optional[str] = None, 
               level: int = logging.INFO) -> logging.Logger:
    """
    Get configured logger instance.
    
    Args:
        name: Logger name
        level: Logging level
        
    Returns:
        Configured logger
    """

class ColoredFormatter(logging.Formatter):
    """Colored console output formatter."""
```

### Model Conversion

```python
def convert_to_anyma(checkpoint_path: str, 
                    output_path: str) -> None:
    """
    Convert PyTorch checkpoint to Anyma format.
    
    Args:
        checkpoint_path: Path to .ckpt file
        output_path: Path to save .pkl file
    """

def extract_state_dict(checkpoint: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """Extract model state dict from checkpoint."""
```

### Metrics

```python
def compute_spatial_similarity(student_feat: torch.Tensor,
                              teacher_feat: torch.Tensor) -> torch.Tensor:
    """
    Compute spatial similarity between features.
    
    Args:
        student_feat: Student feature maps
        teacher_feat: Teacher feature maps
        
    Returns:
        Similarity score [0, 1]
    """

def compute_feature_alignment(student_feat: torch.Tensor,
                             teacher_feat: torch.Tensor) -> torch.Tensor:
    """Compute feature space alignment metric."""
```

### Configuration Utilities

```python
def load_config(config_path: str) -> OmegaConf:
    """Load and validate configuration file."""

def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
    """Merge configuration dictionaries."""

def validate_config(cfg: Dict) -> None:
    """Validate configuration completeness."""
```

## Usage Examples

### Basic Training

```python
from train import DistillationTrainer
from omegaconf import OmegaConf

# Load configuration
cfg = OmegaConf.load('config/config.yaml')

# Create trainer
trainer = DistillationTrainer(cfg)

# Start training
trainer.train()
```

### Custom Model Integration

```python
from models import ModelWrapper

# Register custom model
@ModelWrapper.register_model('my_model')
class MyModelWrapper(ModelWrapper):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize your model
        
    def forward(self, x):
        # Implement feature extraction
        pass
```

### Custom Loss Function

```python
from losses import LOSS_REGISTRY

@LOSS_REGISTRY.register('my_loss')
class MyLoss(nn.Module):
    def forward(self, student_features, teacher_features):
        # Implement loss computation
        return loss_dict
```

### Data Pipeline Customization

```python
from datasets import CustomDataset, DataAugmentationDINO

# Custom augmentation
class MyAugmentation(DataAugmentationDINO):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Add custom transforms
        
# Custom dataset
class MyDataset(CustomDataset):
    def __getitem__(self, idx):
        # Custom loading logic
        pass
```