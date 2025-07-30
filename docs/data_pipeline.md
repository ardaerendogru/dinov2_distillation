# Data Pipeline Documentation

This document covers the data pipeline in the DINOv2 Knowledge Distillation Framework, including dataset handling, augmentations, and data loading strategies.

## Table of Contents
- [Overview](#overview)
- [Dataset Structure](#dataset-structure)
- [Data Augmentations](#data-augmentations)
- [Data Loading](#data-loading)
- [Custom Datasets](#custom-datasets)
- [Performance Optimization](#performance-optimization)
- [Best Practices](#best-practices)

## Overview

The data pipeline is designed to:
- Support flexible directory-based image loading
- Implement DINOv2-style augmentations for optimal distillation
- Enable efficient distributed data loading
- Provide easy customization for different datasets

### Pipeline Architecture

```
Data Pipeline Flow
├── Image Directories
├── CustomDataset
├── DataAugmentationDINO
│   ├── Global Crops
│   └── Local Crops
├── DataLoader
│   ├── Distributed Sampler
│   └── Multi-worker Loading
└── Batched Tensors
```

## Dataset Structure

### CustomDataset Implementation

The framework provides a flexible dataset class that loads images from directories:

```python
# datasets/CustomDataset.py
class CustomDataset(Dataset):
    """
    Custom Dataset for Image Loading.
    
    Args:
        img_dirs: List of directory paths containing images
        transform: Torchvision transforms pipeline
    """
    def __init__(self, img_dirs=["path/to/dir"], transform=None):
        self.img_dirs = img_dirs
        self.transform = transform
        
        # Collect all image paths
        self.images = []
        for img_dir in img_dirs:
            for img_name in os.listdir(img_dir):
                if img_name.endswith(('.jpg', '.png', '.jpeg')):
                    self.images.append(os.path.join(img_dir, img_name))
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image
```

### Directory Organization

Organize your data as follows:

```
data/
├── train/
│   ├── class1/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   ├── class2/
│   │   └── ...
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

Or for unsupervised training:

```
data/
├── images/
│   ├── img001.jpg
│   ├── img002.png
│   └── ...
```

### Configuration

```yaml
data_loader:
  data_dir: ["/path/to/train/images", "/path/to/extra/images"]
  val_dir: ["/path/to/val/images"]  # Optional
  val_split: 0.1  # Use 10% of train for validation if val_dir not provided
```

## Data Augmentations

### DINOv2-Style Augmentations

The framework implements augmentations consistent with DINOv2 training:

```python
# datasets/augmentations.py
class DataAugmentationDINO:
    def __init__(self,
                 global_crops_scale=(0.4, 1.0),
                 local_crops_scale=(0.05, 0.4),
                 global_crops_number=2,
                 local_crops_number=8,
                 global_crops_size=224,
                 local_crops_size=96):
        
        # Global crops: Large views for main features
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(
                global_crops_size,
                scale=global_crops_scale,
                interpolation=InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            self._get_color_distortion(),
            GaussianBlur(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Local crops: Small views for detailed features
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(
                local_crops_size,
                scale=local_crops_scale,
                interpolation=InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            self._get_color_distortion(),
            GaussianBlur(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
```

### Augmentation Components

#### Color Distortion

```python
def _get_color_distortion(self, s=0.5):
    """Color jittering from SimCLR"""
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([
        rnd_color_jitter,
        rnd_gray
    ])
    return color_distort
```

#### Gaussian Blur

```python
class GaussianBlur:
    """Gaussian blur augmentation"""
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.0):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max
        
    def __call__(self, img):
        if random.random() < self.prob:
            radius = random.uniform(self.radius_min, self.radius_max)
            return img.filter(ImageFilter.GaussianBlur(radius=radius))
        return img
```

#### Solarization

```python
class Solarization:
    """Solarization augmentation from BYOL"""
    def __init__(self, p=0.2):
        self.prob = p
        
    def __call__(self, img):
        if random.random() < self.prob:
            return ImageOps.solarize(img)
        return img
```

### Configuration Options

```yaml
data_transform:
  # Global crops configuration
  global_crops_scale: [0.4, 1.0]
  global_crops_size: 224
  global_crops_number: 2
  
  # Local crops configuration
  local_crops_scale: [0.05, 0.4]
  local_crops_size: 96
  local_crops_number: 8
  
  # Augmentation strengths
  color_jitter_strength: 0.4
  blur_prob: 0.5
  solarize_prob: 0.1
  
  # Normalization (ImageNet)
  normalize_mean: [0.485, 0.456, 0.406]
  normalize_std: [0.229, 0.224, 0.225]
```

### Multi-Crop Strategy

The multi-crop strategy creates multiple views of each image:

```python
def __call__(self, image):
    """Generate multiple crops from a single image"""
    crops = []
    
    # Generate global crops
    for i in range(self.global_crops_number):
        if i == 0:
            crop = self.global_transfo1(image)
        else:
            crop = self.global_transfo2(image)
        crops.append(crop)
    
    # Generate local crops
    for i in range(self.local_crops_number):
        crop = self.local_transfo(image)
        crops.append(crop)
    
    return crops
```

## Data Loading

### DataModule Implementation

The framework uses PyTorch Lightning's DataModule pattern:

```python
# datasets/CustomDataset.py
class CustomDataModule(L.LightningDataModule):
    def __init__(self, 
                 data_dir,
                 val_dir=None,
                 batch_size=32,
                 num_workers=8,
                 transform=None,
                 val_split=0.1):
        super().__init__()
        self.data_dir = data_dir
        self.val_dir = val_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.val_split = val_split
    
    def setup(self, stage=None):
        if self.val_dir:
            # Use provided validation directory
            self.train_dataset = CustomDataset(
                self.data_dir, self.transform
            )
            self.val_dataset = CustomDataset(
                self.val_dir, self.transform
            )
        else:
            # Split training data
            full_dataset = CustomDataset(
                self.data_dir, self.transform
            )
            val_size = int(len(full_dataset) * self.val_split)
            train_size = len(full_dataset) - val_size
            
            self.train_dataset, self.val_dataset = random_split(
                full_dataset, [train_size, val_size]
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True
        )
```

### Distributed Data Loading

For multi-GPU training, the framework automatically handles distributed sampling:

```python
def train_dataloader(self):
    # Distributed sampler is automatically added by Lightning
    return DataLoader(
        self.train_dataset,
        batch_size=self.batch_size,
        num_workers=self.num_workers,
        shuffle=False,  # Shuffling handled by DistributedSampler
        pin_memory=True,
        persistent_workers=True
    )
```

### Batch Collation

For multi-crop augmentation, custom collation is needed:

```python
def collate_fn(batch):
    """
    Collate function for multi-crop dataset.
    Each item is a list of crops.
    """
    # batch is a list of crop lists
    # Flatten and stack
    all_crops = []
    for crops in batch:
        all_crops.extend(crops)
    
    # Stack into tensor
    return torch.stack(all_crops)
```

## Custom Datasets

### Creating a Custom Dataset

#### Example: Dataset with Labels

```python
class LabeledDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        # Load image paths and labels
        self.samples = []
        self.class_to_idx = {}
        
        for class_idx, class_name in enumerate(sorted(os.listdir(data_dir))):
            class_dir = os.path.join(data_dir, class_name)
            if os.path.isdir(class_dir):
                self.class_to_idx[class_name] = class_idx
                for img_name in os.listdir(class_dir):
                    if img_name.endswith(('.jpg', '.png', '.jpeg')):
                        img_path = os.path.join(class_dir, img_name)
                        self.samples.append((img_path, class_idx))
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
```

#### Example: Dataset from CSV

```python
class CSVDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Return image and any metadata
        return image, {
            'filename': row['filename'],
            'metadata': row.get('metadata', None)
        }
```

### Integrating Custom Datasets

Update the DataModule to use your custom dataset:

```python
class CustomDataModule(L.LightningDataModule):
    def __init__(self, dataset_class, **kwargs):
        super().__init__()
        self.dataset_class = dataset_class
        self.kwargs = kwargs
    
    def setup(self, stage=None):
        self.train_dataset = self.dataset_class(
            data_dir=self.kwargs['train_dir'],
            transform=self.train_transform
        )
        self.val_dataset = self.dataset_class(
            data_dir=self.kwargs['val_dir'],
            transform=self.val_transform
        )
```

## Performance Optimization

### 1. Efficient Data Loading

```yaml
data_loader:
  num_workers: 8  # Usually 2-4 per GPU
  pin_memory: true  # Faster GPU transfer
  persistent_workers: true  # Keep workers alive
  prefetch_factor: 2  # Prefetch batches
```

### 2. Memory-Efficient Loading

```python
class EfficientDataset(Dataset):
    def __init__(self, img_paths, transform=None):
        self.img_paths = img_paths
        self.transform = transform
        
        # Don't load all images into memory
        # Load on-demand in __getitem__
    
    def __getitem__(self, idx):
        # Load image only when needed
        with Image.open(self.img_paths[idx]) as img:
            img = img.convert('RGB')
            if self.transform:
                img = self.transform(img)
        return img
```

### 3. Caching Strategies

```python
class CachedDataset(Dataset):
    def __init__(self, img_paths, transform=None, cache_size=1000):
        self.img_paths = img_paths
        self.transform = transform
        self.cache = {}
        self.cache_size = cache_size
    
    def __getitem__(self, idx):
        if idx in self.cache:
            image = self.cache[idx]
        else:
            image = Image.open(self.img_paths[idx]).convert('RGB')
            
            # Add to cache if space available
            if len(self.cache) < self.cache_size:
                self.cache[idx] = image
        
        if self.transform:
            image = self.transform(image)
        
        return image
```

### 4. Mixed Precision Data Loading

```python
# Enable automatic mixed precision in transforms
class MixedPrecisionTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform
    
    def __call__(self, image):
        # Apply base transform
        image = self.base_transform(image)
        
        # Convert to half precision if needed
        if torch.cuda.is_available():
            image = image.half()
        
        return image
```

### 5. Data Prefetching

```python
class PrefetchLoader:
    def __init__(self, loader):
        self.loader = loader
        self.stream = torch.cuda.Stream()
    
    def __iter__(self):
        first = True
        for next_data in self.loader:
            with torch.cuda.stream(self.stream):
                next_data = next_data.cuda(non_blocking=True)
            
            if not first:
                yield data
            else:
                first = False
            
            torch.cuda.current_stream().wait_stream(self.stream)
            data = next_data
        
        yield data
```

## Best Practices

### 1. Dataset Organization

- Keep images organized in directories
- Use consistent naming conventions
- Store metadata separately (CSV/JSON)
- Validate data integrity before training

### 2. Augmentation Strategy

```python
# Training augmentations (strong)
train_transform = DataAugmentationDINO(
    global_crops_scale=(0.4, 1.0),
    local_crops_scale=(0.05, 0.4)
)

# Validation augmentations (minimal)
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])
```

### 3. Batch Size Selection

```yaml
# Guidelines for batch size
# Small models (MobileNet): 64-128 per GPU
# Medium models (ResNet50): 32-64 per GPU
# Large models (Swin): 16-32 per GPU

data_loader:
  batch_size: 32  # Per GPU
  
# With gradient accumulation
train:
  accumulate_grad_batches: 4  # Effective batch = 32 * 4 = 128
```

### 4. Debugging Data Pipeline

```python
# Visualize augmentations
def visualize_augmentations(dataset, num_samples=4):
    fig, axes = plt.subplots(num_samples, 10, figsize=(20, 8))
    
    for i in range(num_samples):
        # Get original image
        original = dataset.get_original(i)
        axes[i, 0].imshow(original)
        axes[i, 0].set_title('Original')
        
        # Get augmented versions
        for j in range(1, 10):
            augmented = dataset[i]
            if isinstance(augmented, list):
                augmented = augmented[0]  # First crop
            
            # Denormalize for visualization
            augmented = denormalize(augmented)
            axes[i, j].imshow(augmented.permute(1, 2, 0))
            axes[i, j].set_title(f'Aug {j}')
    
    plt.tight_layout()
    plt.savefig('augmentations.png')
```

### 5. Data Validation

```python
def validate_dataset(dataset):
    """Check dataset integrity"""
    print(f"Dataset size: {len(dataset)}")
    
    # Check sample loading
    errors = []
    for i in range(min(100, len(dataset))):
        try:
            sample = dataset[i]
            if isinstance(sample, tuple):
                img, label = sample
            else:
                img = sample
            
            # Validate tensor shape
            assert img.dim() == 3  # C, H, W
            assert img.shape[0] == 3  # RGB
            
        except Exception as e:
            errors.append((i, str(e)))
    
    if errors:
        print(f"Found {len(errors)} errors:")
        for idx, error in errors[:5]:
            print(f"  Index {idx}: {error}")
    else:
        print("Dataset validation passed!")
```

### 6. Monitoring Data Loading

```python
# In training script
def on_train_batch_start(self, batch, batch_idx):
    # Log data loading time
    if hasattr(self, 'last_batch_end_time'):
        load_time = time.time() - self.last_batch_end_time
        self.log('data_load_time', load_time)

def on_train_batch_end(self, outputs, batch, batch_idx):
    self.last_batch_end_time = time.time()
```