# DINOv2 Knowledge Distillation Framework

A comprehensive framework for knowledge distillation from DINOv2 Vision Transformer models to more efficient architectures like ConvNeXt, Swin Transformer, and STDC networks.

## 📋 Overview

This project implements a flexible and powerful knowledge distillation pipeline for transferring the rich visual representations learned by DINOv2 Vision Transformers to smaller, more efficient models. 

### Key Features

- 🔄 **Flexible Model Support**: Distill from DINOv2 ViT models to various student architectures (ConvNeXt, Swin, STDC)
- 🧠 **Advanced Distillation Techniques**: Implements Scale-KD distillation method
- 📊 **Comprehensive Monitoring**: Integration with Weights & Biases and TensorBoard for experiment tracking
- ⚡ **Efficient Training**: Distributed training support with PyTorch Lightning
- 🛠️ **Highly Configurable**: YAML-based configuration system for easy experiment setup

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU(s) recommended

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ardaerendogru/dinov2_distillation.git
   cd dinov2_distillation
   ```

2. Create and activate a virtual environment using `uv`:
   ```bash
   uv venv .venv
   source .venv/bin/activate  # On Linux/macOS
   ```

3. Install dependencies using `uv`:
   ```bash
   uv pip install -r requirements.txt
   ```

## 🏗️ Project Structure

```
dinov2_distillation/
├── checkpoints              # Pretrained student weights.
├── config/                  # Configuration files for different experiments
│   ├── config.yaml          # Main configuration file
│   └── ...
├── datasets/                # Dataset handling and augmentations
│   ├── CustomDataset.py     # Custom dataset implementation
│   └── augmentations.py     # DINOv2-style data augmentations
├── losses/                  # Distillation loss functions
├── models/                  # Model definitions
│   ├── backbones/           # Backbone architectures
│   │   └── dinov2.py        # DINOv2 ViT implementation
│   │   └── ...              
│   ├── wrappers/            # Model wrapper implementations
│   └── model_zoo.py         # Model factory and registry
├── train/                   # Training modules
│   └── distillation_module.py  # Lightning module for distillation
├── utils/                   # Utility functions
├── scripts/                 # Helper scripts to create weights in anyma format. 
├── logs/                    # Training logs
├── train.py                 # Main training script
└── run.ipynb                # Jupyter notebook for interactive experiments
```


## ⚙️ Configuration

The framework uses YAML configuration files to define all aspects of the distillation process. Key configuration sections include:

- **wandb**: Weights & Biases logging configuration
- **student**: Student model architecture and settings
- **teacher**: Teacher model (DINOv2) configuration
- **data_transform**: Data augmentation settings
- **optimizer**: Optimization settings including learning rate schedules
- **loss**: Distillation loss functions and their weights
- **train**: Training parameters (epochs, devices, etc.)
- **data_loader**: Dataset paths and dataloader settings
- **checkpoints**: Model checkpoint configuration

Example configuration (from `config.yaml`):

```yaml
student:
  model_name: stdc_2
  student_keys: [res5, res4]
  checkpoint_path: path/to/checkpoint # optional

teacher:
  model_name: dinov2_vits14



optimizer:
  type: AdamW
  kwargs:
    lr: 1e-3
    betas: [0.9, 0.999]
    weight_decay: 0.01
  scheduler:
    type: CosineAnnealingLR
    kwargs:
      T_max: 240
      eta_min: 1e-5
    monitor: val_loss
    interval: epoch
    frequency: 1

loss:
  losses:
    - type: scalekd
      weight: 1
      kwargs:
        alpha: [0.08, 0.06]
        window_shapes: [1, 1]
        self_query: True
        softmax_scale: [5.0, 5.0]
        num_heads: 16
        name: scalekd_res4
    - type: scalekd
      weight: 1.0
      kwargs:
        alpha: [0.08, 0.06]
        window_shapes: [1, 1]
        self_query: False
        softmax_scale: [5.0, 5.0]
        num_heads: 24
        name: scalekd_res5

train:
  max_epochs: 240
  accelerator: gpu
  devices: [0,1]
  num_nodes: 1
  strategy: ddp_find_unused_parameters_true
  # resume_from_checkpoint: dinov2_distillation/logs/path/to/ckpt  # Add this line to continue training
  accumulate_grad_batches: 1 \

data_loader:
  data_dir: [/home/arda/data/train2017]
  #val_dir:  also a list
  batch_size: 1 #per gpu
  num_workers: 8

checkpoints:
  dirpath: checkpoints
  monitor:  val_scalekd_res5_spatial_similarity
  mode: max
  save_top_k: 1

wandb:
  project: "distillation"  # Project name in W&B
  tags: ["distillation", "convnext", "dinov2"]  # Searchable tags
  notes: "Knowledge distillation from DINOv2 to convnext"  # Run description

```


### 🧠 Supported Student Models

The framework supports a wide range of efficient architectures as student models for knowledge distillation from DINOv2:

#### ConvNeXt Models
- `convnext_atto`
- `convnext_pico`
- `convnext_nano`
- `convnext_tiny`
- `convnext_base`

#### DarkNet Models
- `darknet_n`
- `darknet_s`
- `darknet_m`
- `darknet_l`
- `darknet_x`

#### MIT (Multiscale Image Transformer) Models
- `mit_b0`
- `mit_b1`
- `mit_b2`
- `mit_b3`
- `mit_b4`
- `mit_b5`

#### MobileNetV2 Models
- `mobilenet_v2`
- `mobilenet_v2_os8`
- `mobilenet_v2_os16`

#### MobileNetV3 Models
- `mobilenet_v3_small`
- `mobilenet_v3_large`
- `mobilenet_v3_small_os8`
- `mobilenet_v3_large_os8`

#### PResNet Models
- `presnet_18`
- `presnet_34`
- `presnet_50`
- `presnet_101`

#### ResNet Models
- `resnet_18`
- `resnet_34`
- `resnet_50`
- `resnet_101`

#### STDC Models
- `stdc_1`
- `stdc_2`

#### Swin Transformer Models
- `swin_tiny`
- `swin_small`

#### TIMM Models
- `efficientnet_b0`
- `efficientnet_b1`
- `efficientnet_b2`
- `efficientnet_b3`
- `efficientnet_b4`
- `edgenext_xx_small`
- `edgenext_x_small`
- `edgenext_small`
- `edgenext_base`
- `mobilenetv3_small_050`
- `mobilenetv3_small_075`
- `mobilenetv3_small_100`
- `mobilenetv3_large_075`
- `mobilenetv3_large_100`



### Scale-KD Loss Configuration

Scale-KD (Scale-aware Knowledge Distillation) transfers attention maps across different scales from teacher to student. Each loss instance should be configured for a specific feature level (res2, res3, res4, or res5).

#### Naming Convention

For proper tracking and visualization, each Scale-KD loss **must** follow this naming convention:

- `scalekd_res2` - For distillation at the res2 feature level
- `scalekd_res3` - For distillation at the res3 feature level
- `scalekd_res4` - For distillation at the res4 feature level
- `scalekd_res5` - For distillation at the res5 feature level

This naming pattern enables the training pipeline to automatically track and log spatial similarity metrics for each feature level.

#### Key Parameters

- **`alpha`**: Controls the contributions of the components of DFM (Direct vs Frequency filtered featuremaps).
- **`window_shapes`**: Defines the window size for computing attention. `[1, 1]` means global attention.
- **`self_query`**: When `True`, uses the feature map from the previous TPP part for as query. When `False`, uses learnable queries.
- **`softmax_scale`**: Temperature parameter for the softmax operation in attention computation.
- **`num_heads`**: Number of attention heads to use. Should generally increase with feature level depth.
- **`name`**: Must follow the `scalekd_res{N}` pattern where N is the feature level (2-5).

## 🔧 Usage

### Training a Model

To start a distillation training run:

```bash
python train.py --config config/config.yaml
```

You can override configuration parameters via command line:

```bash
python train.py --config config/config.yaml --train.max_epochs 100 --optimizer.kwargs.lr 5e-4
```

### 🔄 Converting Trained Models to Anyma Format

After training your distilled model, you may want to use it in other frameworks or applications that support the Anyma weight format. The framework provides a utility script to convert your trained model weights.


The `scripts/convert_to_anyma.py` script converts PyTorch Lightning checkpoints from the distillation process to the Anyma-compatible format:

```bash
python scripts/convert_to_anyma.py <input_checkpoint_path> <output_pickle_path>
```

#### Arguments:

- `input_checkpoint_path`: Path to the PyTorch Lightning checkpoint file (`.ckpt`) containing your trained student model
- `output_pickle_path`: Path where the converted weights will be saved (`.pkl`)

#### Example:

```bash
python scripts/convert_to_anyma.py checkpoints/convnext_tiny_distilled.ckpt checkpoints/convnext_tiny_anyma.pkl
```

### Monitoring Training

The framework integrates with both Weights & Biases and TensorBoard for experiment tracking:

```bash
# Launch TensorBoard
tensorboard --logdir logs/
```




## 🙏 Acknowledgements

- [DINOv2](https://github.com/facebookresearch/dinov2) for the original self-supervised learning framework
- [ScaleKD](https://arxiv.org/abs/2411.06786) for the distillation method.
- [PyTorch Lightning](https://lightning.ai) for the training framework
- [Weights & Biases](https://wandb.ai) for experiment tracking
