# DINOv2 Knowledge Distillation Framework

A comprehensive framework for knowledge distillation from DINOv2 Vision Transformer models to more efficient architectures like ConvNeXt, Swin Transformer, and STDC networks.

## 📋 Overview

This project implements a flexible and powerful knowledge distillation pipeline for transferring the rich visual representations learned by DINOv2 Vision Transformers to smaller, more efficient models. The framework supports various distillation techniques, focusing on spatial and channel attention transfer, feature alignment, and self-supervised learning signals.

### Key Features

- 🔄 **Flexible Model Support**: Distill from DINOv2 ViT models to various student architectures (ConvNeXt, Swin, STDC)
- 🧠 **Advanced Distillation Techniques**: Implements Scale-KD and other state-of-the-art knowledge distillation methods
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
│   ├── wrappers/            # Model wrapper implementations
│   └── model_zoo.py         # Model factory and registry
├── train/                   # Training modules
│   └── distillation_module.py  # Lightning module for distillation
├── utils/                   # Utility functions
├── scripts/                 # Helper scripts
├── checkpoints/             # Saved model checkpoints
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
  model_name: swin_tiny
  student_keys: [res5, res4]

teacher:
  model_name: dinov2_vits14

loss:
  losses:
    - type: scalekd
      weight: 1
      kwargs:
        alpha: [0.08, 0.06]
        window_shapes: [1, 1]
        self_query: True
        softmax_scale: [5.0, 5.0]
        dis_freq: high
        num_heads: 16
        name: scalekd_res4
        use_this: true
```

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

### Monitoring Training

The framework integrates with both Weights & Biases and TensorBoard for experiment tracking:

```bash
# Launch TensorBoard
tensorboard --logdir logs/
```

## 📊 Distillation Methods

The framework implements several distillation techniques:

- **Scale-KD**: A multi-scale knowledge distillation approach that transfers attention maps across different scales
- **Spatial Similarity**: Aligns spatial feature representations between teacher and student
- **Channel Attention**: Transfers channel-wise attention patterns

## 🧪 Experimental Results

Performance comparison of different student models after distillation:

| Student Model | Teacher | Dataset | Performance Metric | Score |
|---------------|---------|---------|-------------------|-------|
| ConvNeXt-Tiny | DINOv2-ViT-S/14 | ImageNet | Feature Alignment | 0.85 |
| Swin-Tiny     | DINOv2-ViT-S/14 | ImageNet | Feature Alignment | 0.82 |
| STDC1         | DINOv2-ViT-S/14 | ImageNet | Feature Alignment | 0.78 |

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- [DINOv2](https://github.com/facebookresearch/dinov2) for the original self-supervised learning framework
- [PyTorch Lightning](https://lightning.ai) for the training framework
- [Weights & Biases](https://wandb.ai) for experiment tracking