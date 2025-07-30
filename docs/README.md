# DINOv2 Knowledge Distillation Framework Documentation

Welcome to the comprehensive documentation for the DINOv2 Knowledge Distillation Framework. This documentation provides detailed information about the framework's architecture, implementation, and usage.

## 📚 Documentation Structure

- **[Getting Started Guide](getting_started.md)** - Installation, setup, and your first distillation run
- **[Architecture Overview](architecture.md)** - Deep dive into the framework's design and components
- **[Configuration Guide](configuration.md)** - Detailed explanation of all configuration options
- **[Models Documentation](models.md)** - Available teacher and student models, architectures, and wrappers
- **[Training Guide](training.md)** - Complete training pipeline, monitoring, and best practices
- **[Loss Functions](losses.md)** - Understanding Scale-KD and other distillation losses
- **[Data Pipeline](data_pipeline.md)** - Dataset handling, augmentations, and data loading
- **[API Reference](api_reference.md)** - Detailed API documentation for all modules
- **[Examples and Tutorials](examples.md)** - Practical examples and step-by-step tutorials
- **[Troubleshooting](troubleshooting.md)** - Common issues and their solutions
- **[Contributing](contributing.md)** - Guidelines for contributing to the project

## 🎯 Quick Links

### For New Users
- [Installation Guide](getting_started.md#installation)
- [Quick Start Tutorial](getting_started.md#quick-start)
- [Basic Configuration](configuration.md#basic-configuration)

### For Advanced Users
- [Custom Model Integration](models.md#custom-models)
- [Advanced Training Strategies](training.md#advanced-strategies)
- [Performance Optimization](training.md#performance-optimization)

### For Developers
- [Architecture Deep Dive](architecture.md)
- [API Reference](api_reference.md)
- [Contributing Guidelines](contributing.md)

## 📊 Framework Overview

The DINOv2 Knowledge Distillation Framework is a comprehensive solution for transferring knowledge from large Vision Transformer models (DINOv2) to smaller, more efficient architectures. Key features include:

- **Flexible Architecture Support**: Distill to ConvNeXt, Swin, STDC, ResNet, MobileNet, and more
- **State-of-the-art Distillation**: Implements Scale-KD method for effective knowledge transfer
- **Production Ready**: Distributed training, experiment tracking, and model conversion utilities
- **Highly Configurable**: YAML-based configuration for easy experimentation

## 🚀 Getting Help

- **Documentation**: You're here! Browse through the sections above
- **Issues**: Report bugs or request features on [GitHub Issues](https://github.com/ardaerendogru/dinov2_distillation/issues)
- **Discussions**: Join the community discussions on [GitHub Discussions](https://github.com/ardaerendogru/dinov2_distillation/discussions)

## 📝 License

This project is licensed under the MIT License. See the [LICENSE](../LICENSE) file for details.