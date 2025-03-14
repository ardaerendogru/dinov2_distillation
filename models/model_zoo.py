import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any
from .wrappers.resnet_wrapper import ResNetWrapper
from .wrappers.stdc_wrapper import STDCWrapper
from .wrappers.convnext_wrapper import ConvNextWrapper
from .wrappers.darknet_wrapper import DarkNetWrapper
from .wrappers.mit_wrapper import MITWrapper
from .wrappers.mobilenet_v2_wrapper import MobileNetV2Wrapper
from .wrappers.mobilenet_v3_wrapper import MobileNetV3Wrapper
from .wrappers.presnet_wrapper import PResNetWrapper
from .wrappers.swin_wrapper import D2SwinTransformerWrapper
from .wrappers.timm_wrapper import TimmWrapper
import torch.nn.functional as F


class ModelWrapper(nn.Module):
    """
    A versatile wrapper class for utilizing various backbone models for feature extraction.

    This class simplifies the process of using different pre-trained models (like ResNet, STDCNet, etc.)
    as feature extractors in your distillation pipeline. It handles model instantiation,
    checkpoint loading, feature extraction, and output formatting.

    Args:
        model_name (str):
            Name of the backbone model architecture to use.
            Must be one of the keys in `ModelWrapper.MODEL_MAP` (e.g., 'resnet50', 'stdc2', 'convnext_tiny').
        n_patches (tuple[int, int]):
            The desired output size (height, width) for the extracted feature maps after interpolation.
            This effectively defines the patch size for feature extraction.
        target_feature (list[str]):
            List of feature levels to extract from the backbone model.
            The available feature levels depend on the specific backbone architecture.
            Defaults to `['res5', 'res4']`, which are common feature levels in many CNNs.
        checkpoint_path (Optional[str]):
            Path to a pre-trained model checkpoint file (`.pth`).
            If provided, the weights from the checkpoint will be loaded into the backbone model.
            Defaults to None, which means using randomly initialized weights or pre-trained weights
            loaded by the backbone itself (if supported).
        **model_kwargs:
            Additional keyword arguments to be passed to the constructor of the specific
            backbone model wrapper class (e.g., `ResNetWrapper`, `STDCWrapper`).
            These arguments can be used to customize the backbone model, such as specifying
            normalization type, activation function, or other model-specific parameters.

    Attributes:
        model (nn.Module):
            The instantiated backbone model (e.g., ResNet, STDCNet) wrapped by its corresponding wrapper class.
        n_patches (tuple[int, int]):
            The target output size (height, width) for the feature maps.
        target_features (list[str]):
            The list of feature levels to be extracted.
        patch_size (tuple[int, int]):
            Alias for `n_patches`, representing the size to which feature maps are interpolated.
    """

    MODEL_MAP = {
        'resnet': ResNetWrapper,
        'stdc': STDCWrapper,
        'convnext': ConvNextWrapper,
        'darknet': DarkNetWrapper,
        'mit': MITWrapper,
        'mobilenet_v2': MobileNetV2Wrapper,
        'mobilenet_v3': MobileNetV3Wrapper,
        'presnet': PResNetWrapper,
        'swin':D2SwinTransformerWrapper,
        'efficientnet' : TimmWrapper,
        'edgenext' : TimmWrapper,
        'mobilenetv3': TimmWrapper


    }
    def __init__(
        self,
        model_name: str,
        n_patches,
        target_feature: list[str] = ['res5', 'res4'],
    ):
        super().__init__()

        model_id = model_name.split('_')[0].lower()
        if model_id == 'mobilenet':
            version = model_name.split('_')[1].lower()
            model_id = model_id + '_' + version
        
        self.model = self.MODEL_MAP[model_id](model_name = model_name, out_features = target_feature)
  
        
        self.n_patches = n_patches
        self.target_features = target_feature
        self.patch_size = n_patches



    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Performs a forward pass through the backbone model and extracts specified feature levels.

        This method takes an input tensor `x`, feeds it through the backbone model,
        and retrieves the feature maps corresponding to the `target_features` specified
        during initialization. The extracted feature maps are then interpolated to the desired
        `patch_size` to ensure consistent output dimensions.

        Args:
            x (torch.Tensor):
                The input tensor, typically an image batch, to be processed by the backbone model.

        Returns:
            Dict[str, torch.Tensor]:
                A dictionary containing the extracted and interpolated feature maps.
                The keys of the dictionary are the feature level names (e.g., 'res5', 'res4'),
                and the values are the corresponding feature tensors, interpolated to `patch_size`.
        """
        features = self.model.get_features(x)
        matched_features = {}
        for feat in self.target_features:
            if feat in features:
                target_feature = features[feat]
                
                # Interpolate to match patch size
                matched_features[feat] = torch.nn.functional.interpolate(
                    target_feature,
                    size=(self.patch_size[0], self.patch_size[1]),
                    mode='bilinear',
                    align_corners=False
                )
        return matched_features


    @property
    def feature_channels(self):
        """
        Retrieves the number of output channels for each feature level from the backbone model.

        This property provides convenient access to the channel dimensions of the extracted
        feature maps. It directly queries the `feature_channels` property of the underlying
        backbone model wrapper to obtain this information.

        Returns:
            dict:
                A dictionary where keys are feature level names (e.g., 'res2', 'res3', 'res4', 'res5')
                and values are the corresponding number of output channels for each feature level.
        """
        return self.model.feature_channels  # Access feature_channels directly from self.model