from typing import List, Optional
import torch.nn as nn
from .base import BaseModel
from ..backbones import D2MobileNetV2
import os

versions = {
    "mobilenet_v2": {
        "strides": (1, 2, 2, 2, 1, 2, 1),
        "dilations": (1, 1, 1, 1, 1, 1, 1),
    },
    "mobilenet_v2_os8": {
        "strides": (1, 2, 2, 1, 1, 1, 1),
        "dilations": (1, 1, 1, 2, 2, 4, 4),
    },
    "mobilenet_v2_os16": {
        "strides": (1, 2, 2, 2, 1, 1, 1),
        "dilations": (1, 1, 1, 1, 1, 2, 2),
    },
}


def list_mobilenet_configs():
    return list(versions.keys())


def get_mobilenet_config(name):
    if name in versions:
        config = versions[name]
    else:
        raise NotImplementedError(
            f"There is not a MobileNet config called {name}."
        )
    checkpoint_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'checkpoints') # Construct path to checkpoints dir
    ckpt_path = os.path.join(checkpoint_dir, 'mobilenet_v2.pkl') # Construct full checkpoint path
    config['ckpt_path'] = ckpt_path # Add ckpt_path to config dictionary
    return config

class MobileNetV2Wrapper(BaseModel):
    """
    Wrapper class for MobileNetV2 models.

    This class allows you to easily use different MobileNetV2 architectures as backbones,
    and configure the output features.

    Args:
        model_name (str):
            Name of the specific MobileNetV2 model to use.
            Must be one of the configurations listed in `list_mobilenet_configs()`.
            Defaults to "mobilenet_v2".
        out_features (Optional[List[str]]):
            List of feature levels to output.
            Each element should be in the format 'res[2-5]'.
            If None, defaults to ['res2', 'res3', 'res4', 'res5'], outputting all available feature levels.
        cfg (dict):
            Configuration dictionary.

    Attributes:
        model (D2MobileNetV2):
            The underlying MobileNetV2 model from the `detectron2.modeling.backbone` library.
        feature_channels (dict):
            A dictionary mapping feature level names (e.g., 'res2') to their corresponding output channel dimensions.
    """
    def __init__(
        self,
        model_name: str = "mobilenet_v2",
        out_features: Optional[List[str]] = None,
    ):
        super().__init__()
        
        # Default output features if none specified
        if out_features is None:
            out_features = ['res2', 'res3', 'res4', 'res5']
        self._out_features = out_features
        
        # Get MobileNet config
        config = get_mobilenet_config(model_name)
        strides = config["strides"]
        dilations = config["dilations"]
        self.ckpt_dir = config['ckpt_path']
        # Create MobileNetV2 model
        self.model = D2MobileNetV2(
            widen_factor=1.0,
            strides=strides,
            dilations=dilations,
            out_features=out_features,
        )

        # Filter feature channels based on out_features
        self._out_feature_channels = {
            feature: self.model._out_feature_channels[feature]
            for feature in out_features
        }

    def get_features(self, x):
        """
        Forward pass through the feature extraction layers of the MobileNetV2 model.

        This method extracts feature maps from the input tensor `x` using the MobileNetV2 backbone.

        Args:
            x (torch.Tensor): The input tensor, typically an image batch.

        Returns:
            dict[str, torch.Tensor]:
                A dictionary of feature maps, where keys are feature level names (e.g., 'res2')
                and values are the corresponding feature tensors.
        """
        return self.model(x)
    
    @property
    def feature_channels(self):
        """
        Returns a dictionary mapping feature levels to their output channels.

        This property provides convenient access to the output channel dimensions of
        each feature level produced by the MobileNetV2 backbone.

        Returns:
            dict[str, int]:
                A dictionary where keys are feature level names (e.g., 'res2') and
                values are the number of output channels for that feature level.
        """
        return self._out_feature_channels