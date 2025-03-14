from typing import List, Optional
import torch.nn as nn
from .base import BaseModel
from ..backbones import D2MobileNetV3
from utils import get_logger
import os

logger = get_logger()
ckpt_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'checkpoints') # Construct path to checkpoints dir

versions = {
    "mobilenet_v3_small": {"size": "small", "dilated": False, 'ckpt_path': os.path.join(ckpt_dir, 'mobilenet_v3_small.pkl')},
    "mobilenet_v3_large": {"size": "large", "dilated": False, 'ckpt_path': os.path.join(ckpt_dir, 'mobilenet_v3_large.pkl')},
    "mobilenet_v3_small_os8": {"size": "small", "dilated": True, 'ckpt_path': os.path.join(ckpt_dir, 'mobilenet_v3_small.pkl')},
    "mobilenet_v3_large_os8": {"size": "large", "dilated": True, 'ckpt_path': os.path.join(ckpt_dir, 'mobilenet_v3_large.pkl')},
}


def list_mobilenet_v3_configs():
    return list(versions.keys())


def get_mobilenet_v3_config(name):
    if name in versions:
        return versions[name]
    else:
        raise NotImplementedError(
            f"There is not a MobileNetV3 config called {name}."
        )

class MobileNetV3Wrapper(BaseModel):
    """
    Wrapper class for MobileNetV3 models.

    This class allows you to easily use different MobileNetV3 architectures as backbones,
    and configure the output features.

    Args:
        model_name (str):
            Name of the specific MobileNetV3 model to use.
            Must be one of the configurations listed in `list_mobilenet_v3_configs()`.
            Defaults to "mobilenet_v3_small".
        out_features (Optional[List[str]]):
            List of feature levels to output.
            Each element should be in the format 'res[2-5]'.
            If None, defaults to ['res2', 'res3', 'res4', 'res5'], outputting all available feature levels.

    Attributes:
        model (D2MobileNetV3):
            The underlying MobileNetV3 model from the `detectron2.modeling.backbone` library.
        feature_channels (dict):
            A dictionary mapping feature level names (e.g., 'res2') to their corresponding output channel dimensions.
    """
    def __init__(
        self,
        model_name: str = "mobilenet_v3_small",
        out_features: Optional[List[str]] = None,
    ):
        super().__init__()
        
        # Default output features if none specified
        if out_features is None:
            out_features = ['res2', 'res3', 'res4', 'res5']
        self._out_features = out_features
        
        # Get MobileNetV3 config
        config = get_mobilenet_v3_config(model_name)
        size = config["size"]
        dilated = config["dilated"]
        self.ckpt_dir = config['ckpt_path']
        # Create MobileNetV3 model
        self.model = D2MobileNetV3(
            size=size,
            width_mult=1.0,
            dilated=dilated,
            out_features=out_features,
        )

        # Filter feature channels based on out_features
        self._out_feature_channels = {
            feature: self.model._out_feature_channels[feature]
            for feature in out_features
        }

    def get_features(self, x):
        """
        Forward pass through the feature extraction layers of the MobileNetV3 model.

        This method extracts feature maps from the input tensor `x` using the MobileNetV3 backbone.

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
        each feature level produced by the MobileNetV3 backbone.

        Returns:
            dict[str, int]:
                A dictionary where keys are feature level names (e.g., 'res2') and
                values are the number of output channels for that feature level.
        """
        return self._out_feature_channels