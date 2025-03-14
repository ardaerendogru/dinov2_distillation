from typing import List, Optional
import torch.nn as nn
from .base import BaseModel
from ..backbones import DarkNet
import os


versions = {
    "n": [[1, 2, 2, 1], [3, 16, 32, 64, 128, 256]],
    "s": [[1, 2, 2, 1], [3, 32, 64, 128, 256, 512]],
    "m": [[2, 4, 4, 2], [3, 48, 96, 192, 384, 576]],
    "l": [[3, 6, 6, 3], [3, 64, 128, 256, 512, 512]],
    "x": [[3, 6, 6, 3], [3, 80, 160, 320, 640, 640]],
}


def list_darknet_configs():
    return ["darknet_n", "darknet_s", "darknet_m", "darknet_l", "darknet_x"]


def get_darknet_config(name):
    version_chosen = name[-1]
    depth, width = [*versions.get(version_chosen)]
    checkpoint_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'checkpoints') # Construct path to checkpoints dir
    ckpt_file = f'yolov8{version_chosen}.pkl' # Construct checkpoint filename
    ckpt_path = os.path.join(checkpoint_dir, ckpt_file) # Construct full checkpoint path
    return {
        "depth": depth,
        "width": width,
        "ckpt_path": ckpt_path # Return ckpt_path
    }

class DarkNetWrapper(BaseModel):
    """
    Wrapper class for DarkNet models.

    This class allows you to easily use different DarkNet architectures as backbones,
    and configure the output features.

    Args:
        model_name (str):
            Name of the specific DarkNet model to use.
            Must be one of the configurations listed in `list_darknet_configs()`.
            Defaults to "darknet_s".
        out_features (Optional[List[str]]):
            List of feature levels to output.
            Each element should be in the format 'res[2-5]'.
            If None, defaults to ['res2', 'res3', 'res4', 'res5'], outputting all available feature levels.
        cfg (dict):
            Configuration dictionary.

    Attributes:
        model (DarkNet):
            The underlying DarkNet model from the `detectron2.modeling.backbone` library.
        feature_channels (dict):
            A dictionary mapping feature level names (e.g., 'res2') to their corresponding output channel dimensions.
    """
    def __init__(
        self,
        model_name: str = "darknet_s",
        out_features: Optional[List[str]] = None,
        cfg = None # Added cfg
    ):
        super().__init__()
        
        # Default output features if none specified
        if out_features is None:
            out_features = ['res2', 'res3', 'res4', 'res5']
        self._out_features = out_features
        
        # Get DarkNet config
        config = get_darknet_config(model_name)
        depth = config["depth"]
        width = config["width"]
        self.ckpt_dir = config['ckpt_path'] # Get ckpt_path from config



        # Create DarkNet model
        self.model = DarkNet(
            depth=depth,
            width=width,
        )

        # Filter feature channels based on out_features
        self._out_feature_channels = {
            feature: self.model._out_feature_channels[feature]
            for feature in out_features
        }

    def get_features(self, x):
        """
        Forward pass through the feature extraction layers of the DarkNet model.

        This method extracts feature maps from the input tensor `x` using the DarkNet backbone.

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
        each feature level produced by the DarkNet backbone.

        Returns:
            dict[str, int]:
                A dictionary where keys are feature level names (e.g., 'res2') and
                values are the number of output channels for that feature level.
        """
        return self._out_feature_channels