from typing import List, Optional
import torch.nn as nn
from .base import BaseModel
# from .resnet import ResNet, BasicStem, make_resnet_stages # Removed ResNet imports
# from torchvision import models # Removed torchvision import
from ..backbones import D2ConvNextV2 # Import ConvNeXt
import os

def list_convnext_configs():
    return [
        "convnext_atto",
        "convnext_pico",
        "convnext_nano",
        "convnext_tiny",
        "convnext_base",
    ]  # , "convnext_large"]


def get_convnext_config(name):
    checkpoint_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'checkpoints') # Construct path to checkpoints dir
    if name == "convnext_atto":
        depths = [2, 2, 6, 2]
        dims = [40, 80, 160, 320]
        ckpt = os.path.join(checkpoint_dir, 'convnextv2_atto_1k_224_ema.pkl') # Use constructed path
    elif name == "convnext_pico":
        depths = [2, 2, 6, 2]
        dims = [64, 128, 256, 512]
        ckpt = os.path.join(checkpoint_dir, 'convnextv2_pico_1k_224_ema.pkl') # Use constructed path
    elif name == "convnext_nano":
        depths = [2, 2, 8, 2]
        dims = [80, 160, 320, 640]
        ckpt = os.path.join(checkpoint_dir, 'convnextv2_nano_1k_224_ema.pkl') # Use constructed path
    elif name == "convnext_tiny":
        depths = [3, 3, 9, 3]
        dims = [96, 192, 384, 768]
        ckpt = os.path.join(checkpoint_dir, 'convnextv2_tiny_1k_224_ema.pkl') # Use constructed path
    elif name == "convnext_base":
        depths = [3, 3, 27, 3]
        dims = [128, 256, 512, 1024]
        ckpt = os.path.join(checkpoint_dir, 'convnextv2_base_1k_224_ema.pkl') # Use constructed path
    else:
        raise NotImplementedError(
            f"There is not a ConvNext_{name}. Please, check the backbone name."
        )
    return {
        "depths": depths,
        "dims": dims,
        'ckpt_dir':ckpt
    }


class ConvNextWrapper(BaseModel):
    """
    Wrapper class for ConvNeXt models.

    This class allows you to easily use different ConvNeXt architectures as backbones,
    and configure the output features.

    Args:
        model_name (str):
            Name of the specific ConvNeXt model to use.
            Must be one of the configurations listed in `list_convnext_configs()`.
            Defaults to "convnext_tiny".
        out_features (Optional[List[str]]):
            List of feature levels to output.
            Each element should be in the format 'res[2-5]'.
            If None, defaults to ['res2', 'res3', 'res4', 'res5'], outputting all available feature levels.

    Attributes:
        model (D2ConvNextV2):
            The underlying ConvNeXt model from the `detectron2.modeling.backbone` library.
        feature_channels (dict):
            A dictionary mapping feature level names (e.g., 'res2') to their corresponding output channel dimensions.
    """
    def __init__(
        self,
        model_name: str = "convnext_tiny",  # Default ConvNeXt model
        out_features: Optional[List[str]] = None,
    ):
        super().__init__()
        
        # Default output features if none specified
        if out_features is None:
            out_features = ['res2', 'res3', 'res4', 'res5']
        self._out_features = out_features
        
        # Get ConvNeXt config
        config = get_convnext_config(model_name)
        depths = config["depths"]
        dims = config["dims"]
        self.ckpt_dir = config['ckpt_dir']

        # Create ConvNeXt model
        self.model = D2ConvNextV2(
            depths=depths,
            embed_dims=dims,
            drop_path_rate=0.0,
            out_features=out_features,
        )

        self._out_feature_channels = {
            f"res{i+2}": dims[i] for i in range(len(dims))
        }





    def get_features(self, x):
        """
        Forward pass through the feature extraction layers of the ConvNeXt model.

        This method extracts feature maps from the input tensor `x` using the ConvNeXt backbone.

        Args:
            x (torch.Tensor): The input tensor, typically an image batch.

        Returns:
            dict[str, torch.Tensor]:
                A dictionary of feature maps, where keys are feature level names (e.g., 'res2')
                and values are the corresponding feature tensors.
        """
        return self.model.forward_features(x)
    
    @property
    def feature_channels(self):
        """
        Returns a dictionary mapping feature levels to their output channels.

        This property provides convenient access to the output channel dimensions of
        each feature level produced by the ConvNeXt backbone.

        Returns:
            dict[str, int]:
                A dictionary where keys are feature level names (e.g., 'res2') and
                values are the number of output channels for that feature level.
        """
        return self._out_feature_channels