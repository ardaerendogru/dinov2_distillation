from typing import List, Optional
import torch.nn as nn
from .base import BaseModel
from ..backbones import STDCNet
import os

ckpt_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'checkpoints') # Construct path to checkpoints dir
versions = {
    "stdc_1": {"layers": [2, 2, 2], 'ckpt_path':os.path.join(ckpt_dir, 'STDCNet1.pkl') },
    "stdc_2": {"layers": [4, 5, 3], 'ckpt_path':os.path.join(ckpt_dir, 'STDCNet2.pkl')},
}

def list_stdc_configs():
    return list(versions.keys())


def get_stdc_config(name):
    if name in versions:
        return versions[name]
    else:
        raise NotImplementedError(
            f"There is not a STDC config called {name}."
        )

class STDCWrapper(BaseModel):
    """
    Wrapper class for STDCNet models.

    This class allows you to easily use different STDCNet architectures as backbones
    for feature extraction. Currently, it supports STDCNet1 and STDCNet2.

    Args:
        model_name (str):
            Name of the specific STDCNet model to use.
            Must be one of the configurations listed in `list_stdc_configs()`.
            Defaults to "stdc_2".
        out_features (Optional[List[str]]):
            List of feature levels to output.
            Possible values are ['res2', 'res3', 'res4', 'res5'].
            If None, defaults to ['res2', 'res3', 'res4', 'res5'], outputting all available feature levels.

    Attributes:
        model (STDCNet):
            The underlying STDCNet model from the `distillation.models.backbones` library.
        out_features (List[str]):
            List of feature levels that will be output by the wrapper.
        feature_channels (dict):
            A dictionary mapping feature level names (e.g., 'res2') to their corresponding output channel dimensions.
    """
    def __init__(
        self,
        model_name: str = "stdc_2",
        out_features: Optional[List[str]] = None,
    ):
        super().__init__()
        
        # Get STDC config
        config = get_stdc_config(model_name)
        layers = config["layers"]
        self.ckpt_dir = config['ckpt_path']
        self.model = STDCNet(
            layers=layers,
        )
        
        # Initialize model
        self.model.init_params()

        self.out_features = out_features if out_features else ['res2', 'res3', 'res4', 'res5']
    
    def get_features(self, x):
        """
        Forward pass through the feature extraction layers of the STDCNet model.

        This method extracts feature maps from the input tensor `x` using the STDCNet backbone.
        It returns a dictionary of feature maps, where each key corresponds to a requested feature level
        (e.g., 'res2', 'res3', etc.).

        Args:
            x (torch.Tensor): The input tensor, typically an image batch.

        Returns:
            dict[str, torch.Tensor]:
                A dictionary of feature maps, where keys are feature level names (e.g., 'res2')
                and values are the corresponding feature tensors.
        """
        features_all =  self.model(x)
        features_out = {}
        for layer in self.out_features:
            features_out[layer] = features_all[layer]
        return features_out
    
    @property
    def feature_channels(self):
        """
        Returns a dictionary mapping feature levels to their output channels.

        This property provides convenient access to the output channel dimensions of
        each feature level produced by the STDCNet backbone.
        The channel counts are predefined based on the STDCNet architecture.

        Returns:
            dict[str, int]:
                A dictionary where keys are feature level names (e.g., 'res2') and
                values are the number of output channels for that feature level.
        """
        channels = {
            'res2': 64 * 1,
            'res3': 64 * 4,
            'res4': 64 * 8,
            'res5': 64 * 16
        }
        return channels