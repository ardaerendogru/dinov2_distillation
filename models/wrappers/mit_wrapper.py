from typing import List, Optional
import torch.nn as nn
from .base import BaseModel
from ..backbones import MultiscaleImageTransformer
import os
versions = {
    "0": [[2, 2, 2, 2], [32, 64, 160, 256]],
    "1": [[2, 2, 2, 2], [64, 128, 320, 512]],
    "2": [[3, 4, 6, 3], [64, 128, 320, 512]],
    "3": [[3, 4, 18, 3], [64, 128, 320, 512]],
    "4": [[3, 8, 27, 3], [64, 128, 320, 512]],
    "5": [[3, 6, 40, 3], [64, 128, 320, 512]],
}


def list_mit_configs():
    return [f"mit_b{i}" for i in range(0, 6)]


def get_mit_config(name):
    checkpoint_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'checkpoints') # Construct path to checkpoints dir
    
    version_chosen = name[-1]
    depths, embed_dims = [*versions.get(version_chosen)]
    ckpt_file = f'mit_b{version_chosen}.pkl' # Construct checkpoint filename
    ckpt_path = os.path.join(checkpoint_dir, ckpt_file) # Construct full checkpoint path
    return {
        "embed_dims": embed_dims,
        "depths": depths,
        "ckpt_path": ckpt_path # Return ckpt_path
    }

class MITWrapper(BaseModel):
    """
    Wrapper class for Multiscale Image Transformer (MIT) models.

    This class allows you to easily use different MIT architectures as backbones,
    and configure the output features.

    Args:
        model_name (str):
            Name of the specific MIT model to use.
            Must be one of the configurations listed in `list_mit_configs()`.
            Defaults to "mit_b0".
        out_features (Optional[List[str]]):
            List of feature levels to output.
            Each element should be in the format 'res[2-5]'.
            If None, defaults to ['res2', 'res3', 'res4', 'res5'], outputting all available feature levels.
        cfg (Any):
            Configuration object.

    Attributes:
        model (MultiscaleImageTransformer):
            The underlying MultiscaleImageTransformer model from the `detectron2.modeling.backbone` library.
        feature_channels (dict):
            A dictionary mapping feature level names (e.g., 'res2') to their corresponding output channel dimensions.
    """
    def __init__(
        self,
        model_name: str = "mit_b0",
        out_features: Optional[List[str]] = None,
    ):
        super().__init__()
        
        # Default output features if none specified
        if out_features is None:
            out_features = ['res2', 'res3', 'res4', 'res5']
        self._out_features = out_features
        
        # Get MIT config
        config = get_mit_config(model_name)
        embed_dims = config["embed_dims"]
        depths = config["depths"]
        self.ckpt_dir = config['ckpt_path']
        # Create MultiscaleImageTransformer model
        self.model = MultiscaleImageTransformer(
            embed_dims=embed_dims,
            depths=depths,
            num_heads=[1, 2, 4, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
        )

        self._out_feature_channels = {
            f"res{i+2}": embed_dims[i] for i in range(len(embed_dims))
        }

    def get_features(self, x):
        """
        Forward pass through the feature extraction layers of the MIT model.

        This method extracts feature maps from the input tensor `x` using the MIT backbone.

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
        each feature level produced by the MIT backbone.

        Returns:
            dict[str, int]:
                A dictionary where keys are feature level names (e.g., 'res2') and
                values are the number of output channels for that feature level.
        """
        return self._out_feature_channels