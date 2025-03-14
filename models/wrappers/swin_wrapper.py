from typing import List, Optional, Tuple
import torch.nn as nn
from .base import BaseModel
from ..backbones import D2SwinTransformer
from fvcore.common.config import CfgNode
import os


ckpt_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'checkpoints') # Construct path to checkpoints dir

versions = {
    "swin_tiny": {
        "embed_dims": 96,
        "depths": [2, 2, 6, 2],
        "pretr_image_size": 224,
        "num_heads": [3, 6, 12, 24],
        "window_size": 7,
        "ckpt_path": os.path.join(ckpt_dir, 'swin_tiny_patch4_window7_224.pkl')
    },
    "swin_small": {
        "embed_dims": 96,
        "depths": [2, 2, 18, 2],
        "pretr_image_size": 224,
        "num_heads": [3, 6, 12, 24],
        "window_size": 7,
        "ckpt_path": os.path.join(ckpt_dir, 'swin_small_patch4_window7_224.pkl')
    },
}


def list_swin_configs():
    return list(versions.keys())


def get_swin_config(name):
    if name in versions:
        return versions[name]
    else:
        raise NotImplementedError(
            f"There is not a Swin config called {name}."
        )

class D2SwinTransformerWrapper(BaseModel):
    """Wrapper class for D2SwinTransformer models, compatible with Detectron2's configuration,
    and configurable via LazyCall.

    Args:
        cfg (L(D2SwinTransformer) or dict): A LazyCall object or a dictionary containing the
            configuration for the D2SwinTransformer model. If a dictionary is provided, it
            should contain the necessary parameters for initializing the D2SwinTransformer.
        out_features (List[str]): Output from which stages. Default: ["res2", "res3", "res4", "res5"]

    Attributes:
        model (D2SwinTransformer): The underlying D2SwinTransformer model.
        feature_channels (dict): Dictionary mapping feature levels to their channel counts.
    """
    
    def __init__(
        self,
        model_name: str = "swin_tiny",
        out_features: List[str] = ["res2", "res3", "res4", "res5"],

    ):
        super().__init__()

        if out_features is None:
            out_features = ['res2', 'res3', 'res4', 'res5']
        self._out_features = out_features
        
        # Get Swin config
        config = get_swin_config(model_name)
        embed_dims = config["embed_dims"]
        depths = config["depths"]
        pretr_image_size = config["pretr_image_size"]
        num_heads = config["num_heads"]
        window_size = config["window_size"]
        self.ckpt_dir = config['ckpt_path']
        
        self.model = D2SwinTransformer(
            patch_size=4,
            pretr_image_size=pretr_image_size,
            embed_dims=embed_dims,
            depths=depths,
            num_heads = num_heads,
            window_size=window_size,
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.3,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            out_features = out_features,

        )


        self._out_feature_channels = self.model._out_feature_channels

    def get_features(self, x):
        return self.model(x)

    @property
    def feature_channels(self):
        return self._out_feature_channels