from typing import List, Optional
import torch.nn as nn
from .base import BaseModel
from ..backbones import D2timm

versions = {
    "efficientnet_b0": {"model_name": "efficientnet_b0"},
    "efficientnet_b1": {"model_name": "efficientnet_b1"},
    "efficientnet_b2": {"model_name": "efficientnet_b2"},
    "efficientnet_b3": {"model_name": "efficientnet_b3"},
    "efficientnet_b4": {"model_name": "efficientnet_b4"},
    "edgenext_xx_small": {"model_name": "edgenext_xx_small"},
    "edgenext_x_small": {"model_name": "edgenext_x_small"},
    "edgenext_small": {"model_name": "edgenext_small"},
    "edgenext_base": {"model_name": "edgenext_base"},
    "mobilenetv3_small_050": {"model_name": "mobilenetv3_small_050"},
    "mobilenetv3_small_075": {"model_name": "mobilenetv3_small_075"},
    "mobilenetv3_small_100": {"model_name": "mobilenetv3_small_100"},
    "mobilenetv3_large_075": {"model_name": "mobilenetv3_large_075"},
    "mobilenetv3_large_100": {"model_name": "mobilenetv3_large_100"},
}


def list_timm_configs():
    return list(versions.keys())


def get_timm_config(name):
    if name in versions:
        return versions[name]
    else:
        raise NotImplementedError(
            f"There is not a Timm config called {name}."
        )

class TimmWrapper(BaseModel):
    """Wrapper class for TimmBackbone models.
    
    Args:
        model_id (str): ID of the Timm model to use.
        pretrained (bool): Whether to use pretrained weights. Default: True
        out_features (Optional[List[str]]): List of feature levels to output.
            Possible values: ['res2', 'res3', 'res4', 'res5']. Default: all levels
    
    Attributes:
        model (TimmBackbone): The underlying TimmBackbone model.
        feature_channels (dict): Dictionary mapping feature levels to their channel counts.
    """
    def __init__(
        self,
        model_name: str = 'efficientnet_b0',
        out_features: Optional[List[str]] = None,
    ):
        super().__init__()
        
        # Default output features if none specified
        if out_features is None:
            out_features = ['res2', 'res3', 'res4', 'res5']
        self._out_features = out_features
        
        # Get Timm config
        config = get_timm_config(model_name)
        model_name = config["model_name"]
        
        # Create TimmBackbone model
        self.model = D2timm(
            name=model_name,
            pretrained=True,
            out_features=out_features
        )

        # Filter feature channels based on out_features
        self._out_feature_channels = self.model._out_feature_channels

    def get_features(self, x):
        return self.model(x)
    
    @property
    def feature_channels(self):
        return self._out_feature_channels