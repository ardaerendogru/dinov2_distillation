from typing import List, Optional
import torch.nn as nn
from .base import BaseModel
from ..backbones import D2Presnet
import os 

ckpt_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'checkpoints') # Construct path to checkpoints dir

versions = {
    "presnet_18": {"depth": 18, "variant": "d", 'ckpt_path': os.path.join(ckpt_dir, 'presnet18.pkl')},
    "presnet_34": {"depth": 34, "variant": "d", 'ckpt_path': os.path.join(ckpt_dir, 'presnet34.pkl')},
    "presnet_50": {"depth": 50, "variant": "d", 'ckpt_path': os.path.join(ckpt_dir, 'presnet50.pkl')},
    "presnet_101": {"depth": 101, "variant": "d", 'ckpt_path': os.path.join(ckpt_dir, 'presnet101.pkl')},
}


def list_presnet_configs():
    return list(versions.keys())


def get_presnet_config(name):
    if name in versions:
        return versions[name]
    else:
        raise NotImplementedError(
            f"There is not a PResNet config called {name}."
        )

class PResNetWrapper(BaseModel):
    """Wrapper class for PResNet models.
    
    Args:
        model_id (str): ID of the PResNet model to use.
        out_features (Optional[List[str]]): List of feature levels to output.
            Possible values: ['res2', 'res3', 'res4', 'res5']. Default: all levels
    
    Attributes:
        model (PResNet): The underlying PResNet model.
        feature_channels (dict): Dictionary mapping feature levels to their channel counts.
    """
    def __init__(
        self,
        model_name: str = "presnet_50",
        out_features: Optional[List[str]] = None,
    ):
        super().__init__()
        
        # Default output features if none specified
        if out_features is None:
            out_features = ['res2', 'res3', 'res4', 'res5']
        self._out_features = out_features
        
        # Get PResNet config
        config = get_presnet_config(model_name)
        depth = config["depth"]
        variant = config["variant"]
        self.ckpt_dir = config['ckpt_path']
        # Create PResNet model
        self.model = D2Presnet(
            depth=depth,
            variant=variant,
            num_stages=4,
            act='relu',
            freeze_at=-1,
            freeze_norm = False,
            pretrained=False,
        )

        # Filter feature channels based on out_features
        self._out_feature_channels = self.model._out_feature_channels

    def get_features(self, x):
        return self.model(x)

    
    @property
    def feature_channels(self):
        return self._out_feature_channels