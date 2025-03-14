import torch
import math
import torch.nn as nn

class DINOv2ViT(nn.Module):
    def __init__(self, model_name='dinov2_vitg14'):
        super().__init__()
        """
        DINOv2 ViT model for distillation.

        Args:
            model_name (str): The name of the DINOv2 ViT model to use.

        Returns:
            dict: A dictionary containing:
                patch_embeddings: torch.Tensor (B, N, D): The patch embeddings excluding the CLS token.
                cls_embedding: torch.Tensor (B, D): The CLS token embedding.
        """
        # Load model from torch hub
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        # Freeze all parameters
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.H = None  # Initialize H
        self.W = None  # Initialize W
    def forward(self, x):



        # Get features from the model's last layer
        patch_embeddings, cls_token = self.model.get_intermediate_layers(x, n=1, return_class_token=True)[0]  # [B, N+1, D]
        # Calculate H and W only once
        if self.H is None:
            self.H = x.shape[2] // 14
            self.W = x.shape[3] // 14
        
        self.B, _, self.D = patch_embeddings.shape
            
        feature_map = patch_embeddings.reshape(self.B, self.H, self.W, self.D).permute(0, 3, 1, 2)  # [B, D, P, P]
        
        return {
            # 'patch_embeddings': patch_embeddings,  # Per-patch embeddings excluding CLS
            # 'embedding': cls_token,        # CLS token embedding
            'feature_map': feature_map,
        }

