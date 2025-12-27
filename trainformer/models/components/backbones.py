"""Vision backbones using timm."""
import logging

import timm
import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)


class TimmBackbone(nn.Module):
    """Wrapper around timm models for feature extraction.

    Args:
        model_name: Name of timm model
        pretrained: Load pretrained weights
        embed_dim: Output embedding dimension (None to use model's native dim)
        dropout: Dropout rate before projection
        normalize: L2 normalize output embeddings
    """

    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        embed_dim: int | None = None,
        dropout: float = 0.0,
        normalize: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.normalize = normalize

        # Create backbone without classification head
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )

        # Get actual output dimension (num_features can be wrong for some models)
        with torch.no_grad():
            self.backbone.eval()
            dummy = torch.zeros(1, 3, 224, 224)
            out = self.backbone(dummy)
            actual_features = out.shape[-1]
            self.backbone.train()

        self.num_features = actual_features
        logger.info(f"Backbone {model_name}: {self.num_features} features")

        # Optional projection layer
        if embed_dim is not None and embed_dim != self.num_features:
            self.proj = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(self.num_features, embed_dim),
                nn.BatchNorm1d(embed_dim),
            )
            self.num_features = embed_dim
        else:
            self.proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        if self.proj is not None:
            x = self.proj(x)
        if self.normalize:
            x = F.normalize(x, dim=-1)
        return x
