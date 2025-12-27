"""Model components."""
from trainformer.models.components.backbones import TimmBackbone
from trainformer.models.components.heads import (
    BYOLHead,
    ClassificationHead,
    DINOHead,
    EmbeddingHead,
    ProjectionHead,
    SimCLRHead,
)
from trainformer.models.components.losses import (
    ArcFaceLoss,
    CosFaceLoss,
    SubcenterArcFace,
    get_loss,
)
from trainformer.models.components.poolers import (
    AdaptiveAvgPool,
    AdaptiveConcatPool,
    AdaptiveMaxPool,
    GeM,
)

__all__ = [
    "TimmBackbone",
    "ArcFaceLoss",
    "CosFaceLoss",
    "SubcenterArcFace",
    "get_loss",
    "GeM",
    "AdaptiveAvgPool",
    "AdaptiveMaxPool",
    "AdaptiveConcatPool",
    "ProjectionHead",
    "ClassificationHead",
    "EmbeddingHead",
    "DINOHead",
    "SimCLRHead",
    "BYOLHead",
]
