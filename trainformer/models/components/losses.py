"""Metric learning loss functions."""
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ArcFaceLoss(nn.Module):
    """ArcFace loss with learnable class centers.

    Additive Angular Margin Loss for face recognition and metric learning.

    Args:
        embedding_dim: Dimension of input embeddings
        num_classes: Number of classes
        margin: Angular margin in radians
        scale: Scaling factor for logits
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        margin: float = 0.5,
        scale: float = 64.0,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale

        # Learnable class centers
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        # Precompute margin values
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

        self.ce = nn.CrossEntropyLoss()

    def forward(self, embeddings: Tensor, labels: Tensor) -> Tensor:
        # Cosine similarity
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))

        # Compute sin from cos
        sine = torch.sqrt(1.0 - cosine.pow(2).clamp(0, 1))

        # cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m

        # Handle edge case where cos(theta) < cos(pi - m)
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # Replace target class logits with margin-adjusted values
        one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        logits = one_hot * phi + (1.0 - one_hot) * cosine

        return self.ce(self.scale * logits, labels)


class CosFaceLoss(nn.Module):
    """CosFace (Large Margin Cosine Loss).

    Simpler alternative to ArcFace with additive cosine margin.

    Args:
        embedding_dim: Dimension of input embeddings
        num_classes: Number of classes
        margin: Cosine margin
        scale: Scaling factor for logits
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        margin: float = 0.35,
        scale: float = 64.0,
    ):
        super().__init__()
        self.margin = margin
        self.scale = scale

        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        self.ce = nn.CrossEntropyLoss()

    def forward(self, embeddings: Tensor, labels: Tensor) -> Tensor:
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))

        # Subtract margin from target class
        one_hot = F.one_hot(labels, num_classes=cosine.size(1)).float()
        logits = cosine - one_hot * self.margin

        return self.ce(self.scale * logits, labels)


class SubcenterArcFace(nn.Module):
    """Subcenter ArcFace for handling intra-class variance.

    Uses K subcenters per class to handle noisy labels and intra-class variations.

    Args:
        embedding_dim: Dimension of input embeddings
        num_classes: Number of classes
        num_subcenters: Number of subcenters per class
        margin: Angular margin
        scale: Scaling factor
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        num_subcenters: int = 3,
        margin: float = 0.5,
        scale: float = 64.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_subcenters = num_subcenters
        self.margin = margin
        self.scale = scale

        # K subcenters per class
        self.weight = nn.Parameter(torch.empty(num_classes * num_subcenters, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

        self.ce = nn.CrossEntropyLoss()

    def forward(self, embeddings: Tensor, labels: Tensor) -> Tensor:
        # Compute cosine to all subcenters
        cosine_all = F.linear(F.normalize(embeddings), F.normalize(self.weight))

        # Reshape and take max over subcenters
        cosine_all = cosine_all.view(-1, self.num_classes, self.num_subcenters)
        cosine, _ = cosine_all.max(dim=2)

        # Apply ArcFace margin
        sine = torch.sqrt(1.0 - cosine.pow(2).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        logits = one_hot * phi + (1.0 - one_hot) * cosine

        return self.ce(self.scale * logits, labels)


def get_loss(
    loss_type: str,
    embedding_dim: int,
    num_classes: int,
    margin: float = 0.5,
    scale: float = 64.0,
    **kwargs,
) -> nn.Module:
    """Factory function for metric learning losses."""
    losses = {
        "arcface": ArcFaceLoss,
        "cosface": CosFaceLoss,
        "subcenter": SubcenterArcFace,
    }

    if loss_type not in losses:
        raise ValueError(f"Unknown loss: {loss_type}. Available: {list(losses.keys())}")

    return losses[loss_type](
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        margin=margin,
        scale=scale,
        **kwargs,
    )
