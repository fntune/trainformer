"""Metric learning task for embedding-based retrieval."""
from dataclasses import dataclass, field
from typing import Any

import torch.nn as nn
from torch import Tensor

from trainformer.models.components.backbones import TimmBackbone
from trainformer.models.components.losses import get_loss
from trainformer.types import DatasetInfo


@dataclass
class MetricLearning:
    """Metric learning task for image retrieval.

    Trains embeddings using margin-based losses like ArcFace or CosFace.
    Evaluates using KNN accuracy and retrieval metrics.

    Args:
        backbone: Name of timm backbone model
        embedding_dim: Dimension of output embeddings
        loss: Loss type ('arcface', 'cosface', 'subcenter')
        margin: Angular margin for loss
        scale: Scaling factor for loss
        pretrained: Use pretrained backbone weights
    """

    backbone: str = "efficientnet_b0"
    embedding_dim: int = 512
    loss: str = "arcface"
    margin: float = 0.5
    scale: float = 64.0
    pretrained: bool = True

    # Internal state (initialized after configure)
    model: nn.Module = field(init=False)
    _loss: nn.Module | None = field(default=None, init=False)
    _num_classes: int | None = field(default=None, init=False)

    def __post_init__(self):
        # Create backbone with projection to embedding space
        self.model = TimmBackbone(
            self.backbone,
            pretrained=self.pretrained,
            embed_dim=self.embedding_dim,
            normalize=True,
        )

    def configure(self, info: DatasetInfo) -> None:
        """Configure loss function with dataset info."""
        if info.num_classes is None:
            raise ValueError("MetricLearning requires dataset with class labels")

        self._num_classes = info.num_classes

        # Initialize loss with correct num_classes
        self._loss = get_loss(
            self.loss,
            embedding_dim=self.embedding_dim,
            num_classes=self._num_classes,
            margin=self.margin,
            scale=self.scale,
        )

    def parameters(self):
        """Return all trainable parameters."""
        params = list(self.model.parameters())
        if self._loss is not None:
            params.extend(self._loss.parameters())
        return params

    def info(self) -> dict[str, Any]:
        """Return task configuration info."""
        return {
            "backbone": self.backbone,
            "embedding_dim": self.embedding_dim,
            "loss_type": self.loss,
            "margin": self.margin,
            "scale": self.scale,
            "num_classes": self._num_classes,
            "num_params": sum(p.numel() for p in self.model.parameters()),
            "trainable_params": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }

    def train_step(self, batch: tuple[Tensor, Tensor]) -> Tensor:
        """Compute loss for a training batch."""
        if self._loss is None:
            raise RuntimeError("Loss not initialized. Trainer must call configure() first.")

        x, y = batch
        embeddings = self.model(x)
        return self._loss(embeddings, y)

    def eval_step(self, batch: tuple[Tensor, Tensor]) -> dict[str, Tensor]:
        """Extract embeddings for evaluation."""
        x, y = batch
        embeddings = self.model(x)
        return {"embeddings": embeddings, "labels": y}

    def load_data(self, path: str):
        """Load image folder dataset."""
        from torchvision import transforms as T
        from torchvision.datasets import ImageFolder

        transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return ImageFolder(path, transform=transform)

    def evaluate_retrieval(self, embeddings: Tensor, labels: Tensor) -> dict[str, float]:
        """Compute retrieval metrics on embeddings."""
        from trainformer.eval.retrieval import knn_accuracy

        metrics = {}

        # KNN accuracy at various k
        for k in [1, 5, 10]:
            acc = knn_accuracy(embeddings, labels, k=k)
            metrics[f"knn@{k}"] = acc

        return metrics
