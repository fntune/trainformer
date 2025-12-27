"""Image classification task."""
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from trainformer.types import DatasetInfo


@dataclass
class ImageClassification:
    """Standard image classification task.

    Args:
        backbone: Name of timm backbone model
        num_classes: Number of output classes (None to infer from data)
        pretrained: Use pretrained backbone weights
        dropout: Dropout rate before classifier
    """

    backbone: str = "efficientnet_b0"
    num_classes: int | None = None
    pretrained: bool = True
    dropout: float = 0.0

    model: nn.Module = field(init=False)
    _num_classes: int | None = field(default=None, init=False)

    def __post_init__(self):
        self._num_classes = self.num_classes
        self._build_model()

    def _build_model(self):
        """Build the classification model."""
        self.model = ClassificationModel(
            backbone=self.backbone,
            num_classes=self._num_classes or 1000,  # Placeholder until configure()
            pretrained=self.pretrained,
            dropout=self.dropout,
        )

    def configure(self, info: DatasetInfo) -> None:
        """Configure with dataset info."""
        if self._num_classes is None:
            if info.num_classes is None:
                raise ValueError("num_classes must be specified or inferable from data")
            self._num_classes = info.num_classes
            # Rebuild model with correct num_classes
            self._build_model()

    def parameters(self):
        return self.model.parameters()

    def train_step(self, batch: tuple[Tensor, Tensor]) -> Tensor:
        x, y = batch
        logits = self.model(x)
        return F.cross_entropy(logits, y)

    def eval_step(self, batch: tuple[Tensor, Tensor]) -> dict[str, Tensor]:
        x, y = batch
        logits = self.model(x)
        preds = logits.argmax(dim=-1)
        return {"predictions": preds, "labels": y, "logits": logits}

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

    def evaluate_classification(
        self, predictions: Tensor, labels: Tensor, logits: Tensor
    ) -> dict[str, float]:
        """Compute classification metrics."""
        correct = (predictions == labels).float()
        accuracy = correct.mean().item()

        # Top-5 accuracy if we have enough classes
        if logits.size(1) >= 5:
            _, top5_preds = logits.topk(5, dim=1)
            top5_correct = (top5_preds == labels.unsqueeze(1)).any(dim=1).float()
            top5_acc = top5_correct.mean().item()
        else:
            top5_acc = accuracy

        return {"accuracy": accuracy, "top5_accuracy": top5_acc}


class ClassificationModel(nn.Module):
    """Backbone + classification head."""

    def __init__(self, backbone: str, num_classes: int, pretrained: bool, dropout: float):
        super().__init__()
        import timm

        # Use timm directly for classification
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )

        # Get actual output dimension by forward pass (num_features can be wrong for some models)
        with torch.no_grad():
            self.backbone.eval()
            dummy = torch.zeros(1, 3, 224, 224)
            out = self.backbone(dummy)
            self.num_features = out.shape[-1]
            self.backbone.train()

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(self.num_features, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        features = self.backbone(x)
        features = self.dropout(features)
        return self.classifier(features)
