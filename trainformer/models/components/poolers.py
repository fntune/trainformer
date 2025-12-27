"""Pooling layers for vision models."""
import torch
import torch.nn.functional as F
from torch import nn


class GeM(nn.Module):
    """Generalized Mean Pooling.

    Reference: Fine-tuning CNN Image Retrieval with No Human Annotation
    https://arxiv.org/abs/1711.02512
    """

    def __init__(self, p: float = 3.0, trainable: bool = False, eps: float = 1e-6):
        super().__init__()
        if trainable:
            self.p = nn.Parameter(torch.ones(1) * p)
        else:
            self.register_buffer("p", torch.tensor(p))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply GeM pooling.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Pooled tensor of shape (B, C)
        """
        return F.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p),
            (x.size(-2), x.size(-1))
        ).pow(1.0 / self.p).squeeze(-1).squeeze(-1)

    def __repr__(self) -> str:
        p_val = self.p.item() if isinstance(self.p, torch.Tensor) else self.p
        return f"{self.__class__.__name__}(p={p_val:.4f}, eps={self.eps})"


class AdaptiveAvgPool(nn.Module):
    """Adaptive average pooling with flatten."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(x, 1).flatten(1)


class AdaptiveMaxPool(nn.Module):
    """Adaptive max pooling with flatten."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_max_pool2d(x, 1).flatten(1)


class AdaptiveConcatPool(nn.Module):
    """Concatenate adaptive avg and max pooling."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = F.adaptive_avg_pool2d(x, 1).flatten(1)
        max_ = F.adaptive_max_pool2d(x, 1).flatten(1)
        return torch.cat([avg, max_], dim=1)
