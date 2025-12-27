"""Image dataset utilities."""
import logging
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.datasets import ImageFolder

logger = logging.getLogger(__name__)


class ImageDataset(Dataset):
    """Generic image dataset from file paths and optional labels."""

    def __init__(
        self,
        image_paths: list[str | Path],
        labels: list[int] | None = None,
        transform: Callable | None = None,
    ):
        self.image_paths = [Path(p) for p in image_paths]
        self.labels = labels
        self.transform = transform

        if labels is not None and len(labels) != len(image_paths):
            raise ValueError("Length of labels must match image_paths")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int] | torch.Tensor:
        img = Image.open(self.image_paths[idx]).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.labels is not None:
            return img, self.labels[idx]
        return img

    @property
    def classes(self) -> list[str] | None:
        """Return unique class names if labels provided."""
        if self.labels is None:
            return None
        return sorted(set(str(label) for label in self.labels))

    @property
    def num_classes(self) -> int | None:
        """Return number of unique classes."""
        if self.labels is None:
            return None
        return len(set(self.labels))


class ImageFolderDataset(ImageFolder):
    """ImageFolder with additional metadata properties."""

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def class_counts(self) -> list[int]:
        """Count samples per class."""
        counts = [0] * len(self.classes)
        for _, label in self.samples:
            counts[label] += 1
        return counts


class TensorDictDataset(Dataset):
    """Dataset wrapping tensor dictionaries (for inference)."""

    def __init__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor | None = None,
        transform: Callable | None = None,
    ):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int] | torch.Tensor:
        img = self.images[idx]

        if self.transform is not None:
            img = self.transform(img)

        if self.labels is not None:
            return img, self.labels[idx].item()
        return img


def get_default_transforms(
    train: bool = True,
    image_size: int = 224,
    mean: tuple[float, ...] = (0.485, 0.456, 0.406),
    std: tuple[float, ...] = (0.229, 0.224, 0.225),
) -> T.Compose:
    """Get standard ImageNet transforms."""
    if train:
        return T.Compose([
            T.RandomResizedCrop(image_size),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    return T.Compose([
        T.Resize(int(image_size * 1.14)),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
