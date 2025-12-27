"""Shared test fixtures."""
import pytest
import torch
from torch.utils.data import Dataset


class DummyImageDataset(Dataset):
    """Dummy dataset for testing."""

    def __init__(self, num_samples: int = 100, num_classes: int = 10, img_size: int = 32):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.img_size = img_size
        self.classes = [f"class_{i}" for i in range(num_classes)]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = torch.randn(3, self.img_size, self.img_size)
        y = idx % self.num_classes
        return x, y


@pytest.fixture
def dummy_dataset():
    """Small dummy dataset for tests."""
    return DummyImageDataset(num_samples=100, num_classes=10, img_size=32)


@pytest.fixture
def tiny_dataset():
    """Very small dataset for fast tests."""
    return DummyImageDataset(num_samples=16, num_classes=4, img_size=32)
