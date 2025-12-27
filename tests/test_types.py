"""Tests for core types."""
import torch

from trainformer.types import DatasetInfo


def test_dataset_info_from_dataset(dummy_dataset):
    """Test DatasetInfo extraction from dataset."""
    info = DatasetInfo.from_dataset(dummy_dataset)

    assert info.num_samples == 100
    assert info.num_classes == 10
    assert info.class_names == [f"class_{i}" for i in range(10)]
    assert info.input_shape == (3, 32, 32)


def test_dataset_info_manual():
    """Test manual DatasetInfo creation."""
    info = DatasetInfo(
        num_samples=1000,
        num_classes=100,
        class_names=None,
        input_shape=(3, 224, 224),
    )

    assert info.num_samples == 1000
    assert info.num_classes == 100
