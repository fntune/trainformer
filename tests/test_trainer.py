"""Tests for Trainer class."""
import torch
from torch.utils.data import Dataset

from trainformer import Trainer
from trainformer.tasks import ImageClassification, MetricLearning


class TinyDataset(Dataset):
    """Tiny dataset for fast integration tests."""

    def __init__(self, num_samples: int = 32, num_classes: int = 4, img_size: int = 224):
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


def test_trainer_classification_one_epoch():
    """Test training ImageClassification for one epoch."""
    dataset = TinyDataset(num_samples=32, num_classes=4)

    task = ImageClassification(
        backbone="mobilenetv3_small_050",  # Tiny model
        num_classes=4,
        pretrained=False,
    )

    trainer = Trainer(
        task=task,
        data=dataset,
        epochs=1,
        batch_size=8,
        lr=1e-3,
        amp=False,  # Disable AMP for CPU tests
        num_workers=0,
    )

    trainer.fit()

    assert trainer.ctx.epoch == 0
    assert trainer.ctx.global_step > 0
    assert "train/loss" in trainer.best_metrics


def test_trainer_metric_learning_one_epoch():
    """Test training MetricLearning for one epoch."""
    dataset = TinyDataset(num_samples=32, num_classes=4)

    task = MetricLearning(
        backbone="mobilenetv3_small_050",
        embedding_dim=64,
        loss="arcface",
        pretrained=False,
    )

    trainer = Trainer(
        task=task,
        data=dataset,
        epochs=1,
        batch_size=8,
        lr=1e-3,
        amp=False,
        num_workers=0,
    )

    trainer.fit()

    assert trainer.ctx.global_step > 0
    assert "train/loss" in trainer.best_metrics


def test_trainer_predict():
    """Test prediction API."""
    dataset = TinyDataset(num_samples=16, num_classes=4)

    task = ImageClassification(
        backbone="mobilenetv3_small_050",
        num_classes=4,
        pretrained=False,
    )

    trainer = Trainer(
        task=task,
        data=dataset,
        epochs=1,
        batch_size=8,
        amp=False,
        num_workers=0,
    )

    trainer.fit()
    outputs = trainer.predict(dataset)

    assert "predictions" in outputs
    assert "labels" in outputs
    assert outputs["predictions"].shape[0] == 16


def test_trainer_state_dict():
    """Test checkpoint save/load."""
    dataset = TinyDataset(num_samples=16, num_classes=4)

    task = ImageClassification(
        backbone="mobilenetv3_small_050",
        num_classes=4,
        pretrained=False,
    )

    trainer = Trainer(
        task=task,
        data=dataset,
        epochs=1,
        batch_size=8,
        amp=False,
        num_workers=0,
    )

    trainer.fit()
    state = trainer.state_dict()

    assert "task" in state
    assert "optimizer" in state
    assert "epoch" in state
    assert "global_step" in state
