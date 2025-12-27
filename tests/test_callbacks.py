"""Tests for callbacks."""
import tempfile
from pathlib import Path

import torch

from trainformer.callbacks import EarlyStopping, ModelCheckpoint


class MockTrainer:
    """Mock trainer for callback tests."""

    def __init__(self):
        self.should_stop = False

    def state_dict(self):
        return {"model": {"weight": torch.randn(10)}}


def test_early_stopping_patience():
    """Test early stopping with patience."""
    callback = EarlyStopping(monitor="val/loss", patience=3)
    trainer = MockTrainer()

    # Improvement
    callback.on_epoch_end(trainer, 0, {"val/loss": 1.0})
    assert not trainer.should_stop

    # No improvement
    callback.on_epoch_end(trainer, 1, {"val/loss": 1.1})
    assert callback.wait_count == 1

    callback.on_epoch_end(trainer, 2, {"val/loss": 1.2})
    assert callback.wait_count == 2

    callback.on_epoch_end(trainer, 3, {"val/loss": 1.3})
    assert callback.wait_count == 3
    assert trainer.should_stop


def test_early_stopping_improvement():
    """Test early stopping resets on improvement."""
    callback = EarlyStopping(monitor="val/loss", patience=3)
    trainer = MockTrainer()

    callback.on_epoch_end(trainer, 0, {"val/loss": 1.0})
    callback.on_epoch_end(trainer, 1, {"val/loss": 1.1})
    assert callback.wait_count == 1

    # Improvement
    callback.on_epoch_end(trainer, 2, {"val/loss": 0.9})
    assert callback.wait_count == 0
    assert not trainer.should_stop


def test_checkpoint_save():
    """Test checkpoint saving."""
    with tempfile.TemporaryDirectory() as tmpdir:
        callback = ModelCheckpoint(
            dirpath=tmpdir,
            filename="epoch_{epoch:02d}",
            monitor="val/loss",
            save_last=True,
        )
        trainer = MockTrainer()

        callback.on_fit_start(trainer)
        callback.on_epoch_end(trainer, 0, {"val/loss": 1.0})

        assert Path(tmpdir, "epoch_00.pt").exists()
        assert Path(tmpdir, "best.pt").exists()
        assert Path(tmpdir, "last.pt").exists()
