"""Early stopping callback."""
import logging
from typing import TYPE_CHECKING

from trainformer.callbacks.base import CallbackBase

if TYPE_CHECKING:
    from trainformer.trainer import Trainer

logger = logging.getLogger(__name__)


class EarlyStopping(CallbackBase):
    """Stop training when a metric stops improving.

    Args:
        monitor: Metric to monitor
        mode: 'min' or 'max' for metric improvement
        patience: Epochs to wait before stopping
        min_delta: Minimum change to qualify as improvement
    """

    def __init__(
        self,
        monitor: str = "val/loss",
        mode: str = "min",
        patience: int = 10,
        min_delta: float = 0.0,
    ):
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta

        self.best_score: float | None = None
        self.wait_count = 0
        self.stopped_epoch: int | None = None

    def on_epoch_end(self, trainer: "Trainer", epoch: int, metrics: dict[str, float]) -> None:
        current = metrics.get(self.monitor)

        if current is None:
            logger.warning(f"Metric '{self.monitor}' not found in metrics")
            return

        if self._is_improvement(current):
            self.best_score = current
            self.wait_count = 0
        else:
            self.wait_count += 1
            logger.info(
                f"EarlyStopping: {self.wait_count}/{self.patience} "
                f"(best {self.monitor}: {self.best_score:.4f})"
            )

            if self.wait_count >= self.patience:
                self.stopped_epoch = epoch
                trainer.should_stop = True
                logger.info(f"Early stopping triggered at epoch {epoch}")

    def _is_improvement(self, current: float) -> bool:
        if self.best_score is None:
            return True

        if self.mode == "min":
            return current < self.best_score - self.min_delta
        return current > self.best_score + self.min_delta
