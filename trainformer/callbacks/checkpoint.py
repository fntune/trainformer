"""Model checkpointing callback."""
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from trainformer.callbacks.base import CallbackBase

if TYPE_CHECKING:
    from trainformer.trainer import Trainer

logger = logging.getLogger(__name__)


class ModelCheckpoint(CallbackBase):
    """Save model checkpoints based on metric improvement.

    Args:
        dirpath: Directory to save checkpoints
        filename: Checkpoint filename template (supports {epoch}, {metric})
        monitor: Metric to monitor for best model
        mode: 'min' or 'max' for metric improvement
        save_top_k: Number of best models to keep (-1 for all)
        save_last: Whether to always save the last checkpoint
    """

    def __init__(
        self,
        dirpath: str | Path = "checkpoints",
        filename: str = "epoch_{epoch:02d}",
        monitor: str = "val/loss",
        mode: str = "min",
        save_top_k: int = 1,
        save_last: bool = True,
    ):
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.save_last = save_last

        self.best_score: float | None = None
        self.best_path: Path | None = None
        self._saved: list[tuple[float, Path]] = []

    def on_fit_start(self, trainer: "Trainer") -> None:
        self.dirpath.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, trainer: "Trainer", epoch: int, metrics: dict[str, float]) -> None:
        current = metrics.get(self.monitor)

        if current is None:
            logger.warning(f"Metric '{self.monitor}' not found in metrics")
            if self.save_last:
                self._save(trainer, epoch, metrics)
            return

        is_better = self._is_better(current)

        if is_better:
            self.best_score = current
            path = self._save(trainer, epoch, metrics)
            self.best_path = path
            logger.info(f"New best {self.monitor}: {current:.4f}")

            # Manage saved checkpoints
            self._saved.append((current, path))
            self._cleanup()
        elif self.save_last:
            self._save(trainer, epoch, metrics, is_best=False)

    def _is_better(self, current: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "min":
            return current < self.best_score
        return current > self.best_score

    def _save(
        self,
        trainer: "Trainer",
        epoch: int,
        metrics: dict[str, float],
        is_best: bool = True,
    ) -> Path:
        metric_val = metrics.get(self.monitor, 0)
        filename = self.filename.format(epoch=epoch, metric=metric_val)
        if not filename.endswith(".pt"):
            filename += ".pt"

        path = self.dirpath / filename
        state = trainer.state_dict()
        state["epoch"] = epoch
        state["metrics"] = metrics
        torch.save(state, path)
        logger.info(f"Saved checkpoint: {path}")

        if is_best:
            best_path = self.dirpath / "best.pt"
            torch.save(state, best_path)

        if self.save_last:
            last_path = self.dirpath / "last.pt"
            torch.save(state, last_path)

        return path

    def _cleanup(self) -> None:
        if self.save_top_k <= 0:
            return

        # Sort by score
        reverse = self.mode == "max"
        self._saved.sort(key=lambda x: x[0], reverse=reverse)

        # Remove excess checkpoints
        while len(self._saved) > self.save_top_k:
            _, path = self._saved.pop()
            if path.exists() and path != self.best_path:
                path.unlink()
                logger.debug(f"Removed checkpoint: {path}")
