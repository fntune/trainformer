"""Monitoring callbacks for training diagnostics."""
import logging
from typing import TYPE_CHECKING, Any

from trainformer.callbacks.base import CallbackBase

if TYPE_CHECKING:
    from trainformer.trainer import Trainer

logger = logging.getLogger(__name__)


class LRMonitor(CallbackBase):
    """Log learning rate to logger.

    Logs the current learning rate at the end of each batch. Useful for
    debugging scheduler behavior and ensuring proper warmup/decay.

    Args:
        log_every: Log every N batches (default: 100)
    """

    def __init__(self, log_every: int = 100):
        self.log_every = log_every

    def on_train_batch_end(
        self, trainer: "Trainer", batch: Any, batch_idx: int, loss: float
    ) -> None:
        if batch_idx % self.log_every != 0:
            return

        if trainer._scheduler is not None:
            lr = trainer._scheduler.get_last_lr()[0]
        else:
            lr = trainer._optimizer.param_groups[0]["lr"]

        trainer._logger.log({"train/lr": lr}, trainer._step)


class GradientMonitor(CallbackBase):
    """Track gradient statistics and detect explosions.

    Computes and logs gradient norms, warns when gradients exceed threshold.
    Useful for debugging training instability and gradient clipping.

    Args:
        log_every: Log every N batches (default: 100)
        explosion_threshold: Warn if grad norm exceeds this (default: 100.0)
    """

    def __init__(self, log_every: int = 100, explosion_threshold: float = 100.0):
        self.log_every = log_every
        self.explosion_threshold = explosion_threshold

    def on_train_batch_end(
        self, trainer: "Trainer", batch: Any, batch_idx: int, loss: float
    ) -> None:
        if batch_idx % self.log_every != 0:
            return

        # Compute gradient norm
        total_norm = 0.0
        params = (
            trainer.task.parameters()
            if hasattr(trainer.task, "parameters")
            else trainer.task.model.parameters()
        )

        for p in params:
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5

        trainer._logger.log({"train/grad_norm": total_norm}, trainer._step)

        if total_norm > self.explosion_threshold:
            logger.warning(
                f"GradientMonitor: gradient explosion detected "
                f"(norm={total_norm:.2f} > threshold={self.explosion_threshold})"
            )
