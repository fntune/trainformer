"""Learning rate finder callback."""
import logging
import math
from typing import TYPE_CHECKING, Any

import torch

from trainformer.callbacks.base import CallbackBase

if TYPE_CHECKING:
    from trainformer.trainer import Trainer

logger = logging.getLogger(__name__)


class LRFinder(CallbackBase):
    """Learning rate range test to find optimal learning rate.

    Runs before training to test a range of learning rates and find the
    optimal value. Uses exponential LR schedule and tracks loss to find
    the steepest descent point.

    Args:
        min_lr: Minimum learning rate to test (default: 1e-7)
        max_lr: Maximum learning rate to test (default: 10.0)
        num_steps: Number of steps to test (default: 100)
        smoothing: Exponential smoothing factor for loss (default: 0.05)
        divergence_factor: Stop if loss exceeds best * factor (default: 4.0)
        auto_update: Automatically update trainer.lr if at default (default: True)
    """

    def __init__(
        self,
        min_lr: float = 1e-7,
        max_lr: float = 10.0,
        num_steps: int = 100,
        smoothing: float = 0.05,
        divergence_factor: float = 4.0,
        auto_update: bool = True,
    ):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.num_steps = num_steps
        self.smoothing = smoothing
        self.divergence_factor = divergence_factor
        self.auto_update = auto_update

        self._losses: list[float] = []
        self._lrs: list[float] = []
        self._best_loss: float = float("inf")
        self.suggested_lr: float | None = None

    def on_fit_start(self, trainer: "Trainer") -> None:
        """Run LR finder before actual training."""
        logger.info("LRFinder: running learning rate range test")

        # Save state
        model_state = {k: v.clone() for k, v in trainer.task.model.state_dict().items()}
        optimizer_state = trainer._optimizer.state_dict()

        # Setup exponential LR schedule
        gamma = (self.max_lr / self.min_lr) ** (1 / self.num_steps)

        # Set initial LR
        for pg in trainer._optimizer.param_groups:
            pg["lr"] = self.min_lr

        # Run test
        trainer.task.model.train()
        data_iter = iter(trainer._train_loader)
        smoothed_loss = 0.0
        device = trainer.device

        for step in range(self.num_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(trainer._train_loader)
                batch = next(data_iter)

            # Move batch to device
            if isinstance(batch, (tuple, list)):
                batch = tuple(
                    b.to(device) if isinstance(b, torch.Tensor) else b for b in batch
                )
            elif isinstance(batch, dict):
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

            # Forward + backward
            trainer._optimizer.zero_grad()

            # Handle AMP if enabled
            if trainer.amp and trainer._scaler is not None:
                with torch.autocast(device_type=device.split(":")[0] if ":" in device else device, dtype=torch.float16):
                    loss = trainer.task.train_step(batch)
                trainer._scaler.scale(loss).backward()
                trainer._scaler.step(trainer._optimizer)
                trainer._scaler.update()
            else:
                loss = trainer.task.train_step(batch)
                loss.backward()
                trainer._optimizer.step()

            # Track
            loss_val = loss.item()
            lr = trainer._optimizer.param_groups[0]["lr"]
            self._losses.append(loss_val)
            self._lrs.append(lr)

            # Smoothed loss for divergence check
            if step > 0:
                smoothed_loss = self.smoothing * loss_val + (1 - self.smoothing) * smoothed_loss
            else:
                smoothed_loss = loss_val

            if smoothed_loss < self._best_loss:
                self._best_loss = smoothed_loss

            # Check for divergence
            if smoothed_loss > self._best_loss * self.divergence_factor:
                logger.info(f"LRFinder: stopping early due to divergence at lr={lr:.2e}")
                break

            # Check for NaN
            if math.isnan(loss_val) or math.isinf(loss_val):
                logger.warning(f"LRFinder: stopping due to invalid loss at lr={lr:.2e}")
                break

            # Update LR
            for pg in trainer._optimizer.param_groups:
                pg["lr"] *= gamma

        # Find suggested LR
        self.suggested_lr = self._find_suggested_lr()
        logger.info(f"LRFinder: suggested LR = {self.suggested_lr:.2e}")

        # Restore state
        trainer.task.model.load_state_dict(model_state)
        trainer._optimizer.load_state_dict(optimizer_state)

        # Update trainer LR if auto_update enabled and at default
        if self.auto_update and trainer.lr == 1e-4:
            logger.info(f"LRFinder: updating trainer.lr to {self.suggested_lr:.2e}")
            trainer.lr = self.suggested_lr
            for pg in trainer._optimizer.param_groups:
                pg["lr"] = self.suggested_lr

    def _find_suggested_lr(self) -> float:
        """Find LR with steepest negative gradient."""
        if len(self._losses) < 10:
            return self.min_lr * 10

        # Compute gradient (derivative of loss w.r.t. log(lr))
        gradients = []
        for i in range(1, len(self._losses)):
            log_lr_diff = math.log(self._lrs[i]) - math.log(self._lrs[i - 1])
            if abs(log_lr_diff) > 1e-10:
                grad = (self._losses[i] - self._losses[i - 1]) / log_lr_diff
                gradients.append(grad)
            else:
                gradients.append(0.0)

        if not gradients:
            return self.min_lr * 10

        # Find steepest descent (most negative gradient)
        min_grad_idx = min(range(len(gradients)), key=lambda i: gradients[i])

        # Return LR slightly before steepest descent for safety
        return self._lrs[max(0, min_grad_idx - 1)]

    @property
    def losses(self) -> list[float]:
        """Get recorded losses."""
        return self._losses.copy()

    @property
    def lrs(self) -> list[float]:
        """Get recorded learning rates."""
        return self._lrs.copy()
