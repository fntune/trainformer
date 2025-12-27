"""Exponential Moving Average callback."""
from typing import TYPE_CHECKING, Any

import torch

from trainformer.callbacks.base import CallbackBase

if TYPE_CHECKING:
    from trainformer.trainer import Trainer


class EMA(CallbackBase):
    """Exponential Moving Average of model weights.

    Maintains a shadow copy of the model with EMA-updated weights.
    Swaps to EMA weights during validation for better metrics.

    Args:
        decay: EMA decay rate (higher = slower update)
        update_after_step: Start EMA updates after this many steps
    """

    def __init__(self, decay: float = 0.999, update_after_step: int = 100):
        self.decay = decay
        self.update_after_step = update_after_step
        self.shadow: dict[str, torch.Tensor] | None = None
        self.backup: dict[str, torch.Tensor] | None = None
        self._step = 0

    def on_fit_start(self, trainer: "Trainer") -> None:
        # Initialize shadow parameters
        self.shadow = {}
        for name, param in trainer.task.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def on_train_batch_end(
        self, trainer: "Trainer", batch: Any, batch_idx: int, loss: float
    ) -> None:
        self._step += 1

        if self._step < self.update_after_step:
            return

        # Update EMA weights
        with torch.no_grad():
            for name, param in trainer.task.model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def on_epoch_end(self, trainer: "Trainer", epoch: int, metrics: dict[str, float]) -> None:
        # Swap back to original weights if we swapped for validation
        self._restore(trainer)

    def apply(self, trainer: "Trainer") -> None:
        """Apply EMA weights to model (for validation/inference)."""
        if self.shadow is None:
            return

        self.backup = {}
        for name, param in trainer.task.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def _restore(self, trainer: "Trainer") -> None:
        """Restore original weights after validation."""
        if self.backup is None:
            return

        for name, param in trainer.task.model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = None

    def state_dict(self) -> dict[str, Any]:
        return {"shadow": self.shadow, "step": self._step}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.shadow = state.get("shadow")
        self._step = state.get("step", 0)
