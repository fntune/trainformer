"""Callback protocol and base implementations."""
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from trainformer.trainer import Trainer


@runtime_checkable
class Callback(Protocol):
    """Protocol for training callbacks."""

    def on_fit_start(self, trainer: "Trainer") -> None:
        """Called at the start of training."""
        ...

    def on_fit_end(self, trainer: "Trainer") -> None:
        """Called at the end of training."""
        ...

    def on_epoch_start(self, trainer: "Trainer", epoch: int) -> None:
        """Called at the start of each epoch."""
        ...

    def on_epoch_end(self, trainer: "Trainer", epoch: int, metrics: dict[str, float]) -> None:
        """Called at the end of each epoch."""
        ...

    def on_train_batch_start(self, trainer: "Trainer", batch: Any, batch_idx: int) -> None:
        """Called before each training batch."""
        ...

    def on_train_batch_end(
        self, trainer: "Trainer", batch: Any, batch_idx: int, loss: float
    ) -> None:
        """Called after each training batch."""
        ...

    def on_val_batch_start(self, trainer: "Trainer", batch: Any, batch_idx: int) -> None:
        """Called before each validation batch."""
        ...

    def on_val_batch_end(
        self, trainer: "Trainer", batch: Any, batch_idx: int, outputs: dict[str, Any]
    ) -> None:
        """Called after each validation batch."""
        ...

    def on_validation_end(self, trainer: "Trainer", metrics: dict[str, float]) -> None:
        """Called after validation epoch completes, before on_epoch_end."""
        ...


class CallbackBase:
    """Base class providing default no-op implementations."""

    def on_fit_start(self, trainer: "Trainer") -> None:
        pass

    def on_fit_end(self, trainer: "Trainer") -> None:
        pass

    def on_epoch_start(self, trainer: "Trainer", epoch: int) -> None:
        pass

    def on_epoch_end(self, trainer: "Trainer", epoch: int, metrics: dict[str, float]) -> None:
        pass

    def on_train_batch_start(self, trainer: "Trainer", batch: Any, batch_idx: int) -> None:
        pass

    def on_train_batch_end(
        self, trainer: "Trainer", batch: Any, batch_idx: int, loss: float
    ) -> None:
        pass

    def on_val_batch_start(self, trainer: "Trainer", batch: Any, batch_idx: int) -> None:
        pass

    def on_val_batch_end(
        self, trainer: "Trainer", batch: Any, batch_idx: int, outputs: dict[str, Any]
    ) -> None:
        pass

    def on_validation_end(self, trainer: "Trainer", metrics: dict[str, float]) -> None:
        pass
