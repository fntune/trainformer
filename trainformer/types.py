"""Core types for trainformer."""
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Literal, Protocol, runtime_checkable

from torch import Tensor
from torch.utils.data import Dataset


class ConfigSource(Enum):
    """Source of configuration values."""
    USER = auto()      # Explicitly set by user
    DATA = auto()      # Inferred from data
    DERIVED = auto()   # Computed from other config
    ENV = auto()       # From environment variables


class Lifecycle(Enum):
    """Lifecycle scope for context values."""
    STEP = auto()      # Reset each step
    EPOCH = auto()     # Reset each epoch
    RUN = auto()       # Persist entire run


@dataclass
class DatasetInfo:
    """Metadata about a dataset, extracted after loading."""
    num_samples: int
    num_classes: int | None = None
    class_names: list[str] | None = None
    input_shape: tuple[int, ...] | None = None
    class_counts: dict[str, int] | None = None

    @property
    def class_weights(self) -> Tensor | None:
        """Inverse frequency weights for class-balanced loss.

        Returns:
            Tensor of shape (num_classes,) with weights, or None if class_counts unavailable
        """
        import torch

        if self.class_counts is None or len(self.class_counts) == 0:
            return None

        # Get counts in order (sorted by class index/name)
        counts = [self.class_counts[k] for k in sorted(self.class_counts.keys())]
        total = sum(counts)

        # Inverse frequency: total / (num_classes * count_per_class)
        num_classes = len(counts)
        weights = [total / (num_classes * c) if c > 0 else 0.0 for c in counts]

        return torch.tensor(weights, dtype=torch.float32)

    @classmethod
    def from_dataset(cls, dataset: Dataset) -> "DatasetInfo":
        """Extract info from a dataset."""
        num_samples = len(dataset)
        num_classes = None
        class_names = None
        class_counts = None

        # Try to get class info from common attributes
        if hasattr(dataset, "classes"):
            class_names = dataset.classes
            num_classes = len(class_names)
        elif hasattr(dataset, "num_classes"):
            num_classes = dataset.num_classes

        # Try to get input shape from first sample
        input_shape = None
        if num_samples > 0:
            sample = dataset[0]
            if isinstance(sample, tuple) and len(sample) >= 1:
                x = sample[0]
                if isinstance(x, Tensor):
                    input_shape = tuple(x.shape)

        return cls(
            num_samples=num_samples,
            num_classes=num_classes,
            class_names=class_names,
            input_shape=input_shape,
            class_counts=class_counts,
        )


@runtime_checkable
class Task(Protocol):
    """Protocol that all tasks must implement."""

    def train_step(self, batch: Any) -> Tensor:
        """Compute loss for a training batch."""
        ...

    def eval_step(self, batch: Any) -> dict[str, Tensor]:
        """Compute outputs for an evaluation batch."""
        ...

    def configure(self, info: DatasetInfo) -> None:
        """Configure task with dataset info (e.g., set num_classes)."""
        ...

    def parameters(self):
        """Return trainable parameters."""
        ...


class TaskBase:
    """Base class with default implementations for Task protocol."""

    def configure(self, info: DatasetInfo) -> None:
        """Configure task with dataset info. Override if needed."""
        pass

    def on_train_begin(self) -> None:
        """Called before training loop starts."""
        pass

    def on_train_end(self) -> None:
        """Called after training loop ends."""
        pass

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> None:
        """Called at end of each epoch with validation metrics."""
        pass

    def on_step_end(self, step: int, max_steps: int) -> None:
        """Post-optimizer hook (EMA, queue updates, etc.)."""
        pass

    def collate_fn(self, examples: list[Any]) -> Any:
        """Custom collation for DataLoader. Return None for default."""
        return None

    def config_dict(self) -> dict[str, Any]:
        """Task config for logging."""
        return {}

    def state_dict(self) -> dict:
        """Task-specific state for checkpointing."""
        return {}

    def load_state_dict(self, state: dict) -> None:
        """Restore task-specific state."""
        pass


Phase = Literal["train", "val", "test"]
