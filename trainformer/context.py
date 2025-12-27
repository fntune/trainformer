"""Runtime context for training pipelines."""
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from trainformer.types import Lifecycle

logger = logging.getLogger(__name__)


@dataclass
class PipelineContext:
    """Runtime state, buffers, and events for training.

    Manages values with different lifecycles:
    - STEP: Reset each training step
    - EPOCH: Reset each epoch
    - RUN: Persist entire training run
    """

    # Current state
    epoch: int = 0
    step: int = 0
    global_step: int = 0

    # Buffers for different lifecycles
    _step_data: dict[str, Any] = field(default_factory=dict)
    _epoch_data: dict[str, Any] = field(default_factory=dict)
    _run_data: dict[str, Any] = field(default_factory=dict)

    # Event handlers
    _handlers: dict[str, list[Callable]] = field(default_factory=dict)

    # Metrics accumulators
    _train_metrics: dict[str, list[float]] = field(default_factory=dict)
    _val_metrics: dict[str, list[float]] = field(default_factory=dict)

    def set(self, key: str, value: Any, lifecycle: Lifecycle = Lifecycle.STEP) -> None:
        """Store a value with specified lifecycle."""
        if lifecycle == Lifecycle.STEP:
            self._step_data[key] = value
        elif lifecycle == Lifecycle.EPOCH:
            self._epoch_data[key] = value
        else:
            self._run_data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value, checking all lifecycles."""
        if key in self._step_data:
            return self._step_data[key]
        if key in self._epoch_data:
            return self._epoch_data[key]
        if key in self._run_data:
            return self._run_data[key]
        return default

    def on_step_end(self) -> None:
        """Reset step-scoped data."""
        self._step_data.clear()

    def on_epoch_end(self) -> None:
        """Reset epoch-scoped data and aggregate metrics."""
        self._step_data.clear()
        self._epoch_data.clear()

    def log_metric(self, name: str, value: float, phase: str = "train") -> None:
        """Accumulate a metric value."""
        metrics = self._train_metrics if phase == "train" else self._val_metrics
        if name not in metrics:
            metrics[name] = []
        metrics[name].append(value)

    def get_epoch_metrics(self, phase: str = "train") -> dict[str, float]:
        """Get averaged metrics for the epoch."""
        metrics = self._train_metrics if phase == "train" else self._val_metrics
        result = {}
        for name, values in metrics.items():
            if values:
                result[name] = sum(values) / len(values)
        return result

    def reset_epoch_metrics(self, phase: str = "train") -> None:
        """Reset metrics for the next epoch."""
        if phase == "train":
            self._train_metrics.clear()
        else:
            self._val_metrics.clear()

    def register_handler(self, event: str, handler: Callable) -> None:
        """Register an event handler."""
        if event not in self._handlers:
            self._handlers[event] = []
        self._handlers[event].append(handler)

    def emit(self, event: str, **kwargs) -> None:
        """Emit an event to all registered handlers."""
        for handler in self._handlers.get(event, []):
            try:
                handler(**kwargs)
            except Exception as e:
                logger.warning(f"Handler for {event} raised: {e}")

    @property
    def is_first_epoch(self) -> bool:
        return self.epoch == 0

    @property
    def is_first_step(self) -> bool:
        return self.step == 0
