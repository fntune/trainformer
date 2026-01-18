"""Logger protocol and base implementations."""
import logging
import sys
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class Logger(Protocol):
    """Protocol for training loggers."""

    def log(self, metrics: dict[str, float], step: int) -> None:
        """Log metrics at a step."""
        ...

    def log_hyperparams(self, params: dict[str, Any]) -> None:
        """Log hyperparameters."""
        ...

    def finish(self) -> None:
        """Cleanup and finalize logging."""
        ...


class ConsoleLogger:
    """Simple console logger with formatted output."""

    def __init__(self, log_every_n_steps: int = 10):
        self.log_every_n_steps = log_every_n_steps
        self._last_logged_step = -1

    def log(self, metrics: dict[str, float], step: int) -> None:
        if step - self._last_logged_step < self.log_every_n_steps:
            return

        parts = [f"step={step}"]
        for key, value in sorted(metrics.items()):
            if isinstance(value, float):
                parts.append(f"{key}={value:.4f}")
            else:
                parts.append(f"{key}={value}")

        print(" | ".join(parts), file=sys.stderr)
        self._last_logged_step = step

    def log_hyperparams(self, params: dict[str, Any]) -> None:
        logger.info("Hyperparameters:")
        for key, value in sorted(params.items()):
            logger.info(f"  {key}: {value}")

    def finish(self) -> None:
        pass


class NoOpLogger:
    """No-operation logger for when logging is disabled."""

    def log(self, metrics: dict[str, float], step: int) -> None:
        pass

    def log_hyperparams(self, params: dict[str, Any]) -> None:
        pass

    def finish(self) -> None:
        pass


class MultiLogger:
    """Combines multiple loggers."""

    def __init__(self, loggers: list[Logger]):
        self.loggers = loggers

    def log(self, metrics: dict[str, float], step: int) -> None:
        for lg in self.loggers:
            lg.log(metrics, step)

    def log_hyperparams(self, params: dict[str, Any]) -> None:
        for lg in self.loggers:
            lg.log_hyperparams(params)

    def finish(self) -> None:
        for lg in self.loggers:
            lg.finish()
