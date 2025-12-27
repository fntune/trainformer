"""Weights & Biases logger."""
import logging
from typing import Any

logger = logging.getLogger(__name__)


class WandbLogger:
    """Weights & Biases logger.

    Args:
        project: W&B project name
        name: Run name
        config: Config dict to log
        **kwargs: Additional arguments passed to wandb.init
    """

    def __init__(
        self,
        project: str,
        name: str | None = None,
        config: dict[str, Any] | None = None,
        **kwargs,
    ):
        import wandb

        self.run = wandb.init(
            project=project,
            name=name,
            config=config,
            **kwargs,
        )

    def log(self, metrics: dict[str, float], step: int) -> None:
        import wandb
        wandb.log(metrics, step=step)

    def log_hyperparams(self, params: dict[str, Any]) -> None:
        import wandb
        wandb.config.update(params, allow_val_change=True)

    def finish(self) -> None:
        import wandb
        wandb.finish()

    @property
    def experiment(self):
        """Access the underlying wandb run."""
        return self.run
