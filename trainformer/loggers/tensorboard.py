"""TensorBoard logger."""
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class TensorBoardLogger:
    """TensorBoard logger for training visualization.

    Args:
        log_dir: Directory for TensorBoard logs
        name: Experiment name (creates subdirectory)
        flush_secs: How often to flush logs to disk (default: 120)
    """

    def __init__(
        self,
        log_dir: str = "runs",
        name: str | None = None,
        flush_secs: int = 120,
    ):
        from torch.utils.tensorboard import SummaryWriter

        if name:
            log_dir = str(Path(log_dir) / name)

        self.writer = SummaryWriter(log_dir=log_dir, flush_secs=flush_secs)
        self.log_dir = log_dir
        logger.info(f"TensorBoard logs at: {log_dir}")

    def log(self, metrics: dict[str, float], step: int) -> None:
        """Log metrics to TensorBoard."""
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)

    def log_hyperparams(self, params: dict[str, Any]) -> None:
        """Log hyperparameters to TensorBoard."""
        # Convert params to string representation for text logging
        text = "\n".join(f"**{k}**: {v}" for k, v in sorted(params.items()))
        self.writer.add_text("hyperparameters", text, 0)

        # Also log as hparams if possible
        try:
            # Filter out non-scalar values for hparams
            scalar_params = {}
            for k, v in params.items():
                if isinstance(v, (int, float, str, bool)):
                    scalar_params[k] = v
            if scalar_params:
                self.writer.add_hparams(scalar_params, {})
        except Exception:
            pass  # hparams logging is optional

    def log_image(self, tag: str, image: Any, step: int) -> None:
        """Log an image to TensorBoard."""
        self.writer.add_image(tag, image, step)

    def log_histogram(self, tag: str, values: Any, step: int) -> None:
        """Log a histogram to TensorBoard."""
        self.writer.add_histogram(tag, values, step)

    def log_graph(self, model: Any, input_to_model: Any = None) -> None:
        """Log model graph to TensorBoard."""
        self.writer.add_graph(model, input_to_model)

    def finish(self) -> None:
        """Close the TensorBoard writer."""
        self.writer.close()

    @property
    def experiment(self):
        """Access the underlying SummaryWriter."""
        return self.writer
