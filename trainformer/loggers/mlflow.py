"""MLflow logger."""
import logging
from typing import Any

logger = logging.getLogger(__name__)


class MLflowLogger:
    """MLflow logger for experiment tracking.

    Args:
        experiment_name: MLflow experiment name
        run_name: Run name
        tracking_uri: MLflow tracking server URI (default: local)
        tags: Additional tags for the run
    """

    def __init__(
        self,
        experiment_name: str,
        run_name: str | None = None,
        tracking_uri: str | None = None,
        tags: dict[str, str] | None = None,
    ):
        import mlflow

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        mlflow.set_experiment(experiment_name)
        self.run = mlflow.start_run(run_name=run_name, tags=tags)
        self.run_id = self.run.info.run_id
        logger.info(f"MLflow run started: {self.run_id}")

    def log(self, metrics: dict[str, float], step: int) -> None:
        """Log metrics to MLflow."""
        import mlflow

        mlflow.log_metrics(metrics, step=step)

    def log_hyperparams(self, params: dict[str, Any]) -> None:
        """Log hyperparameters to MLflow."""
        import mlflow

        # MLflow params must be strings or simple types
        for key, value in params.items():
            try:
                mlflow.log_param(key, value)
            except Exception:
                # Log as string if type not supported
                mlflow.log_param(key, str(value))

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        """Log an artifact file to MLflow."""
        import mlflow

        mlflow.log_artifact(local_path, artifact_path)

    def log_model(self, model: Any, artifact_path: str = "model") -> None:
        """Log a PyTorch model to MLflow."""
        import mlflow.pytorch

        mlflow.pytorch.log_model(model, artifact_path)

    def finish(self) -> None:
        """End the MLflow run."""
        import mlflow

        mlflow.end_run()

    @property
    def experiment(self):
        """Access the underlying MLflow run."""
        return self.run
