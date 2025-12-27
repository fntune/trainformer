"""Universal trainer for all tasks."""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.utils.data import DataLoader, Dataset

from trainformer.callbacks.base import Callback
from trainformer.config import PipelineConfig
from trainformer.context import PipelineContext
from trainformer.loggers.base import ConsoleLogger, Logger, MultiLogger
from trainformer.types import ConfigSource, DatasetInfo, Task

logger = logging.getLogger(__name__)


@dataclass
class Trainer:
    """Universal trainer for deep learning tasks.

    Args:
        task: Task instance (MetricLearning, CausalLM, SSL, etc.)
        data: Training dataset or path to data
        val_data: Validation dataset or path (optional)
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        optimizer: Optimizer type ('adamw', 'adam', 'sgd')
        scheduler: LR scheduler type ('cosine', 'onecycle', None)
        warmup_steps: Warmup steps for scheduler
        weight_decay: Weight decay for optimizer
        grad_clip: Gradient clipping value (None to disable)
        compile: Whether to use torch.compile
        amp: Use automatic mixed precision
        callbacks: List of callbacks
        logger: Logger instance(s)
        name: Experiment name
        seed: Random seed
        device: Device to use ('auto', 'cuda', 'cpu', 'mps')
        num_workers: DataLoader workers
    """

    task: Task
    data: Dataset | str
    val_data: Dataset | str | None = None

    # Training params
    epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-4
    optimizer: str = "adamw"
    scheduler: str | None = "cosine"
    warmup_steps: int = 0
    weight_decay: float = 0.01
    grad_clip: float | None = 1.0

    # Performance
    compile: bool = False
    amp: bool = True

    # Extensibility
    callbacks: list[Callback] = field(default_factory=list)
    logger: Logger | list[Logger] | None = None

    # Metadata
    name: str | None = None
    seed: int = 42

    # Hardware
    device: str = "auto"
    num_workers: int = 4

    # Internal state
    _optimizer: torch.optim.Optimizer | None = field(default=None, init=False)
    _scheduler: Any = field(default=None, init=False)
    _scaler: torch.cuda.amp.GradScaler | None = field(default=None, init=False)
    _train_loader: DataLoader | None = field(default=None, init=False)
    _val_loader: DataLoader | None = field(default=None, init=False)
    _logger: Logger | None = field(default=None, init=False)

    config: PipelineConfig = field(default_factory=PipelineConfig, init=False)
    ctx: PipelineContext = field(default_factory=PipelineContext, init=False)
    should_stop: bool = field(default=False, init=False)
    best_metrics: dict[str, float] = field(default_factory=dict, init=False)

    def __post_init__(self):
        self._setup_device()
        self._setup_config()
        self._setup_logger()

    def _setup_device(self) -> None:
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        logger.info(f"Using device: {self.device}")

    def _setup_config(self) -> None:
        """Store config values with source tracking."""
        self.config.set("epochs", self.epochs, ConfigSource.USER)
        self.config.set("batch_size", self.batch_size, ConfigSource.USER)
        self.config.set("lr", self.lr, ConfigSource.USER)
        self.config.set("optimizer", self.optimizer, ConfigSource.USER)
        self.config.set("scheduler", self.scheduler, ConfigSource.USER)
        self.config.set("weight_decay", self.weight_decay, ConfigSource.USER)
        self.config.set("grad_clip", self.grad_clip, ConfigSource.USER)
        self.config.set("compile", self.compile, ConfigSource.USER)
        self.config.set("amp", self.amp, ConfigSource.USER)
        self.config.set("seed", self.seed, ConfigSource.USER)
        self.config.set("device", self.device, ConfigSource.DERIVED)

    def _setup_logger(self) -> None:
        """Initialize logging."""
        if self.logger is None:
            self._logger = ConsoleLogger()
        elif isinstance(self.logger, list):
            self._logger = MultiLogger(self.logger)
        else:
            self._logger = self.logger

    def _setup_data(self) -> None:
        """Load datasets and create data loaders."""
        # Handle path strings
        if isinstance(self.data, str):
            self.data = self.task.load_data(self.data)
        if isinstance(self.val_data, str):
            self.val_data = self.task.load_data(self.val_data)

        # Extract dataset info
        info = DatasetInfo.from_dataset(self.data)
        self.config.set("num_samples", info.num_samples, ConfigSource.DATA)
        if info.num_classes:
            self.config.set("num_classes", info.num_classes, ConfigSource.DATA)

        # Configure task with data info
        self.task.configure(info)

        # Create data loaders
        self._train_loader = DataLoader(
            self.data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.device == "cuda",
            drop_last=True,
        )

        if self.val_data is not None:
            self._val_loader = DataLoader(
                self.val_data,
                batch_size=self.batch_size * 2,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.device == "cuda",
            )

        # Derive total steps
        steps_per_epoch = len(self._train_loader)
        total_steps = steps_per_epoch * self.epochs
        self.config.set("steps_per_epoch", steps_per_epoch, ConfigSource.DERIVED)
        self.config.set("total_steps", total_steps, ConfigSource.DERIVED)

    def _setup_optimizer(self) -> None:
        """Create optimizer and scheduler."""
        params = self.task.parameters()

        if self.optimizer == "adamw":
            self._optimizer = AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer == "adam":
            self._optimizer = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer == "sgd":
            self._optimizer = torch.optim.SGD(
                params, lr=self.lr, momentum=0.9, weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")

        # Scheduler
        total_steps = self.config["total_steps"]
        if self.scheduler == "cosine":
            self._scheduler = CosineAnnealingLR(self._optimizer, T_max=total_steps)
        elif self.scheduler == "onecycle":
            self._scheduler = OneCycleLR(
                self._optimizer,
                max_lr=self.lr,
                total_steps=total_steps,
            )

    def _setup_amp(self) -> None:
        """Setup automatic mixed precision."""
        if self.amp and self.device == "cuda":
            self._scaler = torch.cuda.amp.GradScaler()

    def _move_to_device(self) -> None:
        """Move task model and loss to device."""
        if hasattr(self.task, "model"):
            self.task.model.to(self.device)

        # Also move loss if it exists (for metric learning, etc.)
        if hasattr(self.task, "_loss") and self.task._loss is not None:
            self.task._loss.to(self.device)

        if self.compile and hasattr(self.task, "model"):
            self.task.model = torch.compile(self.task.model)
            logger.info("Model compiled with torch.compile")

    def fit(self) -> "Trainer":
        """Run the training loop."""
        torch.manual_seed(self.seed)

        # Setup phases
        self._setup_data()
        self._move_to_device()
        self._setup_optimizer()
        self._setup_amp()

        # Log hyperparams
        self._logger.log_hyperparams(self.config.to_dict())

        # Callbacks: fit start
        for cb in self.callbacks:
            cb.on_fit_start(self)

        try:
            for epoch in range(self.epochs):
                if self.should_stop:
                    logger.info("Training stopped early")
                    break

                self.ctx.epoch = epoch

                # Callbacks: epoch start
                for cb in self.callbacks:
                    cb.on_epoch_start(self, epoch)

                # Training epoch
                train_metrics = self._train_epoch()

                # Validation epoch
                val_metrics = {}
                if self._val_loader is not None:
                    val_metrics = self._val_epoch()

                # Combine metrics
                metrics = {**train_metrics, **val_metrics}
                self._update_best_metrics(metrics)

                # Log
                self._logger.log(metrics, step=self.ctx.global_step)

                # Callbacks: epoch end
                for cb in self.callbacks:
                    cb.on_epoch_end(self, epoch, metrics)

                self.ctx.on_epoch_end()

        finally:
            # Callbacks: fit end
            for cb in self.callbacks:
                cb.on_fit_end(self)

            self._logger.finish()

        return self

    def _train_epoch(self) -> dict[str, float]:
        """Run one training epoch."""
        self.task.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(self._train_loader):
            self.ctx.step = batch_idx

            # Move batch to device
            batch = self._to_device(batch)

            # Callbacks: batch start
            for cb in self.callbacks:
                cb.on_train_batch_start(self, batch, batch_idx)

            # Forward pass
            self._optimizer.zero_grad()

            if self._scaler is not None:
                with torch.cuda.amp.autocast():
                    loss = self.task.train_step(batch)
                self._scaler.scale(loss).backward()

                if self.grad_clip is not None:
                    self._scaler.unscale_(self._optimizer)
                    nn.utils.clip_grad_norm_(self.task.parameters(), self.grad_clip)

                self._scaler.step(self._optimizer)
                self._scaler.update()
            else:
                loss = self.task.train_step(batch)
                loss.backward()

                if self.grad_clip is not None:
                    nn.utils.clip_grad_norm_(self.task.parameters(), self.grad_clip)

                self._optimizer.step()

            if self._scheduler is not None:
                self._scheduler.step()

            # Track metrics
            loss_val = loss.item()
            total_loss += loss_val
            num_batches += 1

            self.ctx.log_metric("loss", loss_val, "train")
            self.ctx.global_step += 1

            # Callbacks: batch end
            for cb in self.callbacks:
                cb.on_train_batch_end(self, batch, batch_idx, loss_val)

            # Task hook (for SSL momentum updates, etc.)
            if hasattr(self.task, "on_step_end"):
                total_steps = self.config["total_steps"]
                self.task.on_step_end(self.ctx.global_step, total_steps)

            self.ctx.on_step_end()

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        lr = self._optimizer.param_groups[0]["lr"]

        return {"train/loss": avg_loss, "train/lr": lr}

    @torch.no_grad()
    def _val_epoch(self) -> dict[str, float]:
        """Run one validation epoch."""
        self.task.model.eval()
        all_outputs: dict[str, list[Tensor]] = {}

        for batch_idx, batch in enumerate(self._val_loader):
            batch = self._to_device(batch)

            # Callbacks: batch start
            for cb in self.callbacks:
                cb.on_val_batch_start(self, batch, batch_idx)

            # Forward pass
            if self._scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.task.eval_step(batch)
            else:
                outputs = self.task.eval_step(batch)

            # Accumulate outputs
            for key, value in outputs.items():
                if key not in all_outputs:
                    all_outputs[key] = []
                all_outputs[key].append(value.cpu())

            # Callbacks: batch end
            for cb in self.callbacks:
                cb.on_val_batch_end(self, batch, batch_idx, outputs)

        # Compute final metrics
        metrics = {}

        # If task has custom evaluation
        if hasattr(self.task, "evaluate_retrieval") and "embeddings" in all_outputs:
            embeddings = torch.cat(all_outputs["embeddings"])
            labels = torch.cat(all_outputs["labels"])
            retrieval_metrics = self.task.evaluate_retrieval(embeddings, labels)
            metrics.update({f"val/{k}": v for k, v in retrieval_metrics.items()})

        return metrics

    def _to_device(self, batch: Any) -> Any:
        """Move batch to device."""
        if isinstance(batch, Tensor):
            return batch.to(self.device)
        elif isinstance(batch, (list, tuple)):
            return type(batch)(self._to_device(x) for x in batch)
        elif isinstance(batch, dict):
            return {k: self._to_device(v) for k, v in batch.items()}
        return batch

    def _update_best_metrics(self, metrics: dict[str, float]) -> None:
        """Track best metrics seen during training."""
        for key, value in metrics.items():
            if key not in self.best_metrics:
                self.best_metrics[key] = value
            elif "loss" in key:
                self.best_metrics[key] = min(self.best_metrics[key], value)
            else:
                self.best_metrics[key] = max(self.best_metrics[key], value)

    def state_dict(self) -> dict[str, Any]:
        """Get trainer state for checkpointing."""
        state = {
            "task": self.task.model.state_dict() if hasattr(self.task, "model") else {},
            "optimizer": self._optimizer.state_dict() if self._optimizer else {},
            "scheduler": self._scheduler.state_dict() if self._scheduler else {},
            "epoch": self.ctx.epoch,
            "global_step": self.ctx.global_step,
            "best_metrics": self.best_metrics,
            "config": self.config.to_dict(),
        }

        # Task-specific state
        if hasattr(self.task, "state_dict"):
            state["task_extra"] = self.task.state_dict()

        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Load trainer state from checkpoint."""
        if hasattr(self.task, "model") and "task" in state:
            self.task.model.load_state_dict(state["task"])

        if self._optimizer and "optimizer" in state:
            self._optimizer.load_state_dict(state["optimizer"])

        if self._scheduler and "scheduler" in state:
            self._scheduler.load_state_dict(state["scheduler"])

        self.ctx.epoch = state.get("epoch", 0)
        self.ctx.global_step = state.get("global_step", 0)
        self.best_metrics = state.get("best_metrics", {})

        if hasattr(self.task, "load_state_dict") and "task_extra" in state:
            self.task.load_state_dict(state["task_extra"])

    def load(self, path: str | Path) -> "Trainer":
        """Load state from checkpoint file."""
        state = torch.load(path, map_location=self.device)
        self.load_state_dict(state)
        logger.info(f"Loaded checkpoint from {path}")
        return self

    def save(self, path: str | Path) -> None:
        """Save state to checkpoint file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)
        logger.info(f"Saved checkpoint to {path}")

    @torch.no_grad()
    def predict(self, data: Dataset | DataLoader) -> dict[str, Tensor]:
        """Run inference on data."""
        self.task.model.eval()

        if isinstance(data, Dataset):
            loader = DataLoader(
                data, batch_size=self.batch_size * 2, shuffle=False, num_workers=self.num_workers
            )
        else:
            loader = data

        all_outputs: dict[str, list[Tensor]] = {}

        for batch in loader:
            batch = self._to_device(batch)
            outputs = self.task.eval_step(batch)

            for key, value in outputs.items():
                if key not in all_outputs:
                    all_outputs[key] = []
                all_outputs[key].append(value.cpu())

        return {key: torch.cat(values) for key, values in all_outputs.items()}
