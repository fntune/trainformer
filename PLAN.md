# Trainformer

**Training without the framework overhead.**

A Python-first training library for deep learning: from CNNs to LLMs, classification to SSL, full fine-tuning to LoRA.

## Vision

```python
from trainformer import Trainer
from trainformer.tasks import CausalLM, MetricLearning, SSL
from trainformer.adapters import LoRA

# Fine-tune LLM with LoRA
Trainer(CausalLM("meta-llama/Llama-2-7b", adapter=LoRA(r=8)), data="data/alpaca.json").fit()

# Train ArcFace embeddings
Trainer(MetricLearning("efficientnet_b0", loss="arcface"), data="data/sop").fit()

# Pretrain with DINO
Trainer(SSL.dino("vit_small"), data="data/imagenet").fit()
```

**No YAML. No framework to learn. Just Python.**

---

## Design Principles

1. **Python over YAML** - Config is code. Type-safe, IDE-friendly, debuggable.
2. **Tasks over abstractions** - Bundle model + loss + eval into cohesive units.
3. **Recipes over frameworks** - Complete solutions, not building blocks.
4. **Batteries included** - LoRA, distributed, logging, evaluation built-in.
5. **Build on giants** - Use HuggingFace, timm, Accelerate. Don't reinvent.

---

## Architecture

```
trainformer/
├── __init__.py                   # Public API
├── trainer.py                    # Universal Trainer (~400 lines)
├── context.py                    # PipelineConfig + PipelineContext
│
├── tasks/                        # Task = model + loss + data + eval
│   ├── base.py                   # Task protocol + DatasetInfo
│   ├── vision/
│   │   ├── classification.py     # ImageClassification
│   │   ├── metric_learning.py    # MetricLearning (ArcFace, CosFace)
│   │   └── ssl.py                # SSL (SimCLR, MoCo, DINO, MAE)
│   ├── nlp/
│   │   ├── causal_lm.py          # CausalLM (GPT, Llama, Mistral)
│   │   ├── seq2seq.py            # Seq2Seq (T5, BART)
│   │   └── masked_lm.py          # MaskedLM (BERT)
│   └── multimodal/
│       ├── clip.py               # CLIP
│       └── vlm.py                # VLM (LLaVA, Qwen-VL)
│
├── adapters/                     # Parameter-efficient fine-tuning
│   ├── base.py                   # Adapter protocol
│   ├── lora.py                   # LoRA
│   ├── qlora.py                  # QLoRA (4-bit)
│   └── prefix.py                 # Prefix tuning
│
├── models/                       # Model implementations
│   ├── vision/
│   │   ├── embedding.py          # EmbeddingModel base
│   │   ├── arcface.py            # ArcFace components
│   │   └── ssl/
│   │       ├── simclr.py
│   │       ├── moco.py
│   │       ├── dino.py
│   │       └── mae.py
│   └── components/
│       ├── backbones.py          # TimmBackbone wrapper
│       ├── heads.py              # Projection heads
│       ├── poolers.py            # GeM, etc.
│       └── losses.py             # ArcFaceLoss, NTXentLoss, etc.
│
├── data/                         # Data loading
│   ├── image.py                  # ImageFolder, TensorDictDataset
│   ├── text.py                   # TextDataset, ChatDataset
│   ├── multimodal.py             # ImageTextDataset
│   ├── datasets.py               # JSONLDataset, TextFileDataset
│   └── samplers.py               # ClassBalancedSampler, PKSampler
│
├── eval/                         # Evaluation
│   ├── classification.py         # Accuracy, F1
│   ├── retrieval.py              # KNN, FAISS
│   └── generation.py             # Perplexity, BLEU
│
├── callbacks/                    # Training callbacks
│   ├── base.py                   # Callback protocol, CallbackList
│   ├── early_stopping.py         # EarlyStopping
│   ├── checkpoint.py             # ModelCheckpoint
│   ├── monitors.py               # LRMonitor, GradientMonitor
│   ├── lr_finder.py              # LRFinder
│   ├── ema.py                    # EMA model averaging
│   └── knn.py                    # Online KNN evaluation
│
├── sweep.py                      # Hyperparameter sweeps (cloudpickle)
│
└── utils/
    ├── logging.py                # WandbLogger, ConsoleLogger, TensorBoardLogger, MLflowLogger, MultiLogger
    ├── helpers.py                # seed_everything, count_parameters, profile_memory
    └── distributed.py            # Accelerate helpers
```

---

## Core Components

### 1. DatasetInfo

Metadata extracted from dataset - single source of truth for values that flow through the pipeline:

```python
# tasks/base.py
from dataclasses import dataclass, field
from typing import Any, Iterator
from torch import Tensor
import torch

@dataclass
class DatasetInfo:
    """Metadata extracted from dataset. Single source of truth."""

    # Universal
    num_samples: int = 0

    # Classification / Metric Learning
    num_classes: int | None = None
    class_names: list[str] | None = None
    class_counts: list[int] | None = None

    # Image
    input_shape: tuple[int, ...] | None = None  # (C, H, W)
    mean: tuple[float, ...] | None = None
    std: tuple[float, ...] | None = None

    # Sequence
    max_length: int | None = None

    # Regression
    num_targets: int | None = None
    target_mean: float | None = None
    target_std: float | None = None

    # Extensible
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def class_weights(self) -> Tensor | None:
        """Inverse frequency weights for class-balanced loss."""
        if self.class_counts is None:
            return None
        total = sum(self.class_counts)
        n = len(self.class_counts)
        weights = [total / (n * c) for c in self.class_counts]
        return torch.tensor(weights)
```

### 2. Task Protocol

Tasks bundle everything needed for a training paradigm:

```python
# tasks/base.py
from typing import Protocol, Any, Iterator
from torch import Tensor
from torch.utils.data import Dataset
import torch.nn as nn

class Task(Protocol):
    """What every task must provide."""

    model: nn.Module

    # ─── Core Methods (Required) ────────────────────────────────

    def train_step(self, batch: Any) -> Tensor:
        """Compute loss for one batch. Returns scalar loss."""
        ...

    def eval_step(self, batch: Any) -> dict[str, Tensor]:
        """Compute metrics for one batch. Returns metric dict."""
        ...

    # ─── Lifecycle Hooks (Optional) ─────────────────────────────

    def configure(self, info: DatasetInfo) -> None:
        """Configure task with dataset metadata. Called before training.

        Use this to initialize components that depend on data properties:
        - Loss functions needing num_classes (ArcFace, etc.)
        - Classification heads
        - Class-weighted samplers
        """
        ...

    def on_train_begin(self) -> None:
        """Called before training loop starts."""
        ...

    def on_train_end(self) -> None:
        """Called after training loop ends."""
        ...

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> None:
        """Called at end of each epoch with validation metrics."""
        ...

    def on_step_end(self, step: int, max_steps: int) -> None:
        """Post-optimizer hook (EMA, queue updates, etc.)."""
        ...

    def on_validation_end(self, metrics: dict[str, float]) -> None:
        """Called after validation with computed metrics."""
        ...

    # ─── Data Methods (Optional) ────────────────────────────────

    def load_data(self, path: str) -> Dataset:
        """Load dataset from path. Override for custom loading logic.

        Default: raises NotImplementedError.
        """
        ...

    def collate_fn(self, examples: list[Any]) -> Any:
        """Custom collation for DataLoader.

        Default: None (use torch default_collate).
        """
        ...

    def get_train_transforms(self) -> nn.Module | None:
        """Training augmentations applied on GPU.

        Default: None (no augmentation).
        """
        ...

    def get_eval_transforms(self) -> nn.Module | None:
        """Evaluation transforms applied on GPU.

        Default: None (no transforms).
        """
        ...

    # ─── Parameter Access ───────────────────────────────────────

    def parameters(self) -> Iterator[nn.Parameter]:
        """All trainable parameters (model + loss).

        Override if task has learnable loss (ArcFace weights, etc.).
        Default: yields model.parameters()
        """
        ...

    # ─── Config Contribution ────────────────────────────────────

    def config_dict(self) -> dict[str, Any]:
        """Task config for logging. Merged into wandb config."""
        ...

    # ─── Checkpointing ──────────────────────────────────────────

    def state_dict(self) -> dict:
        """Task-specific state for checkpointing."""
        ...

    def load_state_dict(self, state: dict) -> None:
        """Restore task-specific state."""
        ...


class TaskBase:
    """Optional base class with default implementations."""

    def configure(self, info: DatasetInfo) -> None:
        pass

    def on_train_begin(self) -> None:
        pass

    def on_train_end(self) -> None:
        pass

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> None:
        pass

    def on_step_end(self, step: int, max_steps: int) -> None:
        pass

    def on_validation_end(self, metrics: dict[str, float]) -> None:
        pass

    def load_data(self, path: str) -> Dataset:
        raise NotImplementedError(f"{type(self).__name__} does not implement load_data()")

    def collate_fn(self, examples: list[Any]) -> Any:
        return None  # Use default collate

    def get_train_transforms(self) -> nn.Module | None:
        return None

    def get_eval_transforms(self) -> nn.Module | None:
        return None

    def parameters(self) -> Iterator[nn.Parameter]:
        yield from self.model.parameters()

    def config_dict(self) -> dict[str, Any]:
        return {}

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:
        pass
```

### 3. Trainer

Universal training loop with phased initialization:

```python
# trainer.py
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator
from accelerate import Accelerator
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import torch.nn as nn

@dataclass
class Trainer:
    """Universal trainer for any task."""

    # Required
    task: Task
    data: str | Dataset | DataLoader

    # Optional
    val_data: str | Dataset | DataLoader | None = None
    val_split: float | None = None  # Auto-split if val_data not provided

    # Training
    epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_clip: float = 1.0
    accumulation_steps: int = 1

    # Compute
    precision: str = "bf16"  # no, fp16, bf16
    compile: bool = False
    gradient_checkpointing: bool = False
    activation_offload: bool = False

    # Callbacks
    callbacks: list[Callback] | None = None

    # Logging
    log: str | list[str] = "wandb"  # wandb, console, tensorboard, mlflow, or list
    log_every: int = 100
    project: str = "trainformer"
    name: str | None = None

    # Checkpoints
    save_dir: str = "checkpoints"
    save_every: int | None = None
    save_best: int = 3
    monitor_metric: str = "val/loss"
    monitor_mode: str = "min"

    # Evaluation
    eval_every: int = 1

    # Internal state
    _step: int = field(default=0, init=False)
    _epoch: int = field(default=0, init=False)
    _should_stop: bool = field(default=False, init=False)
    _max_steps: int = field(default=0, init=False)
    best_metrics: dict = field(default_factory=dict, init=False)

    def __post_init__(self):
        self._ctx = PipelineContext()
        self._callbacks = CallbackList(self.callbacks)
        self._setup()

    def _setup(self):
        """Initialize with explicit phases for proper dependency ordering."""

        # ═══ PHASE 1: Record user config ═══
        self._record_user_config()
        self._record_env_config()

        # ═══ PHASE 2: Load data, extract metadata ═══
        train_dataset = self._load_dataset(self.data)

        # Handle val_data or val_split
        if self.val_data is not None:
            val_dataset = self._load_dataset(self.val_data)
        elif self.val_split is not None:
            train_dataset, val_dataset = self._split_dataset(train_dataset, self.val_split)
        else:
            val_dataset = None

        info = self._extract_dataset_info(train_dataset)
        self._record_data_config(info)

        # ═══ PHASE 3: Configure task with data info ═══
        if hasattr(self.task, "configure"):
            self.task.configure(info)
        self._record_task_config()

        # ═══ PHASE 4: Compute derived values ═══
        self._compute_derived(info)

        # ═══ PHASE 5: Apply memory optimizations and compile ═══
        if self.gradient_checkpointing:
            self._enable_gradient_checkpointing()

        if self.compile and hasattr(torch, "compile"):
            self.task.model = torch.compile(self.task.model)

        # ═══ PHASE 6: Create loaders, optimizer, scheduler ═══
        self._train_loader = self._make_loader_from_dataset(train_dataset, shuffle=True)
        self._val_loader = self._make_loader_from_dataset(val_dataset) if val_dataset else None

        self._optimizer = self._make_optimizer()
        self._scheduler = self._make_scheduler()

        # ═══ PHASE 7: Prepare for distributed ═══
        self.accelerator = Accelerator(
            mixed_precision=self.precision if self.precision != "no" else None,
            gradient_accumulation_steps=self.accumulation_steps,
        )

        self.task.model, self._optimizer, self._train_loader, self._scheduler = \
            self.accelerator.prepare(
                self.task.model, self._optimizer, self._train_loader, self._scheduler
            )

        if self._val_loader:
            self._val_loader = self.accelerator.prepare(self._val_loader)

        # Prepare task's loss if it has learnable parameters
        if hasattr(self.task, "_loss") and self.task._loss is not None:
            self.task._loss = self.accelerator.prepare(self.task._loss)

        # ═══ PHASE 8: Initialize logger with full config ═══
        self._logger = self._make_logger()
        self._logger.init(self._ctx.config.to_wandb_config())

        self._max_steps = len(self._train_loader) * self.epochs

    def _split_dataset(self, dataset: Dataset, val_ratio: float) -> tuple[Dataset, Dataset]:
        """Split dataset into train/val."""
        val_size = int(len(dataset) * val_ratio)
        train_size = len(dataset) - val_size
        return random_split(dataset, [train_size, val_size])

    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing on model."""
        if hasattr(self.task.model, "gradient_checkpointing_enable"):
            self.task.model.gradient_checkpointing_enable()
        elif hasattr(self.task.model, "set_grad_checkpointing"):
            self.task.model.set_grad_checkpointing(True)

    def _load_dataset(self, data) -> Dataset | None:
        """Load dataset without creating DataLoader."""
        if data is None:
            return None
        if isinstance(data, DataLoader):
            return data.dataset
        if isinstance(data, Dataset):
            return data
        if isinstance(data, str):
            return self.task.load_data(data)
        raise TypeError(f"Unsupported data type: {type(data)}")

    def _extract_dataset_info(self, dataset: Dataset) -> DatasetInfo:
        """Extract metadata from dataset."""
        from collections import Counter

        info = DatasetInfo(num_samples=len(dataset))

        # Classification / Metric Learning
        if hasattr(dataset, "classes"):
            info.class_names = dataset.classes
            info.num_classes = len(dataset.classes)
        if hasattr(dataset, "targets"):
            counts = Counter(dataset.targets)
            info.class_counts = [counts[i] for i in range(info.num_classes)]

        # Image shape from first sample
        sample = dataset[0]
        x = sample[0] if isinstance(sample, tuple) else sample
        if isinstance(x, Tensor) and x.dim() == 3:
            info.input_shape = tuple(x.shape)

        return info

    def _make_loader_from_dataset(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        """Create DataLoader from dataset."""
        collate = getattr(self.task, "collate_fn", None)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate,
        )

    def fit(self) -> "Trainer":
        """Run training loop."""
        # Notify callbacks and task
        self._callbacks.on_train_begin(self)
        if hasattr(self.task, "on_train_begin"):
            self.task.on_train_begin()

        for epoch in range(self._epoch, self.epochs):
            if self._should_stop:
                break

            self._epoch = epoch
            self._callbacks.on_epoch_begin(self, epoch)
            self._train_epoch()

            metrics = {}
            if self._val_loader and epoch % self.eval_every == 0:
                metrics = self._evaluate()
                self._callbacks.on_validation_end(self, metrics)
                if hasattr(self.task, "on_validation_end"):
                    self.task.on_validation_end(metrics)
                self._update_best(metrics)

            self._callbacks.on_epoch_end(self, epoch, metrics)
            if hasattr(self.task, "on_epoch_end"):
                self.task.on_epoch_end(epoch, metrics)

            # Clear epoch-scoped context
            self._ctx.on_epoch_end()

        # Notify callbacks and task
        self._callbacks.on_train_end(self)
        if hasattr(self.task, "on_train_end"):
            self.task.on_train_end()

        self._logger.finish()
        return self

    def _update_best(self, metrics: dict[str, float]) -> None:
        """Update best metrics tracking."""
        value = metrics.get(self.monitor_metric)
        if value is None:
            return

        is_better = (
            self.monitor_metric not in self.best_metrics or
            (self.monitor_mode == "min" and value < self.best_metrics[self.monitor_metric]) or
            (self.monitor_mode == "max" and value > self.best_metrics[self.monitor_metric])
        )

        if is_better:
            self.best_metrics.update(metrics)

    def _train_epoch(self):
        """Train one epoch."""
        self.task.model.train()

        # Set epoch for distributed sampler
        if hasattr(self._train_loader.sampler, "set_epoch"):
            self._train_loader.sampler.set_epoch(self._epoch)

        for batch in self._train_loader:
            if self._should_stop:
                break

            self._callbacks.on_step_begin(self, self._step)
            loss = self._train_step(batch)

            if self._step % self.log_every == 0:
                lr = self._scheduler.get_last_lr()[0]
                self._logger.log({
                    "train/loss": loss,
                    "train/lr": lr,
                    "train/epoch": self._epoch,
                }, self._step)

            # Notify callbacks
            self._callbacks.on_step_end(self, self._step, loss)

            # Clear step-scoped context
            self._ctx.on_step_end()

            self._step += 1

    def _train_step(self, batch) -> float:
        """Single training step with gradient accumulation."""
        with self.accelerator.accumulate(self.task.model):
            loss = self.task.train_step(batch)
            self.accelerator.backward(loss)

            if self.gradient_clip:
                # Clip both model and loss parameters
                params = self.task.parameters() if hasattr(self.task, "parameters") else self.task.model.parameters()
                self.accelerator.clip_grad_norm_(params, self.gradient_clip)

            self._optimizer.step()
            self._scheduler.step()
            self._optimizer.zero_grad()

        # Task-specific post-step (EMA, queues, etc.)
        if hasattr(self.task, "on_step_end"):
            self.task.on_step_end(self._step, self._max_steps)

        return loss.item()

    def _evaluate(self) -> dict[str, float]:
        """Run evaluation."""
        self.task.model.eval()
        all_metrics = defaultdict(list)

        with torch.no_grad():
            for batch in self._val_loader:
                metrics = self.task.eval_step(batch)
                for k, v in metrics.items():
                    gathered = self.accelerator.gather(v)
                    all_metrics[k].append(gathered)

        # Aggregate
        final = {}
        for k, v in all_metrics.items():
            stacked = torch.cat(v) if v[0].dim() > 0 else torch.stack(v)
            final[k] = stacked.mean().item()

        self._logger.log({f"val/{k}": v for k, v in final.items()}, self._step)
        return final

    def _make_optimizer(self):
        """Create optimizer with all trainable parameters."""
        # Let task define what gets optimized (model + loss params)
        if hasattr(self.task, "parameters"):
            params = self.task.parameters()
        else:
            params = self.task.model.parameters()

        return torch.optim.AdamW(
            params,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

    def _make_scheduler(self):
        """Create LR scheduler with warmup."""
        from torch.optim.lr_scheduler import OneCycleLR

        total_steps = self._ctx.config.get("total_steps", self._max_steps)

        return OneCycleLR(
            self._optimizer,
            max_lr=self.lr,
            total_steps=total_steps,
            pct_start=self.warmup_ratio,
        )

    # ─── Config Recording Methods ───────────────────────────────

    def _record_user_config(self):
        """Record user-specified config values."""
        cfg = self._ctx.config
        cfg.set_user("epochs", self.epochs, filterable=True)
        cfg.set_user("batch_size", self.batch_size, filterable=True)
        cfg.set_user("lr", self.lr, filterable=True)
        cfg.set_user("weight_decay", self.weight_decay)
        cfg.set_user("warmup_ratio", self.warmup_ratio)
        cfg.set_user("gradient_clip", self.gradient_clip)
        cfg.set_user("accumulation_steps", self.accumulation_steps, filterable=True)
        cfg.set_user("precision", self.precision, filterable=True)
        cfg.set_user("compile", self.compile, filterable=True)

    def _record_env_config(self):
        """Record environment info."""
        cfg = self._ctx.config
        cfg.set_env("torch_version", torch.__version__)
        cfg.set_env("cuda_available", torch.cuda.is_available())
        if torch.cuda.is_available():
            cfg.set_env("cuda_version", torch.version.cuda)
            cfg.set_env("gpu_name", torch.cuda.get_device_name(0))
            cfg.set_env("num_gpus", torch.cuda.device_count())

    def _record_data_config(self, info: DatasetInfo):
        """Record dataset-derived values."""
        cfg = self._ctx.config
        cfg.set_from_data("num_samples", info.num_samples)
        if info.num_classes is not None:
            cfg.set_from_data("num_classes", info.num_classes, filterable=True)
        if info.class_names is not None:
            cfg.set_from_data("class_names", info.class_names)
        if info.input_shape is not None:
            cfg.set_from_data("input_shape", info.input_shape)

    def _record_task_config(self):
        """Record task configuration."""
        cfg = self._ctx.config
        cfg.set_user("task_type", type(self.task).__name__, filterable=True)

        if hasattr(self.task, "config_dict"):
            for k, v in self.task.config_dict().items():
                cfg.set_user(f"task/{k}", v, filterable=True)

    def _compute_derived(self, info: DatasetInfo):
        """Compute values that depend on other config."""
        cfg = self._ctx.config
        num_samples = info.num_samples
        batch_size = self.batch_size
        epochs = self.epochs
        accum = self.accumulation_steps

        steps_per_epoch = num_samples // batch_size
        total_steps = steps_per_epoch * epochs // accum
        warmup_steps = int(total_steps * self.warmup_ratio)

        num_gpus = cfg.get("num_gpus", 1)
        effective_batch = batch_size * accum * num_gpus

        cfg.set_derived("steps_per_epoch", steps_per_epoch, ("num_samples", "batch_size"))
        cfg.set_derived("total_steps", total_steps, ("steps_per_epoch", "epochs", "accumulation_steps"))
        cfg.set_derived("warmup_steps", warmup_steps, ("total_steps", "warmup_ratio"))
        cfg.set_derived("effective_batch_size", effective_batch, ("batch_size", "accumulation_steps", "num_gpus"), filterable=True)

    def _make_logger(self):
        """Create logger (supports multiple backends)."""
        backends = self.log if isinstance(self.log, list) else [self.log]

        loggers = []
        for backend in backends:
            if backend == "wandb":
                loggers.append(WandbLogger(project=self.project, name=self.name))
            elif backend == "console":
                loggers.append(ConsoleLogger())
            elif backend == "tensorboard":
                loggers.append(TensorBoardLogger(log_dir=self.save_dir))
            elif backend == "mlflow":
                loggers.append(MLflowLogger(experiment_name=self.project, run_name=self.name))
            elif backend == "none":
                pass

        if len(loggers) == 0:
            return NoOpLogger()
        elif len(loggers) == 1:
            return loggers[0]
        else:
            return MultiLogger(loggers)

    def save(self, path: str):
        """Save checkpoint."""
        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            checkpoint = {
                "model": self.accelerator.unwrap_model(self.task.model).state_dict(),
                "optimizer": self._optimizer.state_dict(),
                "scheduler": self._scheduler.state_dict(),
                "epoch": self._epoch,
                "step": self._step,
                "best_metrics": self.best_metrics,
                "task_state": self.task.state_dict() if hasattr(self.task, "state_dict") else None,
                "config": self._config_dict(),
            }
            self.accelerator.save(checkpoint, path)

    def load(self, path: str) -> "Trainer":
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location="cpu")

        self.accelerator.unwrap_model(self.task.model).load_state_dict(checkpoint["model"])
        self._optimizer.load_state_dict(checkpoint["optimizer"])
        self._scheduler.load_state_dict(checkpoint["scheduler"])
        self._epoch = checkpoint["epoch"]
        self._step = checkpoint["step"]
        self.best_metrics = checkpoint.get("best_metrics", {})

        if checkpoint.get("task_state") and hasattr(self.task, "load_state_dict"):
            self.task.load_state_dict(checkpoint["task_state"])

        # Load callback state
        if checkpoint.get("callbacks_state"):
            self._callbacks.load_state_dict(checkpoint["callbacks_state"])

        return self

    def predict(
        self,
        data: str | Dataset | DataLoader,
        batch_size: int | None = None,
    ) -> list[dict[str, Tensor]]:
        """Run inference on data.

        Args:
            data: Path, Dataset, or DataLoader to predict on
            batch_size: Batch size (defaults to training batch_size)

        Returns:
            List of output dicts from task.eval_step()
        """
        # Prepare data
        if isinstance(data, DataLoader):
            loader = data
        else:
            dataset = self._load_dataset(data)
            loader = self._make_loader_from_dataset(
                dataset,
                shuffle=False,
                batch_size=batch_size or self.batch_size,
            )
            loader = self.accelerator.prepare(loader)

        # Run inference
        self.task.model.eval()
        outputs = []

        with torch.no_grad():
            for batch in loader:
                out = self.task.eval_step(batch)
                # Gather from all processes
                gathered = {k: self.accelerator.gather(v) for k, v in out.items()}
                outputs.append(gathered)

        return outputs

    def export(
        self,
        path: str,
        format: str = "pytorch",  # pytorch, onnx, torchscript
        input_shape: tuple[int, ...] | None = None,
    ) -> str:
        """Export model for deployment.

        Args:
            path: Output path (extension auto-added if missing)
            format: Export format (pytorch, onnx, torchscript)
            input_shape: Input shape for tracing (required for onnx/torchscript)

        Returns:
            Path to exported model
        """
        import logging

        model = self.accelerator.unwrap_model(self.task.model)
        model.eval()

        path = Path(path)

        if format == "pytorch":
            path = path.with_suffix(".pt")
            torch.save(model.state_dict(), path)
            logging.info(f"Exported PyTorch state_dict to {path}")

        elif format == "onnx":
            if input_shape is None:
                raise ValueError("input_shape required for ONNX export")

            path = path.with_suffix(".onnx")
            dummy_input = torch.randn(1, *input_shape, device=next(model.parameters()).device)

            torch.onnx.export(
                model,
                dummy_input,
                str(path),
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
                opset_version=17,
            )
            logging.info(f"Exported ONNX model to {path}")

        elif format == "torchscript":
            if input_shape is None:
                raise ValueError("input_shape required for TorchScript export")

            path = path.with_suffix(".pt")
            dummy_input = torch.randn(1, *input_shape, device=next(model.parameters()).device)

            traced = torch.jit.trace(model, dummy_input)
            traced.save(str(path))
            logging.info(f"Exported TorchScript model to {path}")

        else:
            raise ValueError(f"Unknown format: {format}")

        return str(path)

    def _config_dict(self) -> dict:
        """Get config as dict for logging."""
        return self._ctx.config.to_wandb_config()
```

### 4. PipelineConfig

Config registry with source tracking for proper logging and checkpointing:

```python
# context.py
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

class ConfigSource(Enum):
    USER = auto()      # Explicitly set by user
    DEFAULT = auto()   # Dataclass default
    DATA = auto()      # Derived from dataset
    DERIVED = auto()   # Computed from other values
    ENV = auto()       # Environment/system


@dataclass
class ConfigValue:
    """A config value with provenance tracking."""
    value: Any
    source: ConfigSource
    depends_on: tuple[str, ...] = ()
    filterable: bool = False      # Mark as wandb filter dimension
    checkpoint: bool = True       # Include in checkpoint


class PipelineConfig:
    """
    Central config registry with dependency tracking.

    Accumulates values throughout pipeline lifecycle,
    tracks provenance, and handles serialization for
    logging and checkpointing.
    """

    def __init__(self):
        self._values: dict[str, ConfigValue] = {}

    # ─── Setters with source tracking ───────────────────────────

    def set_user(self, key: str, value: Any, filterable: bool = False):
        """User-specified value."""
        self._values[key] = ConfigValue(value, ConfigSource.USER, filterable=filterable)

    def set_default(self, key: str, value: Any):
        """Default value (only if not already set)."""
        if key not in self._values:
            self._values[key] = ConfigValue(value, ConfigSource.DEFAULT)

    def set_from_data(self, key: str, value: Any, filterable: bool = False):
        """Value derived from dataset."""
        self._values[key] = ConfigValue(value, ConfigSource.DATA, filterable=filterable)

    def set_derived(self, key: str, value: Any, depends_on: tuple[str, ...], filterable: bool = False):
        """Computed value with dependency tracking."""
        self._values[key] = ConfigValue(value, ConfigSource.DERIVED, depends_on, filterable)

    def set_env(self, key: str, value: Any):
        """Environment/system value."""
        self._values[key] = ConfigValue(value, ConfigSource.ENV, checkpoint=False)

    # ─── Getters ────────────────────────────────────────────────

    def get(self, key: str, default: Any = None) -> Any:
        return self._values[key].value if key in self._values else default

    def __getitem__(self, key: str) -> Any:
        return self._values[key].value

    # ─── Serialization ──────────────────────────────────────────

    def to_wandb_config(self) -> dict[str, Any]:
        """Config for wandb.init(config=...)."""
        return {k: v.value for k, v in self._values.items()}

    def get_filter_dimensions(self) -> dict[str, Any]:
        """Values suitable for wandb filtering/grouping."""
        return {k: v.value for k, v in self._values.items() if v.filterable}

    def to_checkpoint(self) -> dict[str, Any]:
        """Config needed to restore training state."""
        return {k: v.value for k, v in self._values.items() if v.checkpoint}

    def user_overrides(self) -> dict[str, Any]:
        """Only user-specified values (for reproducing experiment)."""
        return {k: v.value for k, v in self._values.items() if v.source == ConfigSource.USER}
```

### 5. PipelineContext

Unified context for runtime state, buffers, and events:

```python
# context.py (continued)
from torch import Tensor


class Lifecycle(Enum):
    STEP = auto()      # Cleared every step
    EPOCH = auto()     # Cleared every epoch
    RUN = auto()       # Persists for entire run
    CACHED = auto()    # Explicit invalidation


@dataclass
class StateEntry:
    value: Any
    lifecycle: Lifecycle
    owner: str


class PipelineContext:
    """
    Unified context for all pipeline values.

    Separates:
    - Config: logged to wandb, filterable
    - State: runtime values, not logged but checkpointed
    - Buffers: large tensors with memory management
    - Events: signals between components
    """

    def __init__(self):
        self.config = PipelineConfig()
        self._state: dict[str, StateEntry] = {}
        self._buffers: dict[str, Tensor] = {}
        self._buffer_lifecycle: dict[str, Lifecycle] = {}
        self._subscribers: dict[str, list[Callable]] = {}

    # ─── State API ──────────────────────────────────────────────

    def set(self, key: str, value: Any, lifecycle: Lifecycle = Lifecycle.RUN, owner: str = "trainer"):
        """Set runtime state (NOT logged to wandb)."""
        self._state[key] = StateEntry(value, lifecycle, owner)

    def get(self, key: str, default: Any = None) -> Any:
        """Get runtime state."""
        return self._state[key].value if key in self._state else default

    # ─── Buffer API (large tensors) ─────────────────────────────

    def register_buffer(self, key: str, tensor: Tensor, lifecycle: Lifecycle = Lifecycle.RUN):
        """Register a large tensor buffer."""
        self._buffers[key] = tensor
        self._buffer_lifecycle[key] = lifecycle

    def get_buffer(self, key: str) -> Tensor | None:
        return self._buffers.get(key)

    def accumulate(self, key: str, tensor: Tensor, dim: int = 0, max_size: int | None = None):
        """Accumulate into buffer with optional size limit.

        Args:
            key: Buffer key
            tensor: Tensor to append
            dim: Dimension to concatenate along
            max_size: Maximum number of items. If exceeded, oldest are dropped.
        """
        tensor = tensor.detach()
        if key in self._buffers:
            self._buffers[key] = torch.cat([self._buffers[key], tensor], dim=dim)
            # Enforce max size by dropping oldest
            if max_size is not None and self._buffers[key].size(dim) > max_size:
                self._buffers[key] = self._buffers[key].narrow(
                    dim, self._buffers[key].size(dim) - max_size, max_size
                )
        else:
            self._buffers[key] = tensor
            self._buffer_lifecycle[key] = Lifecycle.EPOCH

    # ─── Event API ──────────────────────────────────────────────

    def subscribe(self, event: str, callback: Callable[[dict], None]):
        """Subscribe to pipeline events."""
        if event not in self._subscribers:
            self._subscribers[event] = []
        self._subscribers[event].append(callback)

    def publish(self, event: str, data: dict[str, Any] | None = None):
        """Publish an event to all subscribers."""
        for callback in self._subscribers.get(event, []):
            callback(data or {})

    # ─── Lifecycle Management ───────────────────────────────────

    def on_step_end(self):
        """Clear STEP-scoped values."""
        self._clear_lifecycle(Lifecycle.STEP)

    def on_epoch_end(self):
        """Clear EPOCH-scoped values."""
        self._clear_lifecycle(Lifecycle.EPOCH)
        self.publish("epoch_end")

    def _clear_lifecycle(self, lifecycle: Lifecycle):
        # Clear state
        to_remove = [k for k, v in self._state.items() if v.lifecycle == lifecycle]
        for k in to_remove:
            del self._state[k]
        # Clear buffers
        buf_remove = [k for k, lc in self._buffer_lifecycle.items() if lc == lifecycle]
        for k in buf_remove:
            del self._buffers[k]
            del self._buffer_lifecycle[k]

    # ─── Scoped Access ──────────────────────────────────────────

    def scope(self, namespace: str) -> "ScopedContext":
        """Get namespaced access for a component."""
        return ScopedContext(self, namespace)

    # ─── Serialization ──────────────────────────────────────────

    def state_dict(self) -> dict:
        """Full checkpoint state."""
        return {
            "config": self.config.to_checkpoint(),
            "state": {k: v.value for k, v in self._state.items() if v.lifecycle == Lifecycle.RUN},
            "buffers": {k: v.cpu() for k, v in self._buffers.items() if self._buffer_lifecycle[k] == Lifecycle.RUN},
        }

    def load_state_dict(self, checkpoint: dict):
        """Restore from checkpoint."""
        for k, v in checkpoint.get("state", {}).items():
            self.set(k, v, Lifecycle.RUN)
        for k, v in checkpoint.get("buffers", {}).items():
            self.register_buffer(k, v, Lifecycle.RUN)


class ScopedContext:
    """Namespaced view into PipelineContext for a component."""

    def __init__(self, parent: PipelineContext, namespace: str):
        self._parent = parent
        self._ns = namespace

    def _key(self, key: str) -> str:
        return f"{self._ns}/{key}"

    def set(self, key: str, value: Any, lifecycle: Lifecycle = Lifecycle.RUN):
        self._parent.set(self._key(key), value, lifecycle, self._ns)

    def get(self, key: str, default: Any = None) -> Any:
        return self._parent.get(self._key(key), default)

    def buffer(self, key: str, tensor: Tensor, lifecycle: Lifecycle = Lifecycle.EPOCH):
        self._parent.register_buffer(self._key(key), tensor, lifecycle)

    def get_buffer(self, key: str) -> Tensor | None:
        return self._parent.get_buffer(self._key(key))

    def accumulate(self, key: str, tensor: Tensor):
        self._parent.accumulate(self._key(key), tensor)

    def subscribe(self, event: str, callback: Callable):
        self._parent.subscribe(event, callback)

    @property
    def config(self) -> PipelineConfig:
        return self._parent.config
```

### 6. Callback Protocol

Hook-based callback system for extensibility:

```python
# callbacks/base.py
from dataclasses import dataclass
from typing import Protocol, Any
from torch import Tensor

class Callback(Protocol):
    """Hook-based callback protocol."""

    def on_train_begin(self, trainer: "Trainer") -> None:
        """Called before training starts."""
        ...

    def on_train_end(self, trainer: "Trainer") -> None:
        """Called after training ends."""
        ...

    def on_epoch_begin(self, trainer: "Trainer", epoch: int) -> None:
        """Called at the start of each epoch."""
        ...

    def on_epoch_end(self, trainer: "Trainer", epoch: int, metrics: dict[str, float]) -> None:
        """Called at the end of each epoch with validation metrics."""
        ...

    def on_step_begin(self, trainer: "Trainer", step: int) -> None:
        """Called before each training step."""
        ...

    def on_step_end(self, trainer: "Trainer", step: int, loss: float) -> None:
        """Called after each training step."""
        ...

    def on_validation_end(self, trainer: "Trainer", metrics: dict[str, float]) -> None:
        """Called after validation with computed metrics."""
        ...

    def state_dict(self) -> dict:
        """Callback state for checkpointing."""
        ...

    def load_state_dict(self, state: dict) -> None:
        """Restore callback state."""
        ...


class CallbackList:
    """Manages multiple callbacks, calls hooks in order."""

    def __init__(self, callbacks: list[Callback] | None = None):
        self.callbacks = callbacks or []

    def add(self, callback: Callback) -> None:
        self.callbacks.append(callback)

    def on_train_begin(self, trainer: "Trainer") -> None:
        for cb in self.callbacks:
            if hasattr(cb, "on_train_begin"):
                cb.on_train_begin(trainer)

    def on_train_end(self, trainer: "Trainer") -> None:
        for cb in self.callbacks:
            if hasattr(cb, "on_train_end"):
                cb.on_train_end(trainer)

    def on_epoch_begin(self, trainer: "Trainer", epoch: int) -> None:
        for cb in self.callbacks:
            if hasattr(cb, "on_epoch_begin"):
                cb.on_epoch_begin(trainer, epoch)

    def on_epoch_end(self, trainer: "Trainer", epoch: int, metrics: dict[str, float]) -> None:
        for cb in self.callbacks:
            if hasattr(cb, "on_epoch_end"):
                cb.on_epoch_end(trainer, epoch, metrics)

    def on_step_begin(self, trainer: "Trainer", step: int) -> None:
        for cb in self.callbacks:
            if hasattr(cb, "on_step_begin"):
                cb.on_step_begin(trainer, step)

    def on_step_end(self, trainer: "Trainer", step: int, loss: float) -> None:
        for cb in self.callbacks:
            if hasattr(cb, "on_step_end"):
                cb.on_step_end(trainer, step, loss)

    def on_validation_end(self, trainer: "Trainer", metrics: dict[str, float]) -> None:
        for cb in self.callbacks:
            if hasattr(cb, "on_validation_end"):
                cb.on_validation_end(trainer, metrics)

    def state_dict(self) -> dict:
        return {
            type(cb).__name__: cb.state_dict()
            for cb in self.callbacks
            if hasattr(cb, "state_dict")
        }

    def load_state_dict(self, state: dict) -> None:
        for cb in self.callbacks:
            name = type(cb).__name__
            if name in state and hasattr(cb, "load_state_dict"):
                cb.load_state_dict(state[name])
```

### 7. Built-in Callbacks

```python
# callbacks/early_stopping.py
from dataclasses import dataclass, field
import logging

@dataclass
class EarlyStopping:
    """Stop training when metric stops improving or diverges."""

    monitor: str = "val/loss"
    patience: int = 5
    mode: str = "min"  # min or max
    min_delta: float = 0.0
    divergence_threshold: float | None = None  # Stop if metric exceeds this

    _best: float = field(init=False, default=None)
    _counter: int = field(init=False, default=0)
    _should_stop: bool = field(init=False, default=False)

    def __post_init__(self):
        self._best = float("inf") if self.mode == "min" else float("-inf")

    def on_validation_end(self, trainer: "Trainer", metrics: dict[str, float]) -> None:
        value = metrics.get(self.monitor)
        if value is None:
            logging.warning(f"EarlyStopping: metric '{self.monitor}' not found")
            return

        # Check for divergence (NaN or explosion)
        if self.divergence_threshold is not None:
            if value != value or value > self.divergence_threshold:  # NaN check
                logging.warning(f"EarlyStopping: divergence detected ({value})")
                self._should_stop = True
                trainer._should_stop = True
                return

        # Check for improvement
        improved = (
            (self.mode == "min" and value < self._best - self.min_delta) or
            (self.mode == "max" and value > self._best + self.min_delta)
        )

        if improved:
            self._best = value
            self._counter = 0
        else:
            self._counter += 1
            if self._counter >= self.patience:
                logging.info(f"EarlyStopping: no improvement for {self.patience} epochs")
                self._should_stop = True
                trainer._should_stop = True

    def state_dict(self) -> dict:
        return {"best": self._best, "counter": self._counter}

    def load_state_dict(self, state: dict) -> None:
        self._best = state["best"]
        self._counter = state["counter"]


# callbacks/checkpoint.py
from dataclasses import dataclass, field
from pathlib import Path
import logging

@dataclass
class ModelCheckpoint:
    """Save best models by metric, manage checkpoint files."""

    monitor: str = "val/loss"
    mode: str = "min"
    save_top_k: int = 3
    save_dir: str = "checkpoints"
    filename: str = "epoch_{epoch}_{monitor}_{value:.4f}.pt"

    _best_scores: list[tuple[float, Path]] = field(init=False, default_factory=list)

    def on_validation_end(self, trainer: "Trainer", metrics: dict[str, float]) -> None:
        value = metrics.get(self.monitor)
        if value is None:
            return

        # Determine if this is a top-k score
        is_better = len(self._best_scores) < self.save_top_k or (
            (self.mode == "min" and value < self._best_scores[-1][0]) or
            (self.mode == "max" and value > self._best_scores[-1][0])
        )

        if is_better:
            # Save checkpoint
            save_dir = Path(self.save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

            filename = self.filename.format(
                epoch=trainer._epoch,
                monitor=self.monitor.replace("/", "_"),
                value=value,
            )
            path = save_dir / filename
            trainer.save(str(path))
            logging.info(f"ModelCheckpoint: saved {path}")

            # Update best scores
            self._best_scores.append((value, path))
            self._best_scores.sort(key=lambda x: x[0], reverse=(self.mode == "max"))

            # Prune old checkpoints
            while len(self._best_scores) > self.save_top_k:
                _, old_path = self._best_scores.pop()
                if old_path.exists():
                    old_path.unlink()
                    logging.info(f"ModelCheckpoint: pruned {old_path}")

    def state_dict(self) -> dict:
        return {"best_scores": [(s, str(p)) for s, p in self._best_scores]}

    def load_state_dict(self, state: dict) -> None:
        self._best_scores = [(s, Path(p)) for s, p in state["best_scores"]]


# callbacks/monitors.py
from dataclasses import dataclass
import torch

@dataclass
class LRMonitor:
    """Log learning rate to logger."""

    def on_step_end(self, trainer: "Trainer", step: int, loss: float) -> None:
        if step % trainer.log_every == 0:
            lr = trainer._scheduler.get_last_lr()[0]
            trainer._logger.log({"train/lr": lr}, step)


@dataclass
class GradientMonitor:
    """Track gradient statistics, detect explosions."""

    log_every: int = 100
    clip_percentile: float = 99.0  # Log gradient at this percentile
    explosion_threshold: float = 100.0  # Warn if grad norm exceeds this

    def on_step_end(self, trainer: "Trainer", step: int, loss: float) -> None:
        if step % self.log_every != 0:
            return

        # Compute gradient norm
        total_norm = 0.0
        for p in trainer.task.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5

        trainer._logger.log({"train/grad_norm": total_norm}, step)

        if total_norm > self.explosion_threshold:
            logging.warning(f"GradientMonitor: gradient explosion detected ({total_norm:.2f})")


# callbacks/lr_finder.py
from dataclasses import dataclass, field
import math
import logging
import torch

@dataclass
class LRFinder:
    """Learning rate range test. Run before training to find optimal LR."""

    min_lr: float = 1e-7
    max_lr: float = 10.0
    num_steps: int = 100
    smoothing: float = 0.05
    divergence_factor: float = 4.0

    _losses: list[float] = field(init=False, default_factory=list)
    _lrs: list[float] = field(init=False, default_factory=list)
    _best_loss: float = field(init=False, default=float("inf"))

    def on_train_begin(self, trainer: "Trainer") -> None:
        """Run LR finder before actual training."""
        logging.info("LRFinder: running learning rate range test")

        # Save state
        model_state = trainer.task.model.state_dict()
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

        for step in range(self.num_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(trainer._train_loader)
                batch = next(data_iter)

            # Forward + backward
            trainer._optimizer.zero_grad()
            loss = trainer.task.train_step(batch)
            trainer.accelerator.backward(loss)
            trainer._optimizer.step()

            # Track
            loss_val = loss.item()
            lr = trainer._optimizer.param_groups[0]["lr"]
            self._losses.append(loss_val)
            self._lrs.append(lr)

            # Smoothed loss for divergence check
            smoothed_loss = self.smoothing * loss_val + (1 - self.smoothing) * smoothed_loss if step > 0 else loss_val

            if smoothed_loss < self._best_loss:
                self._best_loss = smoothed_loss

            # Check for divergence
            if smoothed_loss > self._best_loss * self.divergence_factor:
                logging.info(f"LRFinder: stopping early due to divergence at lr={lr:.2e}")
                break

            # Update LR
            for pg in trainer._optimizer.param_groups:
                pg["lr"] *= gamma

        # Find suggested LR (steepest descent point)
        suggested_lr = self._find_suggested_lr()
        logging.info(f"LRFinder: suggested LR = {suggested_lr:.2e}")

        # Store in context
        trainer._ctx.config.set_derived("suggested_lr", suggested_lr, ("lr_finder",))

        # Restore state
        trainer.task.model.load_state_dict(model_state)
        trainer._optimizer.load_state_dict(optimizer_state)

        # Update trainer LR if user wants
        if trainer.lr == 1e-4:  # Default LR, probably not set
            logging.info(f"LRFinder: updating trainer.lr to {suggested_lr:.2e}")
            trainer.lr = suggested_lr
            for pg in trainer._optimizer.param_groups:
                pg["lr"] = suggested_lr

    def _find_suggested_lr(self) -> float:
        """Find LR with steepest negative gradient."""
        if len(self._losses) < 10:
            return self.min_lr * 10

        # Compute smoothed gradient
        gradients = []
        for i in range(1, len(self._losses)):
            grad = (self._losses[i] - self._losses[i-1]) / (math.log(self._lrs[i]) - math.log(self._lrs[i-1]))
            gradients.append(grad)

        # Find steepest descent
        min_grad_idx = min(range(len(gradients)), key=lambda i: gradients[i])

        # Return LR slightly before steepest descent
        return self._lrs[max(0, min_grad_idx - 1)]
```

### 8. Logger Classes

```python
# utils/logging.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import logging

class BaseLogger(ABC):
    """Base logger interface."""

    @abstractmethod
    def init(self, config: dict[str, Any]) -> None:
        """Initialize logging with config."""
        ...

    @abstractmethod
    def log(self, metrics: dict[str, float], step: int) -> None:
        """Log metrics at step."""
        ...

    @abstractmethod
    def finish(self) -> None:
        """Finalize logging."""
        ...


@dataclass
class WandbLogger(BaseLogger):
    """Weights & Biases logger."""

    project: str = "trainformer"
    name: str | None = None
    entity: str | None = None
    tags: list[str] | None = None

    _run: Any = field(init=False, default=None)

    def init(self, config: dict[str, Any]) -> None:
        import wandb
        self._run = wandb.init(
            project=self.project,
            name=self.name,
            entity=self.entity,
            tags=self.tags,
            config=config,
        )

    def log(self, metrics: dict[str, float], step: int) -> None:
        if self._run:
            self._run.log(metrics, step=step)

    def finish(self) -> None:
        if self._run:
            self._run.finish()


@dataclass
class ConsoleLogger(BaseLogger):
    """Simple console logger with progress bar."""

    log_every: int = 100

    def init(self, config: dict[str, Any]) -> None:
        logging.info(f"Training config: {config}")

    def log(self, metrics: dict[str, float], step: int) -> None:
        parts = [f"{k}={v:.4f}" for k, v in metrics.items()]
        logging.info(f"[step {step}] {' '.join(parts)}")

    def finish(self) -> None:
        logging.info("Training complete")


@dataclass
class TensorBoardLogger(BaseLogger):
    """TensorBoard logger."""

    log_dir: str = "runs"

    _writer: Any = field(init=False, default=None)

    def init(self, config: dict[str, Any]) -> None:
        from torch.utils.tensorboard import SummaryWriter
        self._writer = SummaryWriter(log_dir=self.log_dir)
        # Log config as text
        self._writer.add_text("config", str(config))

    def log(self, metrics: dict[str, float], step: int) -> None:
        if self._writer:
            for k, v in metrics.items():
                self._writer.add_scalar(k, v, step)

    def finish(self) -> None:
        if self._writer:
            self._writer.close()


@dataclass
class MLflowLogger(BaseLogger):
    """MLflow logger."""

    experiment_name: str = "trainformer"
    run_name: str | None = None
    tracking_uri: str | None = None

    def init(self, config: dict[str, Any]) -> None:
        import mlflow
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        mlflow.start_run(run_name=self.run_name)
        mlflow.log_params(config)

    def log(self, metrics: dict[str, float], step: int) -> None:
        import mlflow
        mlflow.log_metrics(metrics, step=step)

    def finish(self) -> None:
        import mlflow
        mlflow.end_run()


@dataclass
class MultiLogger(BaseLogger):
    """Composable logger that wraps multiple backends."""

    loggers: list[BaseLogger] = field(default_factory=list)

    def init(self, config: dict[str, Any]) -> None:
        for logger in self.loggers:
            logger.init(config)

    def log(self, metrics: dict[str, float], step: int) -> None:
        for logger in self.loggers:
            logger.log(metrics, step)

    def finish(self) -> None:
        for logger in self.loggers:
            logger.finish()


class NoOpLogger(BaseLogger):
    """No-op logger for testing or silent runs."""

    def init(self, config: dict[str, Any]) -> None:
        pass

    def log(self, metrics: dict[str, float], step: int) -> None:
        pass

    def finish(self) -> None:
        pass
```

### 9. Sampler Classes

```python
# data/samplers.py
from collections import defaultdict
import random
from typing import Iterator
from torch.utils.data import Sampler, Dataset

class ClassBalancedSampler(Sampler[int]):
    """Oversample minority classes for balanced training."""

    def __init__(self, dataset: Dataset, num_samples: int | None = None):
        self.dataset = dataset
        self.num_samples = num_samples or len(dataset)

        # Build class -> indices mapping
        self.class_indices: dict[int, list[int]] = defaultdict(list)
        for idx in range(len(dataset)):
            _, label = dataset[idx]
            self.class_indices[label].append(idx)

        self.num_classes = len(self.class_indices)

    def __iter__(self) -> Iterator[int]:
        # Sample equally from each class
        samples_per_class = self.num_samples // self.num_classes

        indices = []
        for class_idx in self.class_indices.values():
            # Oversample if needed
            sampled = random.choices(class_idx, k=samples_per_class)
            indices.extend(sampled)

        random.shuffle(indices)
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples


class PKSampler(Sampler[int]):
    """P classes, K samples per class per batch (for metric learning)."""

    def __init__(self, dataset: Dataset, p: int = 8, k: int = 4):
        self.dataset = dataset
        self.p = p  # Classes per batch
        self.k = k  # Samples per class

        # Build class -> indices mapping
        self.class_indices: dict[int, list[int]] = defaultdict(list)
        for idx in range(len(dataset)):
            _, label = dataset[idx]
            self.class_indices[label].append(idx)

        self.classes = list(self.class_indices.keys())
        self.num_batches = len(dataset) // (p * k)

    def __iter__(self) -> Iterator[int]:
        for _ in range(self.num_batches):
            # Sample P classes
            batch_classes = random.sample(self.classes, min(self.p, len(self.classes)))

            # Sample K instances per class
            batch_indices = []
            for cls in batch_classes:
                cls_indices = self.class_indices[cls]
                sampled = random.choices(cls_indices, k=self.k)
                batch_indices.extend(sampled)

            yield from batch_indices

    def __len__(self) -> int:
        return self.num_batches * self.p * self.k
```

### 10. Data Utilities

```python
# data/datasets.py
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
from torch.utils.data import Dataset
from PIL import Image

@dataclass
class JSONLDataset(Dataset):
    """Load data from JSONL file."""

    path: str
    text_key: str = "text"
    label_key: str | None = None

    def __post_init__(self):
        self.data = []
        with open(self.path) as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.data[idx]
        result = {"text": item[self.text_key]}
        if self.label_key and self.label_key in item:
            result["label"] = item[self.label_key]
        return result


@dataclass
class TextFileDataset(Dataset):
    """Load text data from file (one sample per line or chunked)."""

    path: str
    chunk_size: int | None = None  # If set, chunk into fixed-size sequences

    def __post_init__(self):
        with open(self.path) as f:
            text = f.read()

        if self.chunk_size:
            # Chunk into fixed-size pieces
            self.samples = [
                text[i:i + self.chunk_size]
                for i in range(0, len(text), self.chunk_size)
            ]
        else:
            # One sample per line
            self.samples = text.strip().split("\n")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> str:
        return self.samples[idx]


@dataclass
class ImageTextDataset(Dataset):
    """Load image-text pairs from various formats."""

    path: str
    image_key: str = "image"
    text_key: str = "text"
    image_dir: str | None = None

    def __post_init__(self):
        self.data = []
        path = Path(self.path)

        if path.suffix == ".json":
            with open(path) as f:
                self.data = json.load(f)
        elif path.suffix == ".jsonl":
            with open(path) as f:
                self.data = [json.loads(line) for line in f]
        elif path.is_dir():
            # Assume directory with images and captions.json
            captions_file = path / "captions.json"
            with open(captions_file) as f:
                self.data = json.load(f)
            self.image_dir = str(path)

        self.image_dir = self.image_dir or str(path.parent)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.data[idx]
        image_path = Path(self.image_dir) / item[self.image_key]
        image = Image.open(image_path).convert("RGB")
        return {"image": image, "text": item[self.text_key]}
```

### 11. Utility Functions

```python
# utils/helpers.py
import os
import random
import logging
from contextlib import contextmanager
from typing import Iterator
import torch
import numpy as np

def seed_everything(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For deterministic behavior (may impact performance)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logging.info(f"Set random seed to {seed}")


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> str:
    """Count model parameters with nice formatting."""
    if trainable_only:
        total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        total = sum(p.numel() for p in model.parameters())

    if total >= 1e9:
        return f"{total / 1e9:.2f}B"
    elif total >= 1e6:
        return f"{total / 1e6:.2f}M"
    elif total >= 1e3:
        return f"{total / 1e3:.2f}K"
    return str(total)


@contextmanager
def profile_memory(label: str = "Operation") -> Iterator[None]:
    """Context manager to track peak GPU memory usage."""
    if not torch.cuda.is_available():
        yield
        return

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start_mem = torch.cuda.memory_allocated()

    yield

    torch.cuda.synchronize()
    end_mem = torch.cuda.memory_allocated()
    peak_mem = torch.cuda.max_memory_allocated()

    delta = (end_mem - start_mem) / 1e9
    peak = peak_mem / 1e9
    logging.info(f"{label}: delta={delta:.2f}GB, peak={peak:.2f}GB")


def get_device() -> torch.device:
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def move_to_device(batch: dict | tuple | list, device: torch.device) -> dict | tuple | list:
    """Recursively move batch to device."""
    if isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, (tuple, list)):
        return type(batch)(move_to_device(v, device) for v in batch)
    elif isinstance(batch, torch.Tensor):
        return batch.to(device)
    return batch
```

---

## Tasks

### Vision: Image Classification

```python
# tasks/vision/classification.py
from dataclasses import dataclass, field
import timm
import torch.nn as nn
from torch import Tensor

@dataclass
class ImageClassification:
    """Image classification with any timm backbone."""

    backbone: str = "resnet50"
    num_classes: int | None = None  # Inferred from data if None
    pretrained: bool = True

    model: nn.Module = field(init=False)
    _loss: nn.Module = field(init=False)

    def __post_init__(self):
        self.model = timm.create_model(
            self.backbone,
            pretrained=self.pretrained,
            num_classes=self.num_classes or 0,
        )
        self._loss = nn.CrossEntropyLoss()

    def train_step(self, batch: tuple[Tensor, Tensor]) -> Tensor:
        x, y = batch
        logits = self.model(x)
        return self._loss(logits, y)

    def eval_step(self, batch: tuple[Tensor, Tensor]) -> dict[str, Tensor]:
        x, y = batch
        logits = self.model(x)
        loss = self._loss(logits, y)
        acc = (logits.argmax(-1) == y).float().mean()
        return {"loss": loss, "accuracy": acc}

    def load_data(self, path: str):
        from torchvision.datasets import ImageFolder
        from torchvision import transforms as T

        transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return ImageFolder(path, transform=transform)
```

### Vision: Metric Learning

```python
# tasks/vision/metric_learning.py
from dataclasses import dataclass, field
from typing import Iterator, Any
import logging
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

@dataclass
class MetricLearning:
    """Metric learning with ArcFace, CosFace, etc."""

    backbone: str = "efficientnet_b0"
    embedding_dim: int = 512
    loss: str = "arcface"  # arcface, cosface, subcenter
    margin: float = 0.5
    scale: float = 64.0
    pretrained: bool = True

    model: nn.Module = field(init=False)
    _loss: nn.Module | None = field(init=False, default=None)
    _num_classes: int = field(init=False, default=0)
    _class_names: list[str] | None = field(init=False, default=None)

    def __post_init__(self):
        backbone = timm.create_model(self.backbone, pretrained=self.pretrained, num_classes=0)
        self.model = EmbeddingModel(backbone, self.embedding_dim)
        # Loss created in configure() after we know num_classes

    def configure(self, info: DatasetInfo) -> None:
        """Configure task with dataset metadata."""
        if info.num_classes is None:
            raise ValueError(
                f"MetricLearning requires num_classes but dataset has none. "
                f"Ensure dataset has 'classes' or 'num_classes' attribute."
            )

        self._num_classes = info.num_classes
        self._class_names = info.class_names
        self._loss = self._make_loss()

        logging.info(f"MetricLearning: initialized {self.loss} loss with {self._num_classes} classes")

    def _make_loss(self) -> nn.Module:
        if self.loss == "arcface":
            return ArcFaceLoss(self.embedding_dim, self._num_classes, self.margin, self.scale)
        elif self.loss == "cosface":
            return CosFaceLoss(self.embedding_dim, self._num_classes, self.margin, self.scale)
        elif self.loss == "subcenter":
            return SubCenterArcFaceLoss(self.embedding_dim, self._num_classes, self.margin, self.scale)
        else:
            raise ValueError(f"Unknown loss: {self.loss}")

    def parameters(self) -> Iterator[nn.Parameter]:
        """Yield model AND loss parameters for optimizer."""
        yield from self.model.parameters()
        if self._loss is not None:
            yield from self._loss.parameters()

    def config_dict(self) -> dict[str, Any]:
        """Task config for logging."""
        return {
            "backbone": self.backbone,
            "embedding_dim": self.embedding_dim,
            "loss_type": self.loss,
            "margin": self.margin,
            "scale": self.scale,
            "num_classes": self._num_classes,
            "num_params": sum(p.numel() for p in self.model.parameters()),
            "trainable_params": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }

    def train_step(self, batch: tuple[Tensor, Tensor]) -> Tensor:
        if self._loss is None:
            raise RuntimeError("Loss not initialized. Trainer must call configure() before training.")
        x, y = batch
        embeddings = self.model(x)
        return self._loss(embeddings, y)

    def eval_step(self, batch: tuple[Tensor, Tensor]) -> dict[str, Tensor]:
        x, y = batch
        embeddings = self.model(x)
        return {"embeddings": embeddings, "labels": y}

    def load_data(self, path: str):
        from torchvision.datasets import ImageFolder
        from torchvision import transforms as T

        transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return ImageFolder(path, transform=transform)

    def evaluate_retrieval(self, embeddings: Tensor, labels: Tensor) -> dict[str, float]:
        """Full retrieval evaluation with KNN and FAISS."""
        from trainformer.eval.retrieval import knn_accuracy, faiss_retrieval_metrics

        knn_acc = knn_accuracy(embeddings, labels, k=5)
        retrieval = faiss_retrieval_metrics(embeddings, labels)

        return {"knn_accuracy": knn_acc, **retrieval}


class EmbeddingModel(nn.Module):
    """Backbone + projection to embedding space."""

    def __init__(self, backbone: nn.Module, embedding_dim: int):
        super().__init__()
        self.backbone = backbone
        self.proj = nn.Linear(backbone.num_features, embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        features = self.backbone(x)
        embeddings = self.proj(features)
        return F.normalize(embeddings, dim=-1)


class ArcFaceLoss(nn.Module):
    """ArcFace loss with learnable class centers."""

    def __init__(self, embedding_dim: int, num_classes: int, margin: float = 0.5, scale: float = 64.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.margin = margin
        self.scale = scale
        self.ce = nn.CrossEntropyLoss()

    def forward(self, embeddings: Tensor, labels: Tensor) -> Tensor:
        # Cosine similarity
        cosine = F.linear(embeddings, F.normalize(self.weight))

        # Add angular margin to target class
        theta = torch.acos(cosine.clamp(-1 + 1e-7, 1 - 1e-7))
        target_logits = torch.cos(theta + self.margin)

        # Replace target class logits
        one_hot = F.one_hot(labels, num_classes=cosine.size(1)).float()
        logits = cosine * (1 - one_hot) + target_logits * one_hot

        return self.ce(self.scale * logits, labels)
```

### Vision: Self-Supervised Learning

```python
# tasks/vision/ssl.py
from dataclasses import dataclass, field
import copy
import math
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

@dataclass
class SSL:
    """Self-supervised learning tasks."""

    method: str  # simclr, moco, dino, mae, byol
    backbone: str = "resnet50"
    pretrained: bool = False  # Usually train from scratch for SSL

    # Method-specific params
    proj_dim: int = 128
    temperature: float = 0.1
    momentum: float = 0.996

    model: nn.Module = field(init=False)

    def __post_init__(self):
        if self.method == "simclr":
            self.model = SimCLR(self.backbone, self.proj_dim, self.temperature, self.pretrained)
        elif self.method == "moco":
            self.model = MoCo(self.backbone, self.proj_dim, self.temperature, self.momentum, self.pretrained)
        elif self.method == "dino":
            self.model = DINO(self.backbone, self.momentum, self.pretrained)
        elif self.method == "mae":
            self.model = MAE(self.backbone, self.pretrained)
        elif self.method == "byol":
            self.model = BYOL(self.backbone, self.proj_dim, self.momentum, self.pretrained)
        else:
            raise ValueError(f"Unknown SSL method: {self.method}")

    # Factory methods for convenience
    @classmethod
    def simclr(cls, backbone: str = "resnet50", **kw) -> "SSL":
        return cls(method="simclr", backbone=backbone, **kw)

    @classmethod
    def moco(cls, backbone: str = "resnet50", **kw) -> "SSL":
        return cls(method="moco", backbone=backbone, **kw)

    @classmethod
    def dino(cls, backbone: str = "vit_small_patch16_224", **kw) -> "SSL":
        return cls(method="dino", backbone=backbone, **kw)

    @classmethod
    def mae(cls, backbone: str = "vit_base_patch16_224", **kw) -> "SSL":
        return cls(method="mae", backbone=backbone, **kw)

    @classmethod
    def byol(cls, backbone: str = "resnet50", **kw) -> "SSL":
        return cls(method="byol", backbone=backbone, **kw)

    def train_step(self, batch: tuple[Tensor, Tensor]) -> Tensor:
        x, _ = batch  # Labels unused in SSL
        return self.model.training_step(x)

    def eval_step(self, batch: tuple[Tensor, Tensor]) -> dict[str, Tensor]:
        x, y = batch
        embeddings = self.model.embed(x)
        return {"embeddings": embeddings, "labels": y}

    def on_step_end(self, step: int, max_steps: int):
        if hasattr(self.model, "on_step_end"):
            self.model.on_step_end(step, max_steps)

    def state_dict(self) -> dict:
        if hasattr(self.model, "state_dict_extra"):
            return self.model.state_dict_extra()
        return {}

    def load_state_dict(self, state: dict):
        if hasattr(self.model, "load_state_dict_extra"):
            self.model.load_state_dict_extra(state)


class SimCLR(nn.Module):
    """SimCLR: contrastive learning with two views."""

    def __init__(self, backbone: str, proj_dim: int, temperature: float, pretrained: bool):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        self.projector = nn.Sequential(
            nn.Linear(self.backbone.num_features, self.backbone.num_features),
            nn.ReLU(),
            nn.Linear(self.backbone.num_features, proj_dim),
        )
        self.temperature = temperature
        self.augment = SimCLRAugmentation()

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)

    def embed(self, x: Tensor) -> Tensor:
        return self.backbone(x)

    def training_step(self, x: Tensor) -> Tensor:
        # Generate two views
        v1, v2 = self.augment(x)

        # Project
        z1 = F.normalize(self.projector(self.backbone(v1)), dim=-1)
        z2 = F.normalize(self.projector(self.backbone(v2)), dim=-1)

        # NT-Xent loss
        return nt_xent_loss(z1, z2, self.temperature)


class MoCo(nn.Module):
    """MoCo v2: momentum contrast with queue."""

    def __init__(self, backbone: str, proj_dim: int, temperature: float, momentum: float, pretrained: bool):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        self.projector = self._build_projector(proj_dim)

        # Momentum encoder
        self.backbone_m = copy.deepcopy(self.backbone)
        self.projector_m = copy.deepcopy(self.projector)
        self._freeze(self.backbone_m)
        self._freeze(self.projector_m)

        # Queue
        self.register_buffer("queue", F.normalize(torch.randn(proj_dim, 65536), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.temperature = temperature
        self.momentum = momentum
        self.augment = SimCLRAugmentation()

    def _build_projector(self, proj_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(self.backbone.num_features, self.backbone.num_features),
            nn.ReLU(),
            nn.Linear(self.backbone.num_features, proj_dim),
        )

    def _freeze(self, module: nn.Module):
        for p in module.parameters():
            p.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)

    def embed(self, x: Tensor) -> Tensor:
        return self.backbone(x)

    def training_step(self, x: Tensor) -> Tensor:
        v1, v2 = self.augment(x)

        # Query from online encoder
        q = F.normalize(self.projector(self.backbone(v1)), dim=-1)

        # Key from momentum encoder
        with torch.no_grad():
            k = F.normalize(self.projector_m(self.backbone_m(v2)), dim=-1)

        # InfoNCE with queue
        loss = infonce_with_queue(q, k, self.queue, self.temperature)

        # Update queue
        self._enqueue(k)

        return loss

    def on_step_end(self, step: int, max_steps: int):
        # EMA update
        for p_q, p_k in zip(self.backbone.parameters(), self.backbone_m.parameters()):
            p_k.data = self.momentum * p_k.data + (1 - self.momentum) * p_q.data
        for p_q, p_k in zip(self.projector.parameters(), self.projector_m.parameters()):
            p_k.data = self.momentum * p_k.data + (1 - self.momentum) * p_q.data

    @torch.no_grad()
    def _enqueue(self, keys: Tensor):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        queue_size = self.queue.shape[1]

        if ptr + batch_size > queue_size:
            self.queue[:, ptr:] = keys[:queue_size - ptr].T
            self.queue[:, :batch_size - (queue_size - ptr)] = keys[queue_size - ptr:].T
        else:
            self.queue[:, ptr:ptr + batch_size] = keys.T

        self.queue_ptr[0] = (ptr + batch_size) % queue_size

    def state_dict_extra(self) -> dict:
        return {"queue": self.queue, "queue_ptr": self.queue_ptr}

    def load_state_dict_extra(self, state: dict):
        self.queue.copy_(state["queue"])
        self.queue_ptr.copy_(state["queue_ptr"])


class DINO(nn.Module):
    """DINO: self-distillation with no labels."""

    def __init__(self, backbone: str, momentum: float, pretrained: bool):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        self.head = DINOHead(self.backbone.num_features, 65536)

        # Teacher
        self.backbone_t = copy.deepcopy(self.backbone)
        self.head_t = copy.deepcopy(self.head)
        self._freeze(self.backbone_t)
        self._freeze(self.head_t)

        # Center for teacher outputs
        self.register_buffer("center", torch.zeros(1, 65536))

        self.momentum = momentum
        self.center_momentum = 0.9
        self.student_temp = 0.1
        self.teacher_temp = 0.04
        self.augment = MultiCropAugmentation()

    def _freeze(self, module: nn.Module):
        for p in module.parameters():
            p.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)

    def embed(self, x: Tensor) -> Tensor:
        return self.backbone(x)

    def training_step(self, x: Tensor) -> Tensor:
        views = self.augment(x)  # {"global": [g1, g2], "local": [l1, ..., l8]}

        # Student: all views
        student_out = []
        for crop in views["global"] + views.get("local", []):
            student_out.append(self.head(self.backbone(crop)))

        # Teacher: global only, with centering
        with torch.no_grad():
            teacher_out = []
            for crop in views["global"]:
                out = self.head_t(self.backbone_t(crop))
                teacher_out.append(out - self.center)

            # Update center
            teacher_mean = torch.cat(teacher_out).mean(dim=0, keepdim=True)
            self.center = self.center * self.center_momentum + teacher_mean * (1 - self.center_momentum)

        return dino_loss(student_out, teacher_out, self.student_temp, self.teacher_temp)

    def on_step_end(self, step: int, max_steps: int):
        # Cosine momentum schedule
        m = 1 - (1 - self.momentum) * (1 + math.cos(math.pi * step / max_steps)) / 2

        for p_s, p_t in zip(self.backbone.parameters(), self.backbone_t.parameters()):
            p_t.data = m * p_t.data + (1 - m) * p_s.data
        for p_s, p_t in zip(self.head.parameters(), self.head_t.parameters()):
            p_t.data = m * p_t.data + (1 - m) * p_s.data

    def state_dict_extra(self) -> dict:
        return {"center": self.center}

    def load_state_dict_extra(self, state: dict):
        self.center.copy_(state["center"])


class BYOL(nn.Module):
    """BYOL: Bootstrap Your Own Latent."""

    def __init__(self, backbone: str, proj_dim: int, momentum: float, pretrained: bool):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        self.projector = self._build_projector(proj_dim)
        self.predictor = self._build_predictor(proj_dim)

        # Target network
        self.backbone_t = copy.deepcopy(self.backbone)
        self.projector_t = copy.deepcopy(self.projector)
        self._freeze(self.backbone_t)
        self._freeze(self.projector_t)

        self.momentum = momentum
        self.augment = SimCLRAugmentation()

    def _build_projector(self, proj_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(self.backbone.num_features, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Linear(4096, proj_dim),
        )

    def _build_predictor(self, proj_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(proj_dim, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Linear(4096, proj_dim),
        )

    def _freeze(self, module: nn.Module):
        for p in module.parameters():
            p.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)

    def embed(self, x: Tensor) -> Tensor:
        return self.backbone(x)

    def training_step(self, x: Tensor) -> Tensor:
        v1, v2 = self.augment(x)

        # Online network
        z1 = self.projector(self.backbone(v1))
        z2 = self.projector(self.backbone(v2))
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        # Target network
        with torch.no_grad():
            t1 = self.projector_t(self.backbone_t(v1))
            t2 = self.projector_t(self.backbone_t(v2))

        # Symmetric loss
        loss = byol_loss(p1, t2) + byol_loss(p2, t1)
        return loss / 2

    def on_step_end(self, step: int, max_steps: int):
        for p_o, p_t in zip(self.backbone.parameters(), self.backbone_t.parameters()):
            p_t.data = self.momentum * p_t.data + (1 - self.momentum) * p_o.data
        for p_o, p_t in zip(self.projector.parameters(), self.projector_t.parameters()):
            p_t.data = self.momentum * p_t.data + (1 - self.momentum) * p_o.data


class MAE(nn.Module):
    """MAE: Masked Autoencoder for self-supervised ViT pretraining."""

    def __init__(self, backbone: str, pretrained: bool, mask_ratio: float = 0.75):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        self.mask_ratio = mask_ratio

        # Get patch embedding dimensions
        self.patch_size = self.backbone.patch_embed.patch_size[0]
        self.embed_dim = self.backbone.embed_dim

        # Lightweight decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.embed_dim, 512),
            nn.GELU(),
            nn.Linear(512, self.patch_size ** 2 * 3),
        )

        # Learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)

    def embed(self, x: Tensor) -> Tensor:
        return self.backbone(x)

    def training_step(self, x: Tensor) -> Tensor:
        # Patchify
        patches = self._patchify(x)
        B, N, D = patches.shape

        # Random masking
        num_masked = int(N * self.mask_ratio)
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Keep visible patches
        ids_keep = ids_shuffle[:, :N - num_masked]
        visible_patches = torch.gather(patches, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))

        # Encode visible patches
        visible_encoded = self._encode_patches(visible_patches)

        # Add mask tokens and restore order
        mask_tokens = self.mask_token.expand(B, num_masked, -1)
        full = torch.cat([visible_encoded, mask_tokens], dim=1)
        full = torch.gather(full, 1, ids_restore.unsqueeze(-1).expand(-1, -1, self.embed_dim))

        # Decode
        pred = self.decoder(full)

        # Reconstruction loss on masked patches only
        target = patches
        mask = torch.zeros(B, N, device=x.device)
        mask.scatter_(1, ids_shuffle[:, N - num_masked:], 1)

        loss = (pred - target) ** 2
        loss = (loss.mean(dim=-1) * mask).sum() / mask.sum()
        return loss

    def _patchify(self, x: Tensor) -> Tensor:
        """Convert image to patches."""
        B, C, H, W = x.shape
        p = self.patch_size
        h, w = H // p, W // p
        x = x.reshape(B, C, h, p, w, p)
        x = x.permute(0, 2, 4, 3, 5, 1)  # B, h, w, p, p, C
        x = x.reshape(B, h * w, p * p * C)
        return x

    def _encode_patches(self, patches: Tensor) -> Tensor:
        """Encode visible patches through backbone."""
        # This is simplified - full implementation would use backbone's patch embed
        return self.backbone.patch_embed.proj(
            patches.reshape(-1, self.patch_size, self.patch_size, 3).permute(0, 3, 1, 2)
        ).reshape(patches.shape[0], -1, self.embed_dim)


# ─── SSL Components ─────────────────────────────────────────────

class DINOHead(nn.Module):
    """DINO projection head with weight normalization."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 2048, bottleneck_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        self.last_layer.weight_g.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class SimCLRAugmentation(nn.Module):
    """SimCLR-style augmentation (two views)."""

    def __init__(self, size: int = 224):
        super().__init__()
        from torchvision import transforms as T

        self.transform = T.Compose([
            T.RandomResizedCrop(size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(kernel_size=size // 10 * 2 + 1),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return self.transform(x), self.transform(x)


class MultiCropAugmentation(nn.Module):
    """DINO multi-crop augmentation (2 global + N local views)."""

    def __init__(self, global_size: int = 224, local_size: int = 96, num_local: int = 8):
        super().__init__()
        from torchvision import transforms as T

        self.global_transform = T.Compose([
            T.RandomResizedCrop(global_size, scale=(0.4, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(kernel_size=23),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.local_transform = T.Compose([
            T.RandomResizedCrop(local_size, scale=(0.05, 0.4)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(kernel_size=23),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.num_local = num_local

    def forward(self, x: Tensor) -> dict[str, list[Tensor]]:
        return {
            "global": [self.global_transform(x), self.global_transform(x)],
            "local": [self.local_transform(x) for _ in range(self.num_local)],
        }


# ─── SSL Loss Functions ─────────────────────────────────────────

def nt_xent_loss(z1: Tensor, z2: Tensor, temperature: float = 0.1) -> Tensor:
    """NT-Xent loss (SimCLR)."""
    B = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)  # 2B x D

    # Cosine similarity
    sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temperature

    # Mask out self-similarity
    mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
    sim.masked_fill_(mask, float("-inf"))

    # Positive pairs: (i, i+B) and (i+B, i)
    labels = torch.cat([torch.arange(B, 2 * B), torch.arange(B)], dim=0).to(z.device)

    return F.cross_entropy(sim, labels)


def infonce_with_queue(q: Tensor, k: Tensor, queue: Tensor, temperature: float = 0.1) -> Tensor:
    """InfoNCE loss with memory queue (MoCo)."""
    # Positive logits: Bx1
    l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)

    # Negative logits: BxK
    l_neg = torch.einsum("nc,ck->nk", [q, queue])

    # Logits: Bx(1+K)
    logits = torch.cat([l_pos, l_neg], dim=1) / temperature

    # Labels: positives are at index 0
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

    return F.cross_entropy(logits, labels)


def dino_loss(
    student_out: list[Tensor],
    teacher_out: list[Tensor],
    student_temp: float,
    teacher_temp: float,
) -> Tensor:
    """DINO cross-entropy loss between student and teacher."""
    total_loss = 0.0
    n_loss_terms = 0

    for t_idx, t in enumerate(teacher_out):
        t_soft = F.softmax(t / teacher_temp, dim=-1)

        for s_idx, s in enumerate(student_out):
            if t_idx == s_idx:
                continue  # Skip same view
            s_log_soft = F.log_softmax(s / student_temp, dim=-1)
            total_loss += torch.sum(-t_soft * s_log_soft, dim=-1).mean()
            n_loss_terms += 1

    return total_loss / n_loss_terms


def byol_loss(p: Tensor, z: Tensor) -> Tensor:
    """BYOL regression loss."""
    p = F.normalize(p, dim=-1, p=2)
    z = F.normalize(z, dim=-1, p=2)
    return 2 - 2 * (p * z).sum(dim=-1).mean()
```

### NLP: Causal Language Model

```python
# tasks/nlp/causal_lm.py
from dataclasses import dataclass, field
from typing import Any
import torch.nn as nn
from torch import Tensor

@dataclass
class CausalLM:
    """Causal language model (GPT-style) for pretraining or fine-tuning."""

    model_name: str = "gpt2"
    adapter: "Adapter | None" = None
    max_length: int = 2048

    model: nn.Module = field(init=False)
    tokenizer: Any = field(init=False)

    def __post_init__(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Load model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

        # Apply adapter if specified
        if self.adapter:
            self.model = self.adapter.apply(self.model)

    def train_step(self, batch: dict) -> Tensor:
        outputs = self.model(**batch)
        return outputs.loss

    def eval_step(self, batch: dict) -> dict[str, Tensor]:
        outputs = self.model(**batch)
        perplexity = torch.exp(outputs.loss)
        return {"loss": outputs.loss, "perplexity": perplexity}

    def collate_fn(self, examples: list[str]) -> dict:
        """Tokenize and collate text examples."""
        return self.tokenizer(
            examples,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def load_data(self, path: str):
        """Load text data from file or HF dataset."""
        if path.endswith(".json") or path.endswith(".jsonl"):
            return JSONLDataset(path, key="text")
        elif path.endswith(".txt"):
            return TextFileDataset(path)
        else:
            # Assume HuggingFace dataset
            from datasets import load_dataset
            return load_dataset(path, split="train")
```

### NLP: Seq2Seq

```python
# tasks/nlp/seq2seq.py
from dataclasses import dataclass, field
from typing import Any
import torch.nn as nn
from torch import Tensor

@dataclass
class Seq2Seq:
    """Sequence-to-sequence model (T5, BART) for translation, summarization, etc."""

    model_name: str = "t5-small"
    adapter: "Adapter | None" = None
    max_source_length: int = 512
    max_target_length: int = 128

    model: nn.Module = field(init=False)
    tokenizer: Any = field(init=False)

    def __post_init__(self):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

        if self.adapter:
            self.model = self.adapter.apply(self.model)

    def train_step(self, batch: dict) -> Tensor:
        outputs = self.model(**batch)
        return outputs.loss

    def eval_step(self, batch: dict) -> dict[str, Tensor]:
        outputs = self.model(**batch)
        return {"loss": outputs.loss}

    def collate_fn(self, examples: list[dict]) -> dict:
        """Collate source-target pairs."""
        sources = [ex["source"] for ex in examples]
        targets = [ex["target"] for ex in examples]

        model_inputs = self.tokenizer(
            sources,
            padding=True,
            truncation=True,
            max_length=self.max_source_length,
            return_tensors="pt",
        )

        labels = self.tokenizer(
            targets,
            padding=True,
            truncation=True,
            max_length=self.max_target_length,
            return_tensors="pt",
        ).input_ids

        # Replace padding with -100 for loss computation
        labels[labels == self.tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels

        return model_inputs
```

### Multimodal: Vision-Language Model

```python
# tasks/multimodal/vlm.py
from dataclasses import dataclass, field
from typing import Any
import torch.nn as nn
from torch import Tensor

@dataclass
class VLM:
    """Vision-Language Model (LLaVA, Qwen-VL, etc.)."""

    model_name: str = "llava-hf/llava-1.5-7b-hf"
    adapter: "Adapter | None" = None
    max_length: int = 2048

    model: nn.Module = field(init=False)
    processor: Any = field(init=False)

    def __post_init__(self):
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = LlavaForConditionalGeneration.from_pretrained(self.model_name)

        if self.adapter:
            self.model = self.adapter.apply(self.model)

    def train_step(self, batch: dict) -> Tensor:
        outputs = self.model(**batch)
        return outputs.loss

    def eval_step(self, batch: dict) -> dict[str, Tensor]:
        outputs = self.model(**batch)
        return {"loss": outputs.loss}

    def collate_fn(self, examples: list[dict]) -> dict:
        """Collate image-text pairs."""
        images = [ex["image"] for ex in examples]
        texts = [ex["text"] for ex in examples]

        return self.processor(
            images=images,
            text=texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def load_data(self, path: str):
        """Load VLM dataset (image-text pairs with conversations)."""
        return VLMDataset(path)


@dataclass
class CLIP:
    """CLIP for image-text contrastive learning."""

    model_name: str = "openai/clip-vit-base-patch32"
    adapter: "Adapter | None" = None

    model: nn.Module = field(init=False)
    processor: Any = field(init=False)

    def __post_init__(self):
        from transformers import CLIPModel, CLIPProcessor

        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model = CLIPModel.from_pretrained(self.model_name)

        if self.adapter:
            self.model = self.adapter.apply(self.model)

    def train_step(self, batch: dict) -> Tensor:
        outputs = self.model(**batch, return_loss=True)
        return outputs.loss

    def eval_step(self, batch: dict) -> dict[str, Tensor]:
        outputs = self.model(**batch, return_loss=True)
        return {"loss": outputs.loss}

    def collate_fn(self, examples: list[dict]) -> dict:
        images = [ex["image"] for ex in examples]
        texts = [ex["text"] for ex in examples]

        return self.processor(
            images=images,
            text=texts,
            padding=True,
            return_tensors="pt",
        )
```

---

## Adapters

### LoRA

```python
# adapters/lora.py
from dataclasses import dataclass
import torch.nn as nn

@dataclass
class LoRA:
    """Low-Rank Adaptation for efficient fine-tuning."""

    r: int = 8
    alpha: int = 16
    dropout: float = 0.1
    target_modules: list[str] | None = None

    def apply(self, model: nn.Module) -> nn.Module:
        from peft import get_peft_model, LoraConfig, TaskType

        # Infer task type from model
        task_type = self._infer_task_type(model)

        config = LoraConfig(
            r=self.r,
            lora_alpha=self.alpha,
            lora_dropout=self.dropout,
            target_modules=self.target_modules or "all-linear",
            task_type=task_type,
        )

        return get_peft_model(model, config)

    def _infer_task_type(self, model: nn.Module) -> str:
        from peft import TaskType

        class_name = model.__class__.__name__.lower()
        if "causal" in class_name or "gpt" in class_name:
            return TaskType.CAUSAL_LM
        elif "seq2seq" in class_name or "t5" in class_name or "bart" in class_name:
            return TaskType.SEQ_2_SEQ_LM
        elif "classification" in class_name:
            return TaskType.SEQ_CLS
        else:
            return TaskType.CAUSAL_LM  # Default
```

### QLoRA

```python
# adapters/qlora.py
from dataclasses import dataclass
import torch.nn as nn

@dataclass
class QLoRA:
    """4-bit Quantized LoRA for memory-efficient fine-tuning."""

    r: int = 8
    alpha: int = 16
    dropout: float = 0.1
    target_modules: list[str] | None = None

    def apply(self, model: nn.Module) -> nn.Module:
        from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
        from transformers import BitsAndBytesConfig

        # Prepare for k-bit training
        model = prepare_model_for_kbit_training(model)

        config = LoraConfig(
            r=self.r,
            lora_alpha=self.alpha,
            lora_dropout=self.dropout,
            target_modules=self.target_modules or "all-linear",
        )

        return get_peft_model(model, config)

    @staticmethod
    def quantization_config():
        """Return BitsAndBytesConfig for model loading."""
        from transformers import BitsAndBytesConfig
        import torch

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
```

### Prefix Tuning

```python
# adapters/prefix.py
from dataclasses import dataclass
import torch.nn as nn

@dataclass
class PrefixTuning:
    """Prefix tuning for efficient fine-tuning."""

    num_virtual_tokens: int = 20

    def apply(self, model: nn.Module) -> nn.Module:
        from peft import get_peft_model, PrefixTuningConfig

        config = PrefixTuningConfig(
            num_virtual_tokens=self.num_virtual_tokens,
            task_type="CAUSAL_LM",
        )

        return get_peft_model(model, config)
```

---

## Sweeps

```python
# sweep.py
from dataclasses import dataclass
from typing import Any, Callable
import random
import itertools
from concurrent.futures import ProcessPoolExecutor

@dataclass
class Grid:
    values: list

@dataclass
class Choice:
    values: list

@dataclass
class Uniform:
    low: float
    high: float
    log: bool = False

@dataclass
class IntUniform:
    low: int
    high: int

# Convenience constructors
def grid(*values): return Grid(list(values))
def choice(*values): return Choice(list(values))
def uniform(low, high): return Uniform(low, high)
def log_uniform(low, high): return Uniform(low, high, log=True)
def int_uniform(low, high): return IntUniform(low, high)


class Sweep:
    """Hyperparameter sweep runner.

    Uses cloudpickle for parallel execution to handle closures and lambdas.
    """

    def __init__(self, fn: Callable, spaces: dict[str, Any]):
        self.fn = fn
        self.spaces = spaces

    def run(
        self,
        method: str = "random",  # grid, random, bayes
        n_trials: int = 10,
        n_parallel: int = 1,
        seed: int = 42,
    ) -> list[dict]:
        random.seed(seed)

        if method == "grid":
            configs = self._grid_configs()
        elif method == "random":
            configs = [self._sample() for _ in range(n_trials)]
        elif method == "bayes":
            return self._run_bayes(n_trials)
        else:
            raise ValueError(f"Unknown method: {method}")

        if n_parallel == 1:
            results = [self._run_one(c) for c in configs]
        else:
            # Use cloudpickle for proper closure serialization
            import cloudpickle
            from concurrent.futures import ProcessPoolExecutor
            import multiprocessing as mp

            # Serialize function with cloudpickle
            fn_bytes = cloudpickle.dumps(self.fn)

            def worker(config: dict) -> dict:
                fn = cloudpickle.loads(fn_bytes)
                try:
                    metric = fn(**config)
                    return {"config": config, "metric": metric, "status": "completed"}
                except Exception as e:
                    return {"config": config, "metric": None, "status": "failed", "error": str(e)}

            # Use spawn context for clean process state
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(n_parallel, mp_context=ctx) as pool:
                results = list(pool.map(worker, configs))

        return sorted(results, key=lambda r: r.get("metric", float("-inf")), reverse=True)

    def _grid_configs(self) -> list[dict]:
        grid_params = {
            name: space.values
            for name, space in self.spaces.items()
            if isinstance(space, (Grid, Choice))
        }

        configs = []
        for combo in itertools.product(*grid_params.values()):
            config = dict(zip(grid_params.keys(), combo))
            # Add sampled values for non-grid params
            for name, space in self.spaces.items():
                if name not in config:
                    config[name] = self._sample_one(space)
            configs.append(config)

        return configs

    def _sample(self) -> dict:
        return {name: self._sample_one(space) for name, space in self.spaces.items()}

    def _sample_one(self, space) -> Any:
        if isinstance(space, (Grid, Choice)):
            return random.choice(space.values)
        elif isinstance(space, Uniform):
            if space.log:
                import math
                return math.exp(random.uniform(math.log(space.low), math.log(space.high)))
            return random.uniform(space.low, space.high)
        elif isinstance(space, IntUniform):
            return random.randint(space.low, space.high)
        return space

    def _run_one(self, config: dict) -> dict:
        try:
            metric = self.fn(**config)
            return {"config": config, "metric": metric, "status": "completed"}
        except Exception as e:
            return {"config": config, "metric": None, "status": "failed", "error": str(e)}

    def _run_bayes(self, n_trials: int) -> list[dict]:
        import optuna

        def objective(trial):
            config = {}
            for name, space in self.spaces.items():
                if isinstance(space, (Grid, Choice)):
                    config[name] = trial.suggest_categorical(name, space.values)
                elif isinstance(space, Uniform):
                    config[name] = trial.suggest_float(name, space.low, space.high, log=space.log)
                elif isinstance(space, IntUniform):
                    config[name] = trial.suggest_int(name, space.low, space.high)
            return self.fn(**config)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        return [
            {"config": t.params, "metric": t.value, "status": "completed"}
            for t in study.trials
        ]


def sweep(**spaces):
    """Decorator to create a sweep from a function."""
    def decorator(fn):
        return Sweep(fn, spaces)
    return decorator
```

### Sweep Usage

```python
from trainformer import Trainer
from trainformer.tasks import CausalLM
from trainformer.adapters import LoRA
from trainformer.sweep import sweep, choice, log_uniform

@sweep(
    lr=log_uniform(1e-5, 1e-3),
    r=choice(4, 8, 16, 32),
    alpha=choice(16, 32),
)
def experiment(lr: float, r: int, alpha: int) -> float:
    trainer = Trainer(
        task=CausalLM("gpt2", adapter=LoRA(r=r, alpha=alpha)),
        data="data/text",
        epochs=3,
        lr=lr,
        name=f"sweep_r{r}_lr{lr:.0e}",
    )
    trainer.fit()
    return -trainer.best_metrics.get("val/loss", float("inf"))

# Run sweep
results = experiment.run(method="bayes", n_trials=20)
print(f"Best config: {results[0]}")
```

---

## Usage Examples

### Fine-tune LLM with LoRA

```python
from trainformer import Trainer
from trainformer.tasks import CausalLM
from trainformer.adapters import LoRA

Trainer(
    task=CausalLM("meta-llama/Llama-2-7b", adapter=LoRA(r=8)),
    data="data/alpaca.json",
    epochs=3,
    batch_size=4,
    lr=2e-4,
    accumulation_steps=8,
    precision="bf16",
).fit()
```

### Full LLM Fine-tuning

```python
from trainformer import Trainer
from trainformer.tasks import CausalLM

Trainer(
    task=CausalLM("gpt2"),
    data="data/wikitext",
    epochs=10,
    batch_size=32,
    lr=5e-5,
).fit()
```

### Train ArcFace Embeddings

```python
from trainformer import Trainer
from trainformer.tasks import MetricLearning

Trainer(
    task=MetricLearning("efficientnet_b0", loss="arcface", margin=0.5),
    data="data/sop",
    val_data="data/sop_val",
    epochs=100,
    batch_size=64,
    lr=1e-3,
).fit()
```

### SSL Pretraining with DINO

```python
from trainformer import Trainer
from trainformer.tasks import SSL

Trainer(
    task=SSL.dino("vit_small_patch16_224"),
    data="data/imagenet",
    epochs=100,
    batch_size=256,
    lr=5e-4,
    compile=True,
).fit()
```

### Fine-tune Vision-Language Model

```python
from trainformer import Trainer
from trainformer.tasks import VLM
from trainformer.adapters import QLoRA

Trainer(
    task=VLM("llava-hf/llava-1.5-7b", adapter=QLoRA(r=8)),
    data="data/llava_instruct.json",
    epochs=3,
    batch_size=4,
    lr=2e-4,
).fit()
```

### Image Classification

```python
from trainformer import Trainer
from trainformer.tasks import ImageClassification

Trainer(
    task=ImageClassification("vit_base_patch16_224", num_classes=1000),
    data="data/imagenet/train",
    val_data="data/imagenet/val",
    epochs=90,
    batch_size=256,
    lr=1e-3,
).fit()
```

### Resume Training

```python
trainer = Trainer(
    task=CausalLM("gpt2"),
    data="data/text",
    epochs=10,
)

# Resume from checkpoint
trainer.load("checkpoints/epoch_5.pt")
trainer.fit()  # Continues from epoch 5
```

---

## CLI

Simple CLI wrapper using Python's fire:

```python
# cli.py
import fire

def train(
    task: str,
    data: str,
    model: str | None = None,
    adapter: str | None = None,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-4,
    **kwargs,
):
    """Train a model.

    Examples:
        trainformer train --task=CausalLM --model=gpt2 --data=data/text
        trainformer train --task=MetricLearning --model=efficientnet_b0 --data=data/sop
        trainformer train --task=SSL.dino --model=vit_small --data=data/imagenet
    """
    from trainformer import Trainer
    from trainformer import tasks, adapters

    # Parse task
    if "." in task:
        task_cls, method = task.split(".")
        task_obj = getattr(getattr(tasks, task_cls), method)(model or "resnet50", **kwargs)
    else:
        task_cls = getattr(tasks, task)
        task_obj = task_cls(model, **kwargs) if model else task_cls(**kwargs)

    # Parse adapter
    adapter_obj = None
    if adapter:
        adapter_cls = getattr(adapters, adapter)
        adapter_obj = adapter_cls(**{k: v for k, v in kwargs.items() if k in ["r", "alpha", "dropout"]})
        if hasattr(task_obj, "adapter"):
            task_obj.adapter = adapter_obj

    Trainer(
        task=task_obj,
        data=data,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        **{k: v for k, v in kwargs.items() if k not in ["r", "alpha", "dropout"]},
    ).fit()


if __name__ == "__main__":
    fire.Fire({"train": train})
```

Usage:
```bash
# Fine-tune LLM with LoRA
trainformer train --task=CausalLM --model=gpt2 --data=data/text --adapter=LoRA --r=8

# Train ArcFace
trainformer train --task=MetricLearning --model=efficientnet_b0 --data=data/sop --loss=arcface

# SSL pretraining
trainformer train --task=SSL.dino --model=vit_small --data=data/imagenet --epochs=100
```

---

## Testing Strategy

### Test Structure

```
tests/
├── conftest.py               # Shared fixtures
├── test_trainer.py           # Trainer tests
├── test_tasks.py             # Task implementations
├── test_callbacks.py         # Callback tests
├── test_context.py           # PipelineConfig/Context tests
├── test_sweeps.py            # Sweep tests
├── integration/
│   ├── test_vision.py        # Vision pipeline integration
│   ├── test_nlp.py           # NLP pipeline integration
│   └── test_distributed.py   # Multi-GPU tests
└── fixtures/
    ├── tiny_dataset.py       # Small synthetic datasets
    └── mock_models.py        # Lightweight model mocks
```

### Fixtures

```python
# tests/conftest.py
import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset

@pytest.fixture
def tiny_image_dataset() -> Dataset:
    """Tiny image dataset for fast tests."""
    x = torch.randn(100, 3, 32, 32)
    y = torch.randint(0, 10, (100,))
    return TensorDataset(x, y)

@pytest.fixture
def tiny_text_dataset() -> Dataset:
    """Tiny text dataset."""
    texts = ["Hello world"] * 100
    return texts

@pytest.fixture
def mock_backbone() -> nn.Module:
    """Lightweight mock backbone."""
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(16, 64),
    )

@pytest.fixture
def mock_task(mock_backbone):
    """Mock task for Trainer tests."""
    class MockTask:
        def __init__(self, backbone):
            self.model = backbone
            self._loss = nn.CrossEntropyLoss()

        def train_step(self, batch):
            x, y = batch
            logits = self.model(x)
            return self._loss(logits, y)

        def eval_step(self, batch):
            x, y = batch
            logits = self.model(x)
            return {"loss": self._loss(logits, y)}

        def parameters(self):
            return self.model.parameters()

    return MockTask(mock_backbone)


@pytest.fixture
def trainer_kwargs(tiny_image_dataset, mock_task):
    """Common trainer kwargs for testing."""
    return {
        "task": mock_task,
        "data": tiny_image_dataset,
        "epochs": 2,
        "batch_size": 16,
        "log": "none",
    }
```

### Example Tests

```python
# tests/test_trainer.py
import pytest
from trainformer import Trainer

def test_trainer_fit(trainer_kwargs):
    """Test basic training loop."""
    trainer = Trainer(**trainer_kwargs)
    trainer.fit()

    assert trainer._epoch == trainer_kwargs["epochs"] - 1
    assert trainer._step > 0


def test_trainer_val_split(trainer_kwargs):
    """Test automatic train/val split."""
    trainer = Trainer(**trainer_kwargs, val_split=0.2)
    trainer.fit()

    assert trainer._val_loader is not None


def test_trainer_callbacks(trainer_kwargs):
    """Test callback integration."""
    from trainformer.callbacks import EarlyStopping

    early_stop = EarlyStopping(monitor="val/loss", patience=1)
    trainer = Trainer(**trainer_kwargs, callbacks=[early_stop], val_split=0.2)
    trainer.fit()


def test_trainer_checkpoint(trainer_kwargs, tmp_path):
    """Test save/load checkpoint."""
    trainer = Trainer(**trainer_kwargs, save_dir=str(tmp_path))
    trainer.fit()
    trainer.save(str(tmp_path / "checkpoint.pt"))

    # Load into new trainer
    trainer2 = Trainer(**trainer_kwargs)
    trainer2.load(str(tmp_path / "checkpoint.pt"))

    assert trainer2._epoch == trainer._epoch
    assert trainer2._step == trainer._step


def test_trainer_predict(trainer_kwargs, tiny_image_dataset):
    """Test prediction API."""
    trainer = Trainer(**trainer_kwargs)
    trainer.fit()

    outputs = trainer.predict(tiny_image_dataset, batch_size=16)
    assert len(outputs) > 0


def test_trainer_export(trainer_kwargs, tmp_path):
    """Test model export."""
    trainer = Trainer(**trainer_kwargs)
    trainer.fit()

    path = trainer.export(str(tmp_path / "model"), format="pytorch")
    assert Path(path).exists()


# tests/test_context.py
def test_pipeline_config_sources():
    """Test config source tracking."""
    from trainformer.context import PipelineConfig, ConfigSource

    config = PipelineConfig()
    config.set_user("lr", 1e-4, filterable=True)
    config.set_from_data("num_classes", 10, filterable=True)
    config.set_derived("total_steps", 1000, ("num_samples", "batch_size"))

    assert config.get("lr") == 1e-4
    assert config.get("num_classes") == 10
    assert "lr" in config.get_filter_dimensions()
    assert "total_steps" in config.to_wandb_config()


def test_pipeline_context_lifecycle():
    """Test context lifecycle management."""
    from trainformer.context import PipelineContext, Lifecycle

    ctx = PipelineContext()
    ctx.set("step_value", 1, Lifecycle.STEP)
    ctx.set("epoch_value", 2, Lifecycle.EPOCH)
    ctx.set("run_value", 3, Lifecycle.RUN)

    ctx.on_step_end()
    assert ctx.get("step_value") is None
    assert ctx.get("epoch_value") == 2

    ctx.on_epoch_end()
    assert ctx.get("epoch_value") is None
    assert ctx.get("run_value") == 3


# tests/test_callbacks.py
def test_early_stopping():
    """Test early stopping callback."""
    from trainformer.callbacks import EarlyStopping

    es = EarlyStopping(monitor="val/loss", patience=2, mode="min")

    # Mock trainer
    class MockTrainer:
        _should_stop = False

    trainer = MockTrainer()

    # Improving
    es.on_validation_end(trainer, {"val/loss": 1.0})
    assert not trainer._should_stop

    # Not improving
    es.on_validation_end(trainer, {"val/loss": 1.1})
    assert not trainer._should_stop  # patience=2

    es.on_validation_end(trainer, {"val/loss": 1.2})
    assert trainer._should_stop  # patience exceeded


def test_model_checkpoint(tmp_path):
    """Test model checkpoint callback."""
    from trainformer.callbacks import ModelCheckpoint

    ckpt = ModelCheckpoint(
        monitor="val/loss",
        mode="min",
        save_top_k=2,
        save_dir=str(tmp_path),
    )

    # Would need mock trainer with save() method
    # ...
```

### CI Configuration

```yaml
# .github/workflows/test.yml
name: Tests

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          pip install -e ".[dev]"

      - name: Run linting
        run: |
          ruff check .

      - name: Run tests
        run: |
          pytest tests/ -v --cov=trainformer --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
```

---

## Dependencies

```toml
# pyproject.toml
[project]
name = "trainformer"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    # Core
    "torch>=2.0",
    "accelerate>=0.25",

    # Vision
    "timm>=0.9",
    "torchvision>=0.15",

    # NLP
    "transformers>=4.36",
    "peft>=0.7",
    "datasets>=2.14",

    # Evaluation
    "faiss-cpu>=1.7",

    # Logging
    "wandb>=0.16",

    # CLI
    "fire>=0.5",

    # Utilities
    "cloudpickle>=3.0",
    "numpy>=1.24",
]

[project.optional-dependencies]
dev = ["pytest>=7.0", "pytest-cov>=4.0", "ruff>=0.1"]
qlora = ["bitsandbytes>=0.41"]
tensorboard = ["tensorboard>=2.14"]
mlflow = ["mlflow>=2.8"]
all = ["trainformer[qlora,tensorboard,mlflow]"]

[project.scripts]
trainformer = "trainformer.cli:main"
```

---

## Comparison

| Aspect | HuggingFace | Lightning | trainformer |
|--------|-------------|-----------|-------------|
| **Config** | TrainingArguments (100+ params) | YAML or Python | Python dataclass |
| **Abstractions** | Trainer, Tokenizer, Model | Module, Trainer, Callback | Task, Trainer, Adapter |
| **Vision** | Secondary | Good | First-class |
| **NLP** | First-class | Good | Good (uses HF) |
| **SSL** | Limited | Via callbacks | First-class |
| **Metric Learning** | Not supported | DIY | First-class |
| **Adapters** | Separate library (PEFT) | Separate | Integrated |
| **Learning curve** | Medium | High | Low |

---

## Migration Steps

### Phase 1: Core (Week 1)
1. Project setup with pyproject.toml
2. Implement Trainer class
3. Implement Task protocol
4. Basic logging (console, W&B)
5. Checkpoint save/load

### Phase 2: Vision Tasks (Week 2)
6. ImageClassification task
7. MetricLearning task (ArcFace, CosFace)
8. SSL task (SimCLR, MoCo)
9. SSL task (DINO, MAE)
10. Retrieval evaluation (KNN, FAISS)

### Phase 3: NLP Tasks (Week 3)
11. CausalLM task
12. Seq2Seq task
13. LoRA adapter
14. QLoRA adapter
15. Text data loading

### Phase 4: Multimodal (Week 4)
16. CLIP task
17. VLM task
18. Multimodal data loading
19. Cross-modal evaluation

### Phase 5: Polish (Week 5)
20. CLI
21. Sweep support
22. Examples and documentation
23. Tests

---

## Design Decisions

### Why Python configs over YAML?
- Type checking and IDE autocomplete
- No config resolution debugging
- Compose with Python imports
- Expressions and conditionals work naturally

### Why Tasks over separate Model/Loss?
- Bundles related concerns together
- Reduces configuration surface
- Tasks know their evaluation metrics
- Easier to add new paradigms

### Why build on HuggingFace?
- Don't reinvent model loading
- Leverage their ecosystem
- Focus on training experience, not models

### Why Accelerate over raw DDP?
- Handles distributed without code changes
- Mixed precision built-in
- DeepSpeed/FSDP support
- Battle-tested

---

## Changelog

### 2024-12-26: Comprehensive Feature Expansion

**Decision**: Add comprehensive feature set based on user requirements analysis.

**New Features Added**:

1. **Callback System** (§6-7)
   - Hook-based `Callback` protocol with full lifecycle coverage
   - `CallbackList` for managing multiple callbacks
   - Built-in callbacks:
     - `EarlyStopping` - stop on metric plateau with divergence detection
     - `ModelCheckpoint` - save best N models by metric, auto-prune
     - `LRMonitor` - log learning rate
     - `GradientMonitor` - track gradient norms, detect explosions
     - `LRFinder` - learning rate range test before training

2. **Extended Task Protocol** (§2)
   - Additional lifecycle hooks: `on_train_begin`, `on_train_end`, `on_epoch_end`, `on_validation_end`
   - Optional data methods: `load_data`, `collate_fn`, `get_train_transforms`, `get_eval_transforms`
   - `TaskBase` class with default implementations

3. **Trainer Enhancements** (§3)
   - `val_split` parameter for automatic train/val splitting
   - `gradient_checkpointing` and `activation_offload` for memory optimization
   - `predict()` method for inference with batching
   - `export()` method for PyTorch, ONNX, TorchScript export
   - `callbacks` parameter for callback integration
   - `monitor_metric` and `monitor_mode` for best model tracking
   - Multi-backend logging support via list: `log=["wandb", "tensorboard"]`

4. **Logger Classes** (§8)
   - `BaseLogger` abstract interface
   - `WandbLogger`, `ConsoleLogger`, `TensorBoardLogger`, `MLflowLogger`
   - `MultiLogger` for composing multiple backends
   - `NoOpLogger` for testing

5. **Sampler Classes** (§9)
   - `ClassBalancedSampler` for imbalanced datasets
   - `PKSampler` for metric learning (P classes, K samples per batch)

6. **Data Utilities** (§10)
   - `JSONLDataset` for JSONL files
   - `TextFileDataset` for text files (line-based or chunked)
   - `ImageTextDataset` for multimodal data

7. **Utility Functions** (§11)
   - `seed_everything()` for reproducibility
   - `count_parameters()` with nice formatting
   - `profile_memory()` context manager for GPU memory tracking
   - `get_device()` and `move_to_device()`

8. **SSL Components**
   - Full `BYOL` implementation
   - Full `MAE` (Masked Autoencoder) implementation
   - `DINOHead` projection head
   - `SimCLRAugmentation` and `MultiCropAugmentation`
   - Loss functions: `nt_xent_loss`, `infonce_with_queue`, `dino_loss`, `byol_loss`

9. **Sweep Improvements**
   - `cloudpickle` for proper closure serialization in parallel sweeps
   - Spawn context for clean process state

10. **Buffer Management**
    - `max_size` parameter for `accumulate()` to prevent OOM

11. **Testing Strategy**
    - Test structure with fixtures and mock models
    - Example tests for Trainer, Context, Callbacks
    - CI configuration with GitHub Actions

**Dependencies Updated**:
- Added `cloudpickle>=3.0`
- Added optional extras: `tensorboard`, `mlflow`, `all`

---

### 2024-12-19: Pipeline Context System

**Decision**: Unified context system with explicit lifecycle management for config, state, buffers, and events.

**Problem Identified**:
- `num_classes` needs to flow from dataset → loss → optimizer → checkpoint
- Many values have topological dependencies across the pipeline
- Some values should be logged (wandb), some checkpointed, some neither
- Runtime state (MoCo queue, EMA weights, accumulated embeddings) needs lifecycle management

**Solution Components**:

1. **DatasetInfo** - Metadata extracted from dataset (single source of truth)
   - `num_classes`, `class_names`, `class_counts` - for loss/eval/sampling
   - `input_shape`, `mean/std` - for model/transforms
   - `num_samples` - for scheduler total_steps

2. **PipelineConfig** - Config registry with source tracking
   - `ConfigSource`: USER, DEFAULT, DATA, DERIVED, ENV
   - `filterable` flag for wandb grouping dimensions
   - Separate serialization: `to_wandb_config()`, `to_checkpoint()`, `user_overrides()`

3. **PipelineContext** - Unified runtime context
   - `config`: PipelineConfig for logged values
   - `state`: Runtime values with lifecycle (STEP, EPOCH, RUN, CACHED)
   - `buffers`: Large tensors with memory management
   - `events`: Pub/sub for loose coupling between components

4. **Task.configure(info: DatasetInfo)** - Lifecycle hook
   - Called after data loaded, before optimizer created
   - Task initializes components that depend on data (loss, heads, etc.)

5. **Task.parameters()** - Optimizer parameter access
   - Yields model + loss parameters for optimizer
   - Solves ArcFace learnable weights inclusion

**Phased Trainer Initialization**:
```
PHASE 1: Record user config
PHASE 2: Load data, extract DatasetInfo
PHASE 3: Task.configure(info) - initialize data-dependent components
PHASE 4: Compute derived values (total_steps, warmup_steps, etc.)
PHASE 5: Compile model (after task fully configured)
PHASE 6: Create loaders, optimizer, scheduler
PHASE 7: Prepare for distributed
PHASE 8: Initialize logger with full config
```

**Lifecycle Scopes**:
| Lifecycle | Cleared | Examples |
|-----------|---------|----------|
| STEP | Every step | current batch, step gradients |
| EPOCH | Every epoch | accumulated embeddings, confusion matrix |
| RUN | Never (checkpoint) | EMA weights, MoCo queue, best model |
| CACHED | Explicit | tokenized dataset, feature cache |

**Impact**:
- `num_classes` properly propagates to all consumers
- Derived values (total_steps) computed from data
- Full config logged to wandb with source tracking
- Runtime state checkpointed correctly
- Callbacks get scoped context access

---

### 2024-12-19: Major Redesign - Python-First Architecture

**Decision**: Drop Hydra entirely, use Python dataclasses for configuration.

**Hydra Pain Points Identified**:

1. **Learning Curve**
   - `_target_` instantiation pattern is non-obvious
   - Defaults composition (`- model: arcface`) has sharp edges
   - OmegaConf interpolation syntax (`${oc.env:VAR}`) differs from standard
   - `_partial_`, `_recursive_`, `_convert_` flags add complexity

2. **Config Resolution Debugging**
   - "Why isn't my override working?" is common
   - Defaults merge order is hard to reason about
   - Interpolation errors appear at runtime, not import time
   - Stack traces point to Hydra internals, not user code

3. **Working Directory Behavior**
   - Default `chdir: true` breaks relative paths
   - Need `to_absolute_path()` everywhere
   - Output directory structure is opinionated
   - Multirun creates nested directories

4. **Redundant Abstractions**
   - Registry pattern (`name: arcface`) AND Hydra (`_target_: ...`)
   - Two ways to do the same thing → confusion
   - Escape hatches indicate abstraction mismatch

5. **Verbosity**
   - Every class needs `_target_` in config
   - Simple things require multiple files:
     ```yaml
     # conf/model/arcface.yaml
     _target_: trainformer.models.ArcFaceModel
     backbone:
       _target_: trainformer.models.TimmBackbone
       model_name: efficientnet_b0
     ```
   - vs Python: `ArcFaceModel("efficientnet_b0")`

6. **IDE Support Limitations**
   - No autocomplete for YAML config keys
   - Can't cmd-click to navigate to class
   - Type errors discovered at runtime
   - Refactoring doesn't update configs

7. **Testing Friction**
   - Need `initialize_config_dir()` context manager
   - Compose API differs from decorator API
   - Config validation requires running Hydra

**What Hydra Did Well**:
- Multirun sweeps (`-m param=1,2,3`)
- Sweeper plugins (Optuna, Ax, Nevergrad)
- Structured logging to output directories
- Config composition for large projects
- Environment variable interpolation

**Why We Can Live Without It**:
- Sweeps → Python decorator + Optuna directly
- Logging → W&B handles this better
- Composition → Python imports work
- Env vars → `os.environ` or python-dotenv
- Type safety → Dataclasses give this natively

**Alternatives Considered**:
- Keep Hydra with simplified config → Still inherits complexity
- YAML + simple loader → No type checking, no validation
- Pydantic models → Good, but still need override handling
- jsonargparse → Lightning uses this, still YAML-based
- tyro → Interesting but less mature

**Migration Path**:
```python
# Before (Hydra)
@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    model = hydra.utils.instantiate(cfg.model)
    trainer = hydra.utils.instantiate(cfg.trainer, model=model)
    trainer.fit()

# After (Python)
from trainformer import Trainer
from trainformer.tasks import MetricLearning

Trainer(
    task=MetricLearning("efficientnet_b0", loss="arcface"),
    data="data/sop",
    epochs=100,
).fit()
```

---

### 2024-12-19: Task-Based Architecture

**Decision**: Bundle model + loss + data loading + evaluation into Task classes.

**Rationale**:
- Reduces configuration surface (one object vs four)
- Tasks know their own evaluation metrics
- SSL methods are "opinionated bundles", not Lego blocks
- Easier to add new training paradigms
- Clear ownership: Task handles forward/loss, Trainer handles loop

**Alternatives Considered**:
- Separate Model/Loss/Evaluator → Too much configuration, unclear ownership
- SSL as separate abstraction → Creates two code paths in Trainer
- Everything in Trainer → Trainer becomes god class

---

### 2024-12-19: Expanded Scope - LLMs & Multimodal

**Decision**: Support full spectrum from CNNs to LLMs to VLMs.

**Rationale**:
- User needs: classification, metric learning, SSL, LLM fine-tuning, VLM
- Task abstraction scales cleanly to new domains
- Build on HuggingFace for NLP, timm for vision → don't reinvent
- Adapters (LoRA, QLoRA) integrated rather than separate library

**Impact**:
- Added: `CausalLM`, `Seq2Seq`, `VLM`, `CLIP` tasks
- Added: `LoRA`, `QLoRA`, `PrefixTuning` adapters
- Dependencies: transformers, peft, datasets

---

### 2024-12-19: SSL Design - Models Own Training Logic

**Decision**: SSL models implement `training_step()` that returns loss.

**Rationale**:
- Teacher/momentum encoder tightly coupled with loss computation
- Queue/center updates are method-specific state
- Can't cleanly separate "model" from "loss" in SSL
- Users don't mix components: "I want DINO" not "DINO's crops + MoCo's queue"

**Implementation**:
```python
class SimCLR(nn.Module):
    def training_step(self, x: Tensor) -> Tensor:
        v1, v2 = self.augment(x)
        z1 = self.project(self.backbone(v1))
        z2 = self.project(self.backbone(v2))
        return nt_xent_loss(z1, z2, self.temperature)
```

---

### 2024-12-19: Sweep System Without Hydra

**Decision**: Python decorator-based sweeps with optional Optuna integration.

**Rationale**:
- Hydra multirun was the main benefit of Hydra
- W&B sweeps work with any training script
- Optuna for Bayesian optimization
- Shell loops for simple cases
- Python decorators for type-safe sweep definition

**Implementation**:
```python
@sweep(lr=log_uniform(1e-5, 1e-3), r=choice(4, 8, 16))
def experiment(lr, r):
    return Trainer(..., lr=lr).fit().best_metrics["val/loss"]

experiment.run(method="bayes", n_trials=20)
```

---

### 2024-12-19: Removed Registry Pattern

**Decision**: Direct imports instead of registry + `name` lookup.

**Rationale**:
- Registry added indirection without clear benefit
- `_target_` escape hatch meant two ways to do same thing
- Direct imports: `from trainformer.tasks import SSL` is clearer
- IDE can navigate to source, autocomplete works

**Before**:
```yaml
model:
  name: arcface
  backbone: efficientnet_b0
```

**After**:
```python
from trainformer.tasks import MetricLearning
MetricLearning("efficientnet_b0", loss="arcface")
```

---

### 2024-12-19: DeviceTransforms Typing

**Decision**: Use `nn.Module` instead of `Callable` for transforms.

**Rationale**:
- Enables `torch.compile` compatibility
- Proper device handling
- Works with `nn.Sequential`
- Consistent with rest of PyTorch ecosystem

---

### 2024-12-19: Logger Lazy Initialization

**Decision**: Logger receives `is_main_process` flag after Trainer init.

**Rationale**:
- Avoids circular dependency between Logger and Accelerator
- Logger can be created before Accelerator exists
- Trainer sets flag after `accelerator.prepare()`

**Implementation**:
```python
class BaseLogger:
    def __init__(self, ...):
        self._is_main_process = True  # Default

    def set_is_main_process(self, is_main: bool):
        self._is_main_process = is_main
```

---

### 2024-12-19: torch.compile Placement

**Decision**: Compile model, criterion, and device transforms. Not dataloader or callbacks.

**Rationale**:
- Model forward pass: largest compute, most benefit
- Criterion: good for ArcFace with large weight matrix
- Device transforms: if using nn.Module-based transforms
- DataLoader: I/O bound, graph breaks on Python iteration
- Callbacks: control flow heavy, graph breaks everywhere

**Implementation**:
```python
# In Trainer._setup()
if self.compile:
    self.task.model = torch.compile(self.task.model)
```

---

### 2024-12-19: Gradient Clipping with Learnable Loss

**Decision**: Use `itertools.chain()` for clipping model + criterion params.

**Rationale**:
- ArcFaceLoss has learnable weight matrix
- Both model and criterion are prepared by Accelerator
- `itertools.chain` avoids creating intermediate list

**Implementation**:
```python
self.accelerator.clip_grad_norm_(
    itertools.chain(self.model.parameters(), self.criterion.parameters()),
    self.gradient_clip,
)
```

---

### 2024-12-19: No kornia Dependency

**Decision**: Use `torchvision.transforms.v2` instead of kornia.

**Rationale**:
- User preference against kornia
- TV2 works on batched tensors
- TV2 transforms are `nn.Module` compatible
- CutMix/MixUp from TV2 handle `(x, y) -> (x, y)` signature

---

### 2024-12-19: Framework Positioning

**Decision**: Focus on metric learning + SSL + LLM fine-tuning, not general purpose.

**Rationale**:
- Gap exists between timm (scripts) and Lightning (too complex)
- HuggingFace is NLP-first, vision is secondary
- Metric learning not well supported anywhere
- SSL libraries (solo-learn, lightly) require Lightning

**Positioning**:
| Need | Use |
|------|-----|
| General vision | timm |
| NLP fine-tuning | HuggingFace |
| Complex pipelines | Lightning |
| **Metric learning, SSL, LLM fine-tuning** | **trainformer** |

---

### Decision Log Format

Each decision follows this template:
```
### YYYY-MM-DD: Decision Title

**Decision**: What was decided.

**Rationale**: Why this decision was made.

**Alternatives Considered**: What else was evaluated.

**Impact**: What changed as a result.
```
