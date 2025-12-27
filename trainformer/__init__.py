"""Trainformer: Python-first training library for deep learning."""
from trainformer import adapters, callbacks, tasks
from trainformer import sweep as sweep_module
from trainformer.trainer import Trainer
from trainformer.types import DatasetInfo, Task

__version__ = "0.1.0"

__all__ = [
    # Core
    "Trainer",
    "Task",
    "DatasetInfo",
    # Modules
    "tasks",
    "adapters",
    "callbacks",
    "sweep_module",
]
