"""Loggers for training metrics."""
from trainformer.loggers.base import ConsoleLogger, Logger, MultiLogger
from trainformer.loggers.wandb import WandbLogger

__all__ = [
    "Logger",
    "ConsoleLogger",
    "MultiLogger",
    "WandbLogger",
]
