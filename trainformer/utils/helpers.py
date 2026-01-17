"""Utility functions."""
import logging
import os
import random
from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np
import torch
from torch import Tensor, nn

logger = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.debug(f"Seeded everything with seed={seed}")


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def format_params(n: int) -> str:
    """Format parameter count in human-readable form."""
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.2f}K"
    return str(n)


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """Get current learning rate from optimizer."""
    for param_group in optimizer.param_groups:
        return param_group["lr"]
    return 0.0


def get_all_lrs(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    """Get learning rates for all param groups."""
    return {
        f"lr/group_{i}": pg["lr"]
        for i, pg in enumerate(optimizer.param_groups)
    }


def freeze(model: nn.Module) -> None:
    """Freeze all parameters."""
    for p in model.parameters():
        p.requires_grad = False


def unfreeze(model: nn.Module) -> None:
    """Unfreeze all parameters."""
    for p in model.parameters():
        p.requires_grad = True


def freeze_bn(model: nn.Module) -> None:
    """Freeze batch normalization layers."""
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False


def profile_memory() -> dict[str, float]:
    """Profile GPU memory usage (in GB)."""
    if not torch.cuda.is_available():
        return {}
    return {
        "memory/allocated": torch.cuda.memory_allocated() / 1e9,
        "memory/reserved": torch.cuda.memory_reserved() / 1e9,
        "memory/max_allocated": torch.cuda.max_memory_allocated() / 1e9,
    }


def reset_memory_stats() -> None:
    """Reset memory stats for profiling."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def move_to_device(batch: Any, device: torch.device | str) -> Any:
    """Recursively move batch to device.

    Args:
        batch: Tensor, dict, list, or tuple to move
        device: Target device

    Returns:
        Batch with all tensors moved to device
    """
    if isinstance(batch, Tensor):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, (list, tuple)):
        return type(batch)(move_to_device(x, device) for x in batch)
    return batch


@contextmanager
def profile_memory_context(label: str = "Operation") -> Iterator[None]:
    """Context manager to track peak GPU memory usage.

    Args:
        label: Label for logging the memory stats

    Yields:
        None

    Example:
        >>> with profile_memory_context("Forward pass"):
        ...     output = model(input)
        # Logs: "Forward pass - Peak memory: 1.23 GB"
    """
    if not torch.cuda.is_available():
        yield
        return

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    try:
        yield
    finally:
        torch.cuda.synchronize()
        peak_memory = torch.cuda.max_memory_allocated() / 1e9
        logger.info(f"{label} - Peak memory: {peak_memory:.2f} GB")
