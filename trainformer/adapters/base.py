"""Adapter protocol and base types.

This module defines the type system for parameter-efficient fine-tuning adapters.
Adapters modify a pre-trained model to enable efficient fine-tuning while keeping
most parameters frozen.
"""
from typing import Protocol, Union, runtime_checkable


@runtime_checkable
class LoRALike(Protocol):
    """Protocol for LoRA-style adapters (LoRA, QLoRA).

    LoRA adapters add low-rank decomposition matrices to target modules,
    enabling efficient fine-tuning with reduced memory and compute.

    Attributes:
        r: Rank of the low-rank matrices
        alpha: LoRA alpha parameter (scaling factor)
        dropout: Dropout probability for LoRA layers
        target_modules: List of module names to apply LoRA to
    """

    r: int
    alpha: int
    dropout: float
    target_modules: list[str] | None


@runtime_checkable
class PrefixLike(Protocol):
    """Protocol for prefix tuning adapters.

    Prefix tuning prepends learnable virtual tokens to the input sequence,
    enabling task-specific adaptation without modifying model weights.

    Attributes:
        num_virtual_tokens: Number of virtual prefix tokens
        prefix_projection: Whether to use a projection layer
    """

    num_virtual_tokens: int
    prefix_projection: bool


# Type alias for any adapter type
# Used for type hints in task classes
from trainformer.adapters.lora import LoRA, PrefixTuning, QLoRA

Adapter = Union[LoRA, QLoRA, PrefixTuning]


def is_lora_adapter(adapter: object) -> bool:
    """Check if adapter is a LoRA-style adapter (LoRA or QLoRA).

    Args:
        adapter: Object to check

    Returns:
        True if adapter has LoRA-style attributes (r, alpha, dropout)
    """
    return hasattr(adapter, "r") and hasattr(adapter, "alpha")


def is_prefix_adapter(adapter: object) -> bool:
    """Check if adapter is a prefix-style adapter.

    Args:
        adapter: Object to check

    Returns:
        True if adapter has prefix-style attributes (num_virtual_tokens)
    """
    return hasattr(adapter, "num_virtual_tokens")


def get_adapter_type(adapter: object) -> str:
    """Get the type of adapter as a string.

    Args:
        adapter: Adapter object

    Returns:
        String identifier: 'lora', 'qlora', 'prefix', or 'unknown'
    """
    if hasattr(adapter, "compute_dtype"):
        return "qlora"
    elif hasattr(adapter, "r"):
        return "lora"
    elif hasattr(adapter, "num_virtual_tokens"):
        return "prefix"
    return "unknown"
