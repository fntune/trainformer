"""LoRA and QLoRA adapters for efficient fine-tuning."""
from dataclasses import dataclass


@dataclass
class LoRA:
    """Low-Rank Adaptation for efficient fine-tuning.

    Uses PEFT library under the hood.

    Args:
        r: Rank of the low-rank matrices
        alpha: LoRA alpha parameter (scaling factor)
        dropout: Dropout probability for LoRA layers
        target_modules: List of module names to apply LoRA to.
                       If None, uses default for the model type.
    """

    r: int = 8
    alpha: int = 16
    dropout: float = 0.1
    target_modules: list[str] | None = None


@dataclass
class QLoRA:
    """4-bit Quantized LoRA for memory-efficient fine-tuning.

    Combines 4-bit quantization with LoRA for training large models
    on consumer hardware.

    Args:
        r: Rank of the low-rank matrices
        alpha: LoRA alpha parameter
        dropout: Dropout probability
        target_modules: List of module names to apply LoRA to
        compute_dtype: Compute dtype for 4-bit base model
    """

    r: int = 8
    alpha: int = 16
    dropout: float = 0.1
    target_modules: list[str] | None = None
    compute_dtype: str = "float16"


@dataclass
class PrefixTuning:
    """Prefix tuning for efficient fine-tuning.

    Prepends learnable prefix tokens to the input.

    Args:
        num_virtual_tokens: Number of virtual prefix tokens
        prefix_projection: Whether to use a projection layer
    """

    num_virtual_tokens: int = 20
    prefix_projection: bool = False
