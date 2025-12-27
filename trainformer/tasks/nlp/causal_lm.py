"""Causal language model task for pretraining and fine-tuning."""
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from trainformer.types import DatasetInfo


@dataclass
class CausalLM:
    """Causal language model (GPT-style) for pretraining or fine-tuning.

    Uses HuggingFace transformers for model loading.

    Args:
        model_name: HuggingFace model name (e.g., 'gpt2', 'meta-llama/Llama-2-7b')
        adapter: Optional adapter for efficient fine-tuning (LoRA, etc.)
        max_length: Maximum sequence length
        gradient_checkpointing: Enable gradient checkpointing for memory efficiency
    """

    model_name: str = "gpt2"
    adapter: Any = None  # Adapter dataclass (LoRA, QLoRA, etc.)
    max_length: int = 2048
    gradient_checkpointing: bool = False

    model: nn.Module = field(init=False)
    tokenizer: Any = field(init=False)

    def __post_init__(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
        )

        if self.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # Apply adapter if specified
        if self.adapter is not None:
            self._apply_adapter()

    def _apply_adapter(self) -> None:
        """Apply PEFT adapter to model."""
        from peft import LoraConfig, TaskType, get_peft_model

        if hasattr(self.adapter, "r"):  # LoRA-style adapter
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.adapter.r,
                lora_alpha=self.adapter.alpha,
                lora_dropout=self.adapter.dropout,
                target_modules=self.adapter.target_modules,
            )
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()

    def configure(self, info: DatasetInfo) -> None:
        """CausalLM doesn't need dataset configuration."""
        pass

    def parameters(self):
        """Return trainable parameters."""
        return [p for p in self.model.parameters() if p.requires_grad]

    def train_step(self, batch: dict[str, Tensor]) -> Tensor:
        """Compute causal LM loss."""
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            labels=batch["input_ids"],  # Labels are shifted internally
        )
        return outputs.loss

    def eval_step(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Compute evaluation outputs."""
        with torch.no_grad():
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                labels=batch["input_ids"],
            )
        return {
            "loss": outputs.loss.unsqueeze(0),
            "logits": outputs.logits,
        }

    def load_data(self, path: str):
        """Load text dataset."""
        from datasets import load_dataset

        if path.endswith(".txt"):
            dataset = load_dataset("text", data_files=path)["train"]
        elif path.endswith(".json") or path.endswith(".jsonl"):
            dataset = load_dataset("json", data_files=path)["train"]
        else:
            # Assume it's a HuggingFace dataset name
            dataset = load_dataset(path)["train"]

        def tokenize(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )

        dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
        dataset.set_format("torch")
        return dataset

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        **kwargs,
    ) -> str:
        """Generate text from a prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                **kwargs,
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def info(self) -> dict[str, Any]:
        """Return task info."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            "model_name": self.model_name,
            "max_length": self.max_length,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "adapter": type(self.adapter).__name__ if self.adapter else None,
        }
