"""Sequence-to-sequence task for translation, summarization, etc."""
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from trainformer.types import DatasetInfo


@dataclass
class Seq2Seq:
    """Sequence-to-sequence model (T5, BART) for translation, summarization, etc.

    Args:
        model_name: HuggingFace model name (e.g., 't5-small', 'facebook/bart-base')
        adapter: Optional adapter for efficient fine-tuning
        max_source_length: Maximum source sequence length
        max_target_length: Maximum target sequence length
        gradient_checkpointing: Enable gradient checkpointing
    """

    model_name: str = "t5-small"
    adapter: Any = None
    max_source_length: int = 512
    max_target_length: int = 128
    gradient_checkpointing: bool = False

    model: nn.Module = field(init=False)
    tokenizer: Any = field(init=False)

    def __post_init__(self):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
        )

        if self.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        if self.adapter is not None:
            self._apply_adapter()

    def _apply_adapter(self) -> None:
        """Apply PEFT adapter to model."""
        from peft import LoraConfig, TaskType, get_peft_model

        if hasattr(self.adapter, "r"):
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                r=self.adapter.r,
                lora_alpha=self.adapter.alpha,
                lora_dropout=self.adapter.dropout,
                target_modules=self.adapter.target_modules,
            )
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()

    def configure(self, info: DatasetInfo) -> None:
        """Seq2Seq doesn't need dataset configuration."""
        pass

    def parameters(self):
        """Return trainable parameters."""
        return [p for p in self.model.parameters() if p.requires_grad]

    def train_step(self, batch: dict[str, Tensor]) -> Tensor:
        """Compute seq2seq loss."""
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            labels=batch["labels"],
        )
        return outputs.loss

    def eval_step(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Compute evaluation outputs."""
        with torch.no_grad():
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                labels=batch["labels"],
            )
        return {
            "loss": outputs.loss.unsqueeze(0),
            "logits": outputs.logits,
        }

    def load_data(self, path: str):
        """Load seq2seq dataset.

        Expects dataset with 'source' and 'target' columns.
        """
        from datasets import load_dataset

        if path.endswith(".json") or path.endswith(".jsonl"):
            dataset = load_dataset("json", data_files=path)["train"]
        else:
            dataset = load_dataset(path)["train"]

        def tokenize(examples):
            model_inputs = self.tokenizer(
                examples["source"],
                truncation=True,
                max_length=self.max_source_length,
                padding="max_length",
            )

            labels = self.tokenizer(
                examples["target"],
                truncation=True,
                max_length=self.max_target_length,
                padding="max_length",
            )

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
        dataset.set_format("torch")
        return dataset

    def generate(
        self,
        source: str,
        max_new_tokens: int = 128,
        num_beams: int = 4,
        **kwargs,
    ) -> str:
        """Generate output from source text."""
        inputs = self.tokenizer(
            source,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_source_length,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                **kwargs,
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def info(self) -> dict[str, Any]:
        """Return task info."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            "model_name": self.model_name,
            "max_source_length": self.max_source_length,
            "max_target_length": self.max_target_length,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "adapter": type(self.adapter).__name__ if self.adapter else None,
        }
