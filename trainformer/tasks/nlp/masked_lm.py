"""Masked Language Model task (BERT-style)."""
import logging
from dataclasses import dataclass, field
from typing import Any, Iterator

import torch
from torch import Tensor, nn
from torch.utils.data import Dataset

from trainformer.types import TaskBase

logger = logging.getLogger(__name__)


@dataclass
class MaskedLM(TaskBase):
    """Masked Language Modeling task for BERT-style pretraining/finetuning.

    Example:
        task = MaskedLM(
            model_name="bert-base-uncased",
            mlm_probability=0.15,
        )
        trainer = Trainer(task=task, data="data/corpus.txt", epochs=3)
        trainer.fit()
    """

    model_name: str = "bert-base-uncased"
    mlm_probability: float = 0.15
    max_length: int = 512
    adapter: Any = None

    # Internal state
    model: nn.Module = field(init=False, repr=False)
    tokenizer: Any = field(init=False, repr=False)
    _collator: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        from transformers import (
            AutoModelForMaskedLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
        )

        logger.info(f"Loading MaskedLM model: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_name)

        # Apply adapter if provided
        if self.adapter is not None:
            self._apply_adapter()

        # MLM collator handles masking
        self._collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=self.mlm_probability,
        )

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Parameters: {trainable:,} trainable / {total_params:,} total")

    def _apply_adapter(self) -> None:
        """Apply PEFT adapter to model."""
        from peft import LoraConfig, TaskType, get_peft_model

        if hasattr(self.adapter, "r"):  # LoRA or QLoRA
            peft_config = LoraConfig(
                task_type=TaskType.TOKEN_CLS,
                r=self.adapter.r,
                lora_alpha=self.adapter.alpha,
                lora_dropout=getattr(self.adapter, "dropout", 0.0),
                target_modules=getattr(self.adapter, "target_modules", None),
            )
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()

    def load_data(self, path: str) -> Dataset:
        """Load text data for MLM."""
        from trainformer.data.text import TextDataset

        return TextDataset(
            path=path,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )

    def collate_fn(self, examples: list[dict]) -> dict[str, Tensor]:
        """Collate with MLM masking."""
        # Stack input_ids from examples
        if "input_ids" in examples[0]:
            batch = {
                key: torch.stack([ex[key] for ex in examples])
                for key in examples[0].keys()
            }
            # Apply MLM masking
            return self._collator.torch_call([
                {"input_ids": batch["input_ids"][i]}
                for i in range(len(examples))
            ])
        return self._collator(examples)

    def train_step(self, batch: dict[str, Tensor]) -> Tensor:
        """Compute MLM loss."""
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            labels=batch["labels"],
        )
        return outputs.loss

    def eval_step(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Compute evaluation metrics."""
        with torch.no_grad():
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                labels=batch["labels"],
            )

        # Compute accuracy on masked tokens
        predictions = outputs.logits.argmax(dim=-1)
        labels = batch["labels"]
        mask = labels != -100  # -100 is ignore index

        correct = (predictions[mask] == labels[mask]).float().sum()
        total = mask.float().sum()

        return {
            "loss": outputs.loss,
            "accuracy": correct / (total + 1e-8),
            "perplexity": torch.exp(outputs.loss),
        }

    def parameters(self) -> Iterator[nn.Parameter]:
        """Return trainable parameters."""
        return (p for p in self.model.parameters() if p.requires_grad)

    def encode(self, texts: list[str]) -> Tensor:
        """Encode texts to embeddings (CLS token)."""
        self.model.eval()

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.base_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            # Return CLS token embedding
            return outputs.last_hidden_state[:, 0, :]

    def fill_mask(self, text: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Fill [MASK] token in text.

        Args:
            text: Text with [MASK] token
            top_k: Number of predictions to return

        Returns:
            List of {token, score, sequence} dicts
        """
        from transformers import pipeline

        fill_mask = pipeline("fill-mask", model=self.model, tokenizer=self.tokenizer)
        return fill_mask(text, top_k=top_k)

    def config_dict(self) -> dict[str, Any]:
        """Return task config for logging."""
        return {
            "task": "MaskedLM",
            "model_name": self.model_name,
            "mlm_probability": self.mlm_probability,
            "max_length": self.max_length,
            "adapter": type(self.adapter).__name__ if self.adapter else None,
        }

    def state_dict(self) -> dict:
        """Return model state for checkpointing."""
        return {"model": self.model.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        """Load model state."""
        self.model.load_state_dict(state["model"])
