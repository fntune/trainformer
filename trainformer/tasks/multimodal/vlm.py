"""Vision-Language Model task for multimodal understanding."""
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from trainformer.types import DatasetInfo


@dataclass
class VLM:
    """Vision-Language Model (LLaVA, Qwen-VL, etc.) for multimodal tasks.

    Supports image-conditioned text generation for VQA, captioning, etc.

    Args:
        model_name: HuggingFace model name
        adapter: Optional adapter for efficient fine-tuning
        max_length: Maximum sequence length
        gradient_checkpointing: Enable gradient checkpointing
    """

    model_name: str = "llava-hf/llava-1.5-7b-hf"
    adapter: Any = None
    max_length: int = 2048
    gradient_checkpointing: bool = True

    model: nn.Module = field(init=False)
    processor: Any = field(init=False)

    def __post_init__(self):
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )

        if self.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        if self.adapter is not None:
            self._apply_adapter()

    def _apply_adapter(self) -> None:
        """Apply PEFT adapter to language model."""
        from peft import LoraConfig, get_peft_model

        if hasattr(self.adapter, "r"):
            # Only apply LoRA to language model, not vision encoder
            peft_config = LoraConfig(
                r=self.adapter.r,
                lora_alpha=self.adapter.alpha,
                lora_dropout=self.adapter.dropout,
                target_modules=self.adapter.target_modules or ["q_proj", "v_proj", "k_proj", "o_proj"],
            )
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()

    def configure(self, info: DatasetInfo) -> None:
        """VLM doesn't need dataset configuration."""
        pass

    def parameters(self):
        """Return trainable parameters."""
        return [p for p in self.model.parameters() if p.requires_grad]

    def train_step(self, batch: dict[str, Tensor]) -> Tensor:
        """Compute VLM loss."""
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            pixel_values=batch.get("pixel_values"),
            labels=batch["labels"],
        )
        return outputs.loss

    def eval_step(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Compute evaluation outputs."""
        with torch.no_grad():
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                pixel_values=batch.get("pixel_values"),
                labels=batch["labels"],
            )
        return {
            "loss": outputs.loss.unsqueeze(0),
            "logits": outputs.logits,
        }

    def load_data(self, path: str):
        """Load VLM dataset.

        Expects dataset with 'image', 'question', 'answer' columns.
        """
        from datasets import load_dataset

        dataset = load_dataset(path)["train"]

        def process(examples):
            # Format conversation
            conversations = []
            for q, a in zip(examples["question"], examples["answer"]):
                conversations.append(f"USER: <image>\n{q}\nASSISTANT: {a}")

            # Process with processor
            inputs = self.processor(
                text=conversations,
                images=examples["image"],
                return_tensors="pt",
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
            )

            # Create labels (mask image tokens and user prompt)
            inputs["labels"] = inputs["input_ids"].clone()
            return inputs

        dataset = dataset.map(process, batched=True, remove_columns=dataset.column_names)
        dataset.set_format("torch")
        return dataset

    def generate(
        self,
        image: Any,
        prompt: str,
        max_new_tokens: int = 256,
        **kwargs,
    ) -> str:
        """Generate response for image and prompt."""
        conversation = f"USER: <image>\n{prompt}\nASSISTANT:"

        inputs = self.processor(
            text=conversation,
            images=image,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                **kwargs,
            )

        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        # Extract assistant response
        if "ASSISTANT:" in response:
            response = response.split("ASSISTANT:")[-1].strip()
        return response

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
