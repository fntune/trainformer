"""CLIP for image-text contrastive learning."""
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from trainformer.types import DatasetInfo


@dataclass
class CLIP:
    """CLIP for image-text contrastive learning.

    Uses HuggingFace transformers for model loading.

    Args:
        model_name: HuggingFace CLIP model name
        adapter: Optional adapter for fine-tuning
        max_length: Maximum text sequence length
        temperature: Contrastive temperature (learnable if None)
    """

    model_name: str = "openai/clip-vit-base-patch32"
    adapter: Any = None
    max_length: int = 77
    temperature: float | None = None

    model: nn.Module = field(init=False)
    processor: Any = field(init=False)
    _logit_scale: nn.Parameter = field(init=False)

    def __post_init__(self):
        from transformers import CLIPModel, CLIPProcessor

        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model = CLIPModel.from_pretrained(self.model_name)

        # Learnable temperature
        if self.temperature is None:
            self._logit_scale = nn.Parameter(torch.ones([]) * 2.6592)  # ln(1/0.07)
        else:
            self._logit_scale = nn.Parameter(
                torch.ones([]) * torch.log(torch.tensor(1.0 / self.temperature)),
                requires_grad=False,
            )

        if self.adapter is not None:
            self._apply_adapter()

    def _apply_adapter(self) -> None:
        """Apply PEFT adapter."""
        from peft import LoraConfig, get_peft_model

        if hasattr(self.adapter, "r"):
            peft_config = LoraConfig(
                r=self.adapter.r,
                lora_alpha=self.adapter.alpha,
                lora_dropout=self.adapter.dropout,
                target_modules=self.adapter.target_modules or ["q_proj", "v_proj"],
            )
            self.model = get_peft_model(self.model, peft_config)

    def configure(self, info: DatasetInfo) -> None:
        """CLIP doesn't need dataset configuration."""
        pass

    def parameters(self):
        """Return trainable parameters."""
        params = [p for p in self.model.parameters() if p.requires_grad]
        params.append(self._logit_scale)
        return params

    def train_step(self, batch: dict[str, Tensor]) -> Tensor:
        """Compute contrastive loss."""
        # Get embeddings
        image_embeds = self.model.get_image_features(pixel_values=batch["pixel_values"])
        text_embeds = self.model.get_text_features(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
        )

        # Normalize
        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)

        # Compute logits
        logit_scale = self._logit_scale.exp()
        logits_per_image = logit_scale * image_embeds @ text_embeds.T
        logits_per_text = logits_per_image.T

        # Contrastive loss
        batch_size = image_embeds.size(0)
        labels = torch.arange(batch_size, device=image_embeds.device)

        loss_i2t = F.cross_entropy(logits_per_image, labels)
        loss_t2i = F.cross_entropy(logits_per_text, labels)

        return (loss_i2t + loss_t2i) / 2

    def eval_step(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Compute embeddings for evaluation."""
        with torch.no_grad():
            image_embeds = self.model.get_image_features(pixel_values=batch["pixel_values"])
            text_embeds = self.model.get_text_features(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
            )

        return {
            "image_embeddings": F.normalize(image_embeds, dim=-1),
            "text_embeddings": F.normalize(text_embeds, dim=-1),
        }

    def load_data(self, path: str):
        """Load image-text dataset.

        Expects dataset with 'image' and 'text' columns.
        """
        from datasets import load_dataset

        dataset = load_dataset(path)["train"]

        def process(examples):
            images = self.processor(
                images=examples["image"],
                return_tensors="pt",
                padding=True,
            )
            texts = self.processor(
                text=examples["text"],
                return_tensors="pt",
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
            )
            return {**images, **texts}

        dataset = dataset.map(process, batched=True, remove_columns=dataset.column_names)
        dataset.set_format("torch")
        return dataset

    def encode_image(self, images: Tensor) -> Tensor:
        """Encode images to embeddings."""
        with torch.no_grad():
            embeds = self.model.get_image_features(pixel_values=images)
        return F.normalize(embeds, dim=-1)

    def encode_text(self, texts: list[str]) -> Tensor:
        """Encode texts to embeddings."""
        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            embeds = self.model.get_text_features(**inputs)
        return F.normalize(embeds, dim=-1)

    def similarity(self, images: Tensor, texts: list[str]) -> Tensor:
        """Compute image-text similarity scores."""
        image_embeds = self.encode_image(images)
        text_embeds = self.encode_text(texts)
        return image_embeds @ text_embeds.T

    def info(self) -> dict[str, Any]:
        """Return task info."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            "model_name": self.model_name,
            "max_length": self.max_length,
            "total_params": total_params,
            "trainable_params": trainable_params,
        }
