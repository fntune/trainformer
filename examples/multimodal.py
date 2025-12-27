"""Example: Multimodal training with CLIP and VLM."""
import torch

from trainformer import Trainer
from trainformer.adapters import LoRA
from trainformer.tasks import CLIP, VLM  # noqa: F401

# --- CLIP Fine-tuning ---

task = CLIP(
    model_name="openai/clip-vit-base-patch32",
    adapter=LoRA(r=8, alpha=16),
    max_length=77,
)

trainer = Trainer(
    task=task,
    data="data/image_text_pairs",  # {"image": ..., "text": "..."}
    epochs=10,
    batch_size=32,
    lr=1e-5,
)

trainer.fit()

# Compute image-text similarity
images = torch.randn(4, 3, 224, 224)  # Your images
texts = ["a photo of a cat", "a photo of a dog", "sunset", "mountain"]
similarity = task.similarity(images, texts)
print(similarity)  # 4x4 similarity matrix


# --- VLM Fine-tuning (LLaVA) ---

# from PIL import Image
#
# task = VLM(
#     model_name="llava-hf/llava-1.5-7b-hf",
#     adapter=LoRA(r=16, alpha=32),
#     max_length=2048,
#     gradient_checkpointing=True,
# )
#
# trainer = Trainer(
#     task=task,
#     data="data/vqa",  # {"image": ..., "question": "...", "answer": "..."}
#     epochs=1,
#     batch_size=2,
#     lr=1e-4,
# )
#
# trainer.fit()
#
# # Visual question answering
# image = Image.open("photo.jpg")
# response = task.generate(image, "What is in this image?")
# print(response)
