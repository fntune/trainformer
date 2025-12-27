"""Example: Fine-tune LLMs with LoRA."""
from trainformer import Trainer
from trainformer.adapters import LoRA, QLoRA  # noqa: F401
from trainformer.callbacks import ModelCheckpoint
from trainformer.tasks import CausalLM, Seq2Seq  # noqa: F401

# --- LoRA Fine-tuning ---

task = CausalLM(
    model_name="gpt2",
    adapter=LoRA(r=8, alpha=16, dropout=0.1),
    max_length=512,
)

trainer = Trainer(
    task=task,
    data="data/text.jsonl",  # {"text": "..."} format
    epochs=3,
    batch_size=8,
    lr=2e-4,
    callbacks=[
        ModelCheckpoint(monitor="train/loss", mode="min"),
    ],
)

trainer.fit()

# Generate text
response = task.generate("Once upon a time", max_new_tokens=100)
print(response)


# --- QLoRA for Large Models ---

# task = CausalLM(
#     model_name="meta-llama/Llama-2-7b-hf",
#     adapter=QLoRA(r=16, alpha=32),
#     max_length=2048,
#     gradient_checkpointing=True,
# )
#
# trainer = Trainer(
#     task=task,
#     data="data/instructions.jsonl",
#     epochs=1,
#     batch_size=4,
#     lr=1e-4,
# )
#
# trainer.fit()


# --- Seq2Seq (T5, BART) ---

# task = Seq2Seq(
#     model_name="t5-base",
#     adapter=LoRA(r=8, alpha=16),
#     max_source_length=512,
#     max_target_length=128,
# )
#
# trainer = Trainer(
#     task=task,
#     data="data/summarization.jsonl",  # {"source": "...", "target": "..."}
#     epochs=3,
# )
#
# trainer.fit()
#
# # Generate summary
# summary = task.generate("Long article text here...", max_new_tokens=100)
