"""Example: Train ArcFace embeddings for image retrieval."""
from trainformer import Trainer
from trainformer.callbacks import EarlyStopping, ModelCheckpoint
from trainformer.tasks import MetricLearning

# Create task with ArcFace loss
task = MetricLearning(
    backbone="efficientnet_b0",
    embedding_dim=512,
    loss="arcface",
    margin=0.5,
    scale=64.0,
    pretrained=True,
)

# Train
trainer = Trainer(
    task=task,
    data="data/train",  # ImageFolder format
    val_data="data/val",
    epochs=100,
    batch_size=64,
    lr=1e-3,
    callbacks=[
        ModelCheckpoint(monitor="val/knn@5", mode="max"),
        EarlyStopping(monitor="val/knn@5", patience=10, mode="max"),
    ],
)

trainer.fit()

# Extract embeddings for retrieval
outputs = trainer.predict("data/query")
embeddings = outputs["embeddings"]


# --- Alternative losses ---

# CosFace loss
# task = MetricLearning(
#     backbone="resnet50",
#     loss="cosface",
#     margin=0.35,
# )

# Subcenter ArcFace (handles noisy labels)
# task = MetricLearning(
#     backbone="efficientnet_b3",
#     loss="subcenter",
#     margin=0.5,
# )
