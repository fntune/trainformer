"""Example: Self-supervised pretraining with various methods."""
from trainformer import Trainer
from trainformer.callbacks import KNNEvaluator, ModelCheckpoint  # noqa: F401
from trainformer.tasks import SSL

# SimCLR pretraining
task = SSL.simclr(
    backbone="resnet50",
    proj_dim=128,
    temperature=0.1,
)

trainer = Trainer(
    task=task,
    data="data/unlabeled",
    epochs=100,
    batch_size=256,
    lr=3e-4,
    compile=True,  # Use torch.compile for speed
)

trainer.fit()


# --- Other SSL Methods ---

# MoCo v2 with momentum encoder
# task = SSL.moco(
#     backbone="resnet50",
#     proj_dim=128,
#     temperature=0.07,
#     momentum=0.999,
# )

# BYOL (no negatives needed)
# task = SSL.byol(
#     backbone="resnet50",
#     proj_dim=256,
#     momentum=0.996,
# )

# DINO with ViT
# task = SSL.dino(
#     backbone="vit_small_patch16_224",
#     momentum=0.996,
# )

# MAE for ViT pretraining
# task = SSL.mae(
#     backbone="vit_base_patch16_224",
# )


# --- With KNN Evaluation ---

# from torchvision.datasets import ImageFolder
#
# val_dataset = ImageFolder("data/val")
#
# trainer = Trainer(
#     task=SSL.simclr("resnet50"),
#     data="data/unlabeled",
#     epochs=100,
#     callbacks=[
#         KNNEvaluator(val_data=val_dataset, k=20),
#         ModelCheckpoint(monitor="val/knn@20", mode="max"),
#     ],
# )
#
# trainer.fit()
