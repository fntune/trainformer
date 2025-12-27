"""CLI interface for trainformer."""
import logging

import fire

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def train(
    task: str,
    data: str,
    model: str | None = None,
    adapter: str | None = None,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-4,
    val_data: str | None = None,
    name: str | None = None,
    **kwargs,
):
    """Train a model.

    Args:
        task: Task type (ImageClassification, MetricLearning, SSL.simclr, etc.)
        data: Path to training data
        model: Model/backbone name
        adapter: Adapter type (LoRA, QLoRA, etc.)
        epochs: Number of epochs
        batch_size: Batch size
        lr: Learning rate
        val_data: Path to validation data
        name: Experiment name
        **kwargs: Additional task/trainer arguments

    Examples:
        trainformer train --task=ImageClassification --model=resnet50 --data=data/train
        trainformer train --task=MetricLearning --model=efficientnet_b0 --data=data/sop --loss=arcface
        trainformer train --task=SSL.simclr --model=resnet50 --data=data/imagenet
    """
    from trainformer import Trainer, tasks

    # Parse task (handle method syntax like SSL.simclr)
    if "." in task:
        task_cls_name, method = task.split(".", 1)
        task_cls = getattr(tasks, task_cls_name)
        task_obj = getattr(task_cls, method)(model or "resnet50", **kwargs)
    else:
        task_cls = getattr(tasks, task)
        if model:
            task_obj = task_cls(backbone=model, **kwargs)
        else:
            task_obj = task_cls(**kwargs)

    # Build trainer
    trainer = Trainer(
        task=task_obj,
        data=data,
        val_data=val_data,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        name=name,
    )

    trainer.fit()
    print(f"Training complete. Best metrics: {trainer.best_metrics}")


def predict(
    checkpoint: str,
    data: str,
    task: str,
    model: str | None = None,
    output: str | None = None,
    **kwargs,
):
    """Run inference with a trained model.

    Args:
        checkpoint: Path to checkpoint file
        data: Path to data
        task: Task type
        model: Model/backbone name
        output: Output file for predictions
        **kwargs: Additional task arguments
    """
    import torch

    from trainformer import Trainer, tasks

    # Build task
    if "." in task:
        task_cls_name, method = task.split(".", 1)
        task_cls = getattr(tasks, task_cls_name)
        task_obj = getattr(task_cls, method)(model or "resnet50", **kwargs)
    else:
        task_cls = getattr(tasks, task)
        if model:
            task_obj = task_cls(backbone=model, **kwargs)
        else:
            task_obj = task_cls(**kwargs)

    trainer = Trainer(task=task_obj, data=data, epochs=1)
    trainer.load(checkpoint)

    outputs = trainer.predict(trainer.data)

    if output:
        torch.save(outputs, output)
        print(f"Saved predictions to {output}")
    else:
        for key, value in outputs.items():
            print(f"{key}: shape={value.shape}")


def main():
    """Main CLI entry point."""
    fire.Fire({
        "train": train,
        "predict": predict,
    })


if __name__ == "__main__":
    main()
