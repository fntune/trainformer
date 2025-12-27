"""Example: Hyperparameter sweeps."""
from trainformer import Trainer
from trainformer.sweep import Sweep, choice, log_uniform, sweep
from trainformer.tasks import MetricLearning

# --- Decorator Style ---


@sweep(
    lr=log_uniform(1e-5, 1e-3),
    batch_size=choice(32, 64, 128),
    margin=choice(0.3, 0.5, 0.7),
)
def experiment(lr: float, batch_size: int, margin: float) -> float:
    task = MetricLearning(
        backbone="efficientnet_b0",
        loss="arcface",
        margin=margin,
    )

    trainer = Trainer(
        task=task,
        data="data/train",
        val_data="data/val",
        epochs=10,
        batch_size=batch_size,
        lr=lr,
    )

    trainer.fit()

    # Return metric to maximize
    return trainer.best_metrics.get("val/knn@5", 0.0)


# Run sweep
results = experiment.run(method="bayes", n_trials=20)
print(f"Best config: {results[0]}")


# --- Manual Sweep ---


def train_fn(lr: float, backbone: str) -> float:
    task = MetricLearning(backbone=backbone)
    trainer = Trainer(task=task, data="data/train", lr=lr, epochs=5)
    trainer.fit()
    return -trainer.best_metrics["train/loss"]


sweep_runner = Sweep(
    fn=train_fn,
    spaces={
        "lr": log_uniform(1e-5, 1e-2),
        "backbone": choice("resnet18", "resnet50", "efficientnet_b0"),
    },
)

# Grid search
# results = sweep_runner.run(method="grid")

# Random search
# results = sweep_runner.run(method="random", n_trials=50)

# Bayesian optimization
# results = sweep_runner.run(method="bayes", n_trials=30)

# Parallel execution
# results = sweep_runner.run(method="random", n_trials=20, n_parallel=4)
