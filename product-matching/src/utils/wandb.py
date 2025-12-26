import os
import wandb
import pandas as pd
from omegaconf import DictConfig, OmegaConf


def get_latest_model(run_path):
    """
    Fetches the latest model from an existing wandb run.
    Parameters:
            run_path: wandb run path/identifier (e.g. 'sour4bh/aisle3-image-encoder/jasdlk')
    Returns:
            Path to the model checkpoint. None if no run is found.
    """
    run = wandb.Api().run(run_path)
    artifacts = list(run.logged_artifacts())
    if len(artifacts):
        latest_artifact = pd.to_datetime(
            pd.Series([x.created_at for x in artifacts])
        ).argmax()
        latest_artifact = artifacts[latest_artifact]
        model_dir = latest_artifact.download()
        return os.path.join(model_dir, "model.ckpt")


def get_best_model(run_path):
    """
    Fetches the best model from a wandb run.
    Parameters:
            run: wandb run path/identifier (e.g. 'sour4bh/aisle3-image-encoder/jasdlk')
    Returns:
            Path to the model checkpoint. None if no run is found.
    """
    run = wandb.Api().run(run_path)
    best_model = max(run.logged_artifacts(), key=lambda x: x.metadata.get("score", 0.0))
    print(f"Score: {best_model.metadata['score']}")
    model_dir = best_model.download()
    return os.path.join(model_dir, "model.ckpt")

