import os
import random
import numpy as np
import pandas as pd
from typing import Union
from pathlib import Path

import torch
import torchmetrics

from loguru import logger

import hydra
from omegaconf import OmegaConf, DictConfig
from dora import get_xp


def get_lrs(optimizer):
    """Returns current learning rates for all param groups in the optimizer"""
    lrs = dict()
    for i, param_group in enumerate(optimizer.param_groups):
        lrs[f"lr/params_{i}"] = param_group["lr"]
    return lrs


def configure_optimizer(cfg, model, loss_head, total_steps=None):
    """
    instantiate the optimizer and scheduler
    when provided a scheduler that requires total_training_steps to be calculated
    pass total_steps : -1 for calcuation to be done here

    Args:
        cfg (dict): config dict
        model (torch.nn.Module): model to be trained
        total_steps (int): total number of steps received from the main training script
    """
    optim_cfg = dict(cfg.optimizer)
    learnable_params = [
        {
            "params": list(filter(lambda p: p.requires_grad, model.parameters())),
            "lr": optim_cfg["lr"],
        },
        {
            "params": list(filter(lambda p: p.requires_grad, loss_head.parameters())),
            "lr": optim_cfg.pop("loss_lr", optim_cfg["lr"]),
        },
    ]
    optimizer = hydra.utils.instantiate(
        params=learnable_params, config=optim_cfg, _convert_="partial"
    )

    scheduler_cfg = cfg.scheduler.lr_scheduler
    if "warmup_factor" in scheduler_cfg or "total_steps" in scheduler_cfg:
        scheduler_cfg.total_steps = total_steps
    scheduler = hydra.utils.instantiate(optimizer=optimizer, config=scheduler_cfg)
    return optimizer, scheduler


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    logger.debug(f"XXXXXXXXX \t Seeded everything with SEED : {seed} \t XXXXXXXXX")


def save_model(
    name: str,
    xp_sig: str,
    cfg: DictConfig,
    epoch: int,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim,
    objective_metric: torchmetrics.Metric,
):
    ckpt_dir = Path("ckpt")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        f"Saving model checkpoint at epoch : {epoch} to : {ckpt_dir.absolute()}"
    )

    state_dict = {
        "xp_sig": xp_sig,
        "cfg": cfg,
        "score": objective_metric.value,
        "epoch": epoch,
        "model": model.state_dict(),
        "criterion" : criterion.state_dict(),
        "optimizer": optimizer.state_dict(),
        "objective_metric": objective_metric.state_dict(),
    }

    torch.save(state_dict, ckpt_dir / f"{name}.pth")


def get_latest_checkpoint(cfg) -> Union[str, None]:
    last_ckpt = Path("ckpt") / "last.pth"
    if last_ckpt.is_file():
        return last_ckpt.absolute()
