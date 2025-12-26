import os
import time
from typing import Optional

import faiss
import hydra
import numpy as np
import pandas as pd
import torch
import torch.cuda.amp as amp
import torch.nn.functional as F
import torchmetrics
import wandb
from loguru import logger
from omegaconf import DictConfig
from torch import nn

from .core import utils
from .core.extensions.model_summary import print_model_summary
from .core.extensions.prog_bar import ProgressBar
from .core.extensions.xp import format_hyperparams
from .datamodules.image import ImageDataset
from .testing import compute_scores, sweep_matching
from .utils.aws import download_images_from_s3
from .utils.logging import log_retrieval_images

torch.backends.cudnn.benchmark = True


def train_fn(cfg, model, criterion, optimizer, scheduler, dataloader, device, epoch):
    model.train()

    scaler = amp.GradScaler()
    loss_metric = torchmetrics.MeanMetric().to(device)

    with ProgressBar(
        total=len(dataloader),
        description=f"Train Epoch : {epoch} / {cfg.trainer.max_epochs}",
        ignore_prefix="lr",
    ) as prog_bar:
        for step, (x, y) in enumerate(dataloader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            with amp.autocast(enabled=cfg.trainer.amp):
                feats = model(x)
                loss = criterion(feats, y)
                loss = loss / cfg.trainer.n_accumulate

            scaler.scale(loss).backward()
            if (step + 1) % cfg.trainer.n_accumulate == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if cfg.scheduler.lr_scheduler_interval == "step":
                    scheduler.step()

            loss_metric.update(loss.item())
            if step % cfg.logging.log_every_n_steps == 0:
                metrics = {"train/loss": loss_metric.compute().item()}
                metrics.update(utils.get_lrs(optimizer))
                prog_bar.log(metrics)
                wandb.log(metrics)
                loss_metric.reset()
            prog_bar.advance()

        if cfg.scheduler.lr_scheduler_interval == "epoch":
            scheduler.step()
        return


@torch.no_grad()
def val_fn(val_df, model, dataloader, cfg, device, epoch):
    model.eval()

    with ProgressBar(
        total=len(dataloader),
        description=f"Val Epoch : {epoch}",
    ) as prog_bar:
        features = list()
        for batch in dataloader:
            features.append(model(batch.to(device, non_blocking=True)).detach())
            prog_bar.advance()
    features = torch.cat(features, dim=0).cpu().numpy()
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    logger.debug(f"features shape: {features.shape}")

    index = faiss.IndexFlatL2(features.shape[1])
    index.add(features)

    logger.info("searching index:")
    t = time.perf_counter()
    D, NN = index.search(features, cfg.eval.k)
    logger.info(f"index search time : {time.perf_counter() - t}")

    targets = val_df.target.to_numpy()
    scores = sweep_matching(
        NN,
        D,
        targets,
        min_threshold=cfg.eval.min_threshold,
        max_threshold=cfg.eval.max_threshold,
        threshold_step=cfg.eval.threshold_step,
    )
    mean_scores = (
        scores[["iou", "prec", "recall", "f1", "fβ", "threshold"]]
        .groupby("threshold")
        .mean()
    ).reset_index(drop=False)
    print(repr(mean_scores))

    # best_fβ = mean_scores.iloc[mean_scores.fβ.argmax()].to_dict()
    # logger.info(f"best fβ:\n{repr(best_fβ)}")
    # wandb.log({f"val/fβ/{k}": v for k, v in best_fβ.items()})

    best_f1 = mean_scores.iloc[mean_scores.f1.argmax()].to_dict()
    logger.info(f"best f1:\n{repr(best_f1)}")
    wandb.log({f"val/f1/{k}": v for k, v in best_f1.items()})

    if cfg.logging.log_retrieval_images:
        log_retrieval_images(
            NN,
            D,
            targets,
            (cfg.dataloader.data_dir + val_df.image).to_numpy(),
            epoch,
            n_samples=100,
            k=20,
        )
    return best_f1["f1"]


def train(
    cfg: DictConfig,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    xp_sig: str,
    fold: int,
    device: str,
) -> dict:

    if cfg.dataloader.num_workers == -1:
        cfg.dataloader.num_workers = os.cpu_count()
    logger.info(f"Setting num_workers to {cfg.dataloader.num_workers}")

    train_transform = hydra.utils.instantiate(cfg.transforms.train)
    val_transform = hydra.utils.instantiate(cfg.transforms.val)

    train_dataloader = torch.utils.data.DataLoader(
        ImageDataset(
            train_df,
            image_col=cfg.data.image_col,
            label_col=cfg.data.label_col,
            data_dir=cfg.dataloader.data_dir,
            backend=cfg.dataloader.backend,
            transform=train_transform,
        ),
        batch_size=cfg.dataloader.batch_size,
        shuffle=True,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=cfg.dataloader.pin_memory,
        drop_last=True,
    )

    val_dataloader = torch.utils.data.DataLoader(
        ImageDataset(
            val_df,
            image_col=cfg.data.image_col,
            label_col=None,
            data_dir=cfg.dataloader.data_dir,
            backend=cfg.dataloader.backend,
            transform=val_transform,
        ),
        batch_size=cfg.dataloader.batch_size * 2,
        shuffle=False,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=cfg.dataloader.pin_memory,
    )

    logger.info(f"Training on fold: {fold}")

    num_training_steps = int(
        (len(train_dataloader) * cfg.trainer.max_epochs) / cfg.trainer.n_accumulate
    )
    logger.debug(f"num_training_steps : {num_training_steps}")

    model = hydra.utils.instantiate(cfg.model)
    print_model_summary(model, max_depth=cfg.logging.model_summary_max_depth)
    model = model.to(device)

    num_classes = train_df.label.nunique()
    logger.info(f"Number of classes : {num_classes}")

    # criterion = nn.CrossEntropyLoss().to(device)
    criterion = hydra.utils.instantiate(
        cfg.loss, num_classes=num_classes, embedding_size=model.hidden_size
    ).to(device)
    logger.info(f"criterion: \n{criterion}")
    print_model_summary(criterion, 1)

    optimizer, lr_scheduler = utils.configure_optimizer(
        cfg, model, criterion, total_steps=num_training_steps
    )

    start_epoch = 0
    objective_metric = hydra.utils.instantiate(cfg.trainer.objective_metric)

    # load the latest checkpoint
    latest_checkpoint = utils.get_latest_checkpoint(cfg)
    if latest_checkpoint:
        if cfg.xp.resume:
            logger.info(f"Found checkpoint : {latest_checkpoint}")
            state_dict = torch.load(latest_checkpoint)
            assert xp_sig == state_dict["xp_sig"]
            start_epoch = state_dict["epoch"]

            logger.info(
                f"Loaded checkpoint : {latest_checkpoint} xp_sig : {state_dict['xp_sig']}"
            )
            logger.info(
                f"Checkpoint Score : {state_dict.get('score')} at epoch : {start_epoch}"
            )

            if cfg.xp.halt_on_existing_xp and start_epoch >= cfg.trainer.max_epochs:
                logger.warning(f"HALTING! Experiment : {xp_sig} already performed!")
                raise Exception("xp already done")

            model.load_state_dict(state_dict["model"])
            criterion.load_state_dict(state_dict["criterion"])
            optimizer.load_state_dict(state_dict["optimizer"])
            objective_metric.load_state_dict(state_dict["objective_metric"])
            if cfg.scheduler.lr_scheduler_interval == "step":
                lr_scheduler.last_epoch = int(
                    start_epoch * (num_training_steps / cfg.trainer.max_epochs)
                )
            else:
                lr_scheduler.last_epoch = start_epoch
            logger.info(
                f"Resuming lr scheduler at last_epoch/last_step : {lr_scheduler.last_epoch}"
            )
        else:
            logger.info(f"Skipping checkpoint load")
    else:
        logger.info("No checkpoint found! starting from scratch.")

    run = wandb.init(
        project=cfg.project,
        config=format_hyperparams(cfg),
        id=xp_sig,
        resume="allow" if cfg.xp.resume else False,
    )
    xp_sig = run.id
    logger.debug(f"xp_sig: {xp_sig}")

    # wandb.watch(model, log=cfg.logging.wandb.log, log_freq=cfg.logging.wandb.log_freq)
    # training loop
    for epoch in range(start_epoch, cfg.trainer.max_epochs):
        if (epoch != start_epoch and epoch % cfg.trainer.val_every_n_epochs == 0) or (
            epoch == start_epoch and cfg.trainer.val_at_start
        ):
            score = val_fn(val_df, model, val_dataloader, cfg, device, epoch)
            objective_metric.update(score)
            if objective_metric.value == score and cfg.xp.save_ckpt:
                utils.save_model(
                    "best",
                    xp_sig,
                    cfg,
                    epoch,
                    model,
                    criterion,
                    optimizer,
                    objective_metric,
                )

        for param_group, lr in utils.get_lrs(optimizer).items():
            logger.debug(f"{param_group} : {lr:e}")

        train_fn(
            cfg,
            model,
            criterion,
            optimizer,
            lr_scheduler,
            train_dataloader,
            device,
            epoch,
        )

        if cfg.xp.save_ckpt:
            if epoch % cfg.xp.save_ckpt_freq == 0:
                utils.save_model(
                    "last",
                    xp_sig,
                    cfg,
                    epoch,
                    model,
                    criterion,
                    optimizer,
                    objective_metric,
                )

    score = val_fn(val_df, model, val_dataloader, cfg, device, epoch)
    objective_metric.update(score)
    logger.debug(f"Final best score : {objective_metric.compute()}")

    wandb.finish()
    # return value of this function is used
    return objective_metric.compute().item()
