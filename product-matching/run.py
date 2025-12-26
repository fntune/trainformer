import os
from pathlib import Path

import dotenv
import hydra
import numpy as np
import pandas as pd
import termplotlib as tpl
import torch
from loguru import logger
from omegaconf import DictConfig
from sklearn.model_selection import GroupKFold, train_test_split

from src.core.extensions.pretty_config import print_config
from src.core.extensions.xp import (
    get_git_diff,
    get_git_status,
    get_git_version,
    get_xp_sig,
)
from src.train import train
from src.core.utils import seed_everything
from src.utils.aws import download_images_from_s3

dotenv.load_dotenv(override=False)


def read_df(data_path):
    suffix = data_path.suffix
    logger.info(f"Reading dataframe at : {data_path}")

    if suffix == ".csv":
        df = pd.read_csv(data_path)
    elif suffix == ".parquet":
        df = pd.read_parquet(data_path)

    df.info(memory_usage="deep")
    df.to_parquet(data_path.name + ".parquet")
    return df


def prepare_data(df: pd.DataFrame, cfg: DictConfig):
    assert cfg.data.image_col in df.columns
    assert cfg.data.label_col in df.columns

    # download images
    if cfg.dataloader.download_images:
        logger.info(f"Downloading images..")
        # update "image" col with path to downloaded images
        df["image"] = download_images_from_s3(
            df[cfg.data.image_col],
            cfg.dataloader.data_dir,
        )

    # disk check for images
    if not cfg.dataloader.skip_image_check:
        assert df.image.apply(
            lambda img: os.path.isfile(os.path.join(cfg.dataloader.data_dir, img))
        ).all(), "not all images were found"

    df["label"] = df[cfg.data.label_col].astype("category").cat.codes
    return df


@hydra.main(config_path="./conf/", config_name="config")
def main(cfg: DictConfig):
    print_config(cfg)

    xp_sig = get_xp_sig(cfg)
    logger.info(f"XP sig : {xp_sig}")

    logger.info(f"Git head:\n{get_git_version()}")
    logger.info(f"Git status:\n{get_git_status()}")
    git_diff = get_git_diff()
    logger.info(f"Git diff:\n{git_diff}")
    if len(git_diff.split("\n")) > 1 and not cfg.xp.allow_uncommited_changes:
        raise Exception(
            f"Uncommited changes detected. Please commit changes before running."
        )

    logger.info(f"Cuda available : {torch.cuda.is_available()}")
    device = torch.device(cfg.trainer.device)
    logger.info(f"Chosen device : {device}")

    if cfg.seed_everything:
        seed_everything(cfg.seed_everything)

    # read data
    data_path = Path(cfg.data.data_path)
    logger.info(f"reading data at : {str(data_path)}")
    df = read_df(data_path)

    # prepare dataset
    df = prepare_data(df, cfg)

    # prepare folds
    df["fold"] = -1
    logger.info("Preparing {cfg.data.folds} data")
    if cfg.data.folds > 1:
        for fold, (_, val_idxs) in enumerate(
            GroupKFold(n_splits=cfg.data.folds).split(
                df.id, None, df[cfg.data.fold_group]
            )
        ):
            df.loc[val_idxs, "fold"] = fold
    else:
        raise ValueError(f"data.folds should be > 1")

    # train folds
    scores = list()
    folds = df.fold.max()
    for fold in range(folds + 1):
        logger.info(f"Training fold: {fold} / {folds}")
        train_df = df.loc[df.fold != fold].reset_index(drop=True).copy()
        train_df.label = train_df.label.astype("category").cat.codes

        val_df = df.loc[df.fold == fold].reset_index(drop=True).copy()
        val_df.label = val_df.label.astype("category").cat.codes

        # construct target clusters
        logger.info(f"Generating targets for fold : {fold}")
        val_df["target"] = val_df.label.map(
            val_df.reset_index(drop=False).groupby("label").index.agg(list).to_dict()
        )
        fig = tpl.figure()
        target_dist = val_df.target.apply(len).value_counts().sort_index()
        fig.barh(target_dist.values, target_dist.index)
        fig.show()

        # train
        score = train(
            cfg,
            train_df,
            val_df,
            xp_sig + f"-{fold}" if folds > 1 else "",
            fold,
            device,
        )
        logger.info(f"Fold {fold} / {folds} score : {score}")
        scores.append(score)

    scores = pd.DataFrame(scores)
    logger.info(f"Scores: \n{repr(scores)}")
    logger.info(f"Mean score : \n{scores.mean()}")
    return scores.mean()["f1"]


if __name__ == "__main__":
    main()
