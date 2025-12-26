#%%
import os
import gc
import shutil
import time
import json
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import ipyplot as iplt
import wandb
import cv2
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
import sys

sys.path.append("../src/core/")
from feature_index import FeatureIndex
import yaml
import timm
import optuna
import hydra
from omegaconf import DictConfig
from pprint import pprint
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from loguru import logger
from PIL import Image

images_dir = "/data/"
# %%
def compute_scores(pred, gt):
    assert isinstance(pred, (list, set))
    assert isinstance(gt, (list, set))
    pred, gt = set(pred), set(gt)

    tp = gt & pred
    iou = len(gt & pred) / len(gt | pred)
    if not (len(pred) == 0 or iou == 0):
        fp = pred - gt
        prec = len(tp) / len(pred)
        recall = len(tp) / len(gt)
        f1 = (2 * prec * recall) / (prec + recall)
    else:
        prec, recall, f1 = [0.0] * 3
    return {"iou": iou, "prec": prec, "recall": recall, "f1": f1}


def perform_average_embeddings(self, features, df):
    group_feats = df.reset_index(drop=False).groupby("id").index.agg(list)
    group_feats = group_feats.apply(lambda ids: features[ids].mean(axis=0))
    features = np.vstack(group_feats)
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    return (
        features,
        df.drop_duplicates("id")
        .set_index("id", drop=False)
        .loc[group_feats.index]
        .reset_index(drop=True),
    )


# %%
class MatchingImgDataset(Dataset):
    def __init__(self, df, images_dir, transforms=None):
        self.df = df
        self.images_dir = images_dir
        self.transforms = transforms

    def read_image(self, img_path):
        img_path = os.path.join(self.images_dir, img_path)
        # img = cv2.imread(img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.open(img_path).convert("RGB")
        return img

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.at[idx, "image"]
        img = self.read_image(img_path)
        if self.transforms:
            img = self.transforms(img)
        return img


# %%
def get_model(model_name, remove_fc=True):
    if model_name.startswith("dino"):
        model = torch.hub.load("facebookresearch/dino:main", model_name)
    else:
        print(f"Getting timm model: {model_name}")
        model = timm.create_model(model_name, pretrained=True)
        model = model.eval()

        model_cfg = model.default_cfg
        pprint(model_cfg)
        wandb.config.update(model_cfg)  ## Add model config to wandb

        # remove classifier head
        if remove_fc:
            classifier = model_cfg["classifier"]
            if isinstance(classifier, tuple):
                for head in classifier:
                    setattr(model, head, nn.Identity())
            if not "." in classifier:
                setattr(model, classifier, nn.Identity())
            else:
                ptr = model
                for head in classifier.split(".")[:-1]:
                    ptr = getattr(ptr, head)
                setattr(ptr, classifier.split(".")[-1], nn.Identity())

        # warmup and check model output dims
        print("Model input:", model_cfg["input_size"])
        with torch.no_grad():
            out = model(torch.randn(1, *model_cfg["input_size"]))
        print(f"Model output: {out.shape}")
        wandb.log({"model_input" : model_cfg["input_size"][-1], "model_output": out.shape[1]})
        if not (len(out.shape) == 2 and out.shape[0] == 1):
            raise ValueError(f"Model output shape is not (1, X) but {out.shape}")

    return model.cuda(), create_transform(**resolve_data_config({}, model=model))


def build_features(df, model_name):
    model, transforms = get_model(model_name)
    dataset = MatchingImgDataset(df, images_dir, transforms)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=8)
    features = list()

    with torch.no_grad():
        for batch in tqdm(dataloader):
            features.append(model(batch.cuda()))
    features = torch.cat(features, dim=0).cpu().numpy()
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    print("features", features.shape)
    return features

# %%
def sweep_matching(
    test_df,
    queries,
    NN,
    D,
    min_threshold=0.1,
    max_threshold=0.7,
    threshold_step=0.05,
    block_on=["gender", "brand"],
    block_off=[],
    variant_blocking=False,
):
    sweep = list()
    for threshold in np.arange(
        min_threshold, max_threshold + threshold_step, threshold_step
    ).tolist():
        for i, (matches, dists) in enumerate(zip(NN, D)):
            query = test_df.loc[queries[i]]
            matches = matches[np.logical_and(dists < threshold, dists != -1)].tolist()
            for meta_col in block_on:
                matches = [
                    match
                    for match in matches
                    if test_df.at[match, meta_col] == query[meta_col]
                ]
            for meta_col in block_off:
                matches = [
                    match
                    for match in matches
                    if test_df.at[match, meta_col] != query[meta_col]
                ]
            if variant_blocking:
                visited = set()
                final_matches = list()
                for match in matches:
                    pred_merchant = test_df.at[match, "merchant"]
                    if pred_merchant not in visited:
                        final_matches.append(match)
                        visited.add(pred_merchant)
                matches = final_matches
            scores = compute_scores(
                matches, test_df.loc[test_df.label == query.label, "id"].to_list()
            )
            scores["threshold"] = threshold
            sweep.append(scores)
    sweep = pd.DataFrame(sweep).groupby("threshold").mean()
    return sweep


# %%
@hydra.main(config_name="benchmark")
def main(cfg: DictConfig):
    model: str = cfg.model
    pprint(cfg)
    with wandb.init(project=cfg.project, name=model, config=cfg._content) as run:
        try:
            df = pd.read_csv(cfg.data_csv)
            test_df = df.drop_duplicates("id").set_index("id", drop=False)
            features = build_features(df, model_name=cfg.model)
            gc.collect()
            torch.cuda.empty_cache()

            index = FeatureIndex(name="", feature_size=features.shape[1])
            index.add(df.id.tolist(), features)
            NN, D = index.search(test_df.id.tolist(), k=cfg.get("k", 100))
            sweep = sweep_matching(
                test_df,
                test_df.id.to_list(),
                NN,
                D,
                min_threshold=cfg.get("min_threshold", 0.1),
                max_threshold=cfg.get("max_threshold", 0.5),
                threshold_step=cfg.get("threshold_step", 0.05),
                block_on=cfg.get("block_on", ["gender", "brand"]),
                block_off=cfg.get("block_off", []),
                variant_blocking=cfg.get("variant_blocking", False),
            )
            best = sweep.iloc[sweep.f1.argmax()]
            best = dict(threshold=best.name, **best.to_dict())
            wandb.log(best)
            print(best)
            del features
            del index
            gc.collect()
            # shutil.rmtree("/home/sour4bh/.cache/torch/", ignore_errors=True)
            return best["f1"]
        except Exception:
            logger.exception("Exception in main")
            return 0.


if __name__ == "__main__":
    main()
# %%
