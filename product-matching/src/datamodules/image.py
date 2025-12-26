import json
import os
from pathlib import Path
from typing import Union

import cv2
from PIL import Image
import albumentations as A

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

from loguru import logger


class ImageDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_col: str = "image",
        label_col: str = None,
        data_dir="./data/",
        backend: str = "cv2",
        transform: Union[T.Compose, A.Compose, None] = None,
        alpha_fill: int = 255,
    ):
        """
        Pytorch dataset class for Images
        Expects a dataframe containing an {image_col} column with the image file names.

        Parameters:
            df: pandas dataframe containing the image file names
            image_col: name of the column containing the image file names
            label_col: name of the column containing the labels
            data_dir: directory where the images are stored
            backend (str): backend to use for reading the images [cv2, pil].
            transform: albumentations or torchvision transforms to apply to the images
        """
        super().__init__()
        self.df = df
        self.image_col = image_col
        self.label_col = label_col
        self.data_dir = data_dir

        self.backend = backend
        self.alpha_fill = alpha_fill
        assert self.backend in ["cv2", "pil"]
        logger.info(f"Image I/O Backend: {self.backend}")

        logger.info(f"Transform: \n{repr(transform)}")
        if transform is not None:
            if isinstance(transform, A.Compose):
                self.transform = lambda img: transform(image=np.asarray(img))["image"]
            elif isinstance(transform, T.Compose):
                self.transform = transform
                self.backend = "pil"
            else:
                raise ValueError(
                    f"{transform} should be either albumentations or torchvision.transforms"
                )
        else:
            self.transform = None

    def read_cv2_image(self, fimg: str):
        img = cv2.imread(fimg, -1)
        # if image has alpha channel, fill bg with alpha_fill
        if img.shape[-1] == 4:
            mask = img[:, :, 3] == 0
            img[mask] = [self.alpha_fill] * 4
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def read_pil_image(self, fimg: str):
        with open(fimg, "rb") as f:
            return Image.open(f).convert("RGB")

    def read_image(self, fimg: str):
        fimg = os.path.join(self.data_dir, fimg)
        if self.backend == "cv2":
            return self.read_cv2_image(fimg)
        elif self.backend == "pil":
            return self.read_pil_image(fimg)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        fimg = self.df.at[idx, self.image_col]
        img = self.read_image(fimg)

        if self.transform is not None:
            img = self.transform(img)

        if self.label_col:
            label = self.df.at[idx, self.label_col]
            return img, torch.tensor(label, dtype=torch.long)
        return img
