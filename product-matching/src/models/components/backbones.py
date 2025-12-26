import json
from loguru import logger
from typing import Union

import torch
from torch import nn
from torch.functional import F

import timm
from timm.data.transforms_factory import create_transform
from timm.data import resolve_data_config


class TimmBackbone(nn.Module):
    def __init__(
        self,
        model_name: str,
        pretrained=True,
        pooling: Union[str, torch.nn.Module] = "avg",
        apply_bn: bool = True,
        attach_linear_head: bool = True,
        linear_head_dim: int = 512,
        dropout_rate: float = 0.0,
    ):
        """
        Args:
            model_name (str): name of the model to use
            pretrained (bool): if True, loads pretrained weights
            num_classes (int): number of output classes for the classifier (default: 0)
            pooling (str): global pooling type. Can be 'avg', 'max', 'avgmax', 'catavgmax', ''
        Returns:
            nn.Module: a PyTorch model
        """
        super().__init__()

        self.model_name = model_name
        self.pooling = pooling
        self.apply_bn = apply_bn
        self.attach_linear_head = attach_linear_head
        self.linear_head_dim = linear_head_dim
        self.dropout_rate = dropout_rate

        self.backbone = timm.create_model(
            self.model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool=self.pooling if isinstance(self.pooling, str) else "",
        )
        logger.info(f"Pooling: \n{repr(self.pooling)}")
        if not isinstance(self.pooling, str):
            assert isinstance(
                self.pooling, nn.Module
            ), "pooling must be a torch.nn.Module or a string accepted by timm constructor"
        else:
            self.pooling = None

        self.model_cfg = self.backbone.default_cfg
        logger.info(
            f"Model's default timm config:\n{json.dumps(self.model_cfg, indent=2)}"
        )
        self.hidden_size = self.backbone.num_features
        logger.info(f"model hidden size: {self.hidden_size}")

        if pretrained and self.model_cfg["url"] == "":
            logger.error(f"{model_name} does not have pretrained weights available!")

        if self.apply_bn:
            self.bn = nn.BatchNorm1d(self.hidden_size)

        if self.attach_linear_head:
            self.linear_head = nn.Sequential(
                nn.Dropout(p=self.dropout_rate),
                nn.Linear(self.hidden_size, self.linear_head_dim, bias=True),
                nn.BatchNorm1d(self.linear_head_dim),
            )
            self.hidden_size = self.linear_head_dim
        logger.debug(f"Final embedding out dim : {self.hidden_size}")

    def forward(self, x):
        x = self.backbone(x)
        if self.pooling is not None:
            x = self.pooling(x).flatten(1)
        if self.apply_bn:
            x = self.bn(x)
        if self.attach_linear_head:
            x = self.linear_head(x)
        return F.normalize(x)


class HuggingFaceBackbone(nn.Module):
    def __init__(self, model_name, pretrained=False, remove_fc=True):
        super().__init__()
        pass
