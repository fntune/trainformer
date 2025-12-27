"""Callbacks for training customization."""
from trainformer.callbacks.base import Callback, CallbackBase
from trainformer.callbacks.checkpoint import ModelCheckpoint
from trainformer.callbacks.early_stopping import EarlyStopping
from trainformer.callbacks.ema import EMA
from trainformer.callbacks.knn import KNNEvaluator, OnlineKNN

__all__ = [
    "Callback",
    "CallbackBase",
    "ModelCheckpoint",
    "EarlyStopping",
    "EMA",
    "KNNEvaluator",
    "OnlineKNN",
]
