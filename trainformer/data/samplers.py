"""Custom samplers for training."""
import logging
from collections import defaultdict
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

logger = logging.getLogger(__name__)


class ClassBalancedSampler(Sampler[int]):
    """Sampler that balances classes by oversampling minority classes."""

    def __init__(
        self,
        dataset: Dataset,
        labels: list[int] | np.ndarray | None = None,
    ):
        """Initialize class-balanced sampler.

        Args:
            dataset: Dataset with labels
            labels: Optional explicit labels (otherwise extracted from dataset)
        """
        self.dataset = dataset

        if labels is not None:
            self.labels = np.asarray(labels)
        elif hasattr(dataset, "targets"):
            self.labels = np.asarray(dataset.targets)
        elif hasattr(dataset, "labels"):
            self.labels = np.asarray(dataset.labels)
        else:
            # Extract labels by iterating
            self.labels = np.array([dataset[i][1] for i in range(len(dataset))])

        # Compute class weights (inverse frequency)
        class_counts = np.bincount(self.labels)
        weights = 1.0 / class_counts[self.labels]
        self.weights = torch.from_numpy(weights).double()

        self.num_samples = len(dataset)
        logger.info(f"ClassBalancedSampler: {len(class_counts)} classes, {self.num_samples} samples")

    def __iter__(self) -> Iterator[int]:
        indices = torch.multinomial(
            self.weights,
            self.num_samples,
            replacement=True,
        )
        return iter(indices.tolist())

    def __len__(self) -> int:
        return self.num_samples


class PKSampler(Sampler[list[int]]):
    """P-K sampler for metric learning: P classes, K samples per class per batch.

    Each batch contains P*K samples with exactly K samples from P different classes.
    This is useful for triplet/contrastive losses that need multiple samples per class.
    """

    def __init__(
        self,
        dataset: Dataset,
        p: int = 8,
        k: int = 4,
        labels: list[int] | np.ndarray | None = None,
    ):
        """Initialize P-K sampler.

        Args:
            dataset: Dataset with labels
            p: Number of classes per batch
            k: Number of samples per class per batch
            labels: Optional explicit labels
        """
        self.dataset = dataset
        self.p = p
        self.k = k

        if labels is not None:
            labels = np.asarray(labels)
        elif hasattr(dataset, "targets"):
            labels = np.asarray(dataset.targets)
        elif hasattr(dataset, "labels"):
            labels = np.asarray(dataset.labels)
        else:
            labels = np.array([dataset[i][1] for i in range(len(dataset))])

        # Build class -> indices mapping
        self.class_to_indices: dict[int, list[int]] = defaultdict(list)
        for idx, label in enumerate(labels):
            self.class_to_indices[int(label)].append(idx)

        # Filter classes with at least k samples
        self.valid_classes = [
            c for c, indices in self.class_to_indices.items()
            if len(indices) >= k
        ]

        if len(self.valid_classes) < p:
            raise ValueError(
                f"Need at least {p} classes with {k}+ samples, "
                f"but only {len(self.valid_classes)} available"
            )

        self.batch_size = p * k
        self.num_batches = len(dataset) // self.batch_size

        logger.info(
            f"PKSampler: P={p}, K={k}, {len(self.valid_classes)} valid classes, "
            f"{self.num_batches} batches"
        )

    def __iter__(self) -> Iterator[list[int]]:
        for _ in range(self.num_batches):
            # Sample P classes
            classes = np.random.choice(
                self.valid_classes,
                size=self.p,
                replace=False,
            )

            batch_indices = []
            for c in classes:
                indices = self.class_to_indices[c]
                # Sample K indices from this class (with replacement if needed)
                sampled = np.random.choice(
                    indices,
                    size=self.k,
                    replace=len(indices) < self.k,
                )
                batch_indices.extend(sampled.tolist())

            yield batch_indices

    def __len__(self) -> int:
        return self.num_batches


class DistributedPKSampler(PKSampler):
    """Distributed version of PKSampler for multi-GPU training."""

    def __init__(
        self,
        dataset: Dataset,
        p: int = 8,
        k: int = 4,
        labels: list[int] | np.ndarray | None = None,
        num_replicas: int | None = None,
        rank: int | None = None,
        seed: int = 0,
    ):
        super().__init__(dataset, p, k, labels)

        if num_replicas is None:
            if torch.distributed.is_initialized():
                num_replicas = torch.distributed.get_world_size()
            else:
                num_replicas = 1

        if rank is None:
            if torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
            else:
                rank = 0

        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0

        # Adjust num_batches for distributed
        self.num_batches = self.num_batches // num_replicas

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for deterministic shuffling."""
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        # Set seed for reproducibility across ranks
        np.random.seed(self.seed + self.epoch)

        all_batches = list(super().__iter__())

        # Each rank gets a subset of batches
        for i in range(self.rank, len(all_batches), self.num_replicas):
            yield all_batches[i]
