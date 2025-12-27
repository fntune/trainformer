"""KNN evaluation callback for SSL training."""
import logging
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from trainformer.callbacks.base import CallbackBase
from trainformer.eval.retrieval import knn_accuracy

if TYPE_CHECKING:
    from trainformer.trainer import Trainer

logger = logging.getLogger(__name__)


class KNNEvaluator(CallbackBase):
    """Evaluate SSL models using KNN classifier during training.

    Computes KNN accuracy on a validation set at the end of each epoch.
    Useful for monitoring SSL training progress without labels.

    Args:
        val_data: Validation dataset or dataloader
        k: Number of neighbors for KNN
        eval_every_n_epochs: Evaluate every N epochs (1 = every epoch)
        batch_size: Batch size for feature extraction
        num_workers: DataLoader workers
    """

    def __init__(
        self,
        val_data: Any,
        k: int = 20,
        eval_every_n_epochs: int = 1,
        batch_size: int = 256,
        num_workers: int = 4,
    ):
        self.val_data = val_data
        self.k = k
        self.eval_every_n_epochs = eval_every_n_epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self._loader: DataLoader | None = None

    def on_fit_start(self, trainer: "Trainer") -> None:
        """Create validation dataloader."""
        if isinstance(self.val_data, DataLoader):
            self._loader = self.val_data
        else:
            self._loader = DataLoader(
                self.val_data,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=trainer.device == "cuda",
            )

    def on_epoch_end(self, trainer: "Trainer", epoch: int, metrics: dict[str, float]) -> None:
        """Compute KNN accuracy."""
        if (epoch + 1) % self.eval_every_n_epochs != 0:
            return

        embeddings, labels = self._extract_features(trainer)
        acc = knn_accuracy(embeddings, labels, k=self.k)

        metrics[f"val/knn@{self.k}"] = acc
        logger.info(f"KNN@{self.k} accuracy: {acc:.4f}")

    @torch.no_grad()
    def _extract_features(self, trainer: "Trainer") -> tuple[Tensor, Tensor]:
        """Extract features from validation set."""
        trainer.task.model.eval()

        all_embeddings = []
        all_labels = []

        for batch in self._loader:
            x, y = batch
            x = x.to(trainer.device)

            # Get embeddings from task's embed method or forward
            if hasattr(trainer.task.model, "embed"):
                embeddings = trainer.task.model.embed(x)
            else:
                embeddings = trainer.task.model(x)

            all_embeddings.append(embeddings.cpu())
            all_labels.append(y)

        trainer.task.model.train()

        return torch.cat(all_embeddings), torch.cat(all_labels)


class OnlineKNN(CallbackBase):
    """Online KNN using a memory bank.

    Maintains a memory bank of recent embeddings for fast KNN evaluation.
    More memory efficient than full dataset evaluation.

    Args:
        memory_size: Size of the memory bank
        k: Number of neighbors
        temperature: Softmax temperature for weighted KNN
        eval_every_n_steps: Evaluate every N steps
    """

    def __init__(
        self,
        memory_size: int = 65536,
        k: int = 20,
        temperature: float = 0.1,
        eval_every_n_steps: int = 100,
    ):
        self.memory_size = memory_size
        self.k = k
        self.temperature = temperature
        self.eval_every_n_steps = eval_every_n_steps

        self._memory_bank: Tensor | None = None
        self._memory_labels: Tensor | None = None
        self._ptr = 0

    def on_fit_start(self, trainer: "Trainer") -> None:
        """Initialize memory bank."""
        # Will be initialized on first batch when we know embedding dim
        self._memory_bank = None
        self._memory_labels = None
        self._ptr = 0

    def on_train_batch_end(
        self, trainer: "Trainer", batch: Any, batch_idx: int, loss: float
    ) -> None:
        """Update memory bank and optionally evaluate."""
        x, y = batch

        # Get embeddings
        with torch.no_grad():
            if hasattr(trainer.task.model, "embed"):
                embeddings = trainer.task.model.embed(x)
            else:
                embeddings = trainer.task.model(x)
            embeddings = embeddings.cpu()
            y = y.cpu()

        # Initialize memory bank if needed
        if self._memory_bank is None:
            embed_dim = embeddings.size(1)
            self._memory_bank = torch.zeros(self.memory_size, embed_dim)
            self._memory_labels = torch.zeros(self.memory_size, dtype=torch.long)

        # Update memory bank
        batch_size = embeddings.size(0)
        if self._ptr + batch_size > self.memory_size:
            # Wrap around
            first_part = self.memory_size - self._ptr
            self._memory_bank[self._ptr:] = embeddings[:first_part]
            self._memory_labels[self._ptr:] = y[:first_part]
            self._memory_bank[:batch_size - first_part] = embeddings[first_part:]
            self._memory_labels[:batch_size - first_part] = y[first_part:]
        else:
            self._memory_bank[self._ptr:self._ptr + batch_size] = embeddings
            self._memory_labels[self._ptr:self._ptr + batch_size] = y

        self._ptr = (self._ptr + batch_size) % self.memory_size

        # Evaluate
        if trainer.ctx.global_step % self.eval_every_n_steps == 0:
            acc = self._evaluate(embeddings, y)
            trainer.ctx.log_metric(f"knn@{self.k}", acc, "train")

    def _evaluate(self, query: Tensor, labels: Tensor) -> float:
        """Compute KNN accuracy using memory bank."""
        # Use filled portion of memory bank
        filled_size = min(self._ptr + 1, self.memory_size)
        memory = self._memory_bank[:filled_size]
        memory_labels = self._memory_labels[:filled_size]

        # Compute similarity
        sim = query @ memory.T / self.temperature

        # Get top-k
        _, topk_idx = sim.topk(min(self.k, filled_size), dim=1)
        topk_labels = memory_labels[topk_idx]

        # Check accuracy
        correct = (topk_labels == labels.unsqueeze(1)).any(dim=1)
        return correct.float().mean().item()
