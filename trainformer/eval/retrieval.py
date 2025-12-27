"""Retrieval evaluation metrics."""
import torch
from torch import Tensor


def knn_accuracy(embeddings: Tensor, labels: Tensor, k: int = 5) -> float:
    """Compute KNN accuracy using cosine similarity.

    Args:
        embeddings: (N, D) normalized embeddings
        labels: (N,) class labels
        k: Number of neighbors to consider

    Returns:
        Accuracy as float in [0, 1]
    """
    # Compute pairwise cosine similarity
    sim = embeddings @ embeddings.T

    # Exclude self-similarity by setting diagonal to -inf
    sim.fill_diagonal_(-float("inf"))

    # Get top-k indices
    _, topk_idx = sim.topk(k, dim=1)

    # Get labels of neighbors
    neighbor_labels = labels[topk_idx]

    # Check if any neighbor has the same label
    correct = (neighbor_labels == labels.unsqueeze(1)).any(dim=1)

    return correct.float().mean().item()


def precision_at_k(embeddings: Tensor, labels: Tensor, k: int = 10) -> float:
    """Compute Precision@K for retrieval.

    Args:
        embeddings: (N, D) normalized embeddings
        labels: (N,) class labels
        k: Number of retrieved items

    Returns:
        Mean precision@k across all queries
    """
    sim = embeddings @ embeddings.T
    sim.fill_diagonal_(-float("inf"))

    _, topk_idx = sim.topk(k, dim=1)
    neighbor_labels = labels[topk_idx]

    # Count correct retrievals
    correct = (neighbor_labels == labels.unsqueeze(1)).float()
    precision = correct.sum(dim=1) / k

    return precision.mean().item()


def mean_average_precision(embeddings: Tensor, labels: Tensor, k: int | None = None) -> float:
    """Compute Mean Average Precision (mAP) for retrieval.

    Args:
        embeddings: (N, D) normalized embeddings
        labels: (N,) class labels
        k: Optional cutoff (None for full ranking)

    Returns:
        mAP score
    """
    sim = embeddings @ embeddings.T
    sim.fill_diagonal_(-float("inf"))

    n = embeddings.size(0)
    if k is None:
        k = n - 1

    _, sorted_idx = sim.topk(k, dim=1)

    aps = []
    for i in range(n):
        query_label = labels[i]
        retrieved_labels = labels[sorted_idx[i]]

        # Find relevant items
        relevant = (retrieved_labels == query_label).float()
        num_relevant = relevant.sum()

        if num_relevant == 0:
            continue

        # Compute precision at each relevant position
        cumsum = torch.cumsum(relevant, dim=0)
        positions = torch.arange(1, k + 1, device=embeddings.device).float()
        precisions = cumsum / positions

        ap = (precisions * relevant).sum() / num_relevant
        aps.append(ap.item())

    return sum(aps) / len(aps) if aps else 0.0


def recall_at_k(embeddings: Tensor, labels: Tensor, k: int = 10) -> float:
    """Compute Recall@K for retrieval.

    Args:
        embeddings: (N, D) normalized embeddings
        labels: (N,) class labels
        k: Number of retrieved items

    Returns:
        Mean recall@k across all queries
    """
    sim = embeddings @ embeddings.T
    sim.fill_diagonal_(-float("inf"))

    n = embeddings.size(0)
    _, topk_idx = sim.topk(k, dim=1)
    neighbor_labels = labels[topk_idx]

    recalls = []
    for i in range(n):
        query_label = labels[i]
        # Total relevant items (excluding self)
        total_relevant = (labels == query_label).sum().item() - 1

        if total_relevant == 0:
            continue

        # Retrieved relevant items
        retrieved_relevant = (neighbor_labels[i] == query_label).sum().item()
        recalls.append(retrieved_relevant / min(total_relevant, k))

    return sum(recalls) / len(recalls) if recalls else 0.0
