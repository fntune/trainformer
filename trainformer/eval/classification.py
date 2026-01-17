"""Classification evaluation metrics."""
import torch
from torch import Tensor


def accuracy(preds: Tensor, targets: Tensor) -> float:
    """Compute classification accuracy.

    Args:
        preds: (N,) predicted class indices or (N, C) logits
        targets: (N,) ground truth class indices

    Returns:
        Accuracy as float in [0, 1]
    """
    if preds.ndim == 2:
        preds = preds.argmax(dim=-1)

    correct = (preds == targets).float()
    return correct.mean().item()


def precision(
    preds: Tensor,
    targets: Tensor,
    num_classes: int | None = None,
    average: str = "macro",
) -> float:
    """Compute precision for classification.

    Args:
        preds: (N,) predicted class indices or (N, C) logits
        targets: (N,) ground truth class indices
        num_classes: Number of classes (inferred if None)
        average: 'macro' (default), 'micro', or 'weighted'

    Returns:
        Precision score
    """
    if preds.ndim == 2:
        preds = preds.argmax(dim=-1)

    if num_classes is None:
        num_classes = max(preds.max().item(), targets.max().item()) + 1

    precisions = []
    supports = []

    for c in range(num_classes):
        pred_c = (preds == c)
        true_c = (targets == c)
        tp = (pred_c & true_c).sum().float()
        fp = (pred_c & ~true_c).sum().float()

        if tp + fp > 0:
            precisions.append((tp / (tp + fp)).item())
        else:
            precisions.append(0.0)
        supports.append(true_c.sum().item())

    if average == "micro":
        # Global TP / (TP + FP)
        tp_total = sum((preds == targets).sum().item() for _ in [1])
        return tp_total / len(preds)
    elif average == "weighted":
        total = sum(supports)
        if total == 0:
            return 0.0
        return sum(p * s for p, s in zip(precisions, supports)) / total
    else:  # macro
        return sum(precisions) / len(precisions) if precisions else 0.0


def recall(
    preds: Tensor,
    targets: Tensor,
    num_classes: int | None = None,
    average: str = "macro",
) -> float:
    """Compute recall for classification.

    Args:
        preds: (N,) predicted class indices or (N, C) logits
        targets: (N,) ground truth class indices
        num_classes: Number of classes (inferred if None)
        average: 'macro' (default), 'micro', or 'weighted'

    Returns:
        Recall score
    """
    if preds.ndim == 2:
        preds = preds.argmax(dim=-1)

    if num_classes is None:
        num_classes = max(preds.max().item(), targets.max().item()) + 1

    recalls = []
    supports = []

    for c in range(num_classes):
        pred_c = (preds == c)
        true_c = (targets == c)
        tp = (pred_c & true_c).sum().float()
        fn = (~pred_c & true_c).sum().float()

        if tp + fn > 0:
            recalls.append((tp / (tp + fn)).item())
        else:
            recalls.append(0.0)
        supports.append(true_c.sum().item())

    if average == "micro":
        # Micro recall = accuracy for multiclass
        return (preds == targets).float().mean().item()
    elif average == "weighted":
        total = sum(supports)
        if total == 0:
            return 0.0
        return sum(r * s for r, s in zip(recalls, supports)) / total
    else:  # macro
        return sum(recalls) / len(recalls) if recalls else 0.0


def f1_score(
    preds: Tensor,
    targets: Tensor,
    num_classes: int | None = None,
    average: str = "macro",
) -> float:
    """Compute F1 score for classification.

    Args:
        preds: (N,) predicted class indices or (N, C) logits
        targets: (N,) ground truth class indices
        num_classes: Number of classes (inferred if None)
        average: 'macro' (default), 'micro', or 'weighted'

    Returns:
        F1 score
    """
    p = precision(preds, targets, num_classes, average)
    r = recall(preds, targets, num_classes, average)

    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def confusion_matrix(
    preds: Tensor,
    targets: Tensor,
    num_classes: int | None = None,
) -> Tensor:
    """Compute confusion matrix.

    Args:
        preds: (N,) predicted class indices or (N, C) logits
        targets: (N,) ground truth class indices
        num_classes: Number of classes (inferred if None)

    Returns:
        (C, C) confusion matrix where rows are true labels, columns are predictions
    """
    if preds.ndim == 2:
        preds = preds.argmax(dim=-1)

    if num_classes is None:
        num_classes = max(preds.max().item(), targets.max().item()) + 1

    cm = torch.zeros(num_classes, num_classes, dtype=torch.long, device=preds.device)

    for t, p in zip(targets, preds):
        cm[t.item(), p.item()] += 1

    return cm


def top_k_accuracy(preds: Tensor, targets: Tensor, k: int = 5) -> float:
    """Compute top-k accuracy.

    Args:
        preds: (N, C) logits or probabilities
        targets: (N,) ground truth class indices
        k: Number of top predictions to consider

    Returns:
        Top-k accuracy as float in [0, 1]
    """
    if preds.ndim == 1:
        raise ValueError("preds must be 2D (N, C) for top_k_accuracy")

    _, topk_preds = preds.topk(k, dim=-1)
    correct = (topk_preds == targets.unsqueeze(1)).any(dim=1)
    return correct.float().mean().item()
