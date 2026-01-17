"""Evaluation utilities."""
from trainformer.eval.classification import (
    accuracy,
    confusion_matrix,
    f1_score,
    precision,
    recall,
    top_k_accuracy,
)
from trainformer.eval.feature_index import (
    FeatureIndex,
    knn_accuracy_from_index,
)
from trainformer.eval.generation import (
    bleu_score,
    perplexity,
    perplexity_from_loss,
    rouge_l,
)
from trainformer.eval.retrieval import (
    knn_accuracy,
    mean_average_precision,
    precision_at_k,
    recall_at_k,
)

__all__ = [
    # Classification
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "confusion_matrix",
    "top_k_accuracy",
    # Generation
    "perplexity",
    "perplexity_from_loss",
    "bleu_score",
    "rouge_l",
    # Retrieval
    "knn_accuracy",
    "mean_average_precision",
    "precision_at_k",
    "recall_at_k",
    # Feature Index
    "FeatureIndex",
    "knn_accuracy_from_index",
]
