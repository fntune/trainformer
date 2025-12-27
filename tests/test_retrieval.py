"""Tests for retrieval metrics."""
import torch

from trainformer.eval.retrieval import knn_accuracy, precision_at_k, recall_at_k


def test_knn_accuracy_perfect():
    """Test KNN accuracy with perfectly clustered embeddings."""
    # Create perfectly separated embeddings
    embeddings = torch.zeros(20, 16)
    labels = torch.zeros(20, dtype=torch.long)

    # Class 0: embeddings in positive direction
    embeddings[:10, 0] = 1.0
    labels[:10] = 0

    # Class 1: embeddings in negative direction
    embeddings[10:, 0] = -1.0
    labels[10:] = 1

    embeddings = torch.nn.functional.normalize(embeddings, dim=-1)

    acc = knn_accuracy(embeddings, labels, k=5)
    assert acc == 1.0


def test_knn_accuracy_random():
    """Test KNN accuracy with random embeddings."""
    embeddings = torch.randn(100, 32)
    embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
    labels = torch.randint(0, 10, (100,))

    acc = knn_accuracy(embeddings, labels, k=5)

    # Random should be around 10% for 10 classes
    assert 0.0 <= acc <= 1.0


def test_precision_at_k():
    """Test precision@k."""
    embeddings = torch.randn(50, 16)
    embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
    labels = torch.randint(0, 5, (50,))

    prec = precision_at_k(embeddings, labels, k=10)
    assert 0.0 <= prec <= 1.0


def test_recall_at_k():
    """Test recall@k."""
    embeddings = torch.randn(50, 16)
    embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
    labels = torch.randint(0, 5, (50,))

    rec = recall_at_k(embeddings, labels, k=10)
    assert 0.0 <= rec <= 1.0
