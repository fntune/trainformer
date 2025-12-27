"""Tests for loss functions."""
import torch

from trainformer.models.components.losses import ArcFaceLoss, CosFaceLoss, SubcenterArcFace


def test_arcface_loss():
    """Test ArcFace loss forward pass."""
    loss_fn = ArcFaceLoss(embedding_dim=128, num_classes=10, margin=0.5, scale=64.0)

    embeddings = torch.randn(16, 128)
    embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
    labels = torch.randint(0, 10, (16,))

    loss = loss_fn(embeddings, labels)

    assert loss.shape == ()
    assert loss.item() > 0


def test_cosface_loss():
    """Test CosFace loss forward pass."""
    loss_fn = CosFaceLoss(embedding_dim=128, num_classes=10, margin=0.35, scale=64.0)

    embeddings = torch.randn(16, 128)
    embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
    labels = torch.randint(0, 10, (16,))

    loss = loss_fn(embeddings, labels)

    assert loss.shape == ()
    assert loss.item() > 0


def test_subcenter_arcface():
    """Test Subcenter ArcFace loss."""
    loss_fn = SubcenterArcFace(
        embedding_dim=128, num_classes=10, num_subcenters=3, margin=0.5, scale=64.0
    )

    embeddings = torch.randn(16, 128)
    embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
    labels = torch.randint(0, 10, (16,))

    loss = loss_fn(embeddings, labels)

    assert loss.shape == ()
    assert loss.item() > 0
