"""Tests for task implementations."""
import pytest
import torch
from torch import nn

from trainformer.types import DatasetInfo


class TestImageClassification:
    """Tests for ImageClassification task."""

    def test_init(self):
        from trainformer.tasks import ImageClassification

        task = ImageClassification(backbone="mobilenetv3_small_100", num_classes=10)
        assert task.num_classes == 10
        assert hasattr(task, "model")

    def test_train_step(self):
        from trainformer.tasks import ImageClassification

        task = ImageClassification(backbone="mobilenetv3_small_100", num_classes=10)

        batch = (torch.randn(4, 3, 224, 224), torch.randint(0, 10, (4,)))
        loss = task.train_step(batch)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar

    def test_eval_step(self):
        from trainformer.tasks import ImageClassification

        task = ImageClassification(backbone="mobilenetv3_small_100", num_classes=10)

        batch = (torch.randn(4, 3, 224, 224), torch.randint(0, 10, (4,)))
        metrics = task.eval_step(batch)

        # eval_step returns predictions, labels, logits
        assert "predictions" in metrics
        assert "labels" in metrics
        assert "logits" in metrics

    def test_configure(self):
        from trainformer.tasks import ImageClassification

        # Create without num_classes
        task = ImageClassification(backbone="mobilenetv3_small_100")

        # Configure with dataset info
        info = DatasetInfo(num_samples=100, num_classes=5)
        task.configure(info)

        # _num_classes is the internal attribute
        assert task._num_classes == 5


class TestMetricLearning:
    """Tests for MetricLearning task."""

    def test_init(self):
        from trainformer.tasks import MetricLearning

        task = MetricLearning(backbone="mobilenetv3_small_100", embedding_dim=128)
        assert task.embedding_dim == 128

    def test_configure_creates_loss(self):
        from trainformer.tasks import MetricLearning

        task = MetricLearning(backbone="mobilenetv3_small_100", loss="arcface")
        info = DatasetInfo(num_samples=100, num_classes=10)
        task.configure(info)

        assert task._loss is not None
        assert task._num_classes == 10

    def test_train_step(self):
        from trainformer.tasks import MetricLearning

        task = MetricLearning(backbone="mobilenetv3_small_100", embedding_dim=128, loss="arcface")
        info = DatasetInfo(num_samples=100, num_classes=10)
        task.configure(info)

        batch = (torch.randn(4, 3, 224, 224), torch.randint(0, 10, (4,)))
        loss = task.train_step(batch)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

    def test_forward(self):
        from trainformer.tasks import MetricLearning

        task = MetricLearning(backbone="mobilenetv3_small_100", embedding_dim=128)

        images = torch.randn(4, 3, 224, 224)
        # Use model directly for inference
        task.model.eval()
        with torch.no_grad():
            embeddings = task.model(images)

        assert embeddings.shape == (4, 128)
        # Embeddings should be normalized
        norms = torch.norm(embeddings, dim=1)
        assert torch.allclose(norms, torch.ones(4), atol=1e-5)


class TestSSL:
    """Tests for SSL task."""

    def test_simclr(self):
        from trainformer.tasks import SSL

        task = SSL.simclr(backbone="mobilenetv3_small_100", proj_dim=64)
        assert task.method == "simclr"

    def test_moco(self):
        from trainformer.tasks import SSL

        task = SSL.moco(backbone="mobilenetv3_small_100", proj_dim=64)
        assert task.method == "moco"

    def test_byol(self):
        from trainformer.tasks import SSL

        task = SSL.byol(backbone="mobilenetv3_small_100", proj_dim=64)
        assert task.method == "byol"

    @pytest.mark.skip(reason="DINO uses weight_norm which has deepcopy issues")
    def test_dino(self):
        from trainformer.tasks import SSL

        task = SSL.dino(backbone="mobilenetv3_small_100")
        assert task.method == "dino"

    def test_mae(self):
        from trainformer.tasks import SSL

        task = SSL.mae(backbone="vit_tiny_patch16_224")
        assert task.method == "mae"


class TestCausalLM:
    """Tests for CausalLM task."""

    @pytest.mark.slow
    def test_init(self):
        from trainformer.tasks import CausalLM

        task = CausalLM(model_name="gpt2", max_length=128)
        assert task.max_length == 128
        assert task.tokenizer is not None

    @pytest.mark.slow
    def test_generate(self):
        from trainformer.tasks import CausalLM

        task = CausalLM(model_name="gpt2", max_length=128)
        output = task.generate("Hello", max_new_tokens=10)
        assert isinstance(output, str)
        assert len(output) > 0


class TestSeq2Seq:
    """Tests for Seq2Seq task."""

    @pytest.mark.slow
    def test_init(self):
        from trainformer.tasks import Seq2Seq

        task = Seq2Seq(model_name="t5-small", max_source_length=128, max_target_length=64)
        assert task.max_source_length == 128
        assert task.max_target_length == 64


class TestMaskedLM:
    """Tests for MaskedLM task."""

    @pytest.mark.slow
    def test_init(self):
        from trainformer.tasks import MaskedLM

        task = MaskedLM(model_name="bert-base-uncased", max_length=128)
        assert task.max_length == 128
        assert task.mlm_probability == 0.15


class TestCLIP:
    """Tests for CLIP task."""

    @pytest.mark.slow
    def test_init(self):
        from trainformer.tasks import CLIP

        task = CLIP(model_name="openai/clip-vit-base-patch32")
        assert task.model is not None


class TestVLM:
    """Tests for VLM task."""

    @pytest.mark.slow
    def test_init(self):
        from trainformer.tasks import VLM

        # This would require a lot of memory, so just test import
        pass
