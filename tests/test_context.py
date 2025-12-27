"""Tests for pipeline context."""
from trainformer.context import PipelineContext
from trainformer.types import Lifecycle


def test_context_lifecycle_step():
    """Test step-scoped values."""
    ctx = PipelineContext()
    ctx.set("loss", 0.5, Lifecycle.STEP)

    assert ctx.get("loss") == 0.5

    ctx.on_step_end()
    assert ctx.get("loss") is None


def test_context_lifecycle_epoch():
    """Test epoch-scoped values."""
    ctx = PipelineContext()
    ctx.set("best_loss", 0.1, Lifecycle.EPOCH)

    assert ctx.get("best_loss") == 0.1

    ctx.on_step_end()  # Should persist
    assert ctx.get("best_loss") == 0.1

    ctx.on_epoch_end()  # Should clear
    assert ctx.get("best_loss") is None


def test_context_lifecycle_run():
    """Test run-scoped values."""
    ctx = PipelineContext()
    ctx.set("experiment_id", "abc123", Lifecycle.RUN)

    ctx.on_step_end()
    ctx.on_epoch_end()

    assert ctx.get("experiment_id") == "abc123"


def test_context_metrics():
    """Test metric accumulation."""
    ctx = PipelineContext()

    ctx.log_metric("loss", 0.5, "train")
    ctx.log_metric("loss", 0.4, "train")
    ctx.log_metric("loss", 0.3, "train")

    metrics = ctx.get_epoch_metrics("train")
    assert abs(metrics["loss"] - 0.4) < 1e-6

    ctx.reset_epoch_metrics("train")
    assert ctx.get_epoch_metrics("train") == {}
