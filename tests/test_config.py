"""Tests for configuration management."""
from trainformer.config import PipelineConfig
from trainformer.types import ConfigSource


def test_config_set_get():
    """Test basic config set/get."""
    config = PipelineConfig()
    config.set("lr", 0.001, ConfigSource.USER)

    assert config.get("lr") == 0.001
    assert config.source("lr") == ConfigSource.USER


def test_config_dict_interface():
    """Test dict-like interface."""
    config = PipelineConfig()
    config["batch_size"] = 32

    assert config["batch_size"] == 32
    assert "batch_size" in config


def test_config_to_dict():
    """Test export to dict."""
    config = PipelineConfig()
    config.set("lr", 0.001, ConfigSource.USER)
    config.set("batch_size", 32, ConfigSource.USER)

    d = config.to_dict()
    assert d == {"lr": 0.001, "batch_size": 32}


def test_config_derive():
    """Test derived values."""
    config = PipelineConfig()
    config.set("batch_size", 32, ConfigSource.USER)
    config.set("num_samples", 1000, ConfigSource.DATA)

    config.derive("steps_per_epoch", lambda n, b: n // b, "num_samples", "batch_size")

    assert config.get("steps_per_epoch") == 31
    assert config.source("steps_per_epoch") == ConfigSource.DERIVED
