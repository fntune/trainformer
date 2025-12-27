"""Configuration management with source tracking."""
from dataclasses import dataclass
from typing import Any

from trainformer.types import ConfigSource


@dataclass
class ConfigEntry:
    """A single config value with its source."""
    value: Any
    source: ConfigSource


class PipelineConfig:
    """Configuration registry with source tracking.

    Tracks where each config value came from (user, data, derived, env)
    to help with debugging and reproducibility.
    """

    def __init__(self):
        self._entries: dict[str, ConfigEntry] = {}

    def set(self, key: str, value: Any, source: ConfigSource = ConfigSource.USER) -> None:
        """Set a config value with source tracking."""
        self._entries[key] = ConfigEntry(value=value, source=source)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a config value."""
        entry = self._entries.get(key)
        return entry.value if entry else default

    def source(self, key: str) -> ConfigSource | None:
        """Get the source of a config value."""
        entry = self._entries.get(key)
        return entry.source if entry else None

    def __getitem__(self, key: str) -> Any:
        entry = self._entries.get(key)
        if entry is None:
            raise KeyError(key)
        return entry.value

    def __setitem__(self, key: str, value: Any) -> None:
        self.set(key, value, ConfigSource.USER)

    def __contains__(self, key: str) -> bool:
        return key in self._entries

    def items(self):
        """Iterate over key-value pairs."""
        for key, entry in self._entries.items():
            yield key, entry.value

    def to_dict(self) -> dict[str, Any]:
        """Export config as a plain dict."""
        return {key: entry.value for key, entry in self._entries.items()}

    def to_dict_with_sources(self) -> dict[str, dict[str, Any]]:
        """Export config with source info."""
        return {
            key: {"value": entry.value, "source": entry.source.name}
            for key, entry in self._entries.items()
        }

    def update_from_dict(self, d: dict[str, Any], source: ConfigSource = ConfigSource.USER) -> None:
        """Update config from a dict."""
        for key, value in d.items():
            self.set(key, value, source)

    def derive(self, key: str, fn, *deps: str) -> Any:
        """Derive a value from other config values."""
        args = [self.get(dep) for dep in deps]
        value = fn(*args)
        self.set(key, value, ConfigSource.DERIVED)
        return value
