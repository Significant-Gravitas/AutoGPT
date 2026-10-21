"""Helper functions for improved test assertions and error handling."""

from contextlib import contextmanager
from typing import Any, Iterator


@contextmanager
def override_config(settings: Any, attribute: str, value: Any) -> Iterator[None]:
    """Temporarily override a config attribute for testing.

    Warning: Directly mutates settings.config. If config is reloaded or cached
    elsewhere during the test, side effects may leak. Use with caution in
    parallel tests or when config is accessed globally.

    Args:
        settings: The settings object containing .config
        attribute: The config attribute name to override
        value: The temporary value to set
    """
    original = getattr(settings.config, attribute)
    setattr(settings.config, attribute, value)
    try:
        yield
    finally:
        setattr(settings.config, attribute, original)
