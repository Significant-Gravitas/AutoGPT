"""Type definitions for LLM model metadata."""

from typing import NamedTuple


class ModelMetadata(NamedTuple):
    """Metadata for an LLM model."""

    provider: str
    context_window: int
    max_output_tokens: int | None

