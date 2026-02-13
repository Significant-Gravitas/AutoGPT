"""Type definitions for LLM model metadata."""

from typing import Literal, NamedTuple


class ModelMetadata(NamedTuple):
    """Metadata for an LLM model.

    Attributes:
        provider: The provider identifier (e.g., "openai", "anthropic")
        context_window: Maximum context window size in tokens
        max_output_tokens: Maximum output tokens (None if unlimited)
        display_name: Human-readable name for the model
        provider_name: Human-readable provider name (e.g., "OpenAI", "Anthropic")
        creator_name: Name of the organization that created the model
        price_tier: Relative cost tier (1=cheapest, 2=medium, 3=expensive)
    """

    provider: str
    context_window: int
    max_output_tokens: int | None
    display_name: str
    provider_name: str
    creator_name: str
    price_tier: Literal[1, 2, 3]
