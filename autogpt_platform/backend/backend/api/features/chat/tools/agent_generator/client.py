"""OpenRouter client configuration for agent generation."""

import os

from openai import AsyncOpenAI

# Configuration - use OPEN_ROUTER_API_KEY for consistency with chat/config.py
OPENROUTER_API_KEY = os.getenv("OPEN_ROUTER_API_KEY")
AGENT_GENERATOR_MODEL = os.getenv("AGENT_GENERATOR_MODEL", "anthropic/claude-opus-4.5")

# OpenRouter client (OpenAI-compatible API)
_client: AsyncOpenAI | None = None


def get_client() -> AsyncOpenAI:
    """Get or create the OpenRouter client."""
    global _client
    if _client is None:
        if not OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY environment variable is required")
        _client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )
    return _client
