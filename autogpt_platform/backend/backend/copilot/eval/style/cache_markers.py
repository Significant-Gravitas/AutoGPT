"""Anthropic-style prompt-cache markers for the style eval's OpenAI-compatible
requests.

The eval sends many turns that share one system prompt and tool list, so it
marks both as cacheable prefixes. Only models whose endpoint accepts the
``cache_control`` field get the markers; OpenAI, Grok and Gemini reject it.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from backend.copilot.moonshot import is_moonshot_model

# The static prefix stays warm across the eval's many requests.
_PROMPT_CACHE_TTL = "1h"


def supports_prompt_cache_markers(model: str) -> bool:
    """Whether *model*'s endpoint accepts Anthropic-style ``cache_control``:
    Anthropic models, and Moonshot's Anthropic-compatible endpoint."""
    lowered = model.lower()
    return (
        "claude" in lowered
        or lowered.startswith("anthropic")
        or is_moonshot_model(model)
    )


def caching_headers() -> dict[str, str]:
    """A fresh ``extra_headers`` dict requesting the prompt-caching beta.

    Never a shared module-level dict: third-party SDKs may mutate it.
    """
    return {"anthropic-beta": "prompt-caching-2024-07-31"}


def mark_tools_with_cache_control(
    tools: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """A copy of *tools* with ``cache_control`` on the last entry, a cache
    breakpoint covering the whole tool schema block."""
    cached: list[dict[str, Any]] = [dict(t) for t in tools]
    if cached:
        cached[-1] = {**cached[-1], "cache_control": _ephemeral_cache_control()}
    return cached


def cached_system_message(system_message: Mapping[str, Any]) -> dict[str, Any]:
    """A copy of *system_message* whose text carries ``cache_control``.

    With the last-tool marker this gives two cache segments: the system block
    alone, and system plus tools. A message whose content is not a non-empty
    string is returned as a plain copy.
    """
    sys_copy = dict(system_message)
    content = sys_copy.get("content")
    if isinstance(content, str) and content:
        sys_copy["content"] = [
            {
                "type": "text",
                "text": content,
                "cache_control": _ephemeral_cache_control(),
            }
        ]
    return sys_copy


def _ephemeral_cache_control() -> dict[str, str]:
    """A fresh ephemeral ``cache_control`` dict on every call, so no downstream
    mutation can poison a later request's marker."""
    return {"type": "ephemeral", "ttl": _PROMPT_CACHE_TTL}
