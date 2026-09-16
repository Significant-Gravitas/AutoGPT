"""Anthropic prompt-caching markers for OpenAI-compat requests.

Moved verbatim out of the deleted baseline engine
(``backend.copilot.baseline.service``): the style-eval harness
(``eval/style/generation.py``) still issues raw OpenAI-compat calls and
needs the same ``cache_control`` breakpoints + TTL handling.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from backend.copilot.moonshot import is_moonshot_model
from backend.copilot.service import config


def _is_anthropic_model(model: str) -> bool:
    """Return True if *model* routes to Anthropic (native or via OpenRouter).

    Examples that return True:
      - ``anthropic/claude-sonnet-4-6`` (OpenRouter route)
      - ``claude-3-5-sonnet-20241022`` (direct Anthropic API)
      - ``anthropic.claude-3-5-sonnet`` (Bedrock-style)

    False for ``openai/gpt-4o``, ``google/gemini-2.5-pro``, ``xai/grok-4``
    etc.  Moonshot is False here too even though Moonshot's
    Anthropic-compat endpoint honours ``cache_control`` — use
    :func:`_supports_prompt_cache_markers` for the cache-gating decision,
    which also allows Moonshot routes.  This function stays scoped to
    "genuinely Anthropic" so callers that need the stricter check (e.g.
    ``anthropic-beta`` header emission) keep their existing semantics.
    """
    lowered = model.lower()
    return "claude" in lowered or lowered.startswith("anthropic")


def _supports_prompt_cache_markers(model: str) -> bool:
    """Return True when *model* accepts Anthropic-style ``cache_control``.

    Superset of :func:`_is_anthropic_model` — also allows Moonshot
    (``moonshotai/*``), whose OpenRouter Anthropic-compat endpoint
    honours the marker and empirically lifts cache hit rate on
    continuation turns from near-zero (Moonshot's own automatic prefix
    cache, which drifts readily) to the 60-95% Anthropic ballpark.

    OpenAI / Grok / Gemini still 400 on ``cache_control``, so this
    function returns False for those providers — add new vendors here
    only after verifying their endpoint accepts the field.
    """
    return _is_anthropic_model(model) or is_moonshot_model(model)


def _fresh_ephemeral_cache_control() -> dict[str, str]:
    """Return a FRESH ephemeral ``cache_control`` dict each call.

    The ``ttl`` is sourced from :attr:`ChatConfig.baseline_prompt_cache_ttl`
    (default ``1h``) so the static prefix stays warm across many users'
    requests in the same workspace cache.  Anthropic caches are keyed
    per-workspace, so every copilot user reading the same system prompt
    hits the same cached entry.

    Using a shared module-level dict would let any downstream mutation
    (e.g. the OpenAI SDK normalising fields in-place) poison every future
    request's marker.  Construction is O(1) so the safety margin is free.
    """
    return {"type": "ephemeral", "ttl": config.baseline_prompt_cache_ttl}


def _fresh_anthropic_caching_headers() -> dict[str, str]:
    """Return a FRESH ``extra_headers`` dict requesting the Anthropic
    prompt-caching beta.

    Same reasoning as :func:`_fresh_ephemeral_cache_control`: never hand a
    shared module-level dict to third-party SDKs.  OpenRouter auto-forwards
    cache_control for Anthropic routes without this header, but passing it
    makes the intent unambiguous on-wire and is a no-op for non-Anthropic
    providers (unknown headers are dropped).
    """
    return {"anthropic-beta": "prompt-caching-2024-07-31"}


def _mark_tools_with_cache_control(
    tools: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return a copy of *tools* with ``cache_control`` on the last entry.

    Marking the last tool is a cache breakpoint that covers the whole tool
    schema block as a cacheable prefix segment.

    **Only call this for Anthropic model routes.**  Non-Anthropic providers
    (OpenAI, Grok, Gemini) reject the unknown ``cache_control`` field with
    a 400 schema validation error.  Gate via :func:`_is_anthropic_model`.
    """
    cached: list[dict[str, Any]] = [dict(t) for t in tools]
    if cached:
        cached[-1] = {
            **cached[-1],
            "cache_control": _fresh_ephemeral_cache_control(),
        }
    return cached


def _build_cached_system_message(
    system_message: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a copy of *system_message* with ``cache_control`` applied.

    Anthropic's cache uses prefix-match with up to 4 explicit breakpoints.
    Combined with the last-tool marker this gives two cache segments — the
    system block alone, and system+all-tools — so requests that share only
    the system prefix still get a partial cache hit.

    The system message is rebuilt via spread (``{**original, ...}``) so any
    unknown fields the caller set (e.g. ``name``) survive the transformation.
    Non-Anthropic models silently ignore the markers.

    Returns the original dict (shallow-copied) unchanged when the content
    shape is unsupported (missing / non-string / empty) — callers should
    splice it into the message list as-is in that case.
    """
    sys_copy = dict(system_message)
    sys_content = sys_copy.get("content")
    if isinstance(sys_content, str) and sys_content:
        sys_copy["content"] = [
            {
                "type": "text",
                "text": sys_content,
                "cache_control": _fresh_ephemeral_cache_control(),
            }
        ]
    return sys_copy
