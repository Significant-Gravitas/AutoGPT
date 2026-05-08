"""Anthropic rate-card lookup for direct-mode cost computation.

Used by the baseline path when ``CHAT_USE_OPENROUTER=false`` — the
OpenAI-compat endpoint at api.anthropic.com does **not** return a
``usage.cost`` field (that is an OpenRouter extension), so we compute
USD from token counts × per-model rates here.

Rates come from the ``litellm`` package's bundled
``model_prices_and_context_window.json`` via :func:`litellm.get_model_info`.
No vendored JSON, no refresh cron — refreshing the rates is a
``litellm`` version bump in ``pyproject.toml`` (handled by Dependabot).

LiteLLM exposes a ``cost_per_token()`` helper, but it doesn't take a
``cache_ttl`` argument and only consults the 5-minute cache-write
field. Our deployments default to 1-hour ephemeral caches
(``baseline_prompt_cache_ttl='1h'``), which Anthropic prices ~60%
higher, so we read the rates directly and apply the TTL ourselves.

Anthropic prompt-caching docs (rates we apply when LiteLLM omits the
1h cache-write field on a model):
https://docs.claude.com/en/docs/build-with-claude/prompt-caching
"""

from __future__ import annotations

import logging
from typing import Any

import litellm

logger = logging.getLogger(__name__)

# Default cache-read multiplier when LiteLLM omits the field — Anthropic's
# documented rate is 10% of the input rate (0.1×).
_DEFAULT_CACHE_READ_MULTIPLIER = 0.1
# Default 1h cache-write multiplier when LiteLLM omits the field —
# Anthropic's documented rate is 2.0× input.
_DEFAULT_CACHE_WRITE_1H_MULTIPLIER = 2.0
# Default 5m cache-write multiplier — Anthropic's documented 1.25× input.
_DEFAULT_CACHE_WRITE_5M_MULTIPLIER = 1.25


def _resolve_info(model: str) -> dict[str, Any] | None:
    """Return the LiteLLM model_info entry, trying the bare slug first then
    the ``anthropic/`` prefixed variant — LiteLLM keys both shapes
    inconsistently across model generations.
    """
    for candidate in (model, f"anthropic/{model}"):
        try:
            return litellm.get_model_info(candidate)
        except Exception:
            continue
    return None


def compute_anthropic_cost_usd(
    *,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
    cache_ttl: str = "1h",
) -> float | None:
    """Return the USD cost for an Anthropic-direct chat completion.

    ``prompt_tokens`` is the OpenAI-compat top-level total — on
    Anthropic's compat endpoint it **includes** cached + cache-write
    tokens (matching the OpenAI spec, where the cached subset lives
    in ``prompt_tokens_details``).  We subtract those buckets out
    before computing the fresh-input cost so each token is billed
    exactly once at its correct rate.  If the upstream over-reports
    the breakdown (cached + write > total), the fresh-input bucket is
    clamped to zero to avoid flipping the sign.

    *cache_ttl* selects which cache-write field to read from the LiteLLM
    entry: ``5m`` → ``cache_creation_input_token_cost``, ``1h`` →
    ``cache_creation_input_token_cost_above_1hr``. Unknown TTLs fall
    back to the 1h field. When the LiteLLM entry omits the field
    entirely, we fall back to Anthropic's documented multiplier
    (1.25× input for 5m, 2.0× input for 1h, 0.1× input for cache reads).

    Returns ``None`` for unknown models so the caller can decide
    between recording 0 (under-bills) or skipping the row (silent
    miss).  We pick None to surface the misconfiguration upstream.
    """
    info = _resolve_info(model)
    if info is None:
        return None
    input_per_tok = info.get("input_cost_per_token")
    output_per_tok = info.get("output_cost_per_token")
    if not isinstance(input_per_tok, (int, float)) or not isinstance(
        output_per_tok, (int, float)
    ):
        return None

    prompt_tokens = max(0, prompt_tokens)
    completion_tokens = max(0, completion_tokens)
    cache_read_tokens = max(0, cache_read_tokens)
    cache_creation_tokens = max(0, cache_creation_tokens)
    fresh_input_tokens = max(
        0, prompt_tokens - cache_read_tokens - cache_creation_tokens
    )

    cache_read_per_tok = info.get("cache_read_input_token_cost")
    if not isinstance(cache_read_per_tok, (int, float)):
        cache_read_per_tok = input_per_tok * _DEFAULT_CACHE_READ_MULTIPLIER

    if cache_ttl == "5m":
        cache_write_per_tok = info.get("cache_creation_input_token_cost")
        if not isinstance(cache_write_per_tok, (int, float)):
            cache_write_per_tok = input_per_tok * _DEFAULT_CACHE_WRITE_5M_MULTIPLIER
    else:
        # 1h is the documented default TTL for our deployments. Unknown
        # TTLs land here too — over-billing on cache writes is preferable
        # to mis-billing.
        cache_write_per_tok = info.get("cache_creation_input_token_cost_above_1hr")
        if not isinstance(cache_write_per_tok, (int, float)):
            cache_write_per_tok = input_per_tok * _DEFAULT_CACHE_WRITE_1H_MULTIPLIER

    return (
        fresh_input_tokens * input_per_tok
        + completion_tokens * output_per_tok
        + cache_read_tokens * cache_read_per_tok
        + cache_creation_tokens * cache_write_per_tok
    )
