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

# Conservative fallback rates used when LiteLLM has no entry for the
# configured model (a fresh Claude release before a litellm version bump,
# a typo'd slug, or a finetune).  Set to claude-opus-4-1's published
# pricing — the most expensive Claude generally available — so an
# unknown slug **over-bills** rather than silently dropping cost.
# Billing integrity matters more than rate accuracy in the unknown-model
# tail; an alarming bill is recoverable, a missing one is not.
_FALLBACK_INPUT_PER_TOKEN = 15.0 / 1_000_000  # opus-4-1: $15/Mtok
_FALLBACK_OUTPUT_PER_TOKEN = 75.0 / 1_000_000  # opus-4-1: $75/Mtok


# Conservative output-token cap when LiteLLM has no entry / no max_output_tokens
# field — Opus 4.x publishes 32K, Sonnet 4.x publishes 64K.  Pick the lower
# bound so a fresh slug we don't recognise can't push the request over a
# concrete model's hard limit and 400 the API.
_FALLBACK_MAX_OUTPUT_TOKENS = 32_000


def get_max_output_tokens(model: str) -> int:
    """Return the model's max output tokens from LiteLLM, or a conservative
    fallback when LiteLLM has no entry.

    Used by the baseline path to cap ``max_tokens`` on direct-Anthropic
    extended-thinking requests so ``budget_tokens + response margin`` never
    exceeds the model's hard output limit (Anthropic 400s on overflow).
    """
    info = _resolve_info(model) or {}
    raw = info.get("max_output_tokens")
    if isinstance(raw, int) and raw > 0:
        return raw
    return _FALLBACK_MAX_OUTPUT_TOKENS


def _is_anthropic_slug(model: str) -> bool:
    """Quick pre-check: does the slug look like an Anthropic model?

    Lets ``compute_anthropic_cost_usd`` short-circuit non-Anthropic
    slugs (``gpt-4o-mini``, ``google/gemini-2.5-pro``, ...) before
    hitting the LiteLLM lookup — those have no business going through
    the Anthropic rate card and must not trigger the opus fallback.
    """
    lowered = model.lower()
    if lowered.startswith(("anthropic/", "anthropic.")):
        return True
    # Bare ``claude-*`` slug with no provider prefix (direct-API form).
    if "/" in lowered:
        return False
    return lowered.startswith("claude-")


def _resolve_info(model: str) -> dict[str, Any] | None:
    """Return the LiteLLM model_info entry, trying the bare slug first then
    the ``anthropic/`` prefixed variant — LiteLLM keys both shapes
    inconsistently across model generations.
    """
    # ``litellm.get_model_info`` is re-exported from a private submodule and
    # returns a TypedDict; cast to the common ``dict[str, Any]`` shape so call
    # sites can treat absent fields uniformly via ``.get()``.
    get_info = getattr(litellm, "get_model_info")
    for candidate in (model, f"anthropic/{model}"):
        try:
            info = get_info(candidate)
        except Exception:
            continue
        if info is not None:
            return dict(info)
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

    Returns ``None`` for clearly non-Anthropic slugs (``gpt-4o-mini``,
    ``google/gemini-2.5-pro``, ...) — caller should not have invoked the
    Anthropic rate card for those.

    When LiteLLM has no entry for an Anthropic model (or the entry is
    missing the input/output rates), an ERROR is logged and the fallback
    opus pricing applies so cost continues to flow downstream — billing
    must never silently drop on a configuration drift inside the
    Anthropic family.
    """
    if not _is_anthropic_slug(model):
        return None
    info: dict[str, Any] = _resolve_info(model) or {}
    raw_input = info.get("input_cost_per_token")
    raw_output = info.get("output_cost_per_token")
    if isinstance(raw_input, (int, float)) and isinstance(raw_output, (int, float)):
        input_per_tok: float = float(raw_input)
        output_per_tok: float = float(raw_output)
    else:
        logger.error(
            "[rate_card] no LiteLLM entry for model=%s — falling back "
            "to opus-4-1 rates ($15/$75 per Mtok) so cost still records. "
            "Bump litellm in pyproject.toml to add this model.",
            model,
        )
        input_per_tok = _FALLBACK_INPUT_PER_TOKEN
        output_per_tok = _FALLBACK_OUTPUT_PER_TOKEN
        info = {}  # cache fields fall through to default multipliers

    prompt_tokens = max(0, prompt_tokens)
    completion_tokens = max(0, completion_tokens)
    cache_read_tokens = max(0, cache_read_tokens)
    cache_creation_tokens = max(0, cache_creation_tokens)
    fresh_input_tokens = max(
        0, prompt_tokens - cache_read_tokens - cache_creation_tokens
    )

    raw_cache_read = info.get("cache_read_input_token_cost")
    cache_read_per_tok = (
        float(raw_cache_read)
        if isinstance(raw_cache_read, (int, float))
        else input_per_tok * _DEFAULT_CACHE_READ_MULTIPLIER
    )

    if cache_ttl == "5m":
        raw_cache_write = info.get("cache_creation_input_token_cost")
        cache_write_per_tok = (
            float(raw_cache_write)
            if isinstance(raw_cache_write, (int, float))
            else input_per_tok * _DEFAULT_CACHE_WRITE_5M_MULTIPLIER
        )
    else:
        # 1h is the documented default TTL for our deployments. Unknown
        # TTLs land here too — over-billing on cache writes is preferable
        # to mis-billing.
        raw_cache_write = info.get("cache_creation_input_token_cost_above_1hr")
        cache_write_per_tok = (
            float(raw_cache_write)
            if isinstance(raw_cache_write, (int, float))
            else input_per_tok * _DEFAULT_CACHE_WRITE_1H_MULTIPLIER
        )

    return (
        fresh_input_tokens * input_per_tok
        + completion_tokens * output_per_tok
        + cache_read_tokens * cache_read_per_tok
        + cache_creation_tokens * cache_write_per_tok
    )
