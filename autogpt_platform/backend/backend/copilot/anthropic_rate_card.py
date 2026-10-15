"""Anthropic rate-card lookup: the platform's own price for tokens.

Anthropic's API reports tokens, not a cost (``usage.cost`` is an
OpenRouter extension), so a turn on the direct route, an OpenRouter turn
whose bill cannot be read back, and a side call on the direct route are
priced here from token counts × per-model rates. The Claude Code CLI's
own estimate is never charged: its price table can lag Anthropic's.

Rates are vendored from LiteLLM's community-maintained pricing data
(``model_prices_and_context_window.json``) — only the Anthropic
entries are kept, filtered into ``anthropic_rates.json`` next to this
module. The litellm Python package itself is **not** a dependency, to
avoid pulling in its proxy-server module which has had a string of
auth-bypass / password-hash CVEs in stable releases.

Refresh: weekly cron in ``.github/workflows/refresh-anthropic-rates.yml``
re-runs ``scripts/refresh_anthropic_rates.py`` and opens a PR. Manual
refresh: ``poetry run python scripts/refresh_anthropic_rates.py``.

Anthropic prompt-caching docs (rates we apply when the source omits
the 1h cache-write field on a model):
https://docs.claude.com/en/docs/build-with-claude/prompt-caching

Scope note: this rate card prices what the platform's own account
spends on Anthropic models. Per-model catalog costs (the
ones the platform manages) live in ``backend/data/llm_registry/
catalog.py`` — see docs/platform/contributing/managing-llm-models.md.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, NamedTuple

logger = logging.getLogger(__name__)

_RATES_PATH = Path(__file__).parent / "anthropic_rates.json"

# Default cache-read multiplier when the source omits the field — Anthropic's
# documented rate is 10% of the input rate (0.1×).
_DEFAULT_CACHE_READ_MULTIPLIER = 0.1
# Default 1h cache-write multiplier when the source omits the field —
# Anthropic's documented rate is 2.0× input.
_DEFAULT_CACHE_WRITE_1H_MULTIPLIER = 2.0
# Default 5m cache-write multiplier — Anthropic's documented 1.25× input.
_DEFAULT_CACHE_WRITE_5M_MULTIPLIER = 1.25

# Conservative fallback rates used when the rate-card has no entry for the
# configured model (a fresh Claude release before a vendor refresh, a
# typo'd slug, or a finetune).  Set to claude-opus-4-1's published
# pricing — the most expensive Claude generally available — so an
# unknown slug **over-bills** rather than silently dropping cost.
# Billing integrity matters more than rate accuracy in the unknown-model
# tail; an alarming bill is recoverable, a missing one is not.
_FALLBACK_INPUT_PER_TOKEN = 15.0 / 1_000_000  # opus-4-1: $15/Mtok
_FALLBACK_OUTPUT_PER_TOKEN = 75.0 / 1_000_000  # opus-4-1: $75/Mtok


_rates_cache: dict[str, dict[str, Any]] | None = None


def _load_rates() -> dict[str, dict[str, Any]]:
    """Load the vendored Anthropic rate JSON once and cache in memory.

    Returns an empty dict on any failure so callers degrade to fallback
    pricing instead of crashing — billing must never short-circuit on a
    corrupt vendor file.
    """
    global _rates_cache
    if _rates_cache is not None:
        return _rates_cache
    try:
        raw = json.loads(_RATES_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.exception(
            "Failed to load %s; Anthropic direct-mode cost lookups will "
            "use fallback opus rates until the file is restored",
            _RATES_PATH.name,
        )
        _rates_cache = {}
        return _rates_cache
    rates = raw.get("rates") if isinstance(raw, dict) else None
    if not isinstance(rates, dict):
        logger.error(
            "Invalid rates JSON in %s (expected ``rates`` object at root); "
            "direct-mode cost lookups will use fallback opus rates",
            _RATES_PATH.name,
        )
        _rates_cache = {}
        return _rates_cache
    _rates_cache = {k: v for k, v in rates.items() if isinstance(v, dict)}
    return _rates_cache


def reset_cache() -> None:
    """Test-only: drop the in-memory rates cache."""
    global _rates_cache
    _rates_cache = None


def _resolve_info(model: str) -> dict[str, Any] | None:
    """Return the rate entry, trying the bare slug first then the
    ``anthropic/`` prefixed variant — LiteLLM keys both shapes
    inconsistently across model generations.
    """
    rates = _load_rates()
    for candidate in (model, f"anthropic/{model}"):
        entry = rates.get(candidate)
        if isinstance(entry, dict):
            return entry
    return None


def _is_anthropic_slug(model: str) -> bool:
    """Quick pre-check: does the slug look like an Anthropic model?

    Lets ``compute_anthropic_cost_usd`` short-circuit non-Anthropic
    slugs (``gpt-4o-mini``, ``google/gemini-2.5-pro``, ...) before
    hitting the rate-card lookup — those have no business going through
    the Anthropic rate card and must not trigger the opus fallback.
    """
    lowered = model.lower()
    if lowered.startswith(("anthropic/", "anthropic.")):
        return True
    # Bare ``claude-*`` slug with no provider prefix (direct-API form).
    if "/" in lowered:
        return False
    return lowered.startswith("claude-")


def anthropic_usage_cost_usd(
    *,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_5m_tokens: int = 0,
    cache_write_1h_tokens: int = 0,
) -> float | None:
    """USD for Anthropic-native usage: ``input_tokens`` excludes the cache
    buckets, and cache writes come split by their lifetime, as the
    Messages API reports them (``usage.cache_creation``).

    None for a slug that is not Anthropic's; an Anthropic slug the rate
    card lacks is priced at the conservative fallback, as in
    :func:`compute_anthropic_cost_usd`.
    """
    if not _is_anthropic_slug(model):
        return None
    rates = _rates_for(model)
    return (
        max(0, input_tokens) * rates.input
        + max(0, output_tokens) * rates.output
        + max(0, cache_read_tokens) * rates.cache_read
        + max(0, cache_write_5m_tokens) * rates.cache_write_5m
        + max(0, cache_write_1h_tokens) * rates.cache_write_1h
    )


def fallback_usage_cost_usd(
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float:
    """USD for usage on a model no rate card prices: the conservative
    fallback rates, every cache token at the input rate. Over-billing an
    unknown model is recoverable; a missing charge is not."""
    prompt = max(0, input_tokens + cache_read_tokens + cache_write_tokens)
    return (
        prompt * _FALLBACK_INPUT_PER_TOKEN
        + max(0, output_tokens) * _FALLBACK_OUTPUT_PER_TOKEN
    )


class _Rates(NamedTuple):
    input: float
    output: float
    cache_read: float
    cache_write_5m: float
    cache_write_1h: float


def _rates_for(model: str) -> _Rates:
    """Per-token rates for an Anthropic *model*, with the documented
    multipliers where the entry omits a cache field and the fallback
    rates where it has no entry."""
    info: dict[str, Any] = _resolve_info(model) or {}
    raw_input = info.get("input_cost_per_token")
    raw_output = info.get("output_cost_per_token")
    if isinstance(raw_input, (int, float)) and isinstance(raw_output, (int, float)):
        input_per_tok, output_per_tok = float(raw_input), float(raw_output)
    else:
        logger.error(
            "[rate_card] no entry for model=%s in anthropic_rates.json — "
            "falling back to opus-4-1 rates ($15/$75 per Mtok) so cost "
            "still records.  Run scripts/refresh_anthropic_rates.py to "
            "add this model.",
            model,
        )
        input_per_tok = _FALLBACK_INPUT_PER_TOKEN
        output_per_tok = _FALLBACK_OUTPUT_PER_TOKEN
        info = {}

    def _rate(field: str, multiplier: float) -> float:
        raw = info.get(field)
        if isinstance(raw, (int, float)):
            return float(raw)
        return input_per_tok * multiplier

    return _Rates(
        input=input_per_tok,
        output=output_per_tok,
        cache_read=_rate("cache_read_input_token_cost", _DEFAULT_CACHE_READ_MULTIPLIER),
        cache_write_5m=_rate(
            "cache_creation_input_token_cost", _DEFAULT_CACHE_WRITE_5M_MULTIPLIER
        ),
        cache_write_1h=_rate(
            "cache_creation_input_token_cost_above_1hr",
            _DEFAULT_CACHE_WRITE_1H_MULTIPLIER,
        ),
    )


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

    *cache_ttl* selects which cache-write field to read from the rate
    entry: ``5m`` → ``cache_creation_input_token_cost``, ``1h`` →
    ``cache_creation_input_token_cost_above_1hr``. Unknown TTLs fall
    back to the 1h field. When the rate entry omits the field
    entirely, we fall back to Anthropic's documented multiplier
    (1.25× input for 5m, 2.0× input for 1h, 0.1× input for cache reads).

    Returns ``None`` for clearly non-Anthropic slugs (``gpt-4o-mini``,
    ``google/gemini-2.5-pro``, ...) — caller should not have invoked the
    Anthropic rate card for those.

    When the rate card has no entry for an Anthropic model (or the entry
    is missing the input/output rates), an ERROR is logged and the
    fallback opus pricing applies so cost continues to flow downstream
    — billing must never silently drop on a configuration drift inside
    the Anthropic family.
    """
    if not _is_anthropic_slug(model):
        return None
    rates = _rates_for(model)
    prompt_tokens = max(0, prompt_tokens)
    cache_read_tokens = max(0, cache_read_tokens)
    cache_creation_tokens = max(0, cache_creation_tokens)
    fresh_input_tokens = max(
        0, prompt_tokens - cache_read_tokens - cache_creation_tokens
    )
    # 1h is the documented default TTL for our deployments. Unknown TTLs land
    # on it too: over-billing cache writes is preferable to mis-billing them.
    write_rate = rates.cache_write_5m if cache_ttl == "5m" else rates.cache_write_1h
    return (
        fresh_input_tokens * rates.input
        + max(0, completion_tokens) * rates.output
        + cache_read_tokens * rates.cache_read
        + cache_creation_tokens * write_rate
    )
