"""Anthropic rate-card lookup for direct-mode cost computation.

Used by the baseline path when ``CHAT_USE_OPENROUTER=false`` — the
OpenAI-compat endpoint at api.anthropic.com does **not** return a
``usage.cost`` field (that is an OpenRouter extension), so we compute
USD from token counts × per-model rates here.

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
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

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

# Conservative output-token cap when the rate-card has no entry / no
# max_output_tokens field — Opus 4.x publishes 32K, Sonnet 4.x publishes
# 64K.  Pick the lower bound so a fresh slug we don't recognise can't
# push the request over a concrete model's hard limit and 400 the API.
_FALLBACK_MAX_OUTPUT_TOKENS = 32_000


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


def get_max_output_tokens(model: str) -> int:
    """Return the model's max output tokens from the rate card, or a
    conservative fallback when the rate card has no entry.

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
    info: dict[str, Any] = _resolve_info(model) or {}
    raw_input = info.get("input_cost_per_token")
    raw_output = info.get("output_cost_per_token")
    if isinstance(raw_input, (int, float)) and isinstance(raw_output, (int, float)):
        input_per_tok: float = float(raw_input)
        output_per_tok: float = float(raw_output)
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
