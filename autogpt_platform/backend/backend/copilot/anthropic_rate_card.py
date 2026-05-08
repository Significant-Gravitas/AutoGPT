"""Anthropic rate-card lookup for direct-mode cost computation.

Used by the baseline path when ``CHAT_USE_OPENROUTER=false`` — the
OpenAI-compat endpoint at api.anthropic.com does **not** return a
``usage.cost`` field (that is an OpenRouter extension), so we compute
USD from token counts × per-model rates here.

Rates are sourced from LiteLLM's community-maintained model pricing
JSON, vendored at ``litellm_anthropic_rates.json`` next to this module.
The vendored copy is auto-refreshed via the ``refresh-litellm-rates``
GitHub Actions cron — that workflow downloads the upstream JSON,
filters to Claude entries, and opens a PR.  No runtime network
dependency: the JSON is a regular file read at module import.

LiteLLM source:
https://raw.githubusercontent.com/BerriAI/litellm/main/litellm/model_prices_and_context_window_backup.json

Anthropic prompt-caching docs (rates we apply when LiteLLM omits the
1h cache-write field on a model):
https://docs.claude.com/en/docs/build-with-claude/prompt-caching
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TypedDict, cast

logger = logging.getLogger(__name__)

_RATES_PATH = Path(__file__).parent / "litellm_anthropic_rates.json"

# Default cache-read multiplier when LiteLLM omits the field — Anthropic's
# documented rate is 10% of the input rate (0.1×).
_DEFAULT_CACHE_READ_MULTIPLIER = 0.1
# Default 1h cache-write multiplier when LiteLLM omits the field —
# Anthropic's documented rate is 2.0× input.
_DEFAULT_CACHE_WRITE_1H_MULTIPLIER = 2.0
# Default 5m cache-write multiplier — Anthropic's documented 1.25× input.
_DEFAULT_CACHE_WRITE_5M_MULTIPLIER = 1.25


class _RateEntry(TypedDict, total=False):
    input_cost_per_token: float
    output_cost_per_token: float
    cache_read_input_token_cost: float
    cache_creation_input_token_cost: float  # 5m default
    cache_creation_input_token_cost_above_1hr: float  # 1h


_rates_cache: dict[str, _RateEntry] | None = None


def _load_rates() -> dict[str, _RateEntry]:
    """Load + cache the LiteLLM rate JSON. Empty dict on any failure so a
    corrupt or missing file degrades to ``compute_anthropic_cost_usd``
    returning None (callers already handle that as a misconfiguration
    signal — see the warning at ``service.py:_record_title_generation_cost``
    and ``baseline/service.py:cost_missing_logged``).
    """
    global _rates_cache
    if _rates_cache is not None:
        return _rates_cache
    try:
        raw = json.loads(_RATES_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.exception(
            "Failed to load %s; Anthropic direct-mode cost lookups will "
            "return None until the file is restored",
            _RATES_PATH.name,
        )
        _rates_cache = {}
        return _rates_cache
    if not isinstance(raw, dict):
        logger.error(
            "Invalid LiteLLM rates JSON root in %s (expected object); "
            "direct-mode cost lookups disabled",
            _RATES_PATH.name,
        )
        _rates_cache = {}
        return _rates_cache
    # Filter defensively to entries shaped like LiteLLM rate rows.  We don't
    # require every field — `compute_anthropic_cost_usd` falls back per
    # bucket — but the row must at minimum carry the input + output rates.
    valid: dict[str, _RateEntry] = {}
    for key, value in raw.items():
        if not isinstance(value, dict):
            continue
        if not isinstance(value.get("input_cost_per_token"), (int, float)):
            continue
        if not isinstance(value.get("output_cost_per_token"), (int, float)):
            continue
        # Cast: above isinstance gates ensure the row matches _RateEntry's
        # structural shape; runtime keys beyond the TypedDict fields are
        # benign (they're ignored at access time via .get()).
        valid[key] = cast(_RateEntry, value)
    _rates_cache = valid
    return valid


def reset_cache() -> None:
    """Test-only: drop the in-memory rates cache so a freshly-written JSON
    re-loads on the next call."""
    global _rates_cache
    _rates_cache = None


def _resolve_entry(model: str) -> _RateEntry | None:
    """Return the LiteLLM entry for ``model``, trying the post-normalize
    slug first, then the ``anthropic/`` prefixed variant — LiteLLM's keys
    sometimes carry the prefix and sometimes don't, depending on the
    canonical alias the project picked at the time the entry was added.
    """
    rates = _load_rates()
    entry = rates.get(model)
    if entry is None:
        entry = rates.get(f"anthropic/{model}")
    return entry


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
    entry: ``5m`` → ``cache_creation_input_token_cost`` (LiteLLM's
    default-TTL field), ``1h`` → ``cache_creation_input_token_cost_above_1hr``.
    Unknown TTLs fall back to the 1h field.  When the LiteLLM entry omits
    the field entirely, we fall back to Anthropic's documented multiplier
    (1.25× input for 5m, 2.0× input for 1h, 0.1× input for cache reads).

    Returns ``None`` for unknown models so the caller can decide
    between recording 0 (under-bills) or skipping the row (silent
    miss).  We pick None to surface the misconfiguration upstream.
    """
    entry = _resolve_entry(model)
    if entry is None:
        return None
    input_per_tok = entry.get("input_cost_per_token")
    output_per_tok = entry.get("output_cost_per_token")
    if input_per_tok is None or output_per_tok is None:
        return None

    # Clamp each token bucket to ``>= 0`` per the codebase convention —
    # a malformed upstream that reports a negative count must not flip
    # the sign of the recorded cost (which would skew rate-limit and
    # billing accounting).
    prompt_tokens = max(0, prompt_tokens)
    completion_tokens = max(0, completion_tokens)
    cache_read_tokens = max(0, cache_read_tokens)
    cache_creation_tokens = max(0, cache_creation_tokens)
    fresh_input_tokens = max(
        0, prompt_tokens - cache_read_tokens - cache_creation_tokens
    )

    # Cache-read rate: prefer the explicit LiteLLM field, fall back to
    # Anthropic's documented 0.1× input.
    cache_read_per_tok = entry.get("cache_read_input_token_cost")
    if cache_read_per_tok is None:
        cache_read_per_tok = input_per_tok * _DEFAULT_CACHE_READ_MULTIPLIER

    # Cache-write rate: pick the field matching the configured TTL.  5m
    # is LiteLLM's default-TTL field; 1h is the explicit "above_1hr"
    # variant.  Fall back to Anthropic's documented multiplier when
    # absent.
    if cache_ttl == "5m":
        cache_write_per_tok = entry.get("cache_creation_input_token_cost")
        if cache_write_per_tok is None:
            cache_write_per_tok = input_per_tok * _DEFAULT_CACHE_WRITE_5M_MULTIPLIER
    else:
        # 1h is the documented default TTL for our deployments.  Unknown
        # TTLs land here too — over-billing on cache writes is preferable
        # to mis-billing.
        cache_write_per_tok = entry.get("cache_creation_input_token_cost_above_1hr")
        if cache_write_per_tok is None:
            cache_write_per_tok = input_per_tok * _DEFAULT_CACHE_WRITE_1H_MULTIPLIER

    fresh_input_cost = fresh_input_tokens * input_per_tok
    output_cost = completion_tokens * output_per_tok
    cache_read_cost = cache_read_tokens * cache_read_per_tok
    cache_write_cost = cache_creation_tokens * cache_write_per_tok
    return fresh_input_cost + output_cost + cache_read_cost + cache_write_cost
