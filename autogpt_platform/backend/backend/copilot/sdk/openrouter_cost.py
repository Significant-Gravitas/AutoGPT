"""Authoritative per-turn cost for OpenRouter-routed SDK generations.

The Claude Agent SDK CLI's ``ResultMessage.total_cost_usd`` is computed
from a static Anthropic pricing table baked into the binary.  For
non-Anthropic models routed through OpenRouter (e.g. Kimi K2.6) the CLI
silently falls back to Sonnet rates — empirically ~5x too high.  Even
after a rate-card override the estimate is still ~37% off in practice
because OpenRouter's own tokenizer counts, reasoning-token rollup, and
dated-snapshot pricing tiers can't be reconstructed from what the SDK
exposes locally.

This module provides :func:`record_turn_cost_from_openrouter` — an
``asyncio.create_task``-able coroutine that:

1. Queries ``https://openrouter.ai/api/v1/generation?id=<gen-id>`` for
   each generation ID captured during the turn.
2. Sums the authoritative ``total_cost`` across all rounds.
3. Calls :func:`persist_and_record_usage` **once** with the real number,
   updating both the cost-analytics row and the rate-limit counter.

If every lookup fails (404 / timeout / parse error), the caller's
``fallback_cost_usd`` is recorded instead — keeps the rate-limit counter
populated with the best available estimate rather than leaving the turn
uncharged.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import httpx

from backend.copilot.token_tracking import persist_and_record_usage

if TYPE_CHECKING:
    from backend.copilot.model import ChatSession

logger = logging.getLogger(__name__)

# OpenRouter docs:
# https://openrouter.ai/docs/api-reference/get-a-generation
_GENERATION_URL = "https://openrouter.ai/api/v1/generation"

# OpenRouter's generation endpoint indexes the billing row a few seconds
# after the SSE stream closes — observed ~8-12s in practice.  Retry with
# progressive backoff for up to ~30s total before giving up, so the typical
# indexing window (~10s) fits inside the retry envelope.  Backoff values
# in seconds summed: 0.5 + 1 + 2 + 4 + 8 + 15 = 30.5.
_MAX_RETRIES = 7
_BACKOFF_SECONDS = (0.5, 1.0, 2.0, 4.0, 8.0, 15.0)
_REQUEST_TIMEOUT = 10.0


async def _fetch_generation_cost(
    client: httpx.AsyncClient,
    gen_id: str,
    api_key: str,
    log_prefix: str,
) -> float | None:
    """Fetch the ``total_cost`` for one generation, with retries.

    Retries only on transient conditions:

    * HTTP 404 — row not yet indexed server-side (typical ~5-10s lag
      after the SSE stream closes)
    * HTTP 408 / 429 — timeout / rate limit
    * HTTP 5xx — transient OpenRouter outage
    * Network / ``httpx`` exceptions — transport-level retryable

    Fails fast on permanent client errors (401 Unauthorized,
    403 Forbidden, 400 Bad Request, etc.) since they can't recover
    within the retry window and would just burn API quota.

    Returns ``None`` when the endpoint reports no data, on a permanent
    failure, or when every retry attempt hits a transient error.
    """
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {"id": gen_id}
    last_error: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        if attempt > 0:
            await asyncio.sleep(_BACKOFF_SECONDS[attempt - 1])
        try:
            resp = await client.get(
                _GENERATION_URL,
                params=params,
                headers=headers,
                timeout=_REQUEST_TIMEOUT,
            )
            status = resp.status_code
            # Fast-fail on permanent client errors — retrying 401/403/400
            # just burns API quota and delays the fallback.
            if status in (400, 401, 403):
                logger.warning(
                    "%s OpenRouter /generation permanent error %d for %s — "
                    "not retrying (check API key / request shape)",
                    log_prefix,
                    status,
                    gen_id,
                )
                return None
            # Transient retryable: 404 (indexing lag), 408 (timeout),
            # 429 (rate limit), 5xx (server error).
            if status == 404 or status == 408 or status == 429 or status >= 500:
                last_error = RuntimeError(f"HTTP {status} on attempt {attempt + 1}")
                continue
            # Any other 4xx — treat as permanent.
            if status >= 400:
                logger.warning(
                    "%s OpenRouter /generation unexpected status %d for %s — "
                    "not retrying",
                    log_prefix,
                    status,
                    gen_id,
                )
                return None
            payload = resp.json().get("data")
            if not isinstance(payload, dict):
                logger.warning(
                    "%s OpenRouter /generation returned no data for %s",
                    log_prefix,
                    gen_id,
                )
                return None
            cost = payload.get("total_cost")
            if cost is None:
                logger.warning(
                    "%s OpenRouter /generation response missing total_cost "
                    "for %s (keys=%s)",
                    log_prefix,
                    gen_id,
                    sorted(payload.keys())[:10],
                )
                return None
            return float(cost)
        except Exception as exc:  # noqa: BLE001
            # Network / transport errors are retryable.
            last_error = exc
            continue
    logger.warning(
        "%s OpenRouter /generation lookup failed for %s after %d attempts: %s",
        log_prefix,
        gen_id,
        _MAX_RETRIES,
        last_error,
    )
    return None


async def record_turn_cost_from_openrouter(
    *,
    session: "ChatSession",
    user_id: str | None,
    model: str | None,
    prompt_tokens: int,
    completion_tokens: int,
    cache_read_tokens: int,
    cache_creation_tokens: int,
    generation_ids: list[str],
    fallback_cost_usd: float | None,
    api_key: str | None,
    log_prefix: str,
) -> None:
    """Persist turn cost from OpenRouter's authoritative ``/generation``.

    Writes a single cost-analytics row via :func:`persist_and_record_usage`
    — same method used for the Anthropic-direct sync path — so the
    cost-log append and rate-limit counter stay consistent.  No double
    counting: the caller skips its own sync persist for non-Anthropic
    OpenRouter turns and defers entirely to this task.

    Launched via ``asyncio.create_task`` from the stream ``finally`` block
    so the ~500-2000ms ``/generation`` indexing delay doesn't add latency
    to the turn.  During that window the rate-limit counter is briefly
    unaware of the turn's cost; back-to-back turns in that sub-second
    gap see a stale counter.  Acceptable tradeoff — the alternative
    (writing a possibly-wrong estimate synchronously) creates a
    double-count when the reconcile delta arrives.

    Fallback semantics: if every generation lookup fails, records
    ``fallback_cost_usd`` instead so the rate-limit counter isn't left
    completely empty.  Keeps behaviour at-worst equivalent to the
    rate-card estimate that came before this task existed.
    """
    if not generation_ids:
        return
    if not api_key:
        logger.debug(
            "%s OpenRouter cost record skipped: no API key available",
            log_prefix,
        )
        return

    try:
        async with httpx.AsyncClient() as client:
            tasks = [
                _fetch_generation_cost(client, gen_id, api_key, log_prefix)
                for gen_id in generation_ids
            ]
            results = await asyncio.gather(*tasks, return_exceptions=False)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "%s OpenRouter cost record failed to fetch any generation "
            "(falling back to rate-card estimate): %s",
            log_prefix,
            exc,
        )
        results = []

    fetched = [r for r in results if isinstance(r, (int, float))]
    if fetched and len(fetched) == len(generation_ids):
        real_cost: float | None = sum(fetched)
        logger.info(
            "%s[cost-record] OpenRouter real=$%.6f (gen_ids=%d)",
            log_prefix,
            real_cost,
            len(generation_ids),
        )
    else:
        real_cost = fallback_cost_usd
        if fetched:
            # Partial success: some lookups returned a cost, others didn't.
            # Trusting the partial sum would under-report; fall back to the
            # estimate so rate-limit enforcement stays conservative.
            logger.warning(
                "%s[cost-record] OpenRouter partial lookup (%d/%d) — "
                "using fallback estimate=$%s",
                log_prefix,
                len(fetched),
                len(generation_ids),
                real_cost,
            )
        else:
            logger.warning(
                "%s[cost-record] OpenRouter lookup failed for all gens — "
                "using fallback estimate=$%s",
                log_prefix,
                real_cost,
            )

    try:
        await persist_and_record_usage(
            session=session,
            user_id=user_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_creation_tokens=cache_creation_tokens,
            log_prefix=f"{log_prefix}[cost-record]",
            cost_usd=real_cost,
            model=model,
            provider="open_router",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "%s[cost-record] failed to persist: %s",
            log_prefix,
            exc,
        )
