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

If some or all lookups fail (404 / timeout / parse error), the calls that
did not resolve are priced from the turn's own token usage and the LLM
catalog (:mod:`turn_cost_estimate`) — keeps the rate-limit counter
populated with the best available estimate rather than leaving the turn
uncharged.  The caller's ``fallback_cost_usd`` (the CLI's figure, a
running session total on resumed sessions) is recorded only for a model
the catalog cannot price.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
from langfuse import get_client

from backend.copilot.sdk.turn_cost_estimate import (
    FallbackCost,
    TokenUsage,
    fallback_turn_cost,
)
from backend.copilot.token_tracking import persist_and_record_usage
from backend.util import json

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


def _gen_ids_from_jsonl(path: Path) -> set[str]:
    """Extract ``gen-`` message IDs from every assistant entry in a
    Claude CLI JSONL file.

    Tolerant of malformed lines: single bad JSON object doesn't block
    the whole file.  Also reads ``redacted_thinking`` / ``thinking``
    entries that share an ID with their parent (via ``jq -u`` in the
    CLI) and dedups by caller.
    """
    ids: set[str] = set()
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line, fallback=None)
                if not isinstance(entry, dict):
                    continue
                if entry.get("type") != "assistant":
                    continue
                message = entry.get("message")
                if not isinstance(message, dict):
                    continue
                msg_id = message.get("id")
                if isinstance(msg_id, str) and msg_id.startswith("gen-"):
                    ids.add(msg_id)
    except (OSError, UnicodeDecodeError) as exc:
        logger.debug(
            "Failed to scan JSONL for gen-IDs: path=%s err=%s",
            path,
            exc,
        )
    return ids


def _discover_turn_subagent_gen_ids(
    project_dir: Path,
    session_id: str,
    turn_start_ts: float,
    known: list[str],
) -> list[str]:
    """Gen-IDs from this session's subagents created during this turn.

    Main-turn LLM rounds (incl. fallback retries) arrive on the live
    stream as ``AssistantMessage`` and land on ``known`` via
    ``message_id``.  What's NOT on ``known`` is the CLI's subagent LLM
    calls — chiefly auto-compaction, which spawns a fresh JSONL under
    ``<project_dir>/<session_id>/subagents/agent-acompact-*.jsonl``
    whose gen-IDs never touch our main adapter.  OpenRouter bills them
    anyway, so without this sweep compaction turns under-report cost.

    Scoping: ONLY the current session's subagent dir
    (``<project_dir>/<session_id>/subagents/agent-*.jsonl``) and ONLY
    files whose ``mtime >= turn_start_ts``.  Without both guards we'd
    merge prior turns' gen-IDs (main JSONL accumulates forever) and
    foreign sessions' gen-IDs (the project dir contains every session
    for this cwd), double-billing the user.

    Also covers non-compaction subagents (Task tool etc.) when the CLI
    spawns them — their live-stream visibility depends on SDK version,
    so the sweep is a safety net.  The dedup against ``known`` means
    anything already captured live doesn't double count.

    Preserves ``known`` ordering so main-turn IDs stay first; only
    appends truly new IDs from the sweep.
    """
    merged: list[str] = list(known)
    seen = set(merged)
    subagents_dir = project_dir / session_id / "subagents"
    if not subagents_dir.exists():
        return merged
    try:
        for jsonl in subagents_dir.glob("agent-*.jsonl"):
            try:
                if jsonl.stat().st_mtime < turn_start_ts:
                    continue
            except OSError:
                continue
            for gen_id in _gen_ids_from_jsonl(jsonl):
                if gen_id not in seen:
                    seen.add(gen_id)
                    merged.append(gen_id)
    except OSError as exc:
        logger.debug("Failed to walk subagents dir=%s: %s", subagents_dir, exc)
    return merged


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
    cli_project_dir: str | None,
    cli_session_id: str | None,
    turn_start_ts: float | None,
    fallback_cost_usd: float | None,
    api_key: str | None,
    log_prefix: str,
    langfuse_trace_id: str | None = None,
    generation_usage: dict[str, TokenUsage] | None = None,
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

    Fallback semantics: when any generation lookup fails, records the
    provider's cost for the generations that resolved plus a catalog
    estimate for the rest of the turn's tokens (``generation_usage`` maps
    each generation to the tokens the live stream reported for it), so the
    rate-limit counter isn't left empty.  ``fallback_cost_usd`` is recorded
    only when the catalog has no price for ``model``.
    """
    if not api_key:
        logger.debug(
            "%s OpenRouter cost record skipped: no API key available",
            log_prefix,
        )
        return

    # Merge in any gen-IDs from CLI subagent JSONLs the live stream
    # didn't surface — chiefly SDK-internal compaction, which spawns a
    # summarisation LLM call under
    # ``<project_dir>/<cli_session_id>/subagents/...`` that OpenRouter
    # bills but doesn't emit via our main adapter.  Safe no-op when no
    # compaction happened (no subagent files created this turn) or the
    # CLI wrote nothing there.
    #
    # The sweep is SESSION-scoped (``<cli_session_id>/subagents/``, not
    # the whole project dir) and TURN-scoped (mtime >= turn_start_ts).
    # Both guards are load-bearing: the project dir contains every
    # session for this cwd, and subagent files persist across turns,
    # so an unscoped sweep would re-bill prior turns and foreign
    # sessions' gen-IDs.
    if cli_project_dir and cli_session_id and turn_start_ts is not None:
        merged_ids = _discover_turn_subagent_gen_ids(
            Path(os.path.expanduser(cli_project_dir)),
            cli_session_id,
            turn_start_ts,
            generation_ids,
        )
        if len(merged_ids) != len(generation_ids):
            logger.info(
                "%s[cost-record] discovered %d additional gen-IDs in "
                "session subagents (compaction / Task) — reconcile "
                "covers all",
                log_prefix,
                len(merged_ids) - len(generation_ids),
            )
        generation_ids = merged_ids

    if not generation_ids:
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
    cost_source = "fallback"
    fallback: FallbackCost | None = None
    if fetched and len(fetched) == len(generation_ids):
        real_cost: float | None = sum(fetched)
        cost_source = "openrouter"
        # Log real (OpenRouter billed) vs CLI rate-card estimate so an
        # operator can spot divergence without querying OpenRouter by
        # hand.  Under-count typically means a gen-ID source we don't
        # capture live (e.g. title model, background LLM calls running
        # outside the main stream); over-count means the CLI's rate
        # table is stale vs. OpenRouter's current pricing.
        delta_pct: float | None = None
        if fallback_cost_usd and fallback_cost_usd > 0:
            delta_pct = (real_cost - fallback_cost_usd) / fallback_cost_usd * 100
        logger.info(
            "%s[cost-record] OpenRouter real=$%.6f cli_estimate=$%s "
            "delta=%s (gen_ids=%d)",
            log_prefix,
            real_cost,
            f"{fallback_cost_usd:.6f}" if fallback_cost_usd is not None else "?",
            f"{delta_pct:+.1f}%" if delta_pct is not None else "n/a",
            len(generation_ids),
        )
    else:
        # Partial or failed: the partial sum alone would under-report, and
        # the CLI's figure carries the session's earlier turns.  Keep the
        # provider's cost where it resolved and price the rest from the
        # catalog.
        fallback = fallback_turn_cost(
            model=model,
            turn_usage=TokenUsage(
                input_tokens=prompt_tokens,
                cache_read_tokens=cache_read_tokens,
                cache_creation_tokens=cache_creation_tokens,
                output_tokens=completion_tokens,
            ),
            resolved_costs={
                gen_id: float(cost)
                for gen_id, cost in zip(generation_ids, results)
                if isinstance(cost, (int, float))
            },
            call_usage=generation_usage or {},
            cli_cost_usd=fallback_cost_usd,
        )
        real_cost = fallback.cost_usd
        _log_fallback(
            log_prefix=log_prefix,
            resolved=len(fetched),
            total=len(generation_ids),
            fallback=fallback,
            cli_cost_usd=fallback_cost_usd,
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
            extra_metadata={
                "cost_source": fallback.source if fallback else "openrouter"
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "%s[cost-record] failed to persist: %s",
            log_prefix,
            exc,
        )

    # Backfill the Langfuse trace with reconciled cost + token usage.  The
    # OTel span for the turn closes before this background task runs, so the
    # Langfuse trace UI otherwise shows the SDK-CLI rate-card estimate (which
    # for non-Anthropic OpenRouter routes is wildly wrong — Sonnet pricing on
    # Kimi tokens, ~5x too high).  Emitting a child event with the real
    # numbers gives operators a single Langfuse view per turn instead of
    # cross-referencing pod logs.
    if langfuse_trace_id and real_cost is not None:
        try:
            get_client().create_event(
                trace_context={"trace_id": langfuse_trace_id},
                name="openrouter-cost-reconcile",
                metadata={
                    "cost_usd": real_cost,
                    "cost_source": cost_source,
                    "fallback_cost_usd": fallback_cost_usd,
                    "estimate_source": fallback.source if fallback else None,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "cache_read_tokens": cache_read_tokens,
                    "cache_creation_tokens": cache_creation_tokens,
                    "resolved_generation_id_count": len(fetched),
                    "generation_id_count": len(generation_ids),
                    "model": model,
                    "provider": "open_router",
                },
            )
        except Exception:
            logger.debug(
                "%s[cost-record] Langfuse event emit failed",
                log_prefix,
                exc_info=True,
            )


def _log_fallback(
    *,
    log_prefix: str,
    resolved: int,
    total: int,
    fallback: FallbackCost,
    cli_cost_usd: float | None,
) -> None:
    """Warn that the recorded cost is (partly) estimated.

    ``[cost-record]`` lines are parsed by log scripts: the existing fields
    stay as they were and new ones are appended at the end.
    """
    if resolved:
        logger.warning(
            "%s[cost-record] OpenRouter partial lookup (%d/%d) — "
            "using fallback estimate=$%s source=%s cli_estimate=$%s",
            log_prefix,
            resolved,
            total,
            fallback.cost_usd,
            fallback.source,
            cli_cost_usd,
        )
        return
    logger.warning(
        "%s[cost-record] OpenRouter lookup failed for all gens — "
        "using fallback estimate=$%s source=%s cli_estimate=$%s",
        log_prefix,
        fallback.cost_usd,
        fallback.source,
        cli_cost_usd,
    )
