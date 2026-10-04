"""Langfuse trace events for SDK turns — import-safe leaf module.

Compaction cycles and per-turn token usage are otherwise invisible in
Langfuse (the CLI/Codex engines report nothing there), so an operator
reading a turn's trace cannot see them. These helpers emit them as
trace events. Every helper is fail-silent: observability must never
break a turn.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

from langfuse import get_client

if TYPE_CHECKING:
    from backend.copilot.sdk.compaction import CompactionStats

logger = logging.getLogger(__name__)

CompactionPath = Literal["pre_query", "sdk_internal"]

# Event metadata schema version. Bump when a shipped event gains, loses,
# or renames keys so dashboards can split rather than average across the
# boundary. (Schema 1 = the unversioned first deploy.)
EVENT_SCHEMA_VERSION = 2


def emit_compaction_event(
    *,
    path: CompactionPath,
    stats: CompactionStats | None,
    after_source: str | None = None,
    trace_id: str | None = None,
    log_prefix: str = "",
) -> None:
    """Emit one event per completed compaction cycle on the turn's trace.

    *trace_id* is the turn's Langfuse trace id, captured when the turn's
    span opened (see ``CompactionTracker.trace_id``); when given, the
    event is pinned to that trace explicitly rather than relying on the
    ambient span context, which an awaited path may or may not still be
    carrying by the time the cycle closes. Without it the event falls
    back to the current span. *stats* may be None
    when the cycle produced no counts; the event still records that a
    cycle ran, which is what makes back-to-back compactions visible.
    *after_source* names how the post-compaction read resolved (or
    ``compress_result`` when the counts came from our own compressor
    rather than a transcript read); it is what makes a missing
    ``tokensAfter`` diagnosable instead of merely absent.
    """
    metadata: dict[str, Any] = {"schema": EVENT_SCHEMA_VERSION, "path": path}
    if stats is not None:
        # to_wire() carries the camelCase names the tool row's JSON output
        # uses, so the same cycle reads identically in the DB and Langfuse.
        metadata.update(stats.to_wire())
    metadata["dropped"] = stats.dropped if stats is not None else False
    if after_source is not None:
        metadata["after_source"] = after_source
    try:
        if trace_id:
            get_client().create_event(
                trace_context={"trace_id": trace_id},
                name="copilot-compaction",
                metadata=metadata,
            )
        else:
            get_client().create_event(name="copilot-compaction", metadata=metadata)
    except Exception:
        logger.debug(
            "%s Langfuse compaction event emit failed", log_prefix, exc_info=True
        )


def emit_turn_usage_event(
    *,
    trace_id: str | None,
    prompt_tokens: int,
    completion_tokens: int,
    cache_read_tokens: int,
    cache_creation_tokens: int,
    cost_usd: float | None,
    model: str | None,
    provider: str,
    codex_input_tokens: int | None = None,
    codex_cached_input_tokens: int | None = None,
    codex_boundary_peak_estimate: int | None = None,
    log_prefix: str = "",
) -> None:
    """Attach the turn's token usage to its Langfuse trace.

    Takes an explicit *trace_id* (captured before the turn's span
    closes) like the OpenRouter cost-reconcile event. Skipped when no
    trace id was captured. The reconcile path covers OpenRouter turns;
    this covers the Codex/subscription/direct turns it never sees.

    The ``codex_*`` fields carry the inner gauge next to the CLI-side
    numbers, so one event shows both sides of the outer/inner window
    split. Omitted on non-Codex routes. ``codex_input_tokens`` is the
    gateway's turn-aggregate (a billing total, not a size — it sums
    every invoke); ``codex_boundary_peak_estimate`` is the max
    per-request estimate the gateway actually reported, tracked
    independently of completed results so it survives turns whose
    conversations raised before recording anything.
    """
    if trace_id is None:
        return
    metadata: dict[str, Any] = {
        "schema": EVENT_SCHEMA_VERSION,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cache_read_tokens": cache_read_tokens,
        "cache_creation_tokens": cache_creation_tokens,
        "cost_usd": cost_usd,
        "model": model,
        "provider": provider,
    }
    if codex_input_tokens is not None:
        metadata["codex_input_tokens"] = codex_input_tokens
    if codex_cached_input_tokens is not None:
        metadata["codex_cached_input_tokens"] = codex_cached_input_tokens
    if codex_boundary_peak_estimate is not None:
        metadata["codex_boundary_peak_estimate"] = codex_boundary_peak_estimate
    try:
        get_client().create_event(
            trace_context={"trace_id": trace_id},
            name="copilot-turn-usage",
            metadata=metadata,
        )
    except Exception:
        logger.debug(
            "%s Langfuse turn-usage event emit failed", log_prefix, exc_info=True
        )


__all__ = [
    "EVENT_SCHEMA_VERSION",
    "CompactionPath",
    "emit_compaction_event",
    "emit_turn_usage_event",
]
