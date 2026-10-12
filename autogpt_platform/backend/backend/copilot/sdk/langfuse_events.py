"""Langfuse trace events for SDK turns — import-safe leaf module.

Compaction cycles are otherwise invisible in Langfuse (the CLI/Codex
engines report nothing there), so an operator reading a turn's trace
cannot see them; this emits one event per cycle. The turn's own totals
(tokens, cost, the pinned window, compaction outcomes, the Codex gateway's
counts) ride its one ``copilot-turn-summary`` event instead
(``turn_summary.py``). Fail-silent: observability must never break a turn.
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


__all__ = [
    "EVENT_SCHEMA_VERSION",
    "CompactionPath",
    "emit_compaction_event",
]
