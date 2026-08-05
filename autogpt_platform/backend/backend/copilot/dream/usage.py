"""Usage-based protection for dream demotions.

The warm-context hit hook (``ratification.try_ratify_on_hit``) stamps
``recall_count`` / ``last_recalled_at`` on every retrieved edge. The
sanitize prompt surfaces those numbers so the LLM prefers pruning
never-recalled facts, but prompt text is guidance, not enforcement —
this module is the deterministic backstop that runs before any Cypher.

Retrieval ranking is deliberately untouched: usage only ever makes the
nightly pass *less* destructive, it never reorders what a chat turn
recalls.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from .fetch import FactRow, parse_iso_timestamp
from .schemas import DreamDemotion

logger = logging.getLogger(__name__)

# A fact recalled at least MIN_RECALLS_TO_PROTECT times within
# RECENT_RECALL_WINDOW of the pass is protected from demotion.
#
# The window matches the cadence a user is actually observed over: a
# fact pulled into context inside the last week is demonstrably still
# part of their working set, and a nightly pass has seven more chances
# to demote it once it goes quiet. Longer would ossify the graph
# (stale-but-recently-read facts could never be pruned); shorter would
# leave weekly-cadence work (a project touched every Monday)
# unprotected.
RECENT_RECALL_WINDOW = timedelta(days=7)

# Two recalls, not one: a single hit is indistinguishable from an
# incidental embedding-similarity match on an unrelated turn, so
# protecting on it would shield facts the user never really used.
# Two hits inside the window is a repeated, deliberate-looking recall.
MIN_RECALLS_TO_PROTECT = 2


def drop_recently_used_demotions(
    pass_id: str,
    demotions: list[DreamDemotion],
    facts: list[FactRow] | None,
) -> list[DreamDemotion]:
    """Remove demotions targeting facts the user demonstrably still uses.

    ``facts`` is the pass's own fetched fact list — the only source of
    usage data. ``None`` (caller had no bundle, or the persisted one
    expired) means "no usage data available" and the guard fails open,
    keeping every demotion. That matches the known-fact filter's
    posture: a missing input bundle must not silently zero a pass.
    """
    if not demotions or not facts:
        return demotions

    protected = protected_fact_uuids(facts)
    if not protected:
        return demotions

    kept = [d for d in demotions if d.edge_uuid not in protected]
    dropped = len(demotions) - len(kept)
    if dropped:
        logger.warning(
            "Dream pass %s: dropped %d demotion(s) targeting facts recalled "
            "at least %d time(s) in the last %d day(s)",
            pass_id,
            dropped,
            MIN_RECALLS_TO_PROTECT,
            RECENT_RECALL_WINDOW.days,
        )
    return kept


def protected_fact_uuids(
    facts: list[FactRow], *, now: datetime | None = None
) -> set[str]:
    """Uuids of facts that are recently-and-repeatedly recalled."""
    reference = now or datetime.now(timezone.utc)
    cutoff = reference - RECENT_RECALL_WINDOW
    return {
        fact.uuid
        for fact in facts
        if (fact.recall_count or 0) >= MIN_RECALLS_TO_PROTECT
        if fact.last_recalled_at
        if (recalled := parse_iso_timestamp(fact.last_recalled_at)) is not None
        if recalled >= cutoff
    }


def format_usage(fact: FactRow) -> str:
    """Render a fact's usage signal for the dream prompts.

    Rendered for EVERY fact — including never-recalled ones — so the
    model reads absence as an explicit signal rather than having to
    infer it from a missing field.
    """
    count = fact.recall_count or 0
    if not count:
        return "recalls=0(never)"
    return f"recalls={count} last_recall={fact.last_recalled_at or '?'}"
