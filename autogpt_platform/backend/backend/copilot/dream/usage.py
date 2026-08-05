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

# A fact whose two most recent (deduped) recalls both fall inside
# RECENT_RECALL_WINDOW is protected from staleness demotion — i.e. it
# was used on at least two separate days in the last week.
#
# The window matches the cadence a user is actually observed over: a
# fact pulled into context inside the last week is demonstrably still
# part of their working set, and a nightly pass has seven more chances
# to demote it once it goes quiet. Longer would ossify the graph
# (stale-but-recently-read facts could never be pruned); shorter would
# leave weekly-cadence work (a project touched every Monday)
# unprotected.
RECENT_RECALL_WINDOW = timedelta(days=7)

# Recall stamps closer together than this collapse into one "use":
# warm context re-pulls the same edges turn after turn, so raw
# per-turn increments would clear any repetition bar inside a single
# conversation. Day-level dedupe makes ``recall_count`` count distinct
# days of use, and the (last, prev) stamp pair windowed below means
# "used on two separate days this week" — repetition a single
# incidental embedding-similarity match (or one long chat) can't fake.
# Storing only the two latest stamps keeps the windowed predicate
# exact without a full recall log; a lifetime counter alone cannot
# express "N recalls WITHIN the window".
RECALL_DEDUPE_INTERVAL = timedelta(hours=24)

# Demotion reasons that override usage protection. Usage disproves
# *staleness* — a recently-recalled fact isn't stale — but it can't
# disprove a contradiction or an explicit user retraction; a
# frequently-recalled WRONG fact is exactly the memory most worth
# correcting. Anything else (``stale_fact``, ``entity_invalidated:*``,
# free-text reasons the model invents) stays blocked for protected
# facts: fail conservative in the destructive direction.
OVERRIDE_REASONS = frozenset({"user_signal"})
OVERRIDE_REASON_PREFIXES = ("contradicted_by:", "web_contradicted:")


def _reason_overrides_protection(reason: str) -> bool:
    return reason in OVERRIDE_REASONS or reason.startswith(OVERRIDE_REASON_PREFIXES)


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

    kept = [
        d
        for d in demotions
        if d.edge_uuid not in protected or _reason_overrides_protection(d.reason)
    ]
    dropped = len(demotions) - len(kept)
    if dropped:
        logger.warning(
            "Dream pass %s: dropped %d staleness demotion(s) targeting facts "
            "recalled on two separate occasions in the last %d day(s)",
            pass_id,
            dropped,
            RECENT_RECALL_WINDOW.days,
        )
    return kept


def protected_fact_uuids(
    facts: list[FactRow], *, now: datetime | None = None
) -> set[str]:
    """Uuids of facts recalled on two separate (deduped) occasions
    within the window.

    The predicate reads ``prev_recalled_at`` — the second-latest
    deduped stamp — because ``last_recalled_at`` is necessarily no
    older, so ``prev`` inside the window puts BOTH recalls inside it.
    A lifetime ``recall_count`` deliberately plays no part: two recalls
    a year ago plus one incidental hit yesterday must not protect.
    """
    reference = now or datetime.now(timezone.utc)
    cutoff = reference - RECENT_RECALL_WINDOW
    return {
        fact.uuid
        for fact in facts
        if fact.prev_recalled_at
        if (prev := parse_iso_timestamp(fact.prev_recalled_at)) is not None
        if prev >= cutoff
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
