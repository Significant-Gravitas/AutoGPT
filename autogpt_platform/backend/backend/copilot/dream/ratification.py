"""Ratification loop for ``status='tentative'`` dream proposals (P-0.4).

A tentative MemoryFact edge written by ``apply.py`` is on probation:
either warm-context retrieval proves it useful within a grace period
(at which point we promote it to ``status='active'``), or the grace
period elapses with zero hits and the edge is superseded with
``reason='unratified'``. The supersession carries the recall guard in its
own statement (``graphiti/guarded_writes.py``, no override): a proposal the
user recalled within the protection window stays tentative even when its
Redis hit count was lost, and is counted in ``protected_count``. A
promotion or supersession whose write raised has an unknown outcome (it may
have committed, never arrived, or still land), so the counts it would have
moved are provisional: ``accounting_complete`` is False, in the sweep's
``RatificationResult`` and the hit hook's ``HitRatification`` alike.

This module owns the pass logic itself. The Redis hit tracker lives
in ``ratification_hits.py`` so this file stays focused on the
promote-vs-supersede dispatch and fits the file-length budget.

Per ``dream/p0-spec.md`` §5. The metric ``dream_ratification_rate``
(P0.4d) is out of scope for this module — counts are logged at INFO
and surfaced via ``RatificationResult`` for the scheduler wrapper.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from backend.copilot.graphiti.falkordb_driver import AutoGPTFalkorDriver, open_driver
from backend.copilot.graphiti.guarded_writes import (
    WriteOutcome,
    supersede_unless_recalled,
)
from backend.copilot.graphiti.recall_stamp import RecallProtection, stamp_recalls
from backend.copilot.graphiti.scope import HIT_TRACKER_KEY_PREFIX, MemoryScope

from .ratification_hits import (
    RATIFICATION_GRACE_PERIOD,
    get_hit_count,
    parse_created_at,
    record_memory_hit,
)
from .recall_guard import DemotionGuard

logger = logging.getLogger(__name__)

UNRATIFIED_REASON = "unratified"

# Re-export so callers (scheduler wrapper, warm-context retrieval, the
# nightly batch fan-out) only have to know one module name.
__all__ = (
    "HIT_TRACKER_KEY_PREFIX",
    "HitRatification",
    "RATIFICATION_GRACE_PERIOD",
    "RatificationResult",
    "record_memory_hit",
    "run_ratification_pass",
    "try_ratify_on_hit",
)


class RatificationResult(BaseModel):
    """Structured outcome of a single ratification pass.

    Surfaced via the scheduler wrapper so admin / future metrics can
    read promotion vs. supersession counts without re-scanning Graphiti.
    """

    user_id: str
    started_at: datetime
    completed_at: datetime | None = None
    examined_count: int = 0
    ratified_count: int = 0
    superseded_count: int = 0
    # Past the grace period with no hits, but recalled within the protection
    # window: the sweep's guarded write left the proposal tentative.
    protected_count: int = 0
    # False when a promotion's or a supersession's outcome is unknown: its
    # write raised and may have committed, never arrived, or still land, so
    # the counts are provisional. A per-edge error whose outcome is known (a
    # failed hit-count read, a supersession that matched nothing) leaves it
    # True.
    accounting_complete: bool = True
    error: str | None = None
    skipped: bool = False
    skip_reason: str | None = None
    per_edge_errors: list[str] = Field(default_factory=list)


class HitRatification(BaseModel):
    """What one warm-context hit's promotions did."""

    promoted_count: int = 0
    # False when a promotion's outcome is unknown: its write raised and may
    # have committed, never arrived, or still land, so promoted_count is
    # provisional.
    accounting_complete: bool = True


async def run_ratification_pass(
    user_id: str, expert_id: str | None = None
) -> RatificationResult:
    """Promote or supersede every ``status='tentative'`` edge for one user.

    Defensive in two ways:
      * Catastrophic failure (invalid user, no graph, Redis down) is
        captured in ``RatificationResult.error`` rather than raised so
        the scheduler wrapper logs cleanly instead of crashing the job.
      * Per-edge failure is captured in ``per_edge_errors`` so one bad
        edge can't poison the rest of the pass. A write whose outcome is
        unknown is reported there too, and also leaves
        ``accounting_complete`` False.
    """
    started_at = datetime.now(timezone.utc)
    result = RatificationResult(user_id=user_id, started_at=started_at)

    try:
        scope = MemoryScope.build(user_id, expert_id)
    except ValueError as exc:
        result.error = f"invalid_user_id: {exc}"
        result.completed_at = datetime.now(timezone.utc)
        logger.warning(
            "Ratification skipped — invalid user_id %s: %s", user_id[:12], exc
        )
        return result

    driver = open_driver(scope)
    try:
        try:
            tentatives = await _list_tentative_edges(driver)
        except Exception as exc:
            result.error = f"list_tentative_edges_failed: {exc}"
            result.completed_at = datetime.now(timezone.utc)
            logger.warning(
                "Ratification list failed for user %s",
                user_id[:12],
                exc_info=True,
            )
            return result

        result.examined_count = len(tentatives)
        if not tentatives:
            result.completed_at = datetime.now(timezone.utc)
            logger.info(
                "Ratification no-op for user %s — no tentative edges",
                user_id[:12],
            )
            return result

        now = datetime.now(timezone.utc)
        protection = DemotionGuard.at(now, ()).protection(UNRATIFIED_REASON)
        for edge in tentatives:
            try:
                await _process_edge(
                    scope=scope,
                    driver=driver,
                    edge=edge,
                    now=now,
                    protection=protection,
                    result=result,
                )
            except Exception as exc:
                # Per-edge defense: capture and continue so the pass
                # finishes for the rest of the user's tentatives.
                result.per_edge_errors.append(
                    f"{edge.get('uuid', '?')}: {type(exc).__name__}: {exc}"
                )
                logger.warning(
                    "Ratification per-edge failure for user %s edge %s",
                    user_id[:12],
                    edge.get("uuid", "?"),
                    exc_info=True,
                )
    finally:
        await driver.close()

    result.completed_at = datetime.now(timezone.utc)
    logger.info(
        "Ratification complete for user %s: examined=%d ratified=%d superseded=%d "
        "protected=%d errors=%d",
        user_id[:12],
        result.examined_count,
        result.ratified_count,
        result.superseded_count,
        result.protected_count,
        len(result.per_edge_errors),
    )
    return result


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


async def _process_edge(
    *,
    scope: MemoryScope,
    driver: AutoGPTFalkorDriver,
    edge: dict[str, Any],
    now: datetime,
    protection: RecallProtection,
    result: RatificationResult,
) -> None:
    """Promote, supersede, or leave alone one tentative edge.

    Decision table (spec §5):
      * hits >= 1                 → promote to ``status='active'``
      * hits == 0 and past grace  → supersede with ``reason='unratified'``,
        unless *protection* spares it: the user recalled it within the
        window (its stamp is on the edge, so it holds when the Redis hit
        count was lost) → left tentative, counted in ``protected_count``
      * hits == 0 within grace    → no-op (still earning its keep)

    Both writes apply only while the edge is still an unexpired tentative
    one: a user's forget that lands after the listing is never overwritten.
    """
    edge_uuid = edge.get("uuid")
    if not edge_uuid:
        result.per_edge_errors.append("missing_uuid")
        return

    hits = await _get_hit_count(scope, edge_uuid)

    if hits >= 1:
        promoted = await _promote_if_tentative(driver, edge_uuid)
        if promoted is WriteOutcome.CHANGED:
            result.ratified_count += 1
        elif promoted is WriteOutcome.UNKNOWN:
            _unknown_outcome(result, edge_uuid, "promote")
        return

    created_at = parse_created_at(edge.get("created_at"))
    if created_at is None:
        # Edge without a created_at — be safe and leave it alone rather
        # than supersede something we can't date.
        result.per_edge_errors.append(f"{edge_uuid}: missing_created_at")
        return

    if now - created_at <= RATIFICATION_GRACE_PERIOD:
        return

    [outcome] = await supersede_unless_recalled(
        driver,
        [edge_uuid],
        reason=UNRATIFIED_REASON,
        new_status="superseded",
        group_id=scope.group_id,
        protection=protection,
        user_id=scope.owner_user_id,
        expected_status="tentative",
    )
    if outcome is WriteOutcome.CHANGED:
        result.superseded_count += 1
    elif outcome is WriteOutcome.SPARED:
        result.protected_count += 1
    elif outcome is WriteOutcome.UNKNOWN:
        _unknown_outcome(result, edge_uuid, "supersede")
    else:
        # Surface non-matches too: an edge without a group_id property (legacy
        # write) matches nothing under the group-scoped predicate and would
        # otherwise be silently re-examined by every future sweep.
        result.per_edge_errors.append(f"{edge_uuid}: supersede_failed")


def _unknown_outcome(result: RatificationResult, edge_uuid: str, write: str) -> None:
    """The *write* on *edge_uuid* raised and may have committed, never
    arrived, or still land: neither done nor failed as far as the sweep
    knows. It is reported, and the sweep's counts are provisional."""
    result.per_edge_errors.append(f"{edge_uuid}: {write}_outcome_unknown")
    result.accounting_complete = False


async def _list_tentative_edges(
    driver: AutoGPTFalkorDriver,
) -> list[dict[str, Any]]:
    """Return ``status='tentative'`` edges with their uuid and created_at.

    Excludes edges that are already retracted (``expired_at IS NOT NULL``)
    so an edge demoted by a parallel operation isn't ratified back to life.
    """
    query = """
    MATCH ()-[e:RELATES_TO]->()
    WHERE e.status = 'tentative' AND e.expired_at IS NULL
    RETURN e.uuid AS uuid, e.created_at AS created_at
    """
    result = await driver.execute_query(query)
    records = result[0] if result else []
    return [{"uuid": r["uuid"], "created_at": r["created_at"]} for r in records]


async def try_ratify_on_hit(
    scope: MemoryScope, edge_uuids: list[str]
) -> HitRatification:
    """Record warm-context hits and promote any tentative edges inline.

    Called from warm-context retrieval (``graphiti/context.py``) once
    per turn with the list of edge uuids that landed in the user's
    context. For each uuid we:

      1. Bump the ``mem:hits:{scope_key}:{edge_uuid}`` Redis counter
         (so the nightly ratification sweep also sees the hit and
         agrees on promotion if Cypher fails here).
      2. Stamp the recall on every retrieved live edge, in one batched
         write (``graphiti/recall_stamp.py``). The Redis counter expires
         with the grace period; the stamps are the durable usage signal
         the dream pass reads to leave a relied-on fact alone.
      3. Issue a targeted Cypher ``SET status='active'`` filtered by
         ``status='tentative' AND expired_at IS NULL`` — already-active
         and already-retracted edges are no-ops via the WHERE clause.

    Returns the count of edges this call actually promoted, marked
    provisional (``accounting_complete`` False) when a promotion's outcome is
    unknown. The function is **safe to fire-and-forget** from the retrieval
    path: failures are caught and logged, never raised; the user's chat
    turn is never blocked on this.

    Per the architecture plan, this is the sync hit-time half of P0.4
    ratification. The nightly batch's ratification sweep still owns
    grace-period supersession; with this hook landing, the sweep
    rarely promotes (most tentatives get hit at least once within a
    day) and primarily cleans up the truly-unused.
    """
    if not edge_uuids:
        return HitRatification()

    user_id = scope.owner_user_id
    # Step 1: bump hit counters (Redis, best-effort, swallows errors).
    # Done before the Cypher promotion so the counter survives even
    # when the promotion path fails.
    for uuid in edge_uuids:
        await record_memory_hit(scope, uuid)

    # Steps 2 and 3: the recall stamp, then targeted Cypher promotion, on a
    # driver of our own: callers are warm-context retrieval call sites that
    # have a higher-level graphiti client but no raw driver. Neither step
    # takes the graph's write lock (``scope_lock.py``). Each is one statement
    # that writes only over a live edge: the stamp only its usage properties,
    # the promotion only a still-tentative edge's status and ``ratified_at``.
    driver = open_driver(scope)
    try:
        # Never raises; a failed stamp is logged and promotion goes on.
        await stamp_recalls(driver, edge_uuids, owner=user_id)
        # Per edge, never raising: one bad uuid mustn't poison the rest of
        # the retrieved set.
        outcomes = [await _promote_if_tentative(driver, uuid) for uuid in edge_uuids]
    finally:
        await driver.close()

    result = HitRatification(
        promoted_count=outcomes.count(WriteOutcome.CHANGED),
        accounting_complete=WriteOutcome.UNKNOWN not in outcomes,
    )
    if result.promoted_count:
        logger.info(
            "Ratification hit-hook promoted %d edge(s) for user %s",
            result.promoted_count,
            user_id[:12],
        )
    return result


async def _promote_if_tentative(
    driver: AutoGPTFalkorDriver, edge_uuid: str
) -> WriteOutcome:
    """Flip a still-tentative, unexpired edge to ``status='active'`` with a
    ``ratified_at`` stamp: ``CHANGED`` when it did, ``UNMATCHED`` when the
    edge is no longer one to promote, and ``UNKNOWN`` when the write raised
    (logged; it may have committed, never arrived, or still land).

    The one promotion write, for the hit hook and the nightly sweep alike.
    The guard makes a repeat hit on an active edge a no-op (its first
    ``ratified_at`` stays) and keeps a forget that lands between the sweep's
    listing and this write: a retracted or forgotten edge is never made
    active again.
    ``now`` comes from Python: FalkorDB has no no-arg ``datetime()``.
    """
    query = """
    MATCH ()-[e:RELATES_TO {uuid: $uuid}]->()
    WHERE e.status = 'tentative' AND e.expired_at IS NULL
      AND e.forgotten_at IS NULL
    SET e.status = 'active', e.ratified_at = $now
    RETURN e.uuid AS uuid
    """
    try:
        result = await driver.execute_query(
            query, uuid=edge_uuid, now=datetime.now(timezone.utc).isoformat()
        )
    except Exception:
        logger.warning(
            f"Promoting edge {edge_uuid} raised; it may have committed or may "
            "still land",
            exc_info=True,
        )
        return WriteOutcome.UNKNOWN
    records = result[0] if result else []
    return WriteOutcome.CHANGED if records else WriteOutcome.UNMATCHED


# Local indirection so tests can mock ``_get_hit_count`` on this module
# rather than the helper module (matches the poison-pill test pattern).
async def _get_hit_count(scope: MemoryScope, edge_uuid: str) -> int:
    return await get_hit_count(scope, edge_uuid)
