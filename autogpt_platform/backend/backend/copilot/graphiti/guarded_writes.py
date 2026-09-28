"""The dream's destructive writes, each carrying the recall guard in the
statement that writes.

``supersede_unless_recalled`` demotes facts one statement per edge: the dream
pass's demotions, and the ratification sweep's supersession of an unratified
proposal. ``invalidate_entity_direct_neighbors`` demotes every live fact on one
entity, single-hop, for the dream's entity invalidations.

Each statement writes only over a live fact (``recall.live_fact_predicate``),
so it can never overwrite a user's forget. Each also tests
``recall_stamp.spared_by_recall`` for every fact it reaches: a fact the user
recalled within the protection window is left alone unless the write's
``RecallProtection`` overrides it. The statement returns what it changed and
what it spared, so callers count from the write's own result and read
nothing beforehand.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

from .falkordb_driver import AutoGPTFalkorDriver
from .recall import live_fact_predicate
from .recall_stamp import RecallProtection, spared_by_recall

logger = logging.getLogger(__name__)


class WriteOutcome(str, Enum):
    """What one guarded demotion did to one fact."""

    CHANGED = "changed"
    # Live, but the user recalled it within the protection window and the
    # write does not override that: left as it was.
    SPARED = "spared"
    # Not live (or not in the expected status), missing, or the write
    # failed (logged).
    FAILED = "failed"


class NeighbourWrites(BaseModel):
    """What an entity invalidation did: the live neighbours it demoted, and
    those it left alone because the user recalled them within the protection
    window. Each list holds distinct uuids."""

    changed: list[str] = Field(default_factory=list)
    spared: list[str] = Field(default_factory=list)


async def supersede_unless_recalled(
    driver: AutoGPTFalkorDriver,
    uuids: list[str],
    *,
    reason: str,
    new_status: Literal["superseded", "contradicted"],
    group_id: str,
    protection: RecallProtection,
    user_id: str | None = None,
    expected_status: str | None = None,
) -> list[WriteOutcome]:
    """Retire each live fact in *uuids* (``expired_at``, ``status`` and
    ``expiration_reason``; ``invalid_at`` is left alone), unless *protection*
    spares it. The recall test and the write are one statement, so a recall
    stamped before it runs cannot be missed.

    ``group_id`` must match the edge's own, so a caller holding the wrong
    driver cannot touch another scope's facts. With ``expected_status`` the
    fact must also still be in that status (ratification supersedes only a
    still-tentative proposal).

    One outcome per uuid, in order: a fact no longer live (or no longer in
    ``expected_status``), a missing one and a write that failed (logged) are
    ``FAILED``.
    """
    params = {
        "reason": reason,
        "new_status": new_status,
        "group_id": group_id,
        "expected_status": expected_status,
        **protection.params(),
    }
    return [await _supersede_one(driver, uuid, params, user_id) for uuid in uuids]


async def invalidate_entity_direct_neighbors(
    driver: AutoGPTFalkorDriver,
    group_id: str,
    entity_uuid: str,
    reason: str,
    *,
    protection: RecallProtection = RecallProtection(),
) -> NeighbourWrites:
    """Demote every live ``:RELATES_TO`` edge directly attached to an entity,
    but for those *protection* spares.

    **Single-hop only**: it does NOT propagate to neighbours of neighbours.
    The instinct to write ``[r:RELATES_TO*1..N]`` is exactly the
    runaway-demotion bug this protects against (P0.3b in the dream spec).
    Only live neighbours are touched, and the recall guard is tested per
    neighbour in the same statement that demotes it.

    ``DISTINCT`` matters: the undirected ``-[r]-`` pattern can yield the same
    edge from both traversal directions, and duplicate uuids would inflate
    the counts reported in ``DreamPassResult`` and the admin UI. A write that
    fails is logged and reported as touching nothing.
    """
    try:
        result = await driver.execute_query(
            _GUARDED_NEIGHBOURS_QUERY,
            entity_uuid=entity_uuid,
            group_id=group_id,
            reason=reason,
            now=_now_iso(),
            **protection.params(),
        )
        rows = [_GuardedRow.model_validate(r) for r in (result[0] if result else [])]
    except Exception:
        logger.warning(
            f"Failed to invalidate direct neighbors of entity {entity_uuid} "
            f"in group {group_id}",
            exc_info=True,
        )
        return NeighbourWrites()
    return NeighbourWrites(
        changed=[row.uuid for row in rows if not row.spared],
        spared=[row.uuid for row in rows if row.spared],
    )


class _GuardedRow(BaseModel):
    """A row a guarded write returns: a fact it reached, and whether the
    recall guard spared it."""

    uuid: str
    spared: bool


async def _supersede_one(
    driver: AutoGPTFalkorDriver,
    uuid: str,
    params: dict[str, Any],
    user_id: str | None,
) -> WriteOutcome:
    try:
        result = await driver.execute_query(
            _GUARDED_SUPERSEDE_QUERY, uuid=uuid, now=_now_iso(), **params
        )
        rows = [_GuardedRow.model_validate(r) for r in (result[0] if result else [])]
    except Exception:
        logger.warning(
            f"Failed to mark edge {uuid} superseded for user {(user_id or '?')[:12]}",
            exc_info=True,
        )
        return WriteOutcome.FAILED
    if not rows:
        return WriteOutcome.FAILED
    return WriteOutcome.SPARED if rows[0].spared else WriteOutcome.CHANGED


def _now_iso() -> str:
    """Now, as the ISO-8601 string written to ``expired_at``. FalkorDB has no
    no-arg ``datetime()``, so the time is bound from Python."""
    return datetime.now(timezone.utc).isoformat()


# The recall guard and the write are one statement: FOREACH over an empty
# list skips the SET for a spared edge, and the row still comes back so the
# caller can count it.
_GUARDED_SUPERSEDE_QUERY = f"""
MATCH ()-[e:RELATES_TO {{uuid: $uuid, group_id: $group_id}}]->()
WHERE {live_fact_predicate("e")}
  AND ($expected_status IS NULL OR e.status = $expected_status)
WITH e, {spared_by_recall("e")} AS spared
FOREACH (_ IN CASE WHEN spared THEN [] ELSE [1] END |
    SET e.expired_at = $now,
        e.status = $new_status,
        e.expiration_reason = $reason)
RETURN e.uuid AS uuid, spared
"""

_GUARDED_NEIGHBOURS_QUERY = f"""
MATCH (e:Entity {{uuid: $entity_uuid, group_id: $group_id}})
MATCH (e)-[r:RELATES_TO]-(other)
WHERE {live_fact_predicate("r")}
WITH DISTINCT r
WITH r, {spared_by_recall("r")} AS spared
FOREACH (_ IN CASE WHEN spared THEN [] ELSE [1] END |
    SET r.expired_at = $now,
        r.status = 'superseded',
        r.expiration_reason = $reason)
RETURN r.uuid AS uuid, spared
"""
