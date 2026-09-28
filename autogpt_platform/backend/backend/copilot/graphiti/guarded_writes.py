"""The dream's destructive writes, each carrying the recall guard in the
statement that writes, and the read that accounts for them afterwards.

``supersede_unless_recalled`` demotes facts one statement per edge: the dream
pass's demotions, and the ratification sweep's supersession of an unratified
proposal. ``invalidate_entity_direct_neighbors`` demotes every live fact on one
entity, single-hop, for the dream's entity invalidations. It has no degree
cap: a hub is invalidated whole, behind the ``dream-pass-invalidate-entity``
flag, as before recall stamps.

Each statement writes only over a live fact (``recall.live_fact_predicate``),
so it can never overwrite a user's forget. Each also tests
``recall_stamp.spared_by_recall`` for every fact it reaches: a fact the user
recalled within the protection window is left alone unless the write's
``RecallProtection`` overrides it. The statement reports what it changed and
what it spared, and nothing is read beforehand to decide.

A statement that raises has an unknown outcome: the server may have
committed it before the acknowledgement was lost, so it is reported as
``UNKNOWN``, never as having changed nothing. ``live_fact_uuids`` is the
read that settles, afterwards and for accounting only, which of the facts a
pass spared are still live.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, Field

from .falkordb_driver import AutoGPTFalkorDriver
from .recall import live_fact_predicate
from .recall_stamp import RecallProtection, spared_by_recall

logger = logging.getLogger(__name__)

# What the driver returns for one statement: its rows, then two fields this
# module does not read.
QueryResult = tuple[list[dict[str, Any]], list[str], None] | None
_Row = TypeVar("_Row", bound=BaseModel)


class WriteOutcome(str, Enum):
    """What one guarded write did to one fact, as far as its caller knows."""

    CHANGED = "changed"
    # Live, but the user recalled it within the protection window and the
    # write does not override that: left as it was.
    SPARED = "spared"
    # The statement ran and matched nothing to write: the fact is no longer
    # live (or not in the expected status), or missing. Nothing was written.
    UNMATCHED = "unmatched"
    # The statement raised (logged). It may have committed before its
    # acknowledgement was lost, never arrived, or still be queued on the
    # server and land later, so nothing is known about the fact.
    UNKNOWN = "unknown"


class NeighbourWrites(BaseModel):
    """What an entity invalidation did: the live neighbours it demoted, and
    those it left alone because the user recalled them within the protection
    window. Each list holds distinct uuids and is complete, however many
    neighbours the entity has. ``unknown``: the statement raised and may have
    committed, so which neighbours it changed or spared is not known, and
    both lists are empty."""

    changed: list[str] = Field(default_factory=list)
    spared: list[str] = Field(default_factory=list)
    unknown: bool = False


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

    One outcome per uuid, in order: ``UNMATCHED`` when the statement matched
    no fact to write, ``UNKNOWN`` when it raised.
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
    neighbour in the same statement that demotes it. ``DISTINCT`` keeps an
    edge reached from both directions to one outcome, and the statement
    returns every outcome in one row, so the server's result-set row limit
    never truncates the account. A statement that raises is logged and
    reported ``unknown``: it may have committed, never arrived, or still be
    queued on the server.
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
        outcomes = _one_row(result, _NeighbourOutcomes).outcomes
    except Exception:
        logger.warning(
            f"Invalidating the neighbours of entity {entity_uuid} in group "
            f"{group_id} raised; it may have committed or may still land",
            exc_info=True,
        )
        return NeighbourWrites(unknown=True)
    return NeighbourWrites(
        changed=[row.uuid for row in outcomes if not row.spared],
        spared=[row.uuid for row in outcomes if row.spared],
    )


async def live_fact_uuids(
    driver: AutoGPTFalkorDriver, group_id: str, uuids: list[str]
) -> set[str]:
    """Which of *uuids* are live facts of *group_id* now: one statement that
    reads only those facts and returns one row, however many they are.
    Raises on failure, for the caller to decide what an unknown answer
    means."""
    result = await driver.execute_query(
        _LIVE_FACTS_QUERY, uuids=uuids, group_id=group_id
    )
    return set(_one_row(result, _LiveFacts).live)


class _GuardedRow(BaseModel):
    """A fact a guarded write reached, and whether the recall guard spared
    it."""

    uuid: str
    spared: bool


class _NeighbourOutcomes(BaseModel):
    """The one row the neighbour statement returns."""

    outcomes: list[_GuardedRow] = Field(default_factory=list)


class _LiveFacts(BaseModel):
    """The one row the liveness read returns."""

    live: list[str] = Field(default_factory=list)


def _one_row(result: QueryResult, model: type[_Row]) -> _Row:
    """An aggregate statement's single row as *model*; the model's defaults
    when it returned none."""
    rows = result[0] if result else []
    return model.model_validate(rows[0]) if rows else model()


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
            f"Superseding edge {uuid} for user {(user_id or '?')[:12]} raised; "
            "it may have committed or may still land",
            exc_info=True,
        )
        return WriteOutcome.UNKNOWN
    if not rows:
        return WriteOutcome.UNMATCHED
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

# One aggregate row, not one row per neighbour: a hub can have more
# neighbours than the server returns rows (FalkorDB's RESULTSET_SIZE).
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
RETURN collect({{uuid: r.uuid, spared: spared}}) AS outcomes
"""

# The edge lookup is graphiti's range index on ``RELATES_TO.uuid``, as in
# the recall stamp's own query.
_LIVE_FACTS_QUERY = f"""
UNWIND $uuids AS target_uuid
MATCH ()-[e:RELATES_TO]->()
WHERE e.uuid = target_uuid AND e.group_id = $group_id
  AND {live_fact_predicate("e")}
RETURN collect(DISTINCT e.uuid) AS live
"""
