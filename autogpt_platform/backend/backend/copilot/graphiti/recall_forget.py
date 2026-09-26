"""Forgetting under the recall policy.

After ``retract`` the recall paths in ``recall.py`` return neither the
forgotten facts nor the episode text they came from, the sentence is gone
from the fact and from what graphiti read out of it onto entities, and the
audit record stays; ``graphiti/AGENTS.md`` lists the limits. The chat forget
tool and the settings page both forget through here, and an ingestion that
was running when a forget landed applies it again through ``forget_edges``,
so a forget means the same thing wherever it starts. ``forgotten_at``, the
policy's marker for a forgotten fact, originates here and, for forgets made
before it existed, in ``migrations/backfill_legacy_forgets.py``; anything
else that writes it (the ingestion repair) puts back a value that came from
a forget.
"""

import logging
from datetime import datetime, timezone
from typing import Any

from graphiti_core.driver.driver import GraphDriver
from pydantic import BaseModel

from .falkordb_driver import open_driver
from .memory_model import ForgetResult, MemoryForgetFailure
from .recall import FORGOTTEN_FACT, USER_FORGET_REASON
from .recall_hide import Hiding, hide
from .recall_orphans import purge
from .recall_stash import ForgetRecord, read_forgets, stash_forgets
from .scope import MemoryScope

logger = logging.getLogger(__name__)


async def retract(
    scope: MemoryScope,
    uuids: list[str],
    *,
    hard: bool = False,
    reason: str = USER_FORGET_REASON,
) -> ForgetResult:
    """Forget edges: recall stops returning them and the text they came from.

    Soft (the default) stamps ``forgotten_at``, sets ``status='retracted'``
    and ``expiration_reason``, keeps an earlier ``expired_at`` and leaves
    ``invalid_at`` alone: a forget retracts our record of a fact, it does
    not say the world changed (Snodgrass). ``recall_hide.hide`` then moves
    the sentence out of what graphiti reads and redacts every episode citing
    the fact; edges and episodes stay for audit. Hard does the same first,
    then empties and deletes what only those edges kept, the edges last
    (``recall_orphans.purge``), so forgetting again after any failure
    finishes the job. A failed step after the edge write is a
    ``cleanup_error`` on each edge it concerned; recall hides the fact and
    its text regardless.
    """
    requested = list(dict.fromkeys(uuids))
    if not requested:
        return ForgetResult()
    driver = open_driver(scope)
    try:
        return await forget_edges(
            driver, scope.group_id, requested, hard=hard, reason=reason
        )
    finally:
        await driver.close()


async def forget_edges(
    driver: GraphDriver,
    group_id: str,
    uuids: list[str],
    *,
    hard: bool,
    reason: str,
) -> ForgetResult:
    """``retract`` on an open driver. Each edge's ``ForgetRecord`` goes to
    the forget stash before the graph is touched, so an ingestion running
    meanwhile can apply the forget again (``recall_ingest.py``)."""
    result = ForgetResult()
    now = datetime.now(timezone.utc).isoformat()
    found = await _existing_edges(driver, group_id, uuids, result)
    if not found:
        return result
    stashed = await read_forgets(group_id)
    records = [
        forget_record(state, stashed.get(state.uuid), now, reason, hard)
        for state in found
    ]
    await stash_forgets(group_id, records)
    retracted = await _retract_edges(driver, group_id, records, result)
    hiding = Hiding(
        uuids=[record.uuid for record in retracted],
        recovered=[[r.uuid, r.fact_redacted, r.name_redacted] for r in retracted],
    )
    hidden = await hide(driver, group_id, hiding, now, result)
    if not hard:
        result.deleted = hiding.uuids
    elif hidden:
        await purge(driver, group_id, hiding.uuids, now, result)
    return result


class EdgeState(BaseModel):
    """An edge as a forget finds it, with the episodes citing it."""

    uuid: str
    fact: str | None = None
    name: str | None = None
    fact_redacted: str | None = None
    name_redacted: str | None = None
    forgotten_at: str | None = None
    expired_at: str | None = None
    invalid_at: str | None = None
    valid_at: str | None = None
    episodes: list[str] | None = None
    source: str | None = None
    target: str | None = None
    source_name: str | None = None
    target_name: str | None = None
    citing: list[str] = []


def forget_record(
    state: EdgeState,
    stashed: ForgetRecord | None,
    now: str,
    reason: str,
    hard: bool,
) -> ForgetRecord:
    """What this forget sets on the edge. An earlier forget's stashed record
    outranks what graphiti can have rewritten since (a repair that failed):
    its forget and expiry times fill in lost ones, and its valid times and
    original text are kept."""
    earlier = stashed or ForgetRecord(
        uuid=state.uuid,
        forgotten_at=now,
        expired_at=now,
        invalid_at=state.invalid_at,
        valid_at=state.valid_at,
        stashed_at=now,
    )
    dropped = earlier.dropped_episodes
    return ForgetRecord(
        uuid=state.uuid,
        hard=hard,
        forgotten_at=state.forgotten_at or earlier.forgotten_at,
        expiration_reason=reason,
        expired_at=state.expired_at or earlier.expired_at,
        invalid_at=earlier.invalid_at,
        valid_at=earlier.valid_at,
        fact_redacted=_original(state.fact_redacted, state.fact, earlier.fact_redacted),
        name_redacted=_original(state.name_redacted, state.name, earlier.name_redacted),
        episodes=[x for x in state.episodes or [] if x not in dropped],
        redacted_episodes=state.citing,
        dropped_episodes=dropped,
        source=state.source,
        target=state.target,
        source_name=state.source_name,
        target_name=state.target_name,
        stashed_at=now,
    )


def _original(kept: str | None, current: str | None, stashed: str | None) -> str | None:
    """The original text: its audit copy, else the edge's own text unless it
    is already the placeholder, else what an earlier forget stashed."""
    if kept:
        return kept
    if current and current != FORGOTTEN_FACT:
        return current
    return stashed


async def _existing_edges(
    driver: GraphDriver,
    group_id: str,
    uuids: list[str],
    result: ForgetResult,
) -> list[EdgeState]:
    """The requested edges that exist in the graph, as they stand.

    A read, so forgetting in a scope that has no graph never creates one (a
    write would). Every other uuid is recorded on ``result`` as a failure.
    """
    try:
        rows = _rows(
            await driver.execute_query(
                _EXISTING_EDGES_QUERY, uuids=uuids, group_id=group_id
            )
        )
    except Exception as exc:
        logger.warning(f"Forget lookup failed in graph {group_id[:20]}", exc_info=True)
        result.failures.extend(
            MemoryForgetFailure.query_error(edge_uuid, exc) for edge_uuid in uuids
        )
        return []
    found = {row["uuid"]: EdgeState.model_validate(row) for row in rows}
    result.failures.extend(
        MemoryForgetFailure.no_match(edge_uuid)
        for edge_uuid in uuids
        if edge_uuid not in found
    )
    return [found[edge_uuid] for edge_uuid in uuids if edge_uuid in found]


async def _retract_edges(
    driver: GraphDriver,
    group_id: str,
    records: list[ForgetRecord],
    result: ForgetResult,
) -> list[ForgetRecord]:
    """Mark each edge forgotten and retracted; the records of those that were.

    One query per edge, so one bad edge cannot hide what happened to the
    others (SECRT-2371); failures are recorded on ``result``.
    """
    retracted: list[ForgetRecord] = []
    for record in records:
        try:
            rows = _rows(
                await driver.execute_query(
                    _RETRACT_EDGE_QUERY, group_id=group_id, **_retract_params(record)
                )
            )
        except Exception as exc:
            logger.warning(
                f"Forget failed for edge {record.uuid} in graph {group_id[:20]}",
                exc_info=True,
            )
            result.failures.append(MemoryForgetFailure.query_error(record.uuid, exc))
            continue
        if rows:
            retracted.append(record)
        else:
            result.failures.append(MemoryForgetFailure.no_match(record.uuid))
    return retracted


def _retract_params(record: ForgetRecord) -> dict[str, Any]:
    return {
        "uuid": record.uuid,
        "forgotten_at": record.forgotten_at,
        "expired_at": record.expired_at,
        "status": record.status,
        "reason": record.expiration_reason,
        "dropped": record.dropped_episodes,
    }


def _rows(result: Any) -> list[dict[str, Any]]:
    return result[0] if result else []


# The ``group_id`` test is defence in depth on top of the per-scope database;
# an edge with no ``group_id`` (a legacy write) is still forgettable.
_EXISTING_EDGES_QUERY = """
MATCH (source)-[e:MENTIONS|RELATES_TO|HAS_MEMBER]->(target)
WHERE e.uuid IN $uuids AND (e.group_id = $group_id OR e.group_id IS NULL)
OPTIONAL MATCH (ep:Episodic)
WHERE e.uuid IN coalesce(ep.entity_edges, [])
WITH e, source, target, collect(ep.uuid) AS citing
RETURN e.uuid AS uuid, e.fact AS fact, e.name AS name,
       e.fact_redacted AS fact_redacted, e.name_redacted AS name_redacted,
       e.forgotten_at AS forgotten_at, e.expired_at AS expired_at,
       e.invalid_at AS invalid_at, e.valid_at AS valid_at,
       e.episodes AS episodes, source.uuid AS source, target.uuid AS target,
       source.name AS source_name, target.name AS target_name, citing
"""

# ``coalesce`` keeps the first forget's ``forgotten_at`` and the first
# retirement time on an edge that already had one (a graphiti expiry, a
# dream demotion or an earlier forget), so a repeat is harmless. ``$dropped``
# names episodes that stated the fact again after it was forgotten but that a
# failed repair left among its sources (``recall_ingest.py``).
_RETRACT_EDGE_QUERY = """
MATCH ()-[e:MENTIONS|RELATES_TO|HAS_MEMBER {uuid: $uuid}]->()
WHERE e.group_id = $group_id OR e.group_id IS NULL
SET e.forgotten_at = coalesce(e.forgotten_at, $forgotten_at),
    e.expired_at = coalesce(e.expired_at, $expired_at),
    e.status = $status,
    e.expiration_reason = $reason,
    e.episodes = CASE WHEN size($dropped) = 0 THEN e.episodes
                      ELSE [x IN coalesce(e.episodes, []) WHERE NOT x IN $dropped]
                 END
RETURN e.uuid AS uuid
"""
