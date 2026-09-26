"""Forgetting under the recall policy: after ``retract`` no recall path in
``recall.py`` returns the forgotten facts, nor the episode text they came
from, while the audit record stays.

The chat forget tool and the settings page both forget through here, so a
forget means the same thing wherever it starts.
"""

import logging
from datetime import datetime, timezone
from typing import Any

from .falkordb_driver import AutoGPTFalkorDriver, open_driver
from .memory_model import ForgetResult, MemoryForgetFailure, MemoryStatus
from .recall import (
    USER_FORGET_REASON,
    forgotten_facts_clause,
    recallable_episode_predicate,
)
from .recall_orphans import delete_orphans
from .scope import MemoryScope

logger = logging.getLogger(__name__)


async def retract(
    scope: MemoryScope,
    uuids: list[str],
    *,
    hard: bool = False,
    reason: str = USER_FORGET_REASON,
) -> ForgetResult:
    """Forget edges so that no recall path returns them, or their text, again.

    Soft (the default) sets ``status='retracted'`` and ``expiration_reason``,
    sets ``expired_at`` only if the edge had none (the first retirement time
    is kept), and leaves ``invalid_at`` alone: a forget retracts our record of
    a fact, it does not say the world changed (Snodgrass). Every episode that
    names a forgotten edge is then stamped ``redacted_at``. Edges and
    episodes stay for audit.

    Hard does the same first, so a failure part-way still leaves the facts
    forgotten, then deletes the edges and whatever only they referenced.

    A failed redaction or clean-up is a ``cleanup_error`` on each edge it
    concerned: recall hides the text regardless, and forgetting again is safe.
    """
    requested = list(dict.fromkeys(uuids))
    result = ForgetResult()
    if not requested:
        return result
    now = datetime.now(timezone.utc).isoformat()
    driver = open_driver(scope)
    try:
        found = await _existing_edges(driver, scope, requested, result)
        retracted = await _retract_edges(driver, scope, found, reason, now, result)
        redacted = await _redact_episodes(driver, scope, retracted, now, result)
        if not hard:
            result.deleted = retracted
        elif redacted:
            # A deleted edge can no longer hide its episodes on the read
            # side, so nothing is deleted unless the redaction landed.
            await _hard_delete(driver, scope, retracted, result)
    finally:
        await driver.close()
    return result


async def _existing_edges(
    driver: AutoGPTFalkorDriver,
    scope: MemoryScope,
    uuids: list[str],
    result: ForgetResult,
) -> list[str]:
    """The requested uuids that name a forgettable edge in ``scope``.

    A read, so forgetting in a scope that has no graph never creates one (a
    write would). Every other uuid is recorded on ``result`` as a failure.
    """
    try:
        rows = _rows(
            await driver.execute_query(
                _EXISTING_EDGES_QUERY, uuids=uuids, group_id=scope.group_id
            )
        )
    except Exception as exc:
        logger.warning(
            f"Forget lookup failed for user {scope.owner_user_id[:12]}", exc_info=True
        )
        result.failures.extend(
            MemoryForgetFailure.query_error(edge_uuid, exc) for edge_uuid in uuids
        )
        return []
    found = {row["uuid"] for row in rows}
    result.failures.extend(
        MemoryForgetFailure.no_match(edge_uuid)
        for edge_uuid in uuids
        if edge_uuid not in found
    )
    return [edge_uuid for edge_uuid in uuids if edge_uuid in found]


async def _retract_edges(
    driver: AutoGPTFalkorDriver,
    scope: MemoryScope,
    uuids: list[str],
    reason: str,
    now: str,
    result: ForgetResult,
) -> list[str]:
    """Mark each edge retracted; the uuids that were."""
    matched = await _per_edge(
        driver,
        scope,
        uuids,
        result,
        _RETRACT_EDGE_QUERY,
        now=now,
        status=MemoryStatus.retracted.value,
        reason=reason,
    )
    return list(matched)


async def _redact_episodes(
    driver: AutoGPTFalkorDriver,
    scope: MemoryScope,
    uuids: list[str],
    now: str,
    result: ForgetResult,
) -> bool:
    """Stamp ``redacted_at`` on each episode naming one of the retracted
    ``uuids``; False when that write failed."""
    if not uuids:
        return True
    try:
        rows = _rows(
            await driver.execute_query(_REDACT_EPISODES_QUERY, uuids=uuids, now=now)
        )
    except Exception as exc:
        logger.warning(
            f"Edges retracted but episode redaction failed for user "
            f"{scope.owner_user_id[:12]}",
            exc_info=True,
        )
        result.failures.extend(
            MemoryForgetFailure.cleanup_error(edge_uuid, exc) for edge_uuid in uuids
        )
        return False
    result.redacted_episodes = [row["uuid"] for row in rows]
    return True


async def _hard_delete(
    driver: AutoGPTFalkorDriver,
    scope: MemoryScope,
    uuids: list[str],
    result: ForgetResult,
) -> None:
    """Delete the retracted edges, then what they alone kept alive."""
    matched = await _per_edge(driver, scope, uuids, result, _DELETE_EDGE_QUERY)
    result.deleted = list(matched)
    endpoints = {
        row[end]
        for row in matched.values()
        for end in ("source_uuid", "target_uuid")
        if row[end] is not None
    }
    try:
        if result.deleted:
            await delete_orphans(driver, endpoints, result)
    except Exception as exc:
        logger.warning(
            f"Edges deleted but orphan clean-up failed for user "
            f"{scope.owner_user_id[:12]}",
            exc_info=True,
        )
        result.failures.extend(
            MemoryForgetFailure.cleanup_error(edge_uuid, exc)
            for edge_uuid in result.deleted
        )
    deleted = set(result.deleted_episodes)
    result.redacted_episodes = [
        episode for episode in result.redacted_episodes if episode not in deleted
    ]


async def _per_edge(
    driver: AutoGPTFalkorDriver,
    scope: MemoryScope,
    uuids: list[str],
    result: ForgetResult,
    query: str,
    **params: str,
) -> dict[str, dict[str, Any]]:
    """Run ``query`` once per edge uuid; the first row of each match, by uuid.

    One query per uuid, so one bad edge cannot hide what happened to the
    others (SECRT-2371); failures are recorded on ``result``.
    """
    matched: dict[str, dict[str, Any]] = {}
    for edge_uuid in uuids:
        try:
            rows = _rows(
                await driver.execute_query(
                    query, uuid=edge_uuid, group_id=scope.group_id, **params
                )
            )
        except Exception as exc:
            logger.warning(
                f"Forget failed for edge {edge_uuid} of user "
                f"{scope.owner_user_id[:12]}",
                exc_info=True,
            )
            result.failures.append(MemoryForgetFailure.query_error(edge_uuid, exc))
            continue
        if rows:
            matched[edge_uuid] = rows[0]
        else:
            result.failures.append(MemoryForgetFailure.no_match(edge_uuid))
    return matched


def _rows(
    result: tuple[list[dict[str, Any]], list[str], None] | None,
) -> list[dict[str, Any]]:
    return result[0] if result else []


# The ``group_id`` test is defence in depth on top of the per-scope database;
# an edge with no ``group_id`` (a legacy write) is still forgettable.
_EXISTING_EDGES_QUERY = """
MATCH ()-[e:MENTIONS|RELATES_TO|HAS_MEMBER]->()
WHERE e.uuid IN $uuids AND (e.group_id = $group_id OR e.group_id IS NULL)
RETURN DISTINCT e.uuid AS uuid
"""

# ``coalesce`` keeps the first retirement time on an edge that already had
# one (a graphiti expiry, a dream demotion or an earlier forget).
_RETRACT_EDGE_QUERY = """
MATCH ()-[e:MENTIONS|RELATES_TO|HAS_MEMBER {uuid: $uuid}]->()
WHERE e.group_id = $group_id OR e.group_id IS NULL
SET e.expired_at = coalesce(e.expired_at, $now),
    e.status = $status,
    e.expiration_reason = $reason
RETURN e.uuid AS uuid
"""

# Every episode naming one of ``$uuids`` that the recall policy now hides,
# which after the retraction is all of them: the stamp and the read-side test
# cannot disagree. ``coalesce`` keeps the time of the first redaction.
_REDACT_EPISODES_QUERY = (
    forgotten_facts_clause()
    + f"""
MATCH (ep:Episodic)
WHERE any(x IN coalesce(ep.entity_edges, []) WHERE x IN $uuids)
  AND NOT ({recallable_episode_predicate("ep")})
SET ep.redacted_at = coalesce(ep.redacted_at, $now)
RETURN ep.uuid AS uuid
"""
)

# ``WITH`` captures the ids before ``DELETE``: FalkorDB cannot read a deleted
# relationship's properties (FalkorDB #1393).
_DELETE_EDGE_QUERY = """
MATCH (source)-[e:MENTIONS|RELATES_TO|HAS_MEMBER {uuid: $uuid}]->(target)
WHERE e.group_id = $group_id OR e.group_id IS NULL
WITH e, e.uuid AS uuid, source.uuid AS source_uuid, target.uuid AS target_uuid
DELETE e
RETURN uuid, source_uuid, target_uuid
"""
