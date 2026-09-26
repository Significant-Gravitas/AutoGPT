"""Forgetting under the recall policy: after ``retract`` no recall path in
``recall.py`` returns the forgotten facts, nor the episode text they came
from.

The chat forget tool and the settings page both forget through here, so a
forget means the same thing wherever it starts.
"""

import logging
from datetime import datetime, timezone
from typing import Any

from .falkordb_driver import AutoGPTFalkorDriver, open_driver
from .memory_model import ForgetResult, MemoryForgetFailure, MemoryStatus
from .recall import live_fact_predicate
from .scope import MemoryScope

logger = logging.getLogger(__name__)


async def retract(
    scope: MemoryScope,
    uuids: list[str],
    *,
    hard: bool = False,
    reason: str = "user_signal",
) -> ForgetResult:
    """Forget edges so that no recall path returns them again.

    Soft (the default) sets ``expired_at``, ``status='retracted'`` and
    ``expiration_reason`` and leaves ``invalid_at`` alone: a forget retracts
    our record of a fact, it does not say the world changed (Snodgrass). Each
    episode none of whose facts is live any more is then stamped
    ``redacted_at``. The edges stay for audit.

    Hard deletes the edges, then each episode left without a live fact, then
    each entity those deletions leave with no fact and no mention.
    """
    requested = list(dict.fromkeys(uuids))
    result = ForgetResult()
    if not requested:
        return result
    driver = open_driver(scope)
    try:
        found = await _existing_edges(driver, scope, requested, result)
        if hard:
            await _hard_retract(driver, scope, found, result)
        else:
            await _soft_retract(driver, scope, found, reason, result)
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


async def _soft_retract(
    driver: AutoGPTFalkorDriver,
    scope: MemoryScope,
    uuids: list[str],
    reason: str,
    result: ForgetResult,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
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
    result.deleted = list(matched)
    if not result.deleted:
        return
    try:
        rows = _rows(
            await driver.execute_query(
                _REDACT_EPISODES_QUERY, uuids=result.deleted, now=now
            )
        )
    except Exception:
        logger.warning(
            f"Edges retracted but episode redaction failed for user "
            f"{scope.owner_user_id[:12]}",
            exc_info=True,
        )
        return
    result.redacted_episodes = [row["uuid"] for row in rows]


async def _hard_retract(
    driver: AutoGPTFalkorDriver,
    scope: MemoryScope,
    uuids: list[str],
    result: ForgetResult,
) -> None:
    matched = await _per_edge(driver, scope, uuids, result, _DELETE_EDGE_QUERY)
    result.deleted = list(matched)
    if not result.deleted:
        return
    endpoints = {
        row[end]
        for row in matched.values()
        for end in ("source_uuid", "target_uuid")
        if row[end] is not None
    }
    try:
        await _delete_orphans(driver, endpoints, result)
    except Exception:
        # The edges are gone, which is what the caller asked for; report
        # them deleted and leave the orphans for a later clean-up.
        logger.warning(
            f"Edges deleted but orphan clean-up failed for user "
            f"{scope.owner_user_id[:12]}",
            exc_info=True,
        )


async def _delete_orphans(
    driver: AutoGPTFalkorDriver, endpoints: set[str], result: ForgetResult
) -> None:
    """Delete what removing ``result.deleted`` orphaned, in dependency order.

    Episodes left with no live fact go first; they are found through their
    ``entity_edges``, which still name the deleted edges. The surviving
    episodes then drop those uuids, and last go the entities (endpoints of the
    deleted edges, or mentioned by a deleted episode) left with no fact and no
    mention.
    """
    rows = _rows(
        await driver.execute_query(_ORPHANED_EPISODES_QUERY, uuids=result.deleted)
    )
    result.deleted_episodes = [row["uuid"] for row in rows]
    mentioned = {entity for row in rows for entity in row["mentioned"]}
    if result.deleted_episodes:
        await driver.execute_query(
            _DELETE_EPISODES_QUERY, uuids=result.deleted_episodes
        )
    await driver.execute_query(_DROP_EDGE_BACKREFS_QUERY, uuids=result.deleted)
    rows = _rows(
        await driver.execute_query(
            _DELETE_ORPHANED_ENTITIES_QUERY, uuids=sorted(endpoints | mentioned)
        )
    )
    result.deleted_entities = [row["uuid"] for row in rows]


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


def _episodes_without_live_facts() -> str:
    """Cypher prefix binding ``ep`` to each episode that names one of
    ``$uuids`` in ``entity_edges`` and has no live fact left; a named edge
    that is retired or no longer exists does not count."""
    return f"""
    MATCH (ep:Episodic)
    WHERE any(x IN ep.entity_edges WHERE x IN $uuids)
    OPTIONAL MATCH ()-[live:RELATES_TO]->()
    WHERE live.uuid IN ep.entity_edges AND {live_fact_predicate("live")}
    WITH ep, count(live) AS live_facts
    WHERE live_facts = 0
    """


# The ``group_id`` test is defence in depth on top of the per-scope database;
# an edge with no ``group_id`` (a legacy write) is still forgettable.
_EXISTING_EDGES_QUERY = """
MATCH ()-[e:MENTIONS|RELATES_TO|HAS_MEMBER]->()
WHERE e.uuid IN $uuids AND (e.group_id = $group_id OR e.group_id IS NULL)
RETURN DISTINCT e.uuid AS uuid
"""

_RETRACT_EDGE_QUERY = """
MATCH ()-[e:MENTIONS|RELATES_TO|HAS_MEMBER {uuid: $uuid}]->()
WHERE e.group_id = $group_id OR e.group_id IS NULL
SET e.expired_at = $now, e.status = $status, e.expiration_reason = $reason
RETURN e.uuid AS uuid
"""

# ``WITH`` captures the ids before ``DELETE``: FalkorDB cannot read a deleted
# relationship's properties (FalkorDB #1393).
_DELETE_EDGE_QUERY = """
MATCH (source)-[e:MENTIONS|RELATES_TO|HAS_MEMBER {uuid: $uuid}]->(target)
WHERE e.group_id = $group_id OR e.group_id IS NULL
WITH e, e.uuid AS uuid, source.uuid AS source_uuid, target.uuid AS target_uuid
DELETE e
RETURN uuid, source_uuid, target_uuid
"""

# ``coalesce`` keeps the time of the first redaction on a repeat forget.
_REDACT_EPISODES_QUERY = (
    _episodes_without_live_facts()
    + """
    SET ep.redacted_at = coalesce(ep.redacted_at, $now)
    RETURN ep.uuid AS uuid
    """
)

_ORPHANED_EPISODES_QUERY = (
    _episodes_without_live_facts()
    + """
    OPTIONAL MATCH (ep)-[:MENTIONS]->(n:Entity)
    RETURN ep.uuid AS uuid, collect(n.uuid) AS mentioned
    """
)

_DELETE_EPISODES_QUERY = """
MATCH (ep:Episodic)
WHERE ep.uuid IN $uuids
DETACH DELETE ep
"""

_DROP_EDGE_BACKREFS_QUERY = """
MATCH (ep:Episodic)
WHERE any(x IN ep.entity_edges WHERE x IN $uuids)
SET ep.entity_edges = [x IN ep.entity_edges WHERE NOT x IN $uuids]
"""

# A community membership (``HAS_MEMBER``) does not keep an entity: communities
# are rebuilt from the facts, and the entity's own summary may repeat the
# fact that was just deleted.
_DELETE_ORPHANED_ENTITIES_QUERY = """
MATCH (n:Entity)
WHERE n.uuid IN $uuids
OPTIONAL MATCH (n)-[r:RELATES_TO|MENTIONS]-()
WITH n, count(r) AS links
WHERE links = 0
WITH n, n.uuid AS uuid
DETACH DELETE n
RETURN uuid
"""
