"""Forgetting under the recall policy.

After ``retract`` the recall paths in ``recall.py`` return neither the
forgotten facts nor the episode text they came from, the sentence is gone
from the fact and the summaries graphiti's own prompts read, and the audit
record stays; ``graphiti/AGENTS.md`` lists the limits. The chat forget tool
and the settings page both forget through here, so a forget means the same
thing wherever it starts. ``forgotten_at``, the policy's marker for a
forgotten fact, is written here and, for forgets made before it existed, by
``migrations/backfill_legacy_forgets.py``; nothing else writes it.
"""

import logging
from datetime import datetime, timezone
from typing import Any

from .falkordb_driver import AutoGPTFalkorDriver, open_driver
from .memory_model import ForgetResult, MemoryForgetFailure, MemoryStatus
from .recall import USER_FORGET_REASON
from .recall_hide import hide
from .recall_orphans import purge
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
    result = ForgetResult()
    if not requested:
        return result
    now = datetime.now(timezone.utc).isoformat()
    driver = open_driver(scope)
    try:
        found = await _existing_edges(driver, scope, requested, result)
        retracted = await _retract_edges(driver, scope, found, reason, now, result)
        hidden = await hide(driver, scope, retracted, now, result)
        if not hard:
            result.deleted = retracted
        elif hidden:
            await purge(driver, scope, retracted, now, result)
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
    """Mark each edge forgotten and retracted; the uuids that were.

    One query per uuid, so one bad edge cannot hide what happened to the
    others (SECRT-2371); failures are recorded on ``result``.
    """
    retracted: list[str] = []
    for edge_uuid in uuids:
        try:
            rows = _rows(
                await driver.execute_query(
                    _RETRACT_EDGE_QUERY,
                    uuid=edge_uuid,
                    group_id=scope.group_id,
                    now=now,
                    status=MemoryStatus.retracted.value,
                    reason=reason,
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
            retracted.append(edge_uuid)
        else:
            result.failures.append(MemoryForgetFailure.no_match(edge_uuid))
    return retracted


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

# ``coalesce`` keeps the first forget's ``forgotten_at`` and the first
# retirement time on an edge that already had one (a graphiti expiry, a
# dream demotion or an earlier forget), so a repeat is harmless.
_RETRACT_EDGE_QUERY = """
MATCH ()-[e:MENTIONS|RELATES_TO|HAS_MEMBER {uuid: $uuid}]->()
WHERE e.group_id = $group_id OR e.group_id IS NULL
SET e.forgotten_at = coalesce(e.forgotten_at, $now),
    e.expired_at = coalesce(e.expired_at, $now),
    e.status = $status,
    e.expiration_reason = $reason
RETURN e.uuid AS uuid
"""
