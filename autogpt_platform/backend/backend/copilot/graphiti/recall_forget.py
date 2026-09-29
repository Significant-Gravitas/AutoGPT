"""Forgetting under the recall policy.

After ``retract`` the recall paths in ``recall.py`` return neither the
forgotten facts nor the episode text they came from, nor what the dream
derived from them (``recall_cascade.py``), the sentence is gone from the
fact and from what graphiti read out of it onto entities, and the audit
record stays; ``graphiti/AGENTS.md`` lists the limits. The chat forget tool
and the settings page both forget through here, so a forget means the same
thing wherever it starts. ``forgotten_at``, the policy's marker for a
forgotten fact, is written here (for a derived fact, by the cascade this
runs, which ``migrations/backfill_derivations.py`` also runs for forgets
made before it) and, for forgets made before the marker existed, by
``migrations/backfill_legacy_forgets.py``; nothing else writes it.
"""

import logging
from datetime import datetime, timezone
from typing import Any

from graphiti_core.driver.driver import GraphDriver

from .falkordb_driver import open_driver
from .memory_model import (
    ForgetResult,
    MemoryForgetFailure,
    MemoryForgetFailureCode,
    MemoryStatus,
)
from .recall import USER_FORGET_REASON
from .recall_cascade import cascade
from .recall_cascade_queries import NAMED_ROOTS_QUERY
from .recall_cascade_walk import derived_reason
from .recall_hide import hide
from .recall_orphans import purge
from .recall_reconcile import reconcile
from .scope import MemoryScope
from .scope_lock import FORGET_LOCK_WAIT_SECONDS, LockState, graph_write_lock

logger = logging.getLogger(__name__)


async def retract(
    scope: MemoryScope,
    uuids: list[str],
    *,
    hard: bool = False,
    reason: str = USER_FORGET_REASON,
) -> ForgetResult:
    """Forget edges: recall stops returning them and the text they came from.

    The forget holds the graph's write lock (``scope_lock.py``), so it never
    lands while an ingestion is between reading the graph and saving over
    it; when the lock stays busy for ``FORGET_LOCK_WAIT_SECONDS`` every uuid
    fails as ``busy`` and nothing is written. It first completes the
    derivation records of every dream write still marked in the graph
    (``recall_reconcile.reconcile``), so its cascade sees complete
    provenance; if that fails or leaves markers, each fact forgotten is a
    ``cleanup_error`` and forgetting it again finishes the job.

    Soft (the default) stamps ``forgotten_at``, sets ``status='retracted'``
    and ``expiration_reason``, keeps an earlier ``expired_at`` and leaves
    ``invalid_at`` alone: a forget retracts our record of a fact, it does
    not say the world changed (Snodgrass). ``recall_hide.hide`` then moves
    the sentence out of what graphiti reads and redacts every episode citing
    the fact; edges and episodes stay for audit. Then, under the same lock,
    ``recall_cascade.cascade`` retracts the facts the dream derived from the
    forgotten ones, transitively, and hides the dream episodes that did
    (``ForgetResult.derived``). Hard does all that first, the cascade soft
    too but erasing the derived text it reaches (``recall_erase.py``), then
    empties and deletes what only the forgotten edges kept, the edges last
    (``recall_orphans.purge``), so forgetting again after any failure
    finishes the job, a purged fact's cascade included (below). A failed
    step after the edge write is a
    ``cleanup_error`` on each edge it concerned; recall hides the fact and
    its text regardless.

    A uuid that is no longer in the graph but that a derivation record, a
    citation marker, an episode's ``redacted_for`` or an earlier cascade's
    reason still names was a fact a hard forget purged, perhaps before its
    cascade finished: the forget goes on with that cascade, erasing, and
    lists it in ``ForgetResult.resumed`` instead of failing it as no match.
    """
    requested = list(dict.fromkeys(uuids))
    if not requested:
        return ForgetResult()
    wait = FORGET_LOCK_WAIT_SECONDS
    async with graph_write_lock(scope.group_id, wait_seconds=wait) as lock:
        if lock is LockState.BUSY:
            busy = [MemoryForgetFailure.busy(uuid) for uuid in requested]
            return ForgetResult(failures=busy)
        driver = open_driver(scope)
        try:
            return await _forget(driver, scope.group_id, requested, hard, reason)
        finally:
            await driver.close()


async def _forget(
    driver: GraphDriver, group_id: str, uuids: list[str], hard: bool, reason: str
) -> ForgetResult:
    result = ForgetResult()
    now = datetime.now(timezone.utc).isoformat()
    unreconciled = await _reconciled(driver, group_id)
    found = await _existing_edges(driver, group_id, uuids, result)
    result.resumed = await _purged_roots(driver, result)
    retracted = await _retract_edges(driver, group_id, found, reason, now, result)
    hidden = await hide(driver, group_id, retracted, now, result)
    if hidden and retracted:
        await cascade(driver, group_id, retracted, now, result, erase=hard)
    if result.resumed:
        await cascade(driver, group_id, result.resumed, now, result, erase=True)
    if unreconciled is not None:
        roots = [*retracted, *result.resumed]
        _provenance_incomplete(result, roots, unreconciled)
    if not hard:
        result.deleted = retracted
    elif hidden:
        await purge(driver, group_id, retracted, now, result)
    return result


async def _purged_roots(driver: GraphDriver, result: ForgetResult) -> list[str]:
    """The uuids that matched no edge but that something still names as a
    root: taken off ``result.failures``, to resume their cascade."""
    missing = [
        failure.uuid
        for failure in result.failures
        if failure.code is MemoryForgetFailureCode.NO_MATCH
    ]
    if not missing:
        return []
    rows = _rows(
        await driver.execute_query(
            NAMED_ROOTS_QUERY, uuids=missing, prefix=derived_reason("")
        )
    )
    named = {row["uuid"] for row in rows}
    result.failures = [f for f in result.failures if f.uuid not in named]
    return [uuid for uuid in missing if uuid in named]


async def _reconciled(driver: GraphDriver, group_id: str) -> Exception | None:
    """Complete the graph's pending dream records before anything else; what
    went wrong, when some may still be missing: the error, or ``None``."""
    try:
        done = await reconcile(driver, group_id)
    except Exception as exc:
        logger.warning(
            f"Could not reconcile graph {group_id[:20]} before a forget",
            exc_info=True,
        )
        return exc
    if done.left:
        return RuntimeError("more dream writes pending a record than one forget takes")
    return None


def _provenance_incomplete(
    result: ForgetResult, retracted: list[str], exc: Exception
) -> None:
    """A ``cleanup_error`` on each fact forgotten that has no failure yet: its
    cascade may have missed a dream fact whose record was still pending."""
    failed = {failure.uuid for failure in result.failures}
    result.failures.extend(
        MemoryForgetFailure.cleanup_error(uuid, exc)
        for uuid in retracted
        if uuid not in failed
    )


async def _existing_edges(
    driver: GraphDriver,
    group_id: str,
    uuids: list[str],
    result: ForgetResult,
) -> list[str]:
    """The requested uuids that name a forgettable edge in the graph.

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
    found = {row["uuid"] for row in rows}
    result.failures.extend(
        MemoryForgetFailure.no_match(edge_uuid)
        for edge_uuid in uuids
        if edge_uuid not in found
    )
    return [edge_uuid for edge_uuid in uuids if edge_uuid in found]


async def _retract_edges(
    driver: GraphDriver,
    group_id: str,
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
                    group_id=group_id,
                    now=now,
                    status=MemoryStatus.retracted.value,
                    reason=reason,
                )
            )
        except Exception as exc:
            logger.warning(
                f"Forget failed for edge {edge_uuid} in graph {group_id[:20]}",
                exc_info=True,
            )
            result.failures.append(MemoryForgetFailure.query_error(edge_uuid, exc))
            continue
        if rows:
            retracted.append(edge_uuid)
        else:
            result.failures.append(MemoryForgetFailure.no_match(edge_uuid))
    return retracted


def _rows(result: Any) -> list[dict[str, Any]]:
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
