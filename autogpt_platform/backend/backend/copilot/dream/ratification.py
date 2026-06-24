"""Ratification loop for ``status='tentative'`` dream proposals (P-0.4).

A tentative MemoryFact edge written by ``apply.py`` is on probation:
either warm-context retrieval proves it useful within a grace period
(at which point we promote it to ``status='active'``), or the grace
period elapses with zero hits and the edge is superseded with
``reason='unratified'``.

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

from backend.copilot.graphiti.client import derive_group_id
from backend.copilot.graphiti.config import graphiti_config
from backend.copilot.graphiti.falkordb_driver import AutoGPTFalkorDriver
from backend.copilot.tools.graphiti_forget import mark_edges_superseded

from .ratification_hits import (
    HIT_TRACKER_KEY_PREFIX,
    RATIFICATION_GRACE_PERIOD,
    get_hit_count,
    parse_created_at,
    record_memory_hit,
)

logger = logging.getLogger(__name__)

# Re-export so callers (scheduler wrapper, warm-context retrieval, the
# nightly batch fan-out) only have to know one module name.
__all__ = (
    "HIT_TRACKER_KEY_PREFIX",
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
    error: str | None = None
    skipped: bool = False
    skip_reason: str | None = None
    per_edge_errors: list[str] = Field(default_factory=list)


async def run_ratification_pass(user_id: str) -> RatificationResult:
    """Promote or supersede every ``status='tentative'`` edge for one user.

    Defensive in two ways:
      * Catastrophic failure (invalid user, no graph, Redis down) is
        captured in ``RatificationResult.error`` rather than raised so
        the scheduler wrapper logs cleanly instead of crashing the job.
      * Per-edge failure is captured in ``per_edge_errors`` so one bad
        edge can't poison the rest of the pass.
    """
    started_at = datetime.now(timezone.utc)
    result = RatificationResult(user_id=user_id, started_at=started_at)

    try:
        group_id = derive_group_id(user_id)
    except ValueError as exc:
        result.error = f"invalid_user_id: {exc}"
        result.completed_at = datetime.now(timezone.utc)
        logger.warning(
            "Ratification skipped — invalid user_id %s: %s", user_id[:12], exc
        )
        return result

    driver = AutoGPTFalkorDriver(
        host=graphiti_config.falkordb_host,
        port=graphiti_config.falkordb_port,
        password=graphiti_config.falkordb_password or None,
        database=group_id,
        # Indices live with the chat-write client; skip the
        # background-task race that produces "Buffer is closed" spam.
        build_indices=False,
    )
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
        for edge in tentatives:
            try:
                await _process_edge(
                    user_id=user_id,
                    driver=driver,
                    edge=edge,
                    now=now,
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
        "Ratification complete for user %s: examined=%d ratified=%d superseded=%d errors=%d",
        user_id[:12],
        result.examined_count,
        result.ratified_count,
        result.superseded_count,
        len(result.per_edge_errors),
    )
    return result


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


async def _process_edge(
    *,
    user_id: str,
    driver: AutoGPTFalkorDriver,
    edge: dict[str, Any],
    now: datetime,
    result: RatificationResult,
) -> None:
    """Promote, supersede, or leave alone one tentative edge.

    Decision table (spec §5):
      * hits >= 1                 → promote to ``status='active'``
      * hits == 0 and past grace  → supersede with ``reason='unratified'``
      * hits == 0 within grace    → no-op (still earning its keep)
    """
    edge_uuid = edge.get("uuid")
    if not edge_uuid:
        result.per_edge_errors.append("missing_uuid")
        return

    hits = await _get_hit_count(user_id, edge_uuid)

    if hits >= 1:
        promoted = await _promote_edge(driver, edge_uuid)
        if promoted:
            result.ratified_count += 1
        return

    created_at = parse_created_at(edge.get("created_at"))
    if created_at is None:
        # Edge without a created_at — be safe and leave it alone rather
        # than supersede something we can't date.
        result.per_edge_errors.append(f"{edge_uuid}: missing_created_at")
        return

    if now - created_at <= RATIFICATION_GRACE_PERIOD:
        return

    succeeded, _failed = await mark_edges_superseded(
        driver,
        [edge_uuid],
        reason="unratified",
        new_status="superseded",
        user_id=user_id,
    )
    if succeeded:
        result.superseded_count += 1


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


async def _promote_edge(driver: AutoGPTFalkorDriver, edge_uuid: str) -> bool:
    """Flip a tentative edge to ``status='active'`` with a ratified_at stamp.

    Returns True iff Cypher actually touched a row. Sub-10ms per spec §5.

    The ratified_at timestamp is generated in Python because FalkorDB
    does not implement Cypher's no-arg ``datetime()`` function.
    """
    query = """
    MATCH ()-[e:RELATES_TO {uuid: $uuid}]->()
    SET e.status = 'active', e.ratified_at = $now
    RETURN e.uuid AS uuid
    """
    result = await driver.execute_query(
        query, uuid=edge_uuid, now=datetime.now(timezone.utc).isoformat()
    )
    records = result[0] if result else []
    return bool(records)


async def try_ratify_on_hit(user_id: str, edge_uuids: list[str]) -> int:
    """Record warm-context hits and promote any tentative edges inline.

    Called from warm-context retrieval (``graphiti/context.py``) once
    per turn with the list of edge uuids that landed in the user's
    context. For each uuid we:

      1. Bump the ``mem:hits:{user_id}:{edge_uuid}`` Redis counter
         (so the nightly ratification sweep also sees the hit and
         agrees on promotion if Cypher fails here).
      2. Issue a targeted Cypher ``SET status='active'`` filtered by
         ``status='tentative' AND expired_at IS NULL`` — already-active
         and already-retracted edges are no-ops via the WHERE clause.

    Returns the count of edges this call actually promoted. The
    function is **safe to fire-and-forget** from the retrieval path:
    failures are caught and logged, never raised; the user's chat
    turn is never blocked on this.

    Per the architecture plan, this is the sync hit-time half of P0.4
    ratification. The nightly batch's ratification sweep still owns
    grace-period supersession; with this hook landing, the sweep
    rarely promotes (most tentatives get hit at least once within a
    day) and primarily cleans up the truly-unused.
    """
    if not edge_uuids:
        return 0
    if not user_id:
        return 0

    # Step 1: bump hit counters (Redis, best-effort, swallows errors).
    # Done before the Cypher promotion so the counter survives even
    # when the promotion path fails.
    for uuid in edge_uuids:
        await record_memory_hit(user_id, uuid)

    # Step 2: targeted Cypher promotion. We open our own driver here
    # because callers are warm-context retrieval call sites that have
    # a higher-level graphiti client but no raw driver — and we want
    # the brief write-lock semantics to be local to this function.
    try:
        group_id = derive_group_id(user_id)
    except ValueError as exc:
        logger.debug("try_ratify_on_hit: invalid user_id %s: %s", user_id[:12], exc)
        return 0

    promoted_count = 0
    driver = AutoGPTFalkorDriver(
        host=graphiti_config.falkordb_host,
        port=graphiti_config.falkordb_port,
        password=graphiti_config.falkordb_password or None,
        database=group_id,
        # Indices live with the chat-write client; skip the
        # background-task race that produces "Buffer is closed" spam.
        build_indices=False,
    )
    try:
        for uuid in edge_uuids:
            try:
                if await _promote_if_tentative(driver, uuid):
                    promoted_count += 1
            except Exception:
                # Per-edge: log + continue. One bad uuid mustn't poison
                # the rest of the retrieved set.
                logger.debug(
                    "try_ratify_on_hit: Cypher failed for user %s edge %s",
                    user_id[:12],
                    uuid,
                    exc_info=True,
                )
    finally:
        await driver.close()

    if promoted_count:
        logger.info(
            "Ratification hit-hook promoted %d edge(s) for user %s",
            promoted_count,
            user_id[:12],
        )
    return promoted_count


async def _promote_if_tentative(driver: AutoGPTFalkorDriver, edge_uuid: str) -> bool:
    """``_promote_edge`` with a ``status='tentative'`` guard.

    Distinct from ``_promote_edge`` because the hit-hook fires on
    EVERY retrieved edge, including already-active ones — we don't
    want to overwrite an active edge's ``ratified_at`` with a fresh
    timestamp on every retrieval. The guard makes the call idempotent.
    """
    query = """
    MATCH ()-[e:RELATES_TO {uuid: $uuid}]->()
    WHERE e.status = 'tentative' AND e.expired_at IS NULL
    SET e.status = 'active', e.ratified_at = $now
    RETURN e.uuid AS uuid
    """
    result = await driver.execute_query(
        query, uuid=edge_uuid, now=datetime.now(timezone.utc).isoformat()
    )
    records = result[0] if result else []
    return bool(records)


# Local indirection so tests can mock ``_get_hit_count`` on this module
# rather than the helper module (matches the poison-pill test pattern).
async def _get_hit_count(user_id: str, edge_uuid: str) -> int:
    return await get_hit_count(user_id, edge_uuid)
