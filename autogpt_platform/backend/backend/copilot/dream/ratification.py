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

# Re-export so callers (scheduler wrapper, future warm-context wiring)
# only have to know one module name. Trailing-comma tuple silences
# unused-import lint for the re-exports.
__all__ = (
    "HIT_TRACKER_KEY_PREFIX",
    "RATIFICATION_GRACE_PERIOD",
    "RatificationResult",
    "record_memory_hit",
    "run_ratification_pass",
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
    """
    query = """
    MATCH ()-[e:RELATES_TO {uuid: $uuid}]->()
    SET e.status = 'active', e.ratified_at = datetime()
    RETURN e.uuid AS uuid
    """
    result = await driver.execute_query(query, uuid=edge_uuid)
    records = result[0] if result else []
    return bool(records)


# Local indirection so tests can mock ``_get_hit_count`` on this module
# rather than the helper module (matches the poison-pill test pattern).
async def _get_hit_count(user_id: str, edge_uuid: str) -> int:
    return await get_hit_count(user_id, edge_uuid)
