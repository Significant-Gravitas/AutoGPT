"""Graphiti read/write for org shared-memory governance (held-memory review).

Org admins review *tentative* ("held") memories that non-admins wrote into the
org's shared tiers while the hold buffer was on, and either:

- **approve** — ratify the edge ``tentative`` → ``active`` by reusing the dream
  ratification status-flip (``_promote_if_tentative``); or
- **reject** — soft-retract the edge by reusing ``memory_forget``'s supersede
  path (``mark_edges_superseded``), which sets ``expired_at`` + an auditable
  ``status``/``expiration_reason``.

Scope is strictly the org group (``org_<id>``) plus every team group
(``team_<id>``) of *this* org. Personal tiers (``user_<id>``) are never
enumerated or touched here, and a memory id that does not live in one of this
org's shared tiers is a 404 (cross-org / personal / nonexistent alike).
"""

import logging

from redis.exceptions import ResponseError

from backend.copilot.dream.ratification import _promote_if_tentative
from backend.copilot.graphiti.client import derive_org_group_id, derive_team_group_id
from backend.copilot.graphiti.config import graphiti_config
from backend.copilot.graphiti.falkordb_driver import AutoGPTFalkorDriver
from backend.copilot.graphiti.tiers import MemoryTier
from backend.copilot.tools.graphiti_forget import mark_edges_superseded
from backend.data.db import prisma
from backend.util.exceptions import NotFoundError

from .memory_model import HeldMemory, HeldMemoryListResponse, MemoryActionResult

logger = logging.getLogger(__name__)

# Reason stamped on the retracted edge's expiration_reason for the audit trail.
_REJECT_REASON = "org_admin_reject"

_MISSING_GRAPH_MARKERS = ("no such graph", "does not exist", "invalid graph")


def _is_missing_graph_error(exc: BaseException) -> bool:
    """FalkorDB raises ``ResponseError`` when a group's graph was never
    populated (an org/team that has stored no shared memory yet)."""
    if not isinstance(exc, ResponseError):
        return False
    msg = str(exc).lower()
    return any(marker in msg for marker in _MISSING_GRAPH_MARKERS)


def _open_driver(group_id: str) -> AutoGPTFalkorDriver:
    """Read/write-capable, short-lived FalkorDB driver for one group.

    Mirrors the admin memory inspector + ratification pattern: skip full
    Graphiti client construction (LLM/cross-encoder) and the per-init index
    task (``build_indices=False``); ``database=group_id`` scopes every query
    to that tenant's graph.
    """
    return AutoGPTFalkorDriver(
        host=graphiti_config.falkordb_host,
        port=graphiti_config.falkordb_port,
        password=graphiti_config.falkordb_password or None,
        database=group_id,
        build_indices=False,
    )


async def _org_shared_groups(
    org_id: str,
) -> list[tuple[str, MemoryTier, str | None, str | None]]:
    """The org's shared groups: the org group + every team group of the org.

    Returns ``(group_id, tier, team_id, team_name)`` tuples. A team id that
    fails group-id sanitization is skipped rather than sinking the whole
    listing (mirrors tiers.py's per-tier isolation).
    """
    groups: list[tuple[str, MemoryTier, str | None, str | None]] = [
        (derive_org_group_id(org_id), MemoryTier.org, None, None)
    ]
    teams = await prisma.team.find_many(where={"orgId": org_id})
    for team in teams:
        try:
            team_group = derive_team_group_id(team.id)
        except ValueError:
            logger.warning(
                "Skipping team %s from held-memory scope: invalid group id",
                team.id,
            )
            continue
        groups.append((team_group, MemoryTier.team, team.id, team.name))
    return groups


_HELD_LIST_QUERY = """
MATCH (src:Entity)-[e:RELATES_TO]->(tgt:Entity)
WHERE e.group_id = $g AND e.status = 'tentative' AND e.expired_at IS NULL
RETURN e.uuid AS uuid,
       e.name AS name,
       e.fact AS fact,
       e.source_kind AS source_kind,
       e.provenance AS provenance,
       toString(e.created_at) AS created_at
ORDER BY e.created_at DESC
LIMIT $limit
"""

_HELD_LOCATE_QUERY = """
MATCH ()-[e:RELATES_TO {uuid: $u}]->()
WHERE e.group_id = $g AND e.status = 'tentative' AND e.expired_at IS NULL
RETURN e.uuid AS uuid
"""


async def _query_group_tentative(group_id: str, limit: int) -> list[dict]:
    """Tentative, non-retracted edges in one group (empty on a missing graph)."""
    driver = _open_driver(group_id)
    try:
        result = await driver.execute_query(_HELD_LIST_QUERY, g=group_id, limit=limit)
        return result[0] if result else []
    except ResponseError as exc:
        if not _is_missing_graph_error(exc):
            raise
        return []
    finally:
        await driver.close()


async def list_held_memories(org_id: str, limit: int = 50) -> HeldMemoryListResponse:
    """List tentative memories across the org tier and all its team tiers."""
    groups = await _org_shared_groups(org_id)
    items: list[HeldMemory] = []
    for group_id, tier, team_id, team_name in groups:
        for row in await _query_group_tentative(group_id, limit):
            items.append(
                HeldMemory(
                    id=str(row.get("uuid", "")),
                    tier=tier.value,
                    team_id=team_id,
                    team_name=team_name,
                    name=row.get("name"),
                    fact=row.get("fact"),
                    created_at=row.get("created_at"),
                    source_kind=row.get("source_kind"),
                    provenance=row.get("provenance"),
                )
            )
    # Newest first across the merged tiers, then cap.
    items.sort(key=lambda h: h.created_at or "", reverse=True)
    return HeldMemoryListResponse(org_id=org_id, items=items[:limit])


async def _locate_held_edge(
    org_id: str, memory_id: str
) -> tuple[str, MemoryTier, str | None] | None:
    """Find which of *this org's* shared groups holds *memory_id* as tentative.

    Returns ``(group_id, tier, team_id)`` or None. Because it only ever opens
    drivers for this org's groups, a personal or other-org edge is never found
    → the caller 404s. This is the org-ownership check for approve/reject.
    """
    for group_id, tier, team_id, _team_name in await _org_shared_groups(org_id):
        driver = _open_driver(group_id)
        try:
            result = await driver.execute_query(
                _HELD_LOCATE_QUERY, u=memory_id, g=group_id
            )
            rows = result[0] if result else []
        except ResponseError as exc:
            if not _is_missing_graph_error(exc):
                raise
            rows = []
        finally:
            await driver.close()
        if rows:
            return group_id, tier, team_id
    return None


async def approve_held_memory(
    org_id: str, memory_id: str, actor_user_id: str
) -> MemoryActionResult:
    """Ratify a held memory (tentative → active) via the dream status-flip."""
    located = await _locate_held_edge(org_id, memory_id)
    if located is None:
        raise NotFoundError(
            f"Held memory {memory_id} not found in this organization's shared tiers"
        )
    group_id, tier, team_id = located
    driver = _open_driver(group_id)
    try:
        applied = await _promote_if_tentative(driver, memory_id)
    finally:
        await driver.close()
    return MemoryActionResult(
        id=memory_id,
        action="approve",
        applied=applied,
        tier=tier.value,
        team_id=team_id,
    )


async def reject_held_memory(
    org_id: str, memory_id: str, actor_user_id: str
) -> MemoryActionResult:
    """Retract a held memory via memory_forget's soft-retract (supersede)."""
    located = await _locate_held_edge(org_id, memory_id)
    if located is None:
        raise NotFoundError(
            f"Held memory {memory_id} not found in this organization's shared tiers"
        )
    group_id, tier, team_id = located
    driver = _open_driver(group_id)
    try:
        succeeded, _failed = await mark_edges_superseded(
            driver,
            [memory_id],
            reason=_REJECT_REASON,
            new_status="superseded",
            user_id=actor_user_id,
            group_id=group_id,
        )
    finally:
        await driver.close()
    return MemoryActionResult(
        id=memory_id,
        action="reject",
        applied=memory_id in succeeded,
        tier=tier.value,
        team_id=team_id,
    )
