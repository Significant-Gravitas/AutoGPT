"""User-facing memory API: read + forget for the caller's own memory scopes.

Backs the Settings → Memory page (SECRT-2580). Scope is structural, mirroring
the admin memory API: account (AutoPilot) routes live at ``/memory/...`` and
expert routes at ``/memory/experts/{expert_id}/...``. Every route resolves the
scope from the *authenticated caller* — there is no target-user path segment,
so these routes can never read or delete another user's memory by construction.

Deletion semantics match the chat forget tool: single-fact forget is a
bi-temporal system retraction (sets only ``expired_at``; the edge stops being
served but stays for audit), while the scope erase hard-deletes every node and
edge in the scope's graph, raw episode text included.
"""

import logging
from datetime import datetime, timezone
from typing import Annotated

import autogpt_libs.auth as autogpt_auth_lib
from fastapi import APIRouter, HTTPException, Path, Query, Security
from redis.exceptions import ResponseError

from backend.api.features.experts import experts_db
from backend.api.features.memory.models import (
    EraseMemoryResponse,
    ForgetFactResponse,
    MemoryFact,
    MemoryFactListResponse,
    MemoryScopeOverview,
)
from backend.copilot.graphiti.client import derive_memory_group_id
from backend.copilot.graphiti.config import graphiti_config, is_enabled_for_user
from backend.copilot.graphiti.falkordb_driver import AutoGPTFalkorDriver

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/memory",
    tags=["memory", "private"],
    dependencies=[Security(autogpt_auth_lib.requires_user)],
)

_EXPERT_ID_PATH = Path(
    min_length=1, max_length=128, description="Expert id owned by the caller"
)
_FACT_UUID_PATH = Path(min_length=1, max_length=128, description="Fact edge uuid")
_LIMIT_QUERY = Query(ge=1, le=500)

_MISSING_GRAPH_MARKERS = ("no such graph", "does not exist", "invalid graph")


def _now_iso() -> str:
    """FalkorDB has no Cypher ``datetime()``; timestamps are bound in Python."""
    return datetime.now(timezone.utc).isoformat()


def _is_missing_graph_error(exc: BaseException) -> bool:
    if not isinstance(exc, ResponseError):
        return False
    msg = str(exc).lower()
    return any(marker in msg for marker in _MISSING_GRAPH_MARKERS)


def _open_driver(group_id: str) -> AutoGPTFalkorDriver:
    """Cypher-only driver — skips LLM-client construction and the per-init
    index build (indices exist from the long-lived chat-write client)."""
    return AutoGPTFalkorDriver(
        host=graphiti_config.falkordb_host,
        port=graphiti_config.falkordb_port,
        password=graphiti_config.falkordb_password or None,
        database=group_id,
        build_indices=False,
    )


async def _resolve_scope(user_id: str, expert_id: str | None) -> tuple[str, str | None]:
    """Resolve the caller-owned memory group for the requested scope.

    Memory must be enabled for the caller, and an expert scope must name an
    active expert the caller owns — archived experts 404 here, matching the
    settings page's active-experts-only dropdown.
    """
    if not await is_enabled_for_user(user_id):
        raise HTTPException(
            status_code=403, detail="Memory is not enabled for your account"
        )

    if expert_id is None:
        try:
            return derive_memory_group_id(user_id), None
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    expert = await experts_db.get_expert(user_id, expert_id, include_workflows=False)
    if expert is None:
        raise HTTPException(status_code=404, detail="Expert not found")

    try:
        return derive_memory_group_id(user_id, expert.id), expert.id
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


async def _count(driver: AutoGPTFalkorDriver, query: str) -> int:
    try:
        result = await driver.execute_query(query)
        rows = result[0] if result else []
        return int(rows[0]["c"]) if rows else 0
    except ResponseError as exc:
        if _is_missing_graph_error(exc):
            return 0
        raise


async def _get_overview_impl(
    user_id: str, expert_id: str | None
) -> MemoryScopeOverview:
    group_id, resolved_expert_id = await _resolve_scope(user_id, expert_id)
    driver = _open_driver(group_id)
    try:
        facts = await _count(
            driver,
            "MATCH ()-[e:RELATES_TO]->() WHERE e.expired_at IS NULL "
            "RETURN count(e) AS c",
        )
        entities = await _count(driver, "MATCH (n:Entity) RETURN count(n) AS c")
        episodes = await _count(driver, "MATCH (n:Episodic) RETURN count(n) AS c")
    finally:
        await driver.close()
    return MemoryScopeOverview(
        expert_id=resolved_expert_id,
        facts=facts,
        entities=entities,
        episodes=episodes,
    )


@router.get("/overview", operation_id="get_my_memory_overview")
async def get_my_memory_overview(
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> MemoryScopeOverview:
    """Counts for the caller's account (AutoPilot) memory."""
    return await _get_overview_impl(user_id, None)


@router.get(
    "/experts/{expert_id}/overview", operation_id="get_my_expert_memory_overview"
)
async def get_my_expert_memory_overview(
    expert_id: Annotated[str, _EXPERT_ID_PATH],
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> MemoryScopeOverview:
    """Counts for one of the caller's experts' isolated memory."""
    return await _get_overview_impl(user_id, expert_id)


async def _list_facts_impl(
    user_id: str, expert_id: str | None, limit: int
) -> MemoryFactListResponse:
    group_id, resolved_expert_id = await _resolve_scope(user_id, expert_id)
    driver = _open_driver(group_id)
    try:
        result = await driver.execute_query(
            """
            MATCH (src:Entity)-[e:RELATES_TO]->(tgt:Entity)
            WHERE e.group_id = $g AND e.expired_at IS NULL
            RETURN e.uuid AS uuid,
                   e.fact AS fact,
                   e.name AS name,
                   src.name AS source,
                   tgt.name AS target,
                   toString(e.created_at) AS created_at
            ORDER BY e.created_at DESC
            LIMIT $limit
            """,
            g=group_id,
            limit=limit,
        )
        rows = result[0] if result else []
    except ResponseError as exc:
        if not _is_missing_graph_error(exc):
            raise
        rows = []
    finally:
        await driver.close()

    items = [
        MemoryFact(
            uuid=str(r.get("uuid", "")),
            fact=r.get("fact"),
            name=r.get("name"),
            source=str(r.get("source") or ""),
            target=str(r.get("target") or ""),
            created_at=r.get("created_at"),
        )
        for r in rows
    ]
    return MemoryFactListResponse(expert_id=resolved_expert_id, items=items)


@router.get("/facts", operation_id="list_my_memory_facts")
async def list_my_memory_facts(
    limit: Annotated[int, _LIMIT_QUERY] = 50,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> MemoryFactListResponse:
    """Live facts in the caller's account memory, newest first."""
    return await _list_facts_impl(user_id, None, limit)


@router.get("/experts/{expert_id}/facts", operation_id="list_my_expert_memory_facts")
async def list_my_expert_memory_facts(
    expert_id: Annotated[str, _EXPERT_ID_PATH],
    limit: Annotated[int, _LIMIT_QUERY] = 50,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> MemoryFactListResponse:
    """Live facts in one expert's isolated memory, newest first."""
    return await _list_facts_impl(user_id, expert_id, limit)


async def _forget_fact_impl(
    user_id: str, expert_id: str | None, fact_uuid: str
) -> ForgetFactResponse:
    group_id, _ = await _resolve_scope(user_id, expert_id)
    driver = _open_driver(group_id)
    try:
        # Same retraction the chat forget tool performs (``expired_at`` only —
        # a system retraction, not a world change), plus a ``group_id``
        # predicate as defense-in-depth on top of the per-group database.
        result = await driver.execute_query(
            """
            MATCH ()-[e:MENTIONS|RELATES_TO|HAS_MEMBER
                      {uuid: $uuid, group_id: $g}]->()
            SET e.expired_at = $now
            RETURN e.uuid AS uuid
            """,
            uuid=fact_uuid,
            g=group_id,
            now=_now_iso(),
        )
        records = result[0] if result else []
    except ResponseError as exc:
        if not _is_missing_graph_error(exc):
            raise
        records = []
    finally:
        await driver.close()

    if not records:
        raise HTTPException(status_code=404, detail="Memory not found")
    return ForgetFactResponse(uuid=fact_uuid, forgotten=True)


@router.delete("/facts/{fact_uuid}", operation_id="forget_my_memory_fact")
async def forget_my_memory_fact(
    fact_uuid: Annotated[str, _FACT_UUID_PATH],
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> ForgetFactResponse:
    """Forget one fact from the caller's account memory."""
    return await _forget_fact_impl(user_id, None, fact_uuid)


@router.delete(
    "/experts/{expert_id}/facts/{fact_uuid}",
    operation_id="forget_my_expert_memory_fact",
)
async def forget_my_expert_memory_fact(
    expert_id: Annotated[str, _EXPERT_ID_PATH],
    fact_uuid: Annotated[str, _FACT_UUID_PATH],
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> ForgetFactResponse:
    """Forget one fact from an expert's isolated memory."""
    return await _forget_fact_impl(user_id, expert_id, fact_uuid)


async def _erase_scope_impl(user_id: str, expert_id: str | None) -> EraseMemoryResponse:
    group_id, resolved_expert_id = await _resolve_scope(user_id, expert_id)
    driver = _open_driver(group_id)
    deleted = 0
    try:
        deleted = await _count(driver, "MATCH (n) RETURN count(n) AS c")
        if deleted:
            await driver.execute_query("MATCH (n) DETACH DELETE n")
    except ResponseError as exc:
        # A missing-graph error here means the graph vanished between the
        # count and the delete — the scope is gone either way, so keep the
        # counted value instead of misreporting 0.
        if not _is_missing_graph_error(exc):
            raise
    finally:
        await driver.close()

    logger.info(
        f"Memory erase: user {user_id[:12]} wiped scope "
        f"{resolved_expert_id or 'AutoPilot'} ({deleted} nodes)"
    )
    return EraseMemoryResponse(
        expert_id=resolved_expert_id, deleted_nodes=deleted, erased=True
    )


@router.delete("", operation_id="erase_my_memory")
async def erase_my_memory(
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> EraseMemoryResponse:
    """Permanently erase the caller's entire account memory."""
    return await _erase_scope_impl(user_id, None)


@router.delete("/experts/{expert_id}", operation_id="erase_my_expert_memory")
async def erase_my_expert_memory(
    expert_id: Annotated[str, _EXPERT_ID_PATH],
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> EraseMemoryResponse:
    """Permanently erase one expert's entire isolated memory."""
    return await _erase_scope_impl(user_id, expert_id)
