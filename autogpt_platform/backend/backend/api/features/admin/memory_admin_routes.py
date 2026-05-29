"""Admin-only inspector + on-demand rebuild for Graphiti memory.

Powers the admin memory-visualizer page in the frontend (Task #10).
All routes are gated by ``requires_admin_user`` so non-admin callers
get a 403 before any FalkorDB query runs.

Admins can inspect any user's memory by passing the target user id in
the path; ``user_id="me"`` is a convenience shorthand that resolves to
the caller's own id via ``get_user_id`` (which also honours the
``X-Act-As-User-Id`` impersonation header). Cross-user reads/writes are
audited in ``_audit_cross_user_access``.
"""

import logging
from typing import Annotated, Any, Literal

from autogpt_libs.auth import get_user_id, requires_admin_user
from autogpt_libs.auth.jwt_utils import get_jwt_payload
from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request, Security
from pydantic import BaseModel
from redis.exceptions import ResponseError

from backend.copilot.graphiti.client import derive_group_id
from backend.copilot.graphiti.config import graphiti_config
from backend.copilot.graphiti.falkordb_driver import AutoGPTFalkorDriver
from backend.data.redis_client import get_redis_async
from backend.util.clients import get_scheduler_client

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/admin/memory",
    tags=["admin", "memory"],
    dependencies=[Security(requires_admin_user)],
)


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class MemoryOverview(BaseModel):
    """Counts surfaced in the visualizer's header strip."""

    user_id: str
    group_id: str
    entities: int
    episodes: int
    relates_to_edges: int
    mentions_edges: int
    communities: int


class EntitySummary(BaseModel):
    uuid: str
    name: str
    summary: str | None = None


class EntityListResponse(BaseModel):
    user_id: str
    items: list[EntitySummary]


class FactSummary(BaseModel):
    uuid: str
    source: str
    target: str
    name: str | None = None
    fact: str | None = None
    status: str | None = None
    scope: str | None = None
    confidence: float | None = None
    created_at: str | None = None
    expired_at: str | None = None


class FactListResponse(BaseModel):
    user_id: str
    items: list[FactSummary]


class CommunitySummary(BaseModel):
    uuid: str
    name: str | None = None
    summary: str | None = None
    member_count: int = 0


class CommunityListResponse(BaseModel):
    user_id: str
    items: list[CommunitySummary]


class RebuildResponse(BaseModel):
    """Mirror of ``rebuild_communities_for_user``'s return dict."""

    user_id: str
    started_at: str | None = None
    communities_built: Any = None
    elapsed_seconds: float | None = None
    error: str | None = None
    skipped: bool = False
    skip_reason: str | None = None
    activity: dict[str, Any] | None = None
    forced: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_user_id(user_id: str, caller_id: str) -> str:
    """``"me"`` resolves to the calling admin's own id."""
    return caller_id if user_id == "me" else user_id


def _audit_cross_user_access(
    *,
    request: Request,
    caller_id: str,
    target_id: str,
    jwt_payload: dict,
) -> None:
    """Log admin reads/writes against another user's memory.

    ``get_user_id`` only audits impersonation via ``X-Act-As-User-Id``;
    these routes accept the target user id directly in the path, so we
    log here to keep an audit trail for PII access (entity names, fact
    text) and destructive ops (community rebuild).
    """
    if target_id == caller_id:
        return
    caller_email = jwt_payload.get("email") or jwt_payload.get("user_metadata", {}).get(
        "email", ""
    )
    logger.info(
        f"Admin memory access: {caller_id} ({caller_email}) "
        f"acting on user {target_id} for {request.method} {request.url}"
    )


# Per-user lock for admin-triggered rebuilds. The weekly cron is
# serialized by APScheduler's ``max_instances=1``; the admin endpoint
# goes through scheduler RPC and bypasses that, so guard concurrent
# admin POSTs with a Redis NX key to prevent two ``DETACH DELETE`` +
# rebuild passes racing on the same user graph.
_REBUILD_LOCK_TTL_SECONDS = 30 * 60  # rebuild can take many minutes on large graphs


def _rebuild_lock_key(user_id: str) -> str:
    return f"admin:memory:rebuild_lock:{user_id}"


def _open_driver(group_id: str) -> AutoGPTFalkorDriver:
    """Read-only driver — bypasses the full Graphiti client construction.

    The visualizer's read paths only need Cypher; we avoid the
    ~1s LLM-client + cross-encoder setup cost for what should be
    snappy dashboard calls.
    """
    return AutoGPTFalkorDriver(
        host=graphiti_config.falkordb_host,
        port=graphiti_config.falkordb_port,
        password=graphiti_config.falkordb_password or None,
        database=group_id,
    )


_MISSING_GRAPH_MARKERS = ("no such graph", "does not exist", "invalid graph")


def _is_missing_graph_error(exc: BaseException) -> bool:
    """FalkorDB returns ``ResponseError`` with a graph-not-found message
    when querying a database the user has never populated."""
    if not isinstance(exc, ResponseError):
        return False
    msg = str(exc).lower()
    return any(marker in msg for marker in _MISSING_GRAPH_MARKERS)


async def _count(driver, query: str) -> int:
    """Run a ``RETURN count(...) AS c`` query; return 0 on empty.

    Swallows only the "no graph yet" case — the user just hasn't used
    memory. Other Cypher errors (typos, schema issues) propagate so
    they're visible to admins rather than silently zeroed.
    """
    try:
        rows, _, _ = await driver.execute_query(query)
        return int(rows[0]["c"]) if rows else 0
    except ResponseError as exc:
        if _is_missing_graph_error(exc):
            return 0
        raise


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/{user_id}/overview", response_model=MemoryOverview)
async def get_memory_overview(
    request: Request,
    user_id: Annotated[str, Path(description="User id or 'me'")],
    caller_id: Annotated[str, Depends(get_user_id)],
    jwt_payload: Annotated[dict, Security(get_jwt_payload)],
) -> MemoryOverview:
    target = _resolve_user_id(user_id, caller_id)
    try:
        group_id = derive_group_id(target)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    _audit_cross_user_access(
        request=request,
        caller_id=caller_id,
        target_id=target,
        jwt_payload=jwt_payload,
    )
    driver = _open_driver(group_id)
    try:
        entities = await _count(driver, "MATCH (n:Entity) RETURN count(n) AS c")
        episodes = await _count(driver, "MATCH (n:Episodic) RETURN count(n) AS c")
        relates = await _count(
            driver, "MATCH ()-[e:RELATES_TO]->() RETURN count(e) AS c"
        )
        mentions = await _count(
            driver, "MATCH ()-[e:MENTIONS]->() RETURN count(e) AS c"
        )
        communities = await _count(driver, "MATCH (n:Community) RETURN count(n) AS c")
    finally:
        await driver.close()

    return MemoryOverview(
        user_id=target,
        group_id=group_id,
        entities=entities,
        episodes=episodes,
        relates_to_edges=relates,
        mentions_edges=mentions,
        communities=communities,
    )


@router.get("/{user_id}/entities", response_model=EntityListResponse)
async def list_entities(
    request: Request,
    user_id: Annotated[str, Path(description="User id or 'me'")],
    caller_id: Annotated[str, Depends(get_user_id)],
    jwt_payload: Annotated[dict, Security(get_jwt_payload)],
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> EntityListResponse:
    target = _resolve_user_id(user_id, caller_id)
    try:
        group_id = derive_group_id(target)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    _audit_cross_user_access(
        request=request,
        caller_id=caller_id,
        target_id=target,
        jwt_payload=jwt_payload,
    )
    driver = _open_driver(group_id)
    try:
        rows, _, _ = await driver.execute_query(
            """
            MATCH (n:Entity {group_id: $g})
            RETURN n.uuid AS uuid, n.name AS name, n.summary AS summary
            ORDER BY n.name
            LIMIT $limit
            """,
            g=group_id,
            limit=limit,
        )
    except ResponseError as exc:
        if not _is_missing_graph_error(exc):
            raise
        rows = []
    finally:
        await driver.close()

    items = [
        EntitySummary(
            uuid=str(r.get("uuid", "")),
            name=str(r.get("name") or ""),
            summary=r.get("summary"),
        )
        for r in rows
    ]
    return EntityListResponse(user_id=target, items=items)


@router.get("/{user_id}/facts", response_model=FactListResponse)
async def list_facts(
    request: Request,
    user_id: Annotated[str, Path(description="User id or 'me'")],
    caller_id: Annotated[str, Depends(get_user_id)],
    jwt_payload: Annotated[dict, Security(get_jwt_payload)],
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    status: Annotated[
        Literal["active", "superseded", "contradicted", "any"], Query()
    ] = "any",
    scope: Annotated[str | None, Query()] = None,
) -> FactListResponse:
    target = _resolve_user_id(user_id, caller_id)
    try:
        group_id = derive_group_id(target)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    _audit_cross_user_access(
        request=request,
        caller_id=caller_id,
        target_id=target,
        jwt_payload=jwt_payload,
    )
    # Build optional filters
    where_clauses = ["e.group_id = $g"]
    params: dict[str, Any] = {"g": group_id, "limit": limit}
    if status != "any":
        where_clauses.append("e.status = $status")
        params["status"] = status
    if scope:
        where_clauses.append("e.scope = $scope")
        params["scope"] = scope
    where = " AND ".join(where_clauses)

    driver = _open_driver(group_id)
    try:
        rows, _, _ = await driver.execute_query(
            f"""
            MATCH (src:Entity)-[e:RELATES_TO]->(tgt:Entity)
            WHERE {where}
            RETURN e.uuid AS uuid,
                   src.name AS source,
                   tgt.name AS target,
                   e.name AS name,
                   e.fact AS fact,
                   e.status AS status,
                   e.scope AS scope,
                   e.confidence AS confidence,
                   toString(e.created_at) AS created_at,
                   toString(e.expired_at) AS expired_at
            ORDER BY e.created_at DESC
            LIMIT $limit
            """,
            **params,
        )
    except ResponseError as exc:
        if not _is_missing_graph_error(exc):
            raise
        rows = []
    finally:
        await driver.close()

    items = [
        FactSummary(
            uuid=str(r.get("uuid", "")),
            source=str(r.get("source") or ""),
            target=str(r.get("target") or ""),
            name=r.get("name"),
            fact=r.get("fact"),
            status=r.get("status"),
            scope=r.get("scope"),
            confidence=r.get("confidence"),
            created_at=r.get("created_at"),
            expired_at=r.get("expired_at"),
        )
        for r in rows
    ]
    return FactListResponse(user_id=target, items=items)


@router.get("/{user_id}/communities", response_model=CommunityListResponse)
async def list_communities(
    request: Request,
    user_id: Annotated[str, Path(description="User id or 'me'")],
    caller_id: Annotated[str, Depends(get_user_id)],
    jwt_payload: Annotated[dict, Security(get_jwt_payload)],
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> CommunityListResponse:
    target = _resolve_user_id(user_id, caller_id)
    try:
        group_id = derive_group_id(target)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    _audit_cross_user_access(
        request=request,
        caller_id=caller_id,
        target_id=target,
        jwt_payload=jwt_payload,
    )
    driver = _open_driver(group_id)
    try:
        rows, _, _ = await driver.execute_query(
            # Graphiti stores membership as ``(Community)-[HAS_MEMBER]->(Entity)``;
            # match in that direction so member_count + sort-by-size are correct.
            """
            MATCH (c:Community {group_id: $g})
            OPTIONAL MATCH (c)-[:HAS_MEMBER]->(m:Entity)
            WITH c, count(m) AS member_count
            RETURN c.uuid AS uuid,
                   c.name AS name,
                   c.summary AS summary,
                   member_count
            ORDER BY member_count DESC, c.name
            LIMIT $limit
            """,
            g=group_id,
            limit=limit,
        )
    except ResponseError as exc:
        if not _is_missing_graph_error(exc):
            raise
        rows = []
    finally:
        await driver.close()

    items = [
        CommunitySummary(
            uuid=str(r.get("uuid", "")),
            name=r.get("name"),
            summary=r.get("summary"),
            member_count=int(r.get("member_count") or 0),
        )
        for r in rows
    ]
    return CommunityListResponse(user_id=target, items=items)


@router.post("/{user_id}/communities/rebuild", response_model=RebuildResponse)
async def rebuild_communities(
    request: Request,
    user_id: Annotated[str, Path(description="User id or 'me'")],
    caller_id: Annotated[str, Depends(get_user_id)],
    jwt_payload: Annotated[dict, Security(get_jwt_payload)],
    force: Annotated[
        bool,
        Query(
            description="Bypass the activity gate — rebuilds even on unchanged graph."
        ),
    ] = False,
) -> RebuildResponse:
    """Trigger an immediate community rebuild for the user.

    Forwards to ``Scheduler.execute_community_rebuild_pass``. The
    activity gate inside ``rebuild_communities_for_user`` is honoured
    by default — pass ``?force=true`` to bypass it.
    """
    target = _resolve_user_id(user_id, caller_id)
    try:
        derive_group_id(target)  # validate before doing the RPC
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    _audit_cross_user_access(
        request=request,
        caller_id=caller_id,
        target_id=target,
        jwt_payload=jwt_payload,
    )

    # Per-user redis lock: the weekly cron is serialized by APScheduler's
    # ``max_instances=1``; admin POSTs bypass that, so two concurrent
    # rebuilds for the same user would race their ``DETACH DELETE``.
    redis = await get_redis_async()
    lock_key = _rebuild_lock_key(target)
    claimed = await redis.set(lock_key, "1", nx=True, ex=_REBUILD_LOCK_TTL_SECONDS)
    if not claimed:
        raise HTTPException(
            status_code=409,
            detail="A community rebuild is already in progress for this user.",
        )

    try:
        result = await get_scheduler_client().execute_community_rebuild_pass(
            user_id=target, force=force
        )
    except Exception as exc:
        logger.warning(
            "Admin-triggered community rebuild failed for user %s: %s",
            target[:12],
            exc,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Community rebuild failed: {type(exc).__name__}: {exc}",
        )
    finally:
        await redis.delete(lock_key)

    # Promote rebuild-side failures (set in ``result['error']``) to a 500
    # so admin clients don't silently see "success" with a populated error.
    if result.get("error"):
        raise HTTPException(
            status_code=500,
            detail=f"Community rebuild failed: {result['error']}",
        )

    # Normalize to the response model — the scheduler returns a dict.
    return RebuildResponse(**result)
