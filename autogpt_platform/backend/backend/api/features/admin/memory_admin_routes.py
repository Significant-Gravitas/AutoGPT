"""Admin-only inspector + on-demand rebuild for Graphiti memory.

Powers the admin memory-visualizer page in the frontend (Task #10).
All routes are gated by ``requires_admin_user`` so non-admin callers
get a 403 before any FalkorDB query runs.

Initially scoped to "admin views their OWN memory" — every route
accepts ``user_id="me"`` and resolves to the caller's user id via
``get_user_id`` (which also honours the existing admin-impersonation
``X-Act-As-User-Id`` header for cross-user inspection).
"""

import logging
import uuid as _uuid
from datetime import datetime
from typing import Annotated, Any, Literal

from autogpt_libs.auth import get_user_id, requires_admin_user
from autogpt_libs.auth.jwt_utils import get_jwt_payload
from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request, Security
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from redis.exceptions import ResponseError

from backend.copilot.dream.job_status import (
    JobKind,
    JobState,
    JobStatus,
    mark_errored,
    read_status,
    write_initial_status,
)
from backend.copilot.dream.nightly_batch import NightlyBatchResult
from backend.copilot.dream.ratification import RatificationResult
from backend.copilot.dream.schemas import DreamPassResult
from backend.copilot.graphiti.client import derive_group_id
from backend.copilot.graphiti.config import graphiti_config
from backend.copilot.graphiti.falkordb_driver import AutoGPTFalkorDriver
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


class GraphNode(BaseModel):
    """A node in the visualizer graph payload.

    ``label`` is the primary FalkorDB label (Entity / Episodic /
    Community); ``type`` (if present) is the more-specific custom
    entity type from ``MemoryFact``-typed extraction (Person /
    Organization / Project / Concept / Preference / Rule). The frontend
    uses ``type`` for color-coding when present, falls back to ``label``.
    """

    uuid: str
    label: str
    type: str | None = None
    name: str | None = None
    summary: str | None = None


class GraphEdge(BaseModel):
    """An edge in the visualizer graph payload."""

    uuid: str
    label: str  # RELATES_TO | MENTIONS | HAS_MEMBER
    source: str  # source node uuid
    target: str  # target node uuid
    name: str | None = None  # extracted relation name (e.g. "works_on")
    fact: str | None = None
    status: str | None = None
    scope: str | None = None


class GraphResponse(BaseModel):
    user_id: str
    group_id: str
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    truncated: bool = Field(
        default=False,
        description=(
            "True when the result was capped by ``node_limit`` / ``edge_limit``. "
            "Frontend should surface this so the user knows they're not "
            "seeing the full graph."
        ),
    )


class RebuildResult(BaseModel):
    """Typed envelope around ``rebuild_communities_for_user``'s return dict.

    Lives here (admin routes) rather than in ``copilot/graphiti/communities.py``
    because the dict-returning rebuild function predates the admin-API
    typing pass. Hoisting it to ``communities.py`` (and changing the
    function signature) is a follow-up cleanup.
    """

    user_id: str
    started_at: str | None = None
    communities_built: Any = None
    elapsed_seconds: float | None = None
    error: str | None = None
    skipped: bool = False
    skip_reason: str | None = None
    activity: dict[str, Any] | None = None
    forced: bool = False
    execution_path: str | None = None
    """``"flex"`` when the rebuild ran on OpenAI's flex service tier,
    ``"sync"`` for the default. ``None`` when the rebuild was skipped
    before a client was constructed."""


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


async def _mark_schedule_failed(kind: JobKind, job_id: str, exc: Exception) -> None:
    """Best-effort: flip a just-queued JobStatus to errored when the
    scheduler hand-off failed, so a poller doesn't wait on a job that
    will never run."""
    try:
        await mark_errored(
            kind=kind,
            job_id=job_id,
            error=f"scheduling failed: {type(exc).__name__}: {exc}",
        )
    except Exception:
        logger.warning(
            "Failed to mark %s job %s errored after schedule failure",
            kind,
            job_id[:12],
        )


def _open_driver(group_id: str) -> AutoGPTFalkorDriver:
    """Read-only driver — bypasses the full Graphiti client construction.

    The visualizer's read paths only need Cypher; we avoid the
    ~1s LLM-client + cross-encoder setup cost for what should be
    snappy dashboard calls.

    ``build_indices=False`` suppresses graphiti-core's per-init
    fire-and-forget indexing task. For a user whose graph the admin
    is inspecting, the indices are already in place from the
    long-lived chat-write client; firing the index-creation task per
    short-lived admin request creates a race with the route's own
    queries and produces "Buffer is closed" log spam.
    """
    return AutoGPTFalkorDriver(
        host=graphiti_config.falkordb_host,
        port=graphiti_config.falkordb_port,
        password=graphiti_config.falkordb_password or None,
        database=group_id,
        build_indices=False,
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
    _audit_cross_user_access(
        request=request,
        caller_id=caller_id,
        target_id=target,
        jwt_payload=jwt_payload,
    )
    try:
        group_id = derive_group_id(target)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

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
    limit: Annotated[int, Query(ge=1, le=10000)] = 1000,
) -> EntityListResponse:
    target = _resolve_user_id(user_id, caller_id)
    _audit_cross_user_access(
        request=request,
        caller_id=caller_id,
        target_id=target,
        jwt_payload=jwt_payload,
    )
    try:
        group_id = derive_group_id(target)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    driver = _open_driver(group_id)
    try:
        result = await driver.execute_query(
            """
            MATCH (n:Entity {group_id: $g})
            RETURN n.uuid AS uuid, n.name AS name, n.summary AS summary
            ORDER BY n.name
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
    limit: Annotated[int, Query(ge=1, le=10000)] = 1000,
    status: Annotated[
        Literal["active", "superseded", "contradicted", "any"], Query()
    ] = "any",
    scope: Annotated[str | None, Query()] = None,
) -> FactListResponse:
    target = _resolve_user_id(user_id, caller_id)
    _audit_cross_user_access(
        request=request,
        caller_id=caller_id,
        target_id=target,
        jwt_payload=jwt_payload,
    )
    try:
        group_id = derive_group_id(target)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

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
        result = await driver.execute_query(
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
        rows = result[0] if result else []
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
    limit: Annotated[int, Query(ge=1, le=2000)] = 500,
) -> CommunityListResponse:
    target = _resolve_user_id(user_id, caller_id)
    _audit_cross_user_access(
        request=request,
        caller_id=caller_id,
        target_id=target,
        jwt_payload=jwt_payload,
    )
    try:
        group_id = derive_group_id(target)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    driver = _open_driver(group_id)
    try:
        # Graphiti stores membership as (Community)-[HAS_MEMBER]->(Entity); match
        # in that direction so member_count and the size sort are correct.
        result = await driver.execute_query(
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
        rows = result[0] if result else []
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


@router.get("/{user_id}/graph", response_model=GraphResponse)
async def get_graph(
    request: Request,
    user_id: Annotated[str, Path(description="User id or 'me'")],
    caller_id: Annotated[str, Depends(get_user_id)],
    jwt_payload: Annotated[dict, Security(get_jwt_payload)],
    node_limit: Annotated[int, Query(ge=1, le=20000)] = 5000,
    edge_limit: Annotated[int, Query(ge=1, le=50000)] = 10000,
    include_episodes: Annotated[bool, Query()] = False,
    include_communities: Annotated[bool, Query()] = True,
) -> GraphResponse:
    """Single-shot graph payload for the visualizer canvas.

    Returns nodes + edges in one round-trip so the frontend can hand
    the whole thing to a force-directed layout without N+1 fetches.

    Defaults to entities + communities (the "what does my memory know
    about" view). ``include_episodes=True`` adds the temporal
    :Episodic nodes — useful for debugging extraction but noisy for the
    typical inspector view.
    """
    target = _resolve_user_id(user_id, caller_id)
    _audit_cross_user_access(
        request=request,
        caller_id=caller_id,
        target_id=target,
        jwt_payload=jwt_payload,
    )
    try:
        group_id = derive_group_id(target)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # Build the labels-of-interest list for the node queries — one
    # Cypher per label so the label can travel through the result row
    # without an extra ``labels(n)`` call per row.
    labels: list[str] = ["Entity"]
    if include_episodes:
        labels.append("Episodic")
    if include_communities:
        labels.append("Community")

    driver = _open_driver(group_id)
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    truncated = False

    try:
        # Nodes — one query per label so we can carry the label through
        # without an expensive labels() function call per row.
        for label in labels:
            result = await driver.execute_query(
                f"""
                MATCH (n:{label} {{group_id: $g}})
                RETURN n.uuid AS uuid,
                       n.name AS name,
                       n.summary AS summary,
                       labels(n) AS all_labels
                LIMIT $limit
                """,
                g=group_id,
                limit=node_limit,
            )
            rows = result[0] if result else []
            for r in rows:
                # Custom entity types (Person, Organization, etc.) appear
                # alongside the base "Entity" label. Pick the first
                # non-base label as the "type".
                all_labels = r.get("all_labels") or []
                custom_type: str | None = None
                for lbl in all_labels:
                    if lbl not in {"Entity", "Episodic", "Community", "Node"}:
                        custom_type = lbl
                        break
                nodes.append(
                    GraphNode(
                        uuid=str(r.get("uuid", "")),
                        label=label,
                        type=custom_type,
                        name=r.get("name"),
                        summary=r.get("summary"),
                    )
                )
                if len(nodes) >= node_limit:
                    truncated = True
                    break
            if truncated:
                break

        # Build a set of fetched node uuids so we can filter edges
        # to nodes the frontend can actually render (avoids edges
        # dangling into pages of the graph we didn't fetch).
        node_uuids = {n.uuid for n in nodes}

        # Edges — RELATES_TO is the meaty signal; MENTIONS is provenance;
        # HAS_MEMBER ties entities to communities (only meaningful when
        # include_communities=True).
        edge_types = ["RELATES_TO"]
        if include_communities:
            edge_types.append("HAS_MEMBER")
        if include_episodes:
            edge_types.append("MENTIONS")
        edge_label_filter = "|".join(edge_types)

        result = await driver.execute_query(
            f"""
            MATCH (src)-[e:{edge_label_filter} {{group_id: $g}}]->(tgt)
            RETURN e.uuid AS uuid,
                   type(e) AS label,
                   src.uuid AS source,
                   tgt.uuid AS target,
                   e.name AS name,
                   e.fact AS fact,
                   e.status AS status,
                   e.scope AS scope
            LIMIT $limit
            """,
            g=group_id,
            limit=edge_limit,
        )
        rows = result[0] if result else []
        for r in rows:
            src = str(r.get("source", ""))
            tgt = str(r.get("target", ""))
            # Drop edges pointing outside the fetched node set — keeps
            # the rendered graph well-formed when truncation cuts in.
            if src not in node_uuids or tgt not in node_uuids:
                continue
            edges.append(
                GraphEdge(
                    uuid=str(r.get("uuid", "")),
                    label=str(r.get("label", "RELATES_TO")),
                    source=src,
                    target=tgt,
                    name=r.get("name"),
                    fact=r.get("fact"),
                    status=r.get("status"),
                    scope=r.get("scope"),
                )
            )
            if len(edges) >= edge_limit:
                truncated = True
                break
    except ResponseError as exc:
        # Missing graph (new user) — return an empty payload rather than
        # 500; the frontend renders the empty state. Any other FalkorDB
        # error (Cypher typo, schema mismatch) propagates instead of
        # being hidden behind an empty graph.
        if not _is_missing_graph_error(exc):
            raise
    finally:
        await driver.close()

    return GraphResponse(
        user_id=target,
        group_id=group_id,
        nodes=nodes,
        edges=edges,
        truncated=truncated,
    )


class JobTriggerResponse(BaseModel):
    """202 response from a fire-and-forget admin trigger.

    Frontend captures ``job_id`` and polls ``GET .../{job_id}`` for
    progress. ``state`` is always ``"queued"`` at this point — the
    scheduler picks the job up within sub-second and flips it to
    ``running``.
    """

    job_id: str
    user_id: str
    kind: JobKind
    state: JobState
    started_at: datetime


# Typed JobStatus envelopes per job kind. Each is a concrete
# parametrization of ``JobStatus[T]`` from
# ``backend/copilot/dream/job_status.py`` over the work body's result
# shape. Declaring them as named subclasses (rather than inline
# ``JobStatus[DreamPassResult]``) gives FastAPI a clean OpenAPI
# component name (``DreamJobStatus`` instead of
# ``JobStatus_DreamPassResult_``).


class DreamJobStatus(JobStatus[DreamPassResult]):
    """JobStatus envelope for ``kind="dream_pass"``."""


class NightlyJobStatus(JobStatus[NightlyBatchResult]):
    """JobStatus envelope for ``kind="nightly"``."""


class CommunityRebuildJobStatus(JobStatus[RebuildResult]):
    """JobStatus envelope for ``kind="rebuild"``."""


@router.post(
    "/{user_id}/dream",
    response_model=JobTriggerResponse,
    status_code=202,
)
async def trigger_dream_pass(
    request: Request,
    user_id: Annotated[str, Path(description="User id or 'me'")],
    caller_id: Annotated[str, Depends(get_user_id)],
    jwt_payload: Annotated[dict, Security(get_jwt_payload)],
) -> JSONResponse:
    """Fire a dream pass and return 202 + job_id immediately.

    Frontend polls ``GET /api/admin/memory/{user_id}/dream/{job_id}``
    for progress. Runs ONLY the dream pass submitter — for the full
    nightly fan-out use ``POST /{user_id}/nightly``.
    """
    target = _resolve_user_id(user_id, caller_id)
    _audit_cross_user_access(
        request=request,
        caller_id=caller_id,
        target_id=target,
        jwt_payload=jwt_payload,
    )
    try:
        derive_group_id(target)  # validate before kicking off
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    job_id = str(_uuid.uuid4())
    status = await write_initial_status(
        kind="dream_pass", job_id=job_id, user_id=target
    )

    try:
        await get_scheduler_client().schedule_immediate_dream_pass(
            user_id=target, job_id=job_id
        )
    except Exception as exc:
        logger.warning(
            "Failed to schedule dream pass %s for user %s: %s",
            job_id[:12],
            target[:12],
            exc,
        )
        await _mark_schedule_failed("dream_pass", job_id, exc)
        raise HTTPException(
            status_code=500,
            detail=f"Dream pass scheduling failed: {type(exc).__name__}: {exc}",
        )

    payload = JobTriggerResponse(
        job_id=status.job_id,
        user_id=status.user_id,
        kind=status.kind,
        state=status.state,
        started_at=status.started_at,
    )
    return JSONResponse(status_code=202, content=payload.model_dump(mode="json"))


@router.get(
    "/{user_id}/dream/{job_id}",
    response_model=DreamJobStatus,
)
async def get_dream_pass_status(
    request: Request,
    user_id: Annotated[str, Path(description="User id or 'me'")],
    job_id: Annotated[str, Path(description="Job id returned by the POST")],
    caller_id: Annotated[str, Depends(get_user_id)],
    jwt_payload: Annotated[dict, Security(get_jwt_payload)],
) -> DreamJobStatus:
    """Read the current status of a fire-and-forget dream pass job."""
    target = _resolve_user_id(user_id, caller_id)
    _audit_cross_user_access(
        request=request,
        caller_id=caller_id,
        target_id=target,
        jwt_payload=jwt_payload,
    )
    status = await read_status(kind="dream_pass", job_id=job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="job not found")
    if status.user_id != target:
        raise HTTPException(status_code=403, detail="job belongs to a different user")
    return DreamJobStatus.model_validate(status.model_dump())


@router.post("/{user_id}/ratification", response_model=RatificationResult)
async def trigger_ratification_pass(
    request: Request,
    user_id: Annotated[str, Path(description="User id or 'me'")],
    caller_id: Annotated[str, Depends(get_user_id)],
    jwt_payload: Annotated[dict, Security(get_jwt_payload)],
) -> RatificationResult:
    """Trigger an on-demand ratification sweep for the user (in isolation).

    Forwards to ``Scheduler.execute_ratification_pass_now``. Runs ONLY
    the ratification supersession sweep — does NOT run dream pass or
    community rebuild. Useful for testing ratification behavior
    without the full nightly fan-out.
    """
    target = _resolve_user_id(user_id, caller_id)
    _audit_cross_user_access(
        request=request,
        caller_id=caller_id,
        target_id=target,
        jwt_payload=jwt_payload,
    )
    try:
        derive_group_id(target)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    try:
        result = await get_scheduler_client().execute_ratification_pass_now(
            user_id=target
        )
    except Exception as exc:
        logger.warning(
            "Admin-triggered ratification pass failed for user %s: %s",
            target[:12],
            exc,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Ratification pass failed: {type(exc).__name__}: {exc}",
        )
    return RatificationResult.model_validate(result)


@router.post(
    "/{user_id}/nightly",
    response_model=JobTriggerResponse,
    status_code=202,
)
async def trigger_nightly_batch(
    request: Request,
    user_id: Annotated[str, Path(description="User id or 'me'")],
    caller_id: Annotated[str, Depends(get_user_id)],
    jwt_payload: Annotated[dict, Security(get_jwt_payload)],
) -> JSONResponse:
    """Fire the full nightly batch fan-out and return 202 + job_id immediately.

    Frontend polls ``GET /api/admin/memory/{user_id}/nightly/{job_id}``
    for progress. Same composition as the 03:00 cron — every enabled
    batch-family submitter runs in sequence sharing one ``nightly_id``
    for cost-log attribution.
    """
    target = _resolve_user_id(user_id, caller_id)
    _audit_cross_user_access(
        request=request,
        caller_id=caller_id,
        target_id=target,
        jwt_payload=jwt_payload,
    )
    try:
        derive_group_id(target)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    job_id = str(_uuid.uuid4())
    status = await write_initial_status(kind="nightly", job_id=job_id, user_id=target)

    try:
        await get_scheduler_client().schedule_immediate_nightly_batch(
            user_id=target, job_id=job_id
        )
    except Exception as exc:
        logger.warning(
            "Failed to schedule nightly batch %s for user %s: %s",
            job_id[:12],
            target[:12],
            exc,
        )
        await _mark_schedule_failed("nightly", job_id, exc)
        raise HTTPException(
            status_code=500,
            detail=f"Nightly batch scheduling failed: {type(exc).__name__}: {exc}",
        )

    payload = JobTriggerResponse(
        job_id=status.job_id,
        user_id=status.user_id,
        kind=status.kind,
        state=status.state,
        started_at=status.started_at,
    )
    return JSONResponse(status_code=202, content=payload.model_dump(mode="json"))


@router.get(
    "/{user_id}/nightly/{job_id}",
    response_model=NightlyJobStatus,
)
async def get_nightly_batch_status(
    request: Request,
    user_id: Annotated[str, Path(description="User id or 'me'")],
    job_id: Annotated[str, Path(description="Job id returned by the POST")],
    caller_id: Annotated[str, Depends(get_user_id)],
    jwt_payload: Annotated[dict, Security(get_jwt_payload)],
) -> NightlyJobStatus:
    """Read the current status of a fire-and-forget nightly batch job."""
    target = _resolve_user_id(user_id, caller_id)
    _audit_cross_user_access(
        request=request,
        caller_id=caller_id,
        target_id=target,
        jwt_payload=jwt_payload,
    )
    status = await read_status(kind="nightly", job_id=job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="job not found")
    if status.user_id != target:
        raise HTTPException(status_code=403, detail="job belongs to a different user")
    return NightlyJobStatus.model_validate(status.model_dump())


@router.post(
    "/{user_id}/communities/rebuild",
    response_model=JobTriggerResponse,
    status_code=202,
)
async def rebuild_communities(
    request: Request,
    user_id: Annotated[str, Path(description="User id or 'me'")],
    caller_id: Annotated[str, Depends(get_user_id)],
    jwt_payload: Annotated[dict, Security(get_jwt_payload)],
    force: Annotated[
        bool,
        Query(
            description=(
                "Reserved — the fire-and-forget wrapper currently always "
                "honours the activity gate. Kept on the signature so the "
                "frontend hook doesn't need a contract change when the "
                "force flag is threaded through to the wrapper."
            )
        ),
    ] = False,
) -> JSONResponse:
    """Fire a community rebuild and return 202 + job_id immediately.

    Frontend polls
    ``GET /api/admin/memory/{user_id}/communities/rebuild/{job_id}``
    for progress.
    """
    _ = force  # not yet plumbed through the with_status wrapper
    target = _resolve_user_id(user_id, caller_id)
    _audit_cross_user_access(
        request=request,
        caller_id=caller_id,
        target_id=target,
        jwt_payload=jwt_payload,
    )
    try:
        derive_group_id(target)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    job_id = str(_uuid.uuid4())
    status = await write_initial_status(kind="rebuild", job_id=job_id, user_id=target)

    try:
        await get_scheduler_client().schedule_immediate_community_rebuild(
            user_id=target, job_id=job_id
        )
    except Exception as exc:
        logger.warning(
            "Failed to schedule community rebuild %s for user %s: %s",
            job_id[:12],
            target[:12],
            exc,
        )
        await _mark_schedule_failed("rebuild", job_id, exc)
        raise HTTPException(
            status_code=500,
            detail=(
                f"Community rebuild scheduling failed: " f"{type(exc).__name__}: {exc}"
            ),
        )

    payload = JobTriggerResponse(
        job_id=status.job_id,
        user_id=status.user_id,
        kind=status.kind,
        state=status.state,
        started_at=status.started_at,
    )
    return JSONResponse(status_code=202, content=payload.model_dump(mode="json"))


@router.get(
    "/{user_id}/communities/rebuild/{job_id}",
    response_model=CommunityRebuildJobStatus,
)
async def get_community_rebuild_status(
    request: Request,
    user_id: Annotated[str, Path(description="User id or 'me'")],
    job_id: Annotated[str, Path(description="Job id returned by the POST")],
    caller_id: Annotated[str, Depends(get_user_id)],
    jwt_payload: Annotated[dict, Security(get_jwt_payload)],
) -> CommunityRebuildJobStatus:
    """Read the current status of a fire-and-forget community rebuild job."""
    target = _resolve_user_id(user_id, caller_id)
    _audit_cross_user_access(
        request=request,
        caller_id=caller_id,
        target_id=target,
        jwt_payload=jwt_payload,
    )
    status = await read_status(kind="rebuild", job_id=job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="job not found")
    if status.user_id != target:
        raise HTTPException(status_code=403, detail="job belongs to a different user")
    return CommunityRebuildJobStatus.model_validate(status.model_dump())
