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
from typing import Annotated, Any, Literal

from autogpt_libs.auth import get_user_id, requires_admin_user
from fastapi import APIRouter, Depends, HTTPException, Path, Query, Security
from pydantic import BaseModel, Field

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


async def _count(driver, query: str) -> int:
    """Run a ``RETURN count(...) AS c`` query; return 0 on empty."""
    try:
        rows, _, _ = await driver.execute_query(query)
        return int(rows[0]["c"]) if rows else 0
    except Exception:
        # FalkorDB returns an error for queries against a database that
        # doesn't exist yet. Treat as zero — the user just hasn't used
        # memory.
        return 0


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/{user_id}/overview", response_model=MemoryOverview)
async def get_memory_overview(
    user_id: Annotated[str, Path(description="User id or 'me'")],
    caller_id: Annotated[str, Depends(get_user_id)],
) -> MemoryOverview:
    target = _resolve_user_id(user_id, caller_id)
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
    user_id: Annotated[str, Path(description="User id or 'me'")],
    caller_id: Annotated[str, Depends(get_user_id)],
    limit: Annotated[int, Query(ge=1, le=10000)] = 1000,
) -> EntityListResponse:
    target = _resolve_user_id(user_id, caller_id)
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
    except Exception:
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
    user_id: Annotated[str, Path(description="User id or 'me'")],
    caller_id: Annotated[str, Depends(get_user_id)],
    limit: Annotated[int, Query(ge=1, le=10000)] = 1000,
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
    except Exception:
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
    user_id: Annotated[str, Path(description="User id or 'me'")],
    caller_id: Annotated[str, Depends(get_user_id)],
    limit: Annotated[int, Query(ge=1, le=2000)] = 500,
) -> CommunityListResponse:
    target = _resolve_user_id(user_id, caller_id)
    try:
        group_id = derive_group_id(target)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    driver = _open_driver(group_id)
    try:
        result = await driver.execute_query(
            """
            MATCH (c:Community {group_id: $g})
            OPTIONAL MATCH (c)<-[:HAS_MEMBER]-(m:Entity)
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
    except Exception:
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
    user_id: Annotated[str, Path(description="User id or 'me'")],
    caller_id: Annotated[str, Depends(get_user_id)],
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
    except Exception:
        # Missing graph (new user) — return an empty payload rather
        # than 500. Frontend renders the empty state.
        pass
    finally:
        await driver.close()

    return GraphResponse(
        user_id=target,
        group_id=group_id,
        nodes=nodes,
        edges=edges,
        truncated=truncated,
    )


class DreamWriteSummaryResponse(BaseModel):
    edge_uuid: str | None = None
    content: str
    scope: str = "real:global"
    confidence: float | None = None
    status: str = "active"
    source_episode_uuids: list[str] = Field(default_factory=list)


class DreamDemotionSummaryResponse(BaseModel):
    edge_uuid: str
    reason: str
    new_status: str
    applied: bool = True


class DreamEntityInvalidationSummaryResponse(BaseModel):
    entity_uuid: str
    reason: str
    edges_touched: list[str] = Field(default_factory=list)


class DreamOperationsSnapshotResponse(BaseModel):
    writes: list[DreamWriteSummaryResponse] = Field(default_factory=list)
    proposals: list[DreamWriteSummaryResponse] = Field(default_factory=list)
    demotions: list[DreamDemotionSummaryResponse] = Field(default_factory=list)
    entity_invalidations: list[DreamEntityInvalidationSummaryResponse] = Field(
        default_factory=list
    )


class DreamPhaseUsageResponse(BaseModel):
    phase: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    cost_usd: float | None = None


class DreamPassUsageResponse(BaseModel):
    phases: list[DreamPhaseUsageResponse] = Field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_cache_creation_tokens: int = 0
    total_cost_usd: float | None = None
    discount_applied: float = 0.0


class DreamPassResponse(BaseModel):
    """Mirror of ``DreamPassResult`` from the dream orchestrator.

    Kept duplicated rather than importing the source model so the
    admin route stays loosely-coupled to the dream module's internals.
    """

    user_id: str
    pass_id: str
    started_at: str | None = None
    completed_at: str | None = None
    elapsed_seconds: float | None = None
    execution_path: str = "sync_baseline"
    consolidated_count: int = 0
    proposal_count: int = 0
    demotion_count: int = 0
    entity_invalidation_count: int = 0
    summary_for_user: str = ""
    dream_session_id: str | None = None
    # Detailed per-operation rollup — see DreamOperationsSnapshot in
    # backend/copilot/dream/schemas.py. Consumed by the admin
    # visualizer UI and the AgentProbe eval scorers (read via
    # ``rawExchangeKey: "response.body.operations"``).
    operations: DreamOperationsSnapshotResponse | None = None
    # Token + cost telemetry across all phases that ran. ``None`` for
    # skipped passes (lock_held / no_input / insufficient_credits).
    # Populated even on partial failures — we still paid for the
    # phases that ran before the failure, so billing has to charge
    # for them.
    usage: DreamPassUsageResponse | None = None
    error: str | None = None
    skipped: bool = False
    skip_reason: str | None = None


@router.post("/{user_id}/dream", response_model=DreamPassResponse)
async def trigger_dream_pass(
    user_id: Annotated[str, Path(description="User id or 'me'")],
    caller_id: Annotated[str, Depends(get_user_id)],
) -> DreamPassResponse:
    """Trigger an on-demand dream pass for the user.

    Forwards to ``Scheduler.execute_dream_pass_now``. Runs the full
    three-phase pipeline (consolidate → recombine → sanitize) synchronously
    against the OpenRouter-fronted baseline path, applies operations
    to Graphiti + Postgres, and writes a dream-kind ChatSession.
    """
    target = _resolve_user_id(user_id, caller_id)
    try:
        derive_group_id(target)  # validate before the RPC
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    try:
        result = await get_scheduler_client().execute_dream_pass_now(user_id=target)
    except Exception as exc:
        logger.warning(
            "Admin-triggered dream pass failed for user %s: %s",
            target[:12],
            exc,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Dream pass failed: {type(exc).__name__}: {exc}",
        )

    return DreamPassResponse(**result)


@router.post("/{user_id}/communities/rebuild", response_model=RebuildResponse)
async def rebuild_communities(
    user_id: Annotated[str, Path(description="User id or 'me'")],
    caller_id: Annotated[str, Depends(get_user_id)],
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

    # Normalize to the response model — the scheduler returns a dict.
    return RebuildResponse(**result)
