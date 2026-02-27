"""
V2 External API - Graphs Endpoints

Provides endpoints for managing agent graphs (CRUD operations).
"""

import logging
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Path, Query, Security
from prisma.enums import APIKeyPermission

from backend.api.external.middleware import require_permission
from backend.api.features.library import db as library_db
from backend.data import graph as graph_db
from backend.data.auth.base import APIAuthorizationInfo
from backend.integrations.webhooks.graph_lifecycle_hooks import (
    on_graph_activate,
    on_graph_deactivate,
)

from .common import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE
from .integrations.helpers import get_credential_requirements
from .models import (
    BlockInfo,
    CreatableGraph,
    CredentialRequirementsResponse,
    Graph,
    GraphDeleteResponse,
    GraphListResponse,
    GraphMeta,
    GraphSetActiveVersionRequest,
    GraphSettings,
    LibraryAgent,
)

logger = logging.getLogger(__name__)

graphs_router = APIRouter()


@graphs_router.get(
    path="",
    summary="List user's graphs",
)
async def list_graphs(
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.READ_GRAPH)
    ),
    page: int = Query(default=1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(
        default=DEFAULT_PAGE_SIZE,
        ge=1,
        le=MAX_PAGE_SIZE,
        description=f"Items per page (max {MAX_PAGE_SIZE})",
    ),
) -> GraphListResponse:
    """
    List all graphs owned by the authenticated user.

    Returns a paginated list of graph metadata (not full graph details).
    """
    graphs, pagination_info = await graph_db.list_graphs_paginated(
        user_id=auth.user_id,
        page=page,
        page_size=page_size,
        filter_by="active",
    )
    return GraphListResponse(
        graphs=[GraphMeta.from_internal(g) for g in graphs],
        page=pagination_info.current_page,
        page_size=pagination_info.page_size,
        total_count=pagination_info.total_items,
        total_pages=pagination_info.total_pages,
    )


@graphs_router.get(
    path="/{graph_id}",
    summary="Get graph details",
)
async def get_graph(
    graph_id: str,
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.READ_GRAPH)
    ),
    version: int | None = Query(
        default=None,
        description="Specific version to retrieve (default: active version)",
    ),
) -> Graph:
    """
    Get detailed information about a specific graph.

    By default returns the active version. Use the `version` query parameter
    to retrieve a specific version.
    """
    graph = await graph_db.get_graph(
        graph_id,
        version,
        user_id=auth.user_id,
        include_subgraphs=True,
    )
    if not graph:
        raise HTTPException(status_code=404, detail=f"Graph #{graph_id} not found.")
    return Graph.from_internal(graph)


@graphs_router.post(
    path="",
    summary="Create a new graph",
)
async def create_graph(
    create_graph: CreatableGraph,
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.WRITE_GRAPH)
    ),
) -> Graph:
    """
    Create a new agent graph.

    The graph will be validated and assigned a new ID. It will automatically
    be added to the user's library.
    """
    from backend.api.features.library import db as library_db

    internal_graph = create_graph.to_internal(id=str(uuid4()), version=1)

    graph = graph_db.make_graph_model(internal_graph, auth.user_id)
    graph.reassign_ids(user_id=auth.user_id, reassign_graph_id=True)
    graph.validate_graph(for_run=False)

    await graph_db.create_graph(graph, user_id=auth.user_id)
    await library_db.create_library_agent(graph, user_id=auth.user_id)
    activated_graph = await on_graph_activate(graph, user_id=auth.user_id)

    return Graph.from_internal(activated_graph)


@graphs_router.put(
    path="/{graph_id}",
    summary="Update graph (creates new version)",
)
async def update_graph(
    graph_id: str,
    update_graph: CreatableGraph,
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.WRITE_GRAPH)
    ),
) -> Graph:
    """
    Update a graph by creating a new version.

    This does not modify existing versions - it creates a new version with
    the provided content. The new version becomes the active version.
    """
    from backend.api.features.library import db as library_db

    existing_versions = await graph_db.get_graph_all_versions(
        graph_id, user_id=auth.user_id
    )
    if not existing_versions:
        raise HTTPException(404, detail=f"Graph #{graph_id} not found")

    latest_version_number = max(g.version for g in existing_versions)

    internal_graph = update_graph.to_internal(
        id=graph_id, version=latest_version_number + 1
    )

    current_active_version = next((v for v in existing_versions if v.is_active), None)
    graph = graph_db.make_graph_model(internal_graph, auth.user_id)
    graph.reassign_ids(user_id=auth.user_id, reassign_graph_id=False)
    graph.validate_graph(for_run=False)

    new_graph_version = await graph_db.create_graph(graph, user_id=auth.user_id)

    if new_graph_version.is_active:
        await library_db.update_agent_version_in_library(
            auth.user_id, new_graph_version.id, new_graph_version.version
        )
        new_graph_version = await on_graph_activate(
            new_graph_version, user_id=auth.user_id
        )
        await graph_db.set_graph_active_version(
            graph_id=graph_id, version=new_graph_version.version, user_id=auth.user_id
        )
        if current_active_version:
            await on_graph_deactivate(current_active_version, user_id=auth.user_id)

    new_graph_version_with_subgraphs = await graph_db.get_graph(
        graph_id,
        new_graph_version.version,
        user_id=auth.user_id,
        include_subgraphs=True,
    )
    assert new_graph_version_with_subgraphs
    return Graph.from_internal(new_graph_version_with_subgraphs)


@graphs_router.delete(
    path="/{graph_id}",
    summary="Delete graph permanently",
)
async def delete_graph(
    graph_id: str,
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.WRITE_GRAPH)
    ),
) -> GraphDeleteResponse:
    """
    Permanently delete a graph and all its versions.

    This action cannot be undone. All associated executions will remain
    but will reference a deleted graph.
    """
    if active_version := await graph_db.get_graph(
        graph_id=graph_id, version=None, user_id=auth.user_id
    ):
        await on_graph_deactivate(active_version, user_id=auth.user_id)

    version_count = await graph_db.delete_graph(graph_id, user_id=auth.user_id)
    return GraphDeleteResponse(version_count=version_count)


@graphs_router.get(
    path="/{graph_id}/versions",
    summary="List all graph versions",
)
async def list_graph_versions(
    graph_id: str,
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.READ_GRAPH)
    ),
) -> list[Graph]:
    """
    Get all versions of a specific graph.

    Returns a list of all versions, with the active version marked.
    """
    graphs = await graph_db.get_graph_all_versions(graph_id, user_id=auth.user_id)
    if not graphs:
        raise HTTPException(status_code=404, detail=f"Graph #{graph_id} not found.")
    return [Graph.from_internal(g) for g in graphs]


@graphs_router.put(
    path="/{graph_id}/versions/active",
    summary="Set active graph version",
)
async def set_active_version(
    graph_id: str,
    request_body: GraphSetActiveVersionRequest,
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.WRITE_GRAPH)
    ),
) -> None:
    """
    Set which version of a graph is the active version.

    The active version is used when executing the graph without specifying
    a version number.
    """
    from backend.api.features.library import db as library_db

    new_active_version = request_body.active_graph_version
    new_active_graph = await graph_db.get_graph(
        graph_id, new_active_version, user_id=auth.user_id
    )
    if not new_active_graph:
        raise HTTPException(404, f"Graph #{graph_id} v{new_active_version} not found")

    current_active_graph = await graph_db.get_graph(
        graph_id=graph_id,
        version=None,
        user_id=auth.user_id,
    )

    await on_graph_activate(new_active_graph, user_id=auth.user_id)
    await graph_db.set_graph_active_version(
        graph_id=graph_id,
        version=new_active_version,
        user_id=auth.user_id,
    )

    await library_db.update_agent_version_in_library(
        auth.user_id, new_active_graph.id, new_active_graph.version
    )

    if current_active_graph and current_active_graph.version != new_active_version:
        await on_graph_deactivate(current_active_graph, user_id=auth.user_id)


@graphs_router.patch(
    path="/{graph_id}/settings",
    summary="Update graph settings",
)
async def update_graph_settings(
    graph_id: str,
    settings: GraphSettings,
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.WRITE_GRAPH)
    ),
) -> GraphSettings:
    """
    Update settings for a graph.

    Currently supports:
    - human_in_the_loop_safe_mode: Enable/disable safe mode for human-in-the-loop blocks
    """
    from backend.api.features.library import db as library_db

    library_agent = await library_db.get_library_agent_by_graph_id(
        graph_id=graph_id, user_id=auth.user_id
    )
    if not library_agent:
        raise HTTPException(404, f"Graph #{graph_id} not found in user's library")

    updated_agent = await library_db.update_library_agent(
        user_id=auth.user_id,
        library_agent_id=library_agent.id,
        settings=settings.to_internal(),
    )

    return GraphSettings(
        human_in_the_loop_safe_mode=updated_agent.settings.human_in_the_loop_safe_mode
    )


@graphs_router.get(
    path="/{graph_id}/library-agent",
    summary="Get library agent by graph ID",
)
async def get_library_agent_by_graph(
    graph_id: str,
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.READ_LIBRARY)
    ),
) -> LibraryAgent:
    """
    Get the library agent associated with a specific graph.

    Returns the user's library entry for this graph, if it exists.
    """
    agent = await library_db.get_library_agent_by_graph_id(
        graph_id=graph_id,
        user_id=auth.user_id,
    )
    if not agent:
        raise HTTPException(
            status_code=404,
            detail=f"No library agent found for graph #{graph_id}",
        )
    return LibraryAgent.from_internal(agent)


@graphs_router.get(
    path="/{graph_id}/blocks",
    summary="List blocks used by a graph",
)
async def list_graph_blocks(
    graph_id: str = Path(description="Graph ID"),
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.READ_GRAPH)
    ),
) -> list[BlockInfo]:
    """
    List the unique blocks used by a graph.

    Returns metadata for each distinct block type referenced in the graph's nodes.
    """
    from backend.blocks import get_block

    graph = await graph_db.get_graph(
        graph_id,
        version=None,
        user_id=auth.user_id,
        include_subgraphs=True,
    )
    if not graph:
        raise HTTPException(status_code=404, detail=f"Graph #{graph_id} not found.")

    seen_block_ids: set[str] = set()
    blocks: list[BlockInfo] = []

    for node in graph.nodes:
        if node.block_id in seen_block_ids:
            continue
        seen_block_ids.add(node.block_id)

        block = get_block(node.block_id)
        if block and not block.disabled:
            blocks.append(BlockInfo.from_internal(block))

    return blocks


@graphs_router.get(
    path="/{graph_id}/credentials",
    summary="List credentials matching graph requirements",
)
async def list_graph_credential_requirements(
    graph_id: str = Path(description="Graph ID"),
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.READ_INTEGRATIONS)
    ),
) -> CredentialRequirementsResponse:
    """
    List credential requirements for a graph and matching user credentials.

    This helps identify which credentials the user needs to provide
    when executing a graph.
    """
    graph = await graph_db.get_graph(
        graph_id=graph_id,
        version=None,
        user_id=auth.user_id,
        include_subgraphs=True,
    )
    if not graph:
        raise HTTPException(status_code=404, detail=f"Graph #{graph_id} not found")

    requirements = await get_credential_requirements(
        graph.credentials_input_schema, auth.user_id
    )
    return CredentialRequirementsResponse(requirements=requirements)
