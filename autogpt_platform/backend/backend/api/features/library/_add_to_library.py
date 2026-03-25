"""Shared logic for adding store agents to a user's library.

Both `add_store_agent_to_library` and `add_store_agent_to_library_as_admin`
delegate to these helpers so the duplication-prone create/restore/dedup
logic lives in exactly one place.
"""

import logging

import prisma.errors
import prisma.models

import backend.api.features.library.model as library_model
import backend.data.graph as graph_db
from backend.data.graph import GraphModel, GraphSettings
from backend.data.includes import library_agent_include
from backend.util.exceptions import NotFoundError
from backend.util.json import SafeJson

from .db import get_library_agent_by_graph_id, update_library_agent

logger = logging.getLogger(__name__)


async def resolve_graph_for_library(
    store_listing_version_id: str,
    user_id: str,
    *,
    admin: bool,
) -> GraphModel:
    """Look up a StoreListingVersion and resolve its graph.

    When ``admin=True``, uses ``get_graph_as_admin`` to bypass the marketplace
    APPROVED-only check.  Otherwise uses the regular ``get_graph``.
    """
    slv = await prisma.models.StoreListingVersion.prisma().find_unique(
        where={"id": store_listing_version_id}, include={"AgentGraph": True}
    )
    if not slv or not slv.AgentGraph:
        raise NotFoundError(
            f"Store listing version {store_listing_version_id} not found or invalid"
        )

    ag = slv.AgentGraph
    if admin:
        graph_model = await graph_db.get_graph_as_admin(
            graph_id=ag.id, version=ag.version, user_id=user_id
        )
    else:
        graph_model = await graph_db.get_graph(
            graph_id=ag.id, version=ag.version, user_id=user_id
        )

    if not graph_model:
        raise NotFoundError(f"Graph #{ag.id} v{ag.version} not found or accessible")
    return graph_model


async def add_graph_to_library(
    store_listing_version_id: str,
    graph_model: GraphModel,
    user_id: str,
) -> library_model.LibraryAgent:
    """Check existing / restore soft-deleted / create new LibraryAgent."""
    if existing := await get_library_agent_by_graph_id(
        user_id, graph_model.id, graph_model.version
    ):
        return existing

    deleted_agent = await prisma.models.LibraryAgent.prisma().find_unique(
        where={
            "userId_agentGraphId_agentGraphVersion": {
                "userId": user_id,
                "agentGraphId": graph_model.id,
                "agentGraphVersion": graph_model.version,
            }
        },
    )
    if deleted_agent and (deleted_agent.isDeleted or deleted_agent.isArchived):
        return await update_library_agent(
            deleted_agent.id,
            user_id,
            is_deleted=False,
            is_archived=False,
        )

    try:
        added_agent = await prisma.models.LibraryAgent.prisma().create(
            data={
                "User": {"connect": {"id": user_id}},
                "AgentGraph": {
                    "connect": {
                        "graphVersionId": {
                            "id": graph_model.id,
                            "version": graph_model.version,
                        }
                    }
                },
                "isCreatedByUser": False,
                "useGraphIsActiveVersion": False,
                "settings": SafeJson(
                    GraphSettings.from_graph(graph_model).model_dump()
                ),
            },
            include=library_agent_include(
                user_id, include_nodes=False, include_executions=False
            ),
        )
    except prisma.errors.UniqueViolationError:
        # Race condition: concurrent request created the row between our
        # check and create.  Re-read instead of crashing.
        existing = await get_library_agent_by_graph_id(
            user_id, graph_model.id, graph_model.version
        )
        if existing:
            return existing
        raise  # Shouldn't happen, but don't swallow unexpected errors

    logger.debug(
        f"Added graph #{graph_model.id} v{graph_model.version} "
        f"for store listing version #{store_listing_version_id} "
        f"to library for user #{user_id}"
    )
    return library_model.LibraryAgent.from_db(added_agent)
