"""Shared logic for adding store agents to a user's library.

Both `add_store_agent_to_library` and `add_store_agent_to_library_as_admin`
delegate to these helpers so the duplication-prone create/restore/dedup
logic lives in exactly one place.
"""

import logging

import prisma.enums
import prisma.errors
import prisma.models

import backend.api.features.library.model as library_model
from backend.data.graph import GraphSettings
from backend.data.includes import library_agent_include
from backend.util.exceptions import NotFoundError
from backend.util.json import SafeJson

logger = logging.getLogger(__name__)


async def resolve_graph_for_library(
    store_listing_version_id: str,
    user_id: str,
    *,
    admin: bool,
) -> tuple[str, int]:
    """Look up a StoreListingVersion and resolve its graph.

    When ``admin=True``, uses ``get_graph_as_admin`` to bypass the marketplace
    APPROVED-only check.  Otherwise uses the regular ``get_graph``.
    """
    listing_version = await prisma.models.StoreListingVersion.prisma().find_unique(
        where={"id": store_listing_version_id}
    )
    if (
        not listing_version
        or (
            not admin
            and listing_version.submissionStatus
            != prisma.enums.SubmissionStatus.APPROVED
        )
        or listing_version.isDeleted
    ):
        logger.warning(
            "Store listing version not found or not available: "
            f"{store_listing_version_id}"
        )
        raise NotFoundError(
            f"Store listing version {store_listing_version_id} not found "
            "or not available"
        )

    graph_id = listing_version.agentGraphId
    graph_version = listing_version.agentGraphVersion

    return graph_id, graph_version


async def add_graph_to_library(
    graph_id: str,
    graph_version: int,
    user_id: str,
) -> library_model.LibraryAgent:
    """Check existing / restore soft-deleted / create new LibraryAgent.

    Uses a create-then-catch-UniqueViolationError-then-update pattern on
    the (userId, agentGraphId, agentGraphVersion) composite unique constraint.
    This is more robust than ``upsert`` because Prisma's upsert atomicity
    guarantees are not well-documented for all versions.
    """
    settings_json = SafeJson(GraphSettings().model_dump())
    _include = library_agent_include(
        user_id, include_nodes=False, include_executions=False
    )

    try:
        added_agent = await prisma.models.LibraryAgent.prisma().create(
            data={
                "User": {"connect": {"id": user_id}},
                "AgentGraph": {
                    "connect": {
                        "graphVersionId": {
                            "id": graph_id,
                            "version": graph_version,
                        }
                    }
                },
                "isCreatedByUser": False,
                "useGraphIsActiveVersion": False,
                "settings": settings_json,
            },
            include=_include,
        )
    except prisma.errors.UniqueViolationError:
        # Already exists — update to restore if previously soft-deleted/archived
        added_agent = await prisma.models.LibraryAgent.prisma().update(
            where={
                "userId_agentGraphId_agentGraphVersion": {
                    "userId": user_id,
                    "agentGraphId": graph_id,
                    "agentGraphVersion": graph_version,
                }
            },
            data={
                "isDeleted": False,
                "isArchived": False,
                "settings": settings_json,
            },
            include=_include,
        )
        if added_agent is None:
            raise NotFoundError(
                f"LibraryAgent for graph #{graph_id} "
                f"v{graph_version} not found after UniqueViolationError"
            )

    return library_model.LibraryAgent.from_db(added_agent)
