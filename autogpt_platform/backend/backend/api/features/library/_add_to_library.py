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
from backend.api.features.library.db import _fetch_schedule_info
from backend.data.graph import GraphModel, GraphSettings
from backend.data.includes import library_agent_include
from backend.util.exceptions import NotFoundError
from backend.util.json import SafeJson

logger = logging.getLogger(__name__)


async def resolve_graph_for_library(
    store_listing_version_id: str,
    user_id: str,
    *,
    admin: bool,
) -> tuple[GraphModel, prisma.models.StoreListingVersion]:
    """Look up a StoreListingVersion and resolve its graph.

    When ``admin=True``, uses ``get_graph_as_admin`` to bypass the marketplace
    APPROVED-only check.  Otherwise uses the regular ``get_graph``.

    Returns the resolved graph together with the StoreListingVersion, so callers
    can snapshot marketplace metadata without re-querying it.
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
    return graph_model, slv


def _marketplace_metadata(
    store_listing_version: prisma.models.StoreListingVersion,
) -> dict[str, str | None]:
    """Snapshot the marketplace listing's title/description/image.

    Returns the published ``name``, ``description`` and first image URL so a
    downloaded agent shows up in the library exactly as it appears in the
    marketplace.
    """
    return {
        "name": store_listing_version.name,
        "description": store_listing_version.description,
        "imageUrl": (
            store_listing_version.imageUrls[0]
            if store_listing_version.imageUrls
            else None
        ),
    }


async def add_graph_to_library(
    graph_model: GraphModel,
    user_id: str,
    store_listing_version: prisma.models.StoreListingVersion,
) -> library_model.LibraryAgent:
    """Check existing / restore soft-deleted / create new LibraryAgent.

    Uses a create-then-catch-UniqueViolationError-then-update pattern on
    the (userId, agentGraphId, agentGraphVersion) composite unique constraint.
    This is more robust than ``upsert`` because Prisma's upsert atomicity
    guarantees are not well-documented for all versions.
    """
    settings_json = SafeJson(GraphSettings.from_graph(graph_model).model_dump())
    _include = library_agent_include(
        user_id, include_nodes=False, include_executions=False
    )
    marketplace = _marketplace_metadata(store_listing_version)

    # Born-tenanted: mirror create_library_agent — a library entry is the
    # adding user's bookmark, so tag it with their default org/team instead of
    # leaving it untenanted for the startup migration sweep to backfill.
    # get_user_default_team returns (None, None) if bootstrap failed; in that
    # case leave the row untagged (never block a library add on tenancy).
    from backend.api.features.orgs.db import get_user_default_team

    organization_id, team_id = await get_user_default_team(user_id)

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
                "settings": settings_json,
                "name": marketplace["name"],
                "description": marketplace["description"],
                "imageUrl": marketplace["imageUrl"],
                **({"organizationId": organization_id} if organization_id else {}),
                **({"Team": {"connect": {"id": team_id}}} if team_id else {}),
            },
            include=_include,
        )
    except prisma.errors.UniqueViolationError:
        # Already exists — update to restore if previously soft-deleted/archived
        # and refresh the marketplace snapshot in case the listing changed.
        added_agent = await prisma.models.LibraryAgent.prisma().update(
            where={
                "userId_agentGraphId_agentGraphVersion": {
                    "userId": user_id,
                    "agentGraphId": graph_model.id,
                    "agentGraphVersion": graph_model.version,
                }
            },
            data={
                "isDeleted": False,
                "isArchived": False,
                "settings": settings_json,
                "name": marketplace["name"],
                "description": marketplace["description"],
                "imageUrl": marketplace["imageUrl"],
                # Re-adding under a different active org re-tags the row — and
                # resets the team alongside it, since a stale team from the
                # previous org would point across tenants. Untagged callers
                # (bootstrap failure → organization_id None) leave both as-is.
                **(
                    {
                        "organizationId": organization_id,
                        "Team": (
                            {"connect": {"id": team_id}}
                            if team_id
                            else {"disconnect": True}
                        ),
                    }
                    if organization_id
                    else {}
                ),
            },
            include=_include,
        )
        if added_agent is None:
            raise NotFoundError(
                f"LibraryAgent for graph #{graph_model.id} "
                f"v{graph_model.version} not found after UniqueViolationError"
            )

    logger.debug(
        f"Added graph #{graph_model.id} v{graph_model.version} "
        f"for store listing version #{store_listing_version.id} "
        f"to library for user #{user_id}"
    )
    schedule_info = await _fetch_schedule_info(user_id, graph_id=graph_model.id)
    return library_model.LibraryAgent.from_db(added_agent, schedule_info=schedule_info)
