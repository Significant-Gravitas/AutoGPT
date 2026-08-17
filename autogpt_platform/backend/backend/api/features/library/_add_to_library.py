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
from backend.api.features.orgs.db import resolve_default_tenancy
from backend.api.features.store.store_listing_versions import (
    installable_store_version_where,
)
from backend.data.graph import GraphModel, GraphSettings
from backend.data.includes import library_agent_include
from backend.util.exceptions import NotFoundError
from backend.util.json import SafeJson

from ._schedule_info import _fetch_schedule_info

logger = logging.getLogger(__name__)


def _require_graph(
    store_listing_version: prisma.models.StoreListingVersion,
) -> prisma.models.AgentGraph:
    graph = store_listing_version.AgentGraph
    if not graph:
        raise NotFoundError(
            f"Store listing version {store_listing_version.id} has no graph"
        )
    return graph


async def resolve_store_version_for_library(
    store_listing_version_id: str,
    *,
    admin: bool,
    tx: prisma.Prisma | None = None,
) -> prisma.models.StoreListingVersion:
    """Look up an installable StoreListingVersion and its graph reference."""
    delegate = (
        prisma.models.StoreListingVersion.prisma(tx)
        if tx is not None
        else prisma.models.StoreListingVersion.prisma()
    )
    if admin:
        slv = await delegate.find_unique(
            where={"id": store_listing_version_id}, include={"AgentGraph": True}
        )
    else:
        slv = await delegate.find_first(
            where={
                "id": store_listing_version_id,
                **installable_store_version_where(),
            },
            include={"AgentGraph": True},
        )
    if not slv:
        raise NotFoundError(
            f"Store listing version {store_listing_version_id} not found or invalid"
        )
    _require_graph(slv)
    return slv


async def resolve_graph_model_for_library(
    store_listing_version: prisma.models.StoreListingVersion,
    user_id: str,
    *,
    admin: bool,
) -> GraphModel:
    """Resolve the graph for an already-validated marketplace version."""
    ag = _require_graph(store_listing_version)

    graph_model = (
        await graph_db.get_graph_as_admin(
            graph_id=ag.id, version=ag.version, user_id=user_id
        )
        if admin
        else await graph_db.get_graph(
            graph_id=ag.id, version=ag.version, user_id=user_id
        )
    )
    if not graph_model:
        raise NotFoundError(f"Graph #{ag.id} v{ag.version} not found or accessible")
    return graph_model


async def resolve_graph_for_library(
    store_listing_version_id: str,
    user_id: str,
    *,
    admin: bool,
    tx: prisma.Prisma | None = None,
) -> tuple[GraphModel, prisma.models.StoreListingVersion]:
    """Look up a StoreListingVersion and resolve its graph.

    When ``admin=True``, uses ``get_graph_as_admin`` to bypass the marketplace
    APPROVED-only check.  Otherwise uses the regular ``get_graph``.

    Returns the resolved graph together with the StoreListingVersion, so callers
    can snapshot marketplace metadata without re-querying it.
    """
    slv = await resolve_store_version_for_library(
        store_listing_version_id, admin=admin, tx=tx
    )
    graph_model = await resolve_graph_model_for_library(slv, user_id, admin=admin)
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


async def restore_existing_library_agent(
    store_listing_version: prisma.models.StoreListingVersion,
    user_id: str,
) -> library_model.LibraryAgent | None:
    """Restore and return an existing library entry without loading its graph."""
    ag = _require_graph(store_listing_version)

    client = prisma.models.LibraryAgent.prisma()
    where = {
        "userId_agentGraphId_agentGraphVersion": {
            "userId": user_id,
            "agentGraphId": ag.id,
            "agentGraphVersion": ag.version,
        }
    }
    existing = await client.find_unique(where=where)
    if existing is None:
        return None

    restored = await client.update(
        where=where,
        data={
            "isDeleted": False,
            "isArchived": False,
            **_marketplace_metadata(store_listing_version),
        },
        include=library_agent_include(
            user_id, include_nodes=False, include_executions=False
        ),
    )
    if restored is None:
        return None

    schedule_info = await _fetch_schedule_info(user_id, graph_id=ag.id)
    return library_model.LibraryAgent.from_db(
        restored,
        schedule_info=schedule_info,
        store_listing_version_id=store_listing_version.id,
    )


async def add_graph_to_library(
    graph_model: GraphModel,
    user_id: str,
    store_listing_version: prisma.models.StoreListingVersion,
    *,
    tx: prisma.Prisma | None = None,
) -> library_model.LibraryAgent:
    """Check existing / restore soft-deleted / create new LibraryAgent.

    The standalone path preserves the established create-then-update behavior.
    When a transaction client is supplied, an upsert keeps the library write
    atomic with the caller's other writes; catching a uniqueness error inside
    PostgreSQL's transaction would leave that transaction aborted.
    """
    settings_json = SafeJson(GraphSettings.from_graph(graph_model).model_dump())
    _include = library_agent_include(
        user_id, include_nodes=False, include_executions=False
    )
    marketplace = _marketplace_metadata(store_listing_version)
    create_data, update_data = await _library_agent_payloads(
        graph_model, user_id, settings_json, marketplace
    )

    if tx is not None:
        added_agent = await _upsert_library_agent(
            tx, graph_model, user_id, create_data, update_data, _include
        )
    else:
        added_agent = await _create_or_restore_library_agent(
            graph_model, user_id, create_data, update_data, _include
        )

    logger.debug(
        f"Added graph #{graph_model.id} v{graph_model.version} "
        f"for store listing version #{store_listing_version.id} "
        f"to library for user #{user_id}"
    )
    schedule_info = await _fetch_schedule_info(user_id, graph_id=graph_model.id)
    return library_model.LibraryAgent.from_db(
        added_agent,
        schedule_info=schedule_info,
        store_listing_version_id=store_listing_version.id,
    )


async def _library_agent_payloads(
    graph_model: GraphModel,
    user_id: str,
    settings_json: SafeJson,
    marketplace: dict[str, str | None],
) -> tuple[dict, dict]:
    organization_id, team_id = await resolve_default_tenancy(user_id)
    create_data = {
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
    }
    # Deliberately leave settings and organizationId/Team untouched on re-add:
    # an existing entry already carries its safety/tenancy configuration.
    update_data = {
        "isDeleted": False,
        "isArchived": False,
        "name": marketplace["name"],
        "description": marketplace["description"],
        "imageUrl": marketplace["imageUrl"],
    }
    return create_data, update_data


def _library_agent_where(graph_model: GraphModel, user_id: str) -> dict:
    return {
        "userId_agentGraphId_agentGraphVersion": {
            "userId": user_id,
            "agentGraphId": graph_model.id,
            "agentGraphVersion": graph_model.version,
        }
    }


async def _upsert_library_agent(
    tx: prisma.Prisma,
    graph_model: GraphModel,
    user_id: str,
    create_data: dict,
    update_data: dict,
    include: dict,
) -> prisma.models.LibraryAgent:
    return await prisma.models.LibraryAgent.prisma(tx).upsert(
        where=_library_agent_where(graph_model, user_id),
        data={"create": create_data, "update": update_data},
        include=include,
    )


async def _create_or_restore_library_agent(
    graph_model: GraphModel,
    user_id: str,
    create_data: dict,
    update_data: dict,
    include: dict,
) -> prisma.models.LibraryAgent:
    try:
        return await prisma.models.LibraryAgent.prisma().create(
            data=create_data,
            include=include,
        )
    except prisma.errors.UniqueViolationError:
        added_agent = await prisma.models.LibraryAgent.prisma().update(
            where=_library_agent_where(graph_model, user_id),
            data=update_data,
            include=include,
        )
        if added_agent is None:
            raise NotFoundError(
                f"LibraryAgent for graph #{graph_model.id} "
                f"v{graph_model.version} not found after UniqueViolationError"
            )
        return added_agent
