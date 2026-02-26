import asyncio
import itertools
import logging
from typing import Literal, Optional

import fastapi
import prisma.errors
import prisma.models
import prisma.types

import backend.api.features.store.exceptions as store_exceptions
import backend.api.features.store.image_gen as store_image_gen
import backend.api.features.store.media as store_media
import backend.data.graph as graph_db
import backend.data.integrations as integrations_db
from backend.api.features.library.exceptions import (
    FolderAlreadyExistsError,
    FolderValidationError,
)
from backend.data.db import transaction
from backend.data.execution import get_graph_execution
from backend.data.graph import GraphSettings
from backend.data.includes import (
    AGENT_PRESET_INCLUDE,
    LIBRARY_FOLDER_INCLUDE,
    library_agent_include,
)
from backend.data.model import CredentialsMetaInput, GraphInput
from backend.integrations.creds_manager import IntegrationCredentialsManager
from backend.integrations.webhooks.graph_lifecycle_hooks import (
    on_graph_activate,
    on_graph_deactivate,
)
from backend.util.clients import get_scheduler_client
from backend.util.exceptions import InvalidInputError, NotFoundError
from backend.util.json import SafeJson
from backend.util.models import Pagination
from backend.util.settings import Config

from . import model as library_model

logger = logging.getLogger(__name__)
config = Config()
integration_creds_manager = IntegrationCredentialsManager()


async def list_library_agents(
    user_id: str,
    search_term: Optional[str] = None,
    sort_by: library_model.LibraryAgentSort = library_model.LibraryAgentSort.UPDATED_AT,
    page: int = 1,
    page_size: int = 50,
    include_executions: bool = False,
    folder_id: Optional[str] = None,
    include_root_only: bool = False,
) -> library_model.LibraryAgentResponse:
    """
    Retrieves a paginated list of LibraryAgent records for a given user.

    Args:
        user_id: The ID of the user whose LibraryAgents we want to retrieve.
        search_term: Optional string to filter agents by name/description.
        sort_by: Sorting field (createdAt, updatedAt, isFavorite, isCreatedByUser).
        page: Current page (1-indexed).
        page_size: Number of items per page.
        folder_id: Filter by folder ID. If provided, only returns agents in this folder.
        include_root_only: If True, only returns agents without a folder (root-level).
        include_executions: Whether to include execution data for status calculation.
            Defaults to False for performance (UI fetches status separately).
            Set to True when accurate status/metrics are needed (e.g., agent generator).

    Returns:
        A LibraryAgentResponse containing the list of agents and pagination details.

    Raises:
        DatabaseError: If there is an issue fetching from Prisma.
    """
    logger.debug(
        f"Fetching library agents for user_id={user_id}, "
        f"search_term={repr(search_term)}, "
        f"sort_by={sort_by}, page={page}, page_size={page_size}"
    )

    if page < 1 or page_size < 1:
        logger.warning(f"Invalid pagination: page={page}, page_size={page_size}")
        raise InvalidInputError("Invalid pagination input")

    if search_term and len(search_term.strip()) > 100:
        logger.warning(f"Search term too long: {repr(search_term)}")
        raise InvalidInputError("Search term is too long")

    where_clause: prisma.types.LibraryAgentWhereInput = {
        "userId": user_id,
        "isDeleted": False,
        "isArchived": False,
    }

    # Apply folder filter (skip when searching — search spans all folders)
    if folder_id is not None and not search_term:
        where_clause["folderId"] = folder_id
    elif include_root_only and not search_term:
        where_clause["folderId"] = None

    # Build search filter if applicable
    if search_term:
        where_clause["OR"] = [
            {
                "AgentGraph": {
                    "is": {"name": {"contains": search_term, "mode": "insensitive"}}
                }
            },
            {
                "AgentGraph": {
                    "is": {
                        "description": {"contains": search_term, "mode": "insensitive"}
                    }
                }
            },
        ]

    order_by: prisma.types.LibraryAgentOrderByInput | None = None

    if sort_by == library_model.LibraryAgentSort.CREATED_AT:
        order_by = {"createdAt": "asc"}
    elif sort_by == library_model.LibraryAgentSort.UPDATED_AT:
        order_by = {"updatedAt": "desc"}

    library_agents = await prisma.models.LibraryAgent.prisma().find_many(
        where=where_clause,
        include=library_agent_include(
            user_id, include_nodes=False, include_executions=include_executions
        ),
        order=order_by,
        skip=(page - 1) * page_size,
        take=page_size,
    )
    agent_count = await prisma.models.LibraryAgent.prisma().count(where=where_clause)

    logger.debug(f"Retrieved {len(library_agents)} library agents for user #{user_id}")

    # Only pass valid agents to the response
    valid_library_agents: list[library_model.LibraryAgent] = []

    for agent in library_agents:
        try:
            library_agent = library_model.LibraryAgent.from_db(agent)
            valid_library_agents.append(library_agent)
        except Exception as e:
            # Skip this agent if there was an error
            logger.error(f"Error parsing LibraryAgent #{agent.id} from DB item: {e}")
            continue

    # Return the response with only valid agents
    return library_model.LibraryAgentResponse(
        agents=valid_library_agents,
        pagination=Pagination(
            total_items=agent_count,
            total_pages=(agent_count + page_size - 1) // page_size,
            current_page=page,
            page_size=page_size,
        ),
    )


async def list_favorite_library_agents(
    user_id: str,
    page: int = 1,
    page_size: int = 50,
) -> library_model.LibraryAgentResponse:
    """
    Retrieves a paginated list of favorite LibraryAgent records for a given user.

    Args:
        user_id: The ID of the user whose favorite LibraryAgents we want to retrieve.
        page: Current page (1-indexed).
        page_size: Number of items per page.

    Returns:
        A LibraryAgentResponse containing the list of favorite agents and pagination details.

    Raises:
        DatabaseError: If there is an issue fetching from Prisma.
    """
    logger.debug(
        f"Fetching favorite library agents for user_id={user_id}, "
        f"page={page}, page_size={page_size}"
    )

    if page < 1 or page_size < 1:
        logger.warning(f"Invalid pagination: page={page}, page_size={page_size}")
        raise InvalidInputError("Invalid pagination input")

    where_clause: prisma.types.LibraryAgentWhereInput = {
        "userId": user_id,
        "isDeleted": False,
        "isArchived": False,
        "isFavorite": True,  # Only fetch favorites
    }

    # Sort favorites by updated date descending
    order_by: prisma.types.LibraryAgentOrderByInput = {"updatedAt": "desc"}

    library_agents = await prisma.models.LibraryAgent.prisma().find_many(
        where=where_clause,
        include=library_agent_include(
            user_id, include_nodes=False, include_executions=False
        ),
        order=order_by,
        skip=(page - 1) * page_size,
        take=page_size,
    )
    agent_count = await prisma.models.LibraryAgent.prisma().count(where=where_clause)

    logger.debug(
        f"Retrieved {len(library_agents)} favorite library agents for user #{user_id}"
    )

    # Only pass valid agents to the response
    valid_library_agents: list[library_model.LibraryAgent] = []

    for agent in library_agents:
        try:
            library_agent = library_model.LibraryAgent.from_db(agent)
            valid_library_agents.append(library_agent)
        except Exception as e:
            # Skip this agent if there was an error
            logger.error(f"Error parsing LibraryAgent #{agent.id} from DB item: {e}")
            continue

    # Return the response with only valid agents
    return library_model.LibraryAgentResponse(
        agents=valid_library_agents,
        pagination=Pagination(
            total_items=agent_count,
            total_pages=(agent_count + page_size - 1) // page_size,
            current_page=page,
            page_size=page_size,
        ),
    )


async def get_library_agent(id: str, user_id: str) -> library_model.LibraryAgent:
    """
    Get a specific agent from the user's library.

    Args:
        id: ID of the library agent to retrieve.
        user_id: ID of the authenticated user.

    Returns:
        The requested LibraryAgent.

    Raises:
        AgentNotFoundError: If the specified agent does not exist.
        DatabaseError: If there's an error during retrieval.
    """
    library_agent = await prisma.models.LibraryAgent.prisma().find_first(
        where={
            "id": id,
            "userId": user_id,
            "isDeleted": False,
        },
        include=library_agent_include(user_id),
    )

    if not library_agent:
        raise NotFoundError(f"Library agent #{id} not found")

    # Fetch marketplace listing if the agent has been published
    store_listing = None
    profile = None
    if library_agent.AgentGraph:
        store_listing = await prisma.models.StoreListing.prisma().find_first(
            where={
                "agentGraphId": library_agent.AgentGraph.id,
                "isDeleted": False,
                "hasApprovedVersion": True,
            },
            include={
                "ActiveVersion": True,
            },
        )
        if store_listing and store_listing.ActiveVersion and store_listing.owningUserId:
            # Fetch Profile separately since User doesn't have a direct Profile relation
            profile = await prisma.models.Profile.prisma().find_first(
                where={"userId": store_listing.owningUserId}
            )

    return library_model.LibraryAgent.from_db(
        library_agent,
        sub_graphs=(
            await graph_db.get_sub_graphs(library_agent.AgentGraph)
            if library_agent.AgentGraph
            else None
        ),
        store_listing=store_listing,
        profile=profile,
    )


async def get_library_agent_by_store_version_id(
    store_listing_version_id: str,
    user_id: str,
) -> library_model.LibraryAgent | None:
    """
    Get the library agent metadata for a given store listing version ID and user ID.
    """
    logger.debug(
        f"Getting library agent for store listing ID: {store_listing_version_id}"
    )

    store_listing_version = (
        await prisma.models.StoreListingVersion.prisma().find_unique(
            where={"id": store_listing_version_id},
        )
    )
    if not store_listing_version:
        logger.warning(f"Store listing version not found: {store_listing_version_id}")
        raise NotFoundError(
            f"Store listing version {store_listing_version_id} not found or invalid"
        )

    # Check if user already has this agent
    agent = await prisma.models.LibraryAgent.prisma().find_first(
        where={
            "userId": user_id,
            "agentGraphId": store_listing_version.agentGraphId,
            "agentGraphVersion": store_listing_version.agentGraphVersion,
            "isDeleted": False,
        },
        include=library_agent_include(user_id),
    )
    return library_model.LibraryAgent.from_db(agent) if agent else None


async def get_library_agent_by_graph_id(
    user_id: str,
    graph_id: str,
    graph_version: Optional[int] = None,
) -> library_model.LibraryAgent | None:
    filter: prisma.types.LibraryAgentWhereInput = {
        "agentGraphId": graph_id,
        "userId": user_id,
        "isDeleted": False,
    }
    if graph_version is not None:
        filter["agentGraphVersion"] = graph_version

    agent = await prisma.models.LibraryAgent.prisma().find_first(
        where=filter,
        include=library_agent_include(user_id),
    )
    if not agent:
        return None

    assert agent.AgentGraph  # make type checker happy
    # Include sub-graphs so we can make a full credentials input schema
    sub_graphs = await graph_db.get_sub_graphs(agent.AgentGraph)
    return library_model.LibraryAgent.from_db(agent, sub_graphs=sub_graphs)


async def add_generated_agent_image(
    graph: graph_db.GraphBaseMeta,
    user_id: str,
    library_agent_id: str,
) -> Optional[prisma.models.LibraryAgent]:
    """
    Generates an image for the specified LibraryAgent and updates its record.
    """
    graph_id = graph.id

    # Use .jpeg here since we are generating JPEG images
    filename = f"agent_{graph_id}.jpeg"
    try:
        if not (image_url := await store_media.check_media_exists(user_id, filename)):
            # Generate agent image as JPEG
            image = await store_image_gen.generate_agent_image(graph)

            # Create UploadFile with the correct filename and content_type
            image_file = fastapi.UploadFile(file=image, filename=filename)

            image_url = await store_media.upload_media(
                user_id=user_id, file=image_file, use_file_name=True
            )
    except Exception as e:
        logger.warning(f"Error generating and uploading agent image: {e}")
        return None

    return await prisma.models.LibraryAgent.prisma().update(
        where={"id": library_agent_id},
        data={"imageUrl": image_url},
    )


async def create_library_agent(
    graph: graph_db.GraphModel,
    user_id: str,
    hitl_safe_mode: bool = True,
    sensitive_action_safe_mode: bool = False,
    create_library_agents_for_sub_graphs: bool = True,
) -> list[library_model.LibraryAgent]:
    """
    Adds an agent to the user's library (LibraryAgent table).

    Args:
        agent: The agent/Graph to add to the library.
        user_id: The user to whom the agent will be added.
        hitl_safe_mode: Whether HITL blocks require manual review (default True).
        sensitive_action_safe_mode: Whether sensitive action blocks require review.
        create_library_agents_for_sub_graphs: If True, creates LibraryAgent records for sub-graphs as well.

    Returns:
        The newly created LibraryAgent records.
        If the graph has sub-graphs, the parent graph will always be the first entry in the list.

    Raises:
        AgentNotFoundError: If the specified agent does not exist.
        DatabaseError: If there's an error during creation or if image generation fails.
    """
    logger.info(
        f"Creating library agent for graph #{graph.id} v{graph.version}; user:<redacted>"
    )
    graph_entries = (
        [graph, *graph.sub_graphs] if create_library_agents_for_sub_graphs else [graph]
    )

    async with transaction() as tx:
        library_agents = await asyncio.gather(
            *(
                prisma.models.LibraryAgent.prisma(tx).create(
                    data=prisma.types.LibraryAgentCreateInput(
                        isCreatedByUser=(user_id == user_id),
                        useGraphIsActiveVersion=True,
                        User={"connect": {"id": user_id}},
                        # Creator={"connect": {"id": user_id}},
                        AgentGraph={
                            "connect": {
                                "graphVersionId": {
                                    "id": graph_entry.id,
                                    "version": graph_entry.version,
                                }
                            }
                        },
                        settings=SafeJson(
                            GraphSettings.from_graph(
                                graph_entry,
                                hitl_safe_mode=hitl_safe_mode,
                                sensitive_action_safe_mode=sensitive_action_safe_mode,
                            ).model_dump()
                        ),
                    ),
                    include=library_agent_include(
                        user_id, include_nodes=False, include_executions=False
                    ),
                )
                for graph_entry in graph_entries
            )
        )

    # Generate images for the main graph and sub-graphs
    for agent, graph in zip(library_agents, graph_entries):
        asyncio.create_task(add_generated_agent_image(graph, user_id, agent.id))

    return [library_model.LibraryAgent.from_db(agent) for agent in library_agents]


async def update_agent_version_in_library(
    user_id: str,
    agent_graph_id: str,
    agent_graph_version: int,
) -> library_model.LibraryAgent:
    """
    Updates the agent version in the library for any agent owned by the user.

    Args:
        user_id: Owner of the LibraryAgent.
        agent_graph_id: The agent graph's ID to update.
        agent_graph_version: The new version of the agent graph.

    Raises:
        DatabaseError: If there's an error with the update.
        NotFoundError: If no library agent is found for this user and agent.
    """
    logger.debug(
        f"Updating agent version in library for user #{user_id}, "
        f"agent #{agent_graph_id} v{agent_graph_version}"
    )
    async with transaction() as tx:
        library_agent = await prisma.models.LibraryAgent.prisma(tx).find_first_or_raise(
            where={
                "userId": user_id,
                "agentGraphId": agent_graph_id,
            },
        )

        # Delete any conflicting LibraryAgent for the target version
        await prisma.models.LibraryAgent.prisma(tx).delete_many(
            where={
                "userId": user_id,
                "agentGraphId": agent_graph_id,
                "agentGraphVersion": agent_graph_version,
                "id": {"not": library_agent.id},
            }
        )

        lib = await prisma.models.LibraryAgent.prisma(tx).update(
            where={"id": library_agent.id},
            data={
                "AgentGraph": {
                    "connect": {
                        "graphVersionId": {
                            "id": agent_graph_id,
                            "version": agent_graph_version,
                        }
                    },
                },
            },
            include={"AgentGraph": True},
        )

    if lib is None:
        raise NotFoundError(
            f"Failed to update library agent for {agent_graph_id} v{agent_graph_version}"
        )

    return library_model.LibraryAgent.from_db(lib)


async def create_graph_in_library(
    graph: graph_db.Graph,
    user_id: str,
) -> tuple[graph_db.GraphModel, library_model.LibraryAgent]:
    """Create a new graph and add it to the user's library."""
    graph.version = 1
    graph_model = graph_db.make_graph_model(graph, user_id)
    graph_model.reassign_ids(user_id=user_id, reassign_graph_id=True)

    created_graph = await graph_db.create_graph(graph_model, user_id)

    library_agents = await create_library_agent(
        graph=created_graph,
        user_id=user_id,
        sensitive_action_safe_mode=True,
        create_library_agents_for_sub_graphs=False,
    )

    if created_graph.is_active:
        created_graph = await on_graph_activate(created_graph, user_id=user_id)

    return created_graph, library_agents[0]


async def update_graph_in_library(
    graph: graph_db.Graph,
    user_id: str,
) -> tuple[graph_db.GraphModel, library_model.LibraryAgent]:
    """Create a new version of an existing graph and update the library entry."""
    existing_versions = await graph_db.get_graph_all_versions(graph.id, user_id)
    current_active_version = (
        next((v for v in existing_versions if v.is_active), None)
        if existing_versions
        else None
    )
    graph.version = (
        max(v.version for v in existing_versions) + 1 if existing_versions else 1
    )

    graph_model = graph_db.make_graph_model(graph, user_id)
    graph_model.reassign_ids(user_id=user_id, reassign_graph_id=False)

    created_graph = await graph_db.create_graph(graph_model, user_id)

    library_agent = await get_library_agent_by_graph_id(user_id, created_graph.id)
    if not library_agent:
        raise NotFoundError(f"Library agent not found for graph {created_graph.id}")

    library_agent = await update_library_agent_version_and_settings(
        user_id, created_graph
    )

    if created_graph.is_active:
        created_graph = await on_graph_activate(created_graph, user_id=user_id)
        await graph_db.set_graph_active_version(
            graph_id=created_graph.id,
            version=created_graph.version,
            user_id=user_id,
        )
        if current_active_version:
            await on_graph_deactivate(current_active_version, user_id=user_id)

    return created_graph, library_agent


async def update_library_agent_version_and_settings(
    user_id: str, agent_graph: graph_db.GraphModel
) -> library_model.LibraryAgent:
    """Update library agent to point to new graph version and sync settings."""
    library = await update_agent_version_in_library(
        user_id, agent_graph.id, agent_graph.version
    )
    updated_settings = GraphSettings.from_graph(
        graph=agent_graph,
        hitl_safe_mode=library.settings.human_in_the_loop_safe_mode,
        sensitive_action_safe_mode=library.settings.sensitive_action_safe_mode,
    )
    if updated_settings != library.settings:
        library = await update_library_agent(
            library_agent_id=library.id,
            user_id=user_id,
            settings=updated_settings,
        )
    return library


async def update_library_agent(
    library_agent_id: str,
    user_id: str,
    auto_update_version: Optional[bool] = None,
    graph_version: Optional[int] = None,
    is_favorite: Optional[bool] = None,
    is_archived: Optional[bool] = None,
    is_deleted: Optional[Literal[False]] = None,
    settings: Optional[GraphSettings] = None,
    folder_id: Optional[str] = None,
) -> library_model.LibraryAgent:
    """
    Updates the specified LibraryAgent record.

    Args:
        library_agent_id: The ID of the LibraryAgent to update.
        user_id: The owner of this LibraryAgent.
        auto_update_version: Whether the agent should auto-update to active version.
        graph_version: Specific graph version to update to.
        is_favorite: Whether this agent is marked as a favorite.
        is_archived: Whether this agent is archived.
        settings: User-specific settings for this library agent.
        folder_id: Folder ID to move agent to (None to skip).

    Returns:
        The updated LibraryAgent.

    Raises:
        NotFoundError: If the specified LibraryAgent does not exist.
        DatabaseError: If there's an error in the update operation.
    """
    logger.debug(
        f"Updating library agent {library_agent_id} for user {user_id} with "
        f"auto_update_version={auto_update_version}, graph_version={graph_version}, "
        f"is_favorite={is_favorite}, is_archived={is_archived}, settings={settings}"
    )
    update_fields: prisma.types.LibraryAgentUpdateManyMutationInput = {}
    if auto_update_version is not None:
        update_fields["useGraphIsActiveVersion"] = auto_update_version
    if is_favorite is not None:
        update_fields["isFavorite"] = is_favorite
    if is_archived is not None:
        update_fields["isArchived"] = is_archived
    if is_deleted is not None:
        if is_deleted is True:
            raise RuntimeError(
                "Use delete_library_agent() to (soft-)delete library agents"
            )
        update_fields["isDeleted"] = is_deleted
    if settings is not None:
        existing_agent = await get_library_agent(id=library_agent_id, user_id=user_id)
        current_settings_dict = (
            existing_agent.settings.model_dump() if existing_agent.settings else {}
        )
        new_settings = settings.model_dump(exclude_unset=True)
        merged_settings = {**current_settings_dict, **new_settings}
        update_fields["settings"] = SafeJson(merged_settings)
    if folder_id is not None:
        # Authorization: FK only checks existence, not ownership.
        # Verify the folder belongs to this user to prevent cross-user nesting.
        await get_folder(folder_id, user_id)
        update_fields["folderId"] = folder_id

    # If graph_version is provided, update to that specific version
    if graph_version is not None:
        # Apply any other field updates first so they aren't lost
        if update_fields:
            await prisma.models.LibraryAgent.prisma().update_many(
                where={"id": library_agent_id, "userId": user_id},
                data=update_fields,
            )
        # Get the current agent to find its graph_id
        agent = await get_library_agent(id=library_agent_id, user_id=user_id)
        # Update to the specified version using existing function
        return await update_agent_version_in_library(
            user_id=user_id,
            agent_graph_id=agent.graph_id,
            agent_graph_version=graph_version,
        )

    # Otherwise, just update the simple fields
    if not update_fields:
        raise ValueError("No values were passed to update")

    n_updated = await prisma.models.LibraryAgent.prisma().update_many(
        where={"id": library_agent_id, "userId": user_id},
        data=update_fields,
    )
    if n_updated < 1:
        raise NotFoundError(f"Library agent {library_agent_id} not found")

    return await get_library_agent(
        id=library_agent_id,
        user_id=user_id,
    )


async def delete_library_agent(
    library_agent_id: str, user_id: str, soft_delete: bool = True
) -> None:
    # First get the agent to find the graph_id for cleanup
    library_agent = await prisma.models.LibraryAgent.prisma().find_unique(
        where={"id": library_agent_id}, include={"AgentGraph": True}
    )

    if not library_agent or library_agent.userId != user_id:
        raise NotFoundError(f"Library agent #{library_agent_id} not found")

    graph_id = library_agent.agentGraphId

    # Clean up associated schedules and webhooks BEFORE deleting the agent
    # This prevents executions from starting after agent deletion
    await _cleanup_schedules_for_graph(graph_id=graph_id, user_id=user_id)
    await _cleanup_webhooks_for_graph(graph_id=graph_id, user_id=user_id)

    # Delete the library agent after cleanup
    if soft_delete:
        deleted_count = await prisma.models.LibraryAgent.prisma().update_many(
            where={"id": library_agent_id, "userId": user_id},
            data={"isDeleted": True},
        )
    else:
        deleted_count = await prisma.models.LibraryAgent.prisma().delete_many(
            where={"id": library_agent_id, "userId": user_id}
        )

    if deleted_count < 1:
        raise NotFoundError(f"Library agent #{library_agent_id} not found")


async def _cleanup_schedules_for_graph(graph_id: str, user_id: str) -> None:
    """
    Clean up all schedules for a specific graph and user.

    Args:
        graph_id: The ID of the graph
        user_id: The ID of the user
    """
    scheduler_client = get_scheduler_client()
    schedules = await scheduler_client.get_execution_schedules(
        graph_id=graph_id, user_id=user_id
    )

    for schedule in schedules:
        try:
            await scheduler_client.delete_schedule(
                schedule_id=schedule.id, user_id=user_id
            )
            logger.info(f"Deleted schedule {schedule.id} for graph {graph_id}")
        except Exception:
            logger.exception(
                f"Failed to delete schedule {schedule.id} for graph {graph_id}"
            )


async def _cleanup_webhooks_for_graph(graph_id: str, user_id: str) -> None:
    """
    Clean up webhook connections for a specific graph and user.
    Unlinks webhooks from this graph and deletes them if no other triggers remain.

    Args:
        graph_id: The ID of the graph
        user_id: The ID of the user
    """
    # Find all webhooks that trigger nodes in this graph
    webhooks = await integrations_db.find_webhooks_by_graph_id(
        graph_id=graph_id, user_id=user_id
    )

    for webhook in webhooks:
        try:
            # Unlink webhook from this graph's nodes and presets
            await integrations_db.unlink_webhook_from_graph(
                webhook_id=webhook.id, graph_id=graph_id, user_id=user_id
            )
            logger.info(f"Unlinked webhook {webhook.id} from graph {graph_id}")
        except Exception:
            logger.exception(
                f"Failed to unlink webhook {webhook.id} from graph {graph_id}"
            )


async def delete_library_agent_by_graph_id(graph_id: str, user_id: str) -> None:
    """
    Deletes a library agent for the given user
    """
    await prisma.models.LibraryAgent.prisma().delete_many(
        where={"agentGraphId": graph_id, "userId": user_id}
    )


async def add_store_agent_to_library(
    store_listing_version_id: str, user_id: str
) -> library_model.LibraryAgent:
    """
    Adds an agent from a store listing version to the user's library if they don't already have it.

    Args:
        store_listing_version_id: The ID of the store listing version containing the agent.
        user_id: The user’s library to which the agent is being added.

    Returns:
        The newly created LibraryAgent if successfully added, the existing corresponding one if any.

    Raises:
        AgentNotFoundError: If the store listing or associated agent is not found.
        DatabaseError: If there's an issue creating the LibraryAgent record.
    """
    logger.debug(
        f"Adding agent from store listing version #{store_listing_version_id} "
        f"to library for user #{user_id}"
    )

    store_listing_version = (
        await prisma.models.StoreListingVersion.prisma().find_unique(
            where={"id": store_listing_version_id}, include={"AgentGraph": True}
        )
    )
    if not store_listing_version or not store_listing_version.AgentGraph:
        logger.warning(f"Store listing version not found: {store_listing_version_id}")
        raise store_exceptions.AgentNotFoundError(
            f"Store listing version {store_listing_version_id} not found or invalid"
        )

    graph = store_listing_version.AgentGraph

    # Convert to GraphModel to check for HITL blocks
    graph_model = await graph_db.get_graph(
        graph_id=graph.id,
        version=graph.version,
        user_id=user_id,
        include_subgraphs=False,
    )
    if not graph_model:
        raise store_exceptions.AgentNotFoundError(
            f"Graph #{graph.id} v{graph.version} not found or accessible"
        )

    # Check if user already has this agent (non-deleted)
    if existing := await get_library_agent_by_graph_id(
        user_id, graph.id, graph.version
    ):
        return existing

    # Check for soft-deleted version and restore it
    deleted_agent = await prisma.models.LibraryAgent.prisma().find_unique(
        where={
            "userId_agentGraphId_agentGraphVersion": {
                "userId": user_id,
                "agentGraphId": graph.id,
                "agentGraphVersion": graph.version,
            }
        },
    )
    if deleted_agent and deleted_agent.isDeleted:
        return await update_library_agent(deleted_agent.id, user_id, is_deleted=False)

    # Create LibraryAgent entry
    added_agent = await prisma.models.LibraryAgent.prisma().create(
        data={
            "User": {"connect": {"id": user_id}},
            "AgentGraph": {
                "connect": {
                    "graphVersionId": {"id": graph.id, "version": graph.version}
                }
            },
            "isCreatedByUser": False,
            "useGraphIsActiveVersion": False,
            "settings": SafeJson(GraphSettings.from_graph(graph_model).model_dump()),
        },
        include=library_agent_include(
            user_id, include_nodes=False, include_executions=False
        ),
    )
    logger.debug(
        f"Added graph #{graph.id} v{graph.version}"
        f"for store listing version #{store_listing_version.id} "
        f"to library for user #{user_id}"
    )
    return library_model.LibraryAgent.from_db(added_agent)


##############################################
############ Folder DB Functions #############
##############################################


async def _fetch_user_folders(
    user_id: str,
    extra_where: Optional[prisma.types.LibraryFolderWhereInput] = None,
    include_relations: bool = True,
) -> list[prisma.models.LibraryFolder]:
    """
    Shared helper to fetch folders for a user with consistent query params.

    Args:
        user_id: The ID of the user.
        extra_where: Additional where-clause filters to merge in.
        include_relations: Whether to include LibraryAgents and Children relations
            (used to derive counts via len(); Prisma Python has no _count include).

    Returns:
        A list of raw Prisma LibraryFolder records.
    """
    where_clause: prisma.types.LibraryFolderWhereInput = {
        "userId": user_id,
        "isDeleted": False,
    }
    if extra_where:
        where_clause.update(extra_where)

    return await prisma.models.LibraryFolder.prisma().find_many(
        where=where_clause,
        order={"createdAt": "asc"},
        include=LIBRARY_FOLDER_INCLUDE if include_relations else None,
    )


async def list_folders(
    user_id: str,
    parent_id: Optional[str] = None,
    include_relations: bool = True,
) -> list[library_model.LibraryFolder]:
    """
    Lists folders for a user, optionally filtered by parent.

    Args:
        user_id: The ID of the user.
        parent_id: If provided, only returns folders with this parent.
                   If None, returns root-level folders.
        include_relations: Whether to include agent and subfolder relations for counts.

    Returns:
        A list of LibraryFolder objects.
    """
    logger.debug(f"Listing folders for user #{user_id}, parent_id={parent_id}")

    folders = await _fetch_user_folders(
        user_id,
        extra_where={"parentId": parent_id},
        include_relations=include_relations,
    )

    return [
        library_model.LibraryFolder.from_db(
            folder,
            agent_count=len(folder.LibraryAgents) if folder.LibraryAgents else 0,
            subfolder_count=len(folder.Children) if folder.Children else 0,
        )
        for folder in folders
    ]


async def get_folder_tree(
    user_id: str,
) -> list[library_model.LibraryFolderTree]:
    """
    Gets the full folder tree for a user.

    Args:
        user_id: The ID of the user.

    Returns:
        A list of LibraryFolderTree objects (root folders with nested children).
    """
    logger.debug(f"Getting folder tree for user #{user_id}")

    # Fetch all folders for the user
    all_folders = await _fetch_user_folders(user_id)

    # Build a map of folder ID to folder data
    folder_map: dict[str, library_model.LibraryFolderTree] = {
        folder.id: library_model.LibraryFolderTree(
            **library_model.LibraryFolder.from_db(
                folder,
                agent_count=len(folder.LibraryAgents) if folder.LibraryAgents else 0,
                subfolder_count=len(folder.Children) if folder.Children else 0,
            ).model_dump(),
            children=[],
        )
        for folder in all_folders
    }

    # Build the tree structure
    root_folders: list[library_model.LibraryFolderTree] = []
    for folder in all_folders:
        tree_folder = folder_map[folder.id]
        if folder.parentId and folder.parentId in folder_map:
            folder_map[folder.parentId].children.append(tree_folder)
        else:
            root_folders.append(tree_folder)

    return root_folders


async def get_folder(
    folder_id: str,
    user_id: str,
) -> library_model.LibraryFolder:
    """
    Gets a single folder by ID.

    Args:
        folder_id: The ID of the folder.
        user_id: The ID of the user (for ownership verification).

    Returns:
        The LibraryFolder object.

    Raises:
        NotFoundError: If the folder doesn't exist or doesn't belong to the user.
    """
    folder = await prisma.models.LibraryFolder.prisma().find_first(
        where={
            "id": folder_id,
            "userId": user_id,
            "isDeleted": False,
        },
        include=LIBRARY_FOLDER_INCLUDE,
    )

    if not folder:
        raise NotFoundError(f"Folder #{folder_id} not found")

    return library_model.LibraryFolder.from_db(
        folder,
        agent_count=len(folder.LibraryAgents) if folder.LibraryAgents else 0,
        subfolder_count=len(folder.Children) if folder.Children else 0,
    )


async def _is_descendant_of(
    folder_id: str,
    potential_ancestor_id: str,
    user_id: str,
) -> bool:
    """
    Check if folder_id is a descendant of (or equal to) potential_ancestor_id.

    Fetches all user folders in a single query and walks the parent chain
    in memory to avoid N database round-trips.

    Args:
        folder_id: The ID of the folder to check.
        potential_ancestor_id: The ID of the potential ancestor.
        user_id: The ID of the user.

    Returns:
        True if folder_id is a descendant of (or equal to) potential_ancestor_id.
    """
    all_folders = await prisma.models.LibraryFolder.prisma().find_many(
        where={"userId": user_id, "isDeleted": False},
    )
    parent_map = {f.id: f.parentId for f in all_folders}

    visited: set[str] = set()
    current_id: str | None = folder_id
    while current_id:
        if current_id == potential_ancestor_id:
            return True
        if current_id in visited:
            break  # cycle detected
        visited.add(current_id)
        current_id = parent_map.get(current_id)

    return False


async def create_folder(
    user_id: str,
    name: str,
    parent_id: Optional[str] = None,
    icon: Optional[str] = None,
    color: Optional[str] = None,
) -> library_model.LibraryFolder:
    """
    Creates a new folder for the user.

    Args:
        user_id: The ID of the user.
        name: The folder name.
        parent_id: Optional parent folder ID.
        icon: Optional icon identifier.
        color: Optional hex color code.

    Returns:
        The created LibraryFolder.

    Raises:
        FolderAlreadyExistsError: If a folder with this name already exists.
        NotFoundError: If the parent folder doesn't exist.
    """
    logger.debug(f"Creating folder '{name}' for user #{user_id}")

    # Authorization: FK only checks existence, not ownership.
    # Verify the parent belongs to this user to prevent cross-user nesting.
    if parent_id:
        await get_folder(parent_id, user_id)

    # Build data dict conditionally - don't include Parent key if no parent_id
    create_data: dict = {
        "name": name,
        "User": {"connect": {"id": user_id}},
    }
    if icon is not None:
        create_data["icon"] = icon
    if color is not None:
        create_data["color"] = color
    if parent_id:
        create_data["Parent"] = {"connect": {"id": parent_id}}

    try:
        folder = await prisma.models.LibraryFolder.prisma().create(data=create_data)
    except prisma.errors.UniqueViolationError:
        raise FolderAlreadyExistsError(
            "A folder with this name already exists in this location"
        )

    return library_model.LibraryFolder.from_db(folder)


async def create_folder_with_unique_name(
    user_id: str,
    base_name: str,
    parent_id: Optional[str] = None,
    icon: Optional[str] = None,
    color: Optional[str] = None,
) -> library_model.LibraryFolder:
    """
    Creates a folder, appending (2), (3), etc. if name exists.

    Args:
        user_id: The ID of the user.
        base_name: The base folder name.
        parent_id: Optional parent folder ID.
        icon: Optional icon identifier.
        color: Optional hex color code.

    Returns:
        The created LibraryFolder.
    """
    for i in itertools.count():
        name = base_name if i == 0 else f"{base_name} ({i + 1})"
        try:
            return await create_folder(
                user_id=user_id,
                name=name,
                parent_id=parent_id,
                icon=icon,
                color=color,
            )
        except FolderAlreadyExistsError:
            continue

    raise RuntimeError("Unreachable")


async def update_folder(
    folder_id: str,
    user_id: str,
    name: Optional[str] = None,
    icon: Optional[str] = None,
    color: Optional[str] = None,
) -> library_model.LibraryFolder:
    """
    Updates a folder's properties.

    Args:
        folder_id: The ID of the folder to update.
        user_id: The ID of the user.
        name: New folder name.
        icon: New icon identifier.
        color: New hex color code.

    Returns:
        The updated LibraryFolder.

    Raises:
        NotFoundError: If the folder doesn't exist.
        DatabaseError: If there's a database error.
    """
    logger.debug(f"Updating folder #{folder_id} for user #{user_id}")

    # Authorization: update uses where={"id": ...} without userId,
    # so we must verify ownership first.
    await get_folder(folder_id, user_id)

    update_data: prisma.types.LibraryFolderUpdateInput = {}
    if name is not None:
        update_data["name"] = name
    if icon is not None:
        update_data["icon"] = icon
    if color is not None:
        update_data["color"] = color

    if not update_data:
        return await get_folder(folder_id, user_id)

    try:
        folder = await prisma.models.LibraryFolder.prisma().update(
            where={"id": folder_id},
            data=update_data,
            include=LIBRARY_FOLDER_INCLUDE,
        )
    except prisma.errors.UniqueViolationError:
        raise FolderAlreadyExistsError(
            "A folder with this name already exists in this location"
        )

    if not folder:
        raise NotFoundError(f"Folder #{folder_id} not found")

    return library_model.LibraryFolder.from_db(
        folder,
        agent_count=len(folder.LibraryAgents) if folder.LibraryAgents else 0,
        subfolder_count=len(folder.Children) if folder.Children else 0,
    )


async def move_folder(
    folder_id: str,
    user_id: str,
    target_parent_id: Optional[str],
) -> library_model.LibraryFolder:
    """
    Moves a folder to a new parent.

    Args:
        folder_id: The ID of the folder to move.
        user_id: The ID of the user.
        target_parent_id: The target parent ID (None for root).

    Returns:
        The moved LibraryFolder.

    Raises:
        FolderValidationError: If the move is invalid.
        NotFoundError: If the folder doesn't exist.
        DatabaseError: If there's a database error.
    """
    logger.debug(f"Moving folder #{folder_id} to parent #{target_parent_id}")

    # Authorization: update uses where={"id": ...} without userId,
    # so we must verify ownership first.
    await get_folder(folder_id, user_id)

    # Authorization: FK only checks existence, not ownership.
    # Verify the target parent belongs to this user to prevent cross-user nesting.
    if target_parent_id:
        await get_folder(target_parent_id, user_id)

    # Validate no circular reference
    if target_parent_id:
        if await _is_descendant_of(target_parent_id, folder_id, user_id):
            raise FolderValidationError("Cannot move folder into its own descendant")

    try:
        folder = await prisma.models.LibraryFolder.prisma().update(
            where={"id": folder_id},
            data={
                "parentId": target_parent_id,
            },
            include=LIBRARY_FOLDER_INCLUDE,
        )
    except prisma.errors.UniqueViolationError:
        raise FolderAlreadyExistsError(
            "A folder with this name already exists in this location"
        )

    if not folder:
        raise NotFoundError(f"Folder #{folder_id} not found")

    return library_model.LibraryFolder.from_db(
        folder,
        agent_count=len(folder.LibraryAgents) if folder.LibraryAgents else 0,
        subfolder_count=len(folder.Children) if folder.Children else 0,
    )


async def delete_folder(
    folder_id: str,
    user_id: str,
    soft_delete: bool = True,
) -> None:
    """
    Deletes a folder and all its contents (cascade).

    Args:
        folder_id: The ID of the folder to delete.
        user_id: The ID of the user.
        soft_delete: If True, soft-deletes; otherwise hard-deletes.

    Raises:
        NotFoundError: If the folder doesn't exist.
        DatabaseError: If there's a database error.
    """
    logger.debug(f"Deleting folder #{folder_id} for user #{user_id}")

    # Authorization: verify folder exists and belongs to user.
    await get_folder(folder_id, user_id)

    # Collect all folder IDs (target + descendants) — single query, in-memory walk
    descendant_ids = await _get_descendant_folder_ids(folder_id, user_id)
    all_folder_ids = [folder_id] + descendant_ids

    async with transaction() as tx:
        if soft_delete:
            # Move agents to root so they aren't lost when the folder is deleted
            await prisma.models.LibraryAgent.prisma(tx).update_many(
                where={
                    "folderId": {"in": all_folder_ids},
                    "userId": user_id,
                },
                data={"folderId": None},
            )

            # Soft-delete all folders
            await prisma.models.LibraryFolder.prisma(tx).update_many(
                where={
                    "id": {"in": all_folder_ids},
                    "userId": user_id,
                },
                data={"isDeleted": True},
            )
        else:
            # Move agents to root (or could hard-delete them)
            await prisma.models.LibraryAgent.prisma(tx).update_many(
                where={
                    "folderId": {"in": all_folder_ids},
                    "userId": user_id,
                },
                data={"folderId": None},
            )

            # Hard-delete folders (children first due to FK constraints)
            for fid in reversed(all_folder_ids):
                await prisma.models.LibraryFolder.prisma(tx).delete(where={"id": fid})


async def _get_descendant_folder_ids(
    folder_id: str,
    user_id: str,
) -> list[str]:
    """
    Get all descendant folder IDs in a single query + in-memory walk,
    same approach as _is_descendant_of to avoid N recursive DB round-trips.

    Args:
        folder_id: The ID of the parent folder.
        user_id: The ID of the user.

    Returns:
        A list of descendant folder IDs.
    """
    all_folders = await prisma.models.LibraryFolder.prisma().find_many(
        where={"userId": user_id, "isDeleted": False},
    )

    # Build children map: parent_id -> [child_ids]
    children_map: dict[str, list[str]] = {}
    for f in all_folders:
        if f.parentId:
            children_map.setdefault(f.parentId, []).append(f.id)

    # Walk the tree in memory
    result: list[str] = []
    visited: set[str] = set()
    stack = list(children_map.get(folder_id, []))
    while stack:
        current = stack.pop()
        if current in visited:
            continue  # cycle guard
        visited.add(current)
        result.append(current)
        stack.extend(children_map.get(current, []))

    return result


async def move_agent_to_folder(
    library_agent_id: str,
    folder_id: Optional[str],
    user_id: str,
) -> library_model.LibraryAgent:
    """
    Moves a library agent to a folder.

    Args:
        library_agent_id: The ID of the library agent.
        folder_id: The target folder ID (None for root).
        user_id: The ID of the user.

    Returns:
        The updated LibraryAgent.

    Raises:
        NotFoundError: If the agent or folder doesn't exist.
        DatabaseError: If there's a database error.
    """
    logger.debug(f"Moving agent #{library_agent_id} to folder #{folder_id}")

    # Authorization: verify agent belongs to user before updating.
    # update() uses where={"id": ...} without userId, so check ownership first.
    await get_library_agent(library_agent_id, user_id)

    # Authorization: folderId is set directly, FK only checks existence
    # not ownership, so verify the folder belongs to the user.
    if folder_id:
        await get_folder(folder_id, user_id)

    await prisma.models.LibraryAgent.prisma().update(
        where={"id": library_agent_id},
        data={"folderId": folder_id},
    )

    return await get_library_agent(library_agent_id, user_id)


async def bulk_move_agents_to_folder(
    agent_ids: list[str],
    folder_id: Optional[str],
    user_id: str,
) -> list[library_model.LibraryAgent]:
    """
    Moves multiple library agents to a folder.

    Args:
        agent_ids: The IDs of the library agents.
        folder_id: The target folder ID (None for root).
        user_id: The ID of the user.

    Returns:
        The updated LibraryAgents.

    Raises:
        NotFoundError: If any agent or the folder doesn't exist.
        DatabaseError: If there's a database error.
    """
    logger.debug(f"Bulk moving {len(agent_ids)} agents to folder #{folder_id}")

    # Authorization: folderId is set directly, FK only checks existence
    # not ownership, so verify the folder belongs to the user.
    if folder_id:
        await get_folder(folder_id, user_id)

    # Update all agents
    await prisma.models.LibraryAgent.prisma().update_many(
        where={
            "id": {"in": agent_ids},
            "userId": user_id,
            "isDeleted": False,
        },
        data={"folderId": folder_id},
    )

    # Fetch and return updated agents
    agents = await prisma.models.LibraryAgent.prisma().find_many(
        where={
            "id": {"in": agent_ids},
            "userId": user_id,
        },
        include=library_agent_include(
            user_id, include_nodes=False, include_executions=False
        ),
    )

    return [library_model.LibraryAgent.from_db(agent) for agent in agents]


##############################################
########### Presets DB Functions #############
##############################################


async def list_presets(
    user_id: str, page: int, page_size: int, graph_id: Optional[str] = None
) -> library_model.LibraryAgentPresetResponse:
    """
    Retrieves a paginated list of AgentPresets for the specified user.

    Args:
        user_id: The user ID whose presets are being retrieved.
        page: The current page index (1-based).
        page_size: Number of items to retrieve per page.
        graph_id: Agent Graph ID to filter by.

    Returns:
        A LibraryAgentPresetResponse containing a list of presets and pagination info.

    Raises:
        DatabaseError: If there's a database error during the operation.
    """
    logger.debug(
        f"Fetching presets for user #{user_id}, page={page}, page_size={page_size}"
    )

    if page < 1 or page_size < 1:
        logger.warning(
            "Invalid pagination input: page=%d, page_size=%d", page, page_size
        )
        raise InvalidInputError("Invalid pagination parameters")

    query_filter: prisma.types.AgentPresetWhereInput = {
        "userId": user_id,
        "isDeleted": False,
    }
    if graph_id:
        query_filter["agentGraphId"] = graph_id

    presets_records = await prisma.models.AgentPreset.prisma().find_many(
        where=query_filter,
        skip=(page - 1) * page_size,
        take=page_size,
        include=AGENT_PRESET_INCLUDE,
    )
    total_items = await prisma.models.AgentPreset.prisma().count(where=query_filter)
    total_pages = (total_items + page_size - 1) // page_size

    presets = [
        library_model.LibraryAgentPreset.from_db(preset) for preset in presets_records
    ]

    return library_model.LibraryAgentPresetResponse(
        presets=presets,
        pagination=Pagination(
            total_items=total_items,
            total_pages=total_pages,
            current_page=page,
            page_size=page_size,
        ),
    )


async def get_preset(
    user_id: str, preset_id: str
) -> library_model.LibraryAgentPreset | None:
    """
    Retrieves a single AgentPreset by its ID for a given user.

    Args:
        user_id: The user that owns the preset.
        preset_id: The ID of the preset.

    Returns:
        A LibraryAgentPreset if it exists and matches the user, otherwise None.

    Raises:
        DatabaseError: If there's a database error during the fetch.
    """
    logger.debug(f"Fetching preset #{preset_id} for user #{user_id}")
    preset = await prisma.models.AgentPreset.prisma().find_unique(
        where={"id": preset_id},
        include=AGENT_PRESET_INCLUDE,
    )
    if not preset or preset.userId != user_id or preset.isDeleted:
        return None
    return library_model.LibraryAgentPreset.from_db(preset)


async def create_preset(
    user_id: str,
    preset: library_model.LibraryAgentPresetCreatable,
) -> library_model.LibraryAgentPreset:
    """
    Creates a new AgentPreset for a user.

    Args:
        user_id: The ID of the user creating the preset.
        preset: The preset data used for creation.

    Returns:
        The newly created LibraryAgentPreset.

    Raises:
        DatabaseError: If there's a database error in creating the preset.
    """
    logger.debug(
        f"Creating preset ({repr(preset.name)}) for user #{user_id}",
    )
    new_preset = await prisma.models.AgentPreset.prisma().create(
        data=prisma.types.AgentPresetCreateInput(
            userId=user_id,
            name=preset.name,
            description=preset.description,
            agentGraphId=preset.graph_id,
            agentGraphVersion=preset.graph_version,
            isActive=preset.is_active,
            webhookId=preset.webhook_id,
            InputPresets={
                "create": [
                    prisma.types.AgentNodeExecutionInputOutputCreateWithoutRelationsInput(  # noqa
                        name=name, data=SafeJson(data)
                    )
                    for name, data in {
                        **preset.inputs,
                        **preset.credentials,
                    }.items()
                ]
            },
        ),
        include=AGENT_PRESET_INCLUDE,
    )
    return library_model.LibraryAgentPreset.from_db(new_preset)


async def create_preset_from_graph_execution(
    user_id: str,
    create_request: library_model.LibraryAgentPresetCreatableFromGraphExecution,
) -> library_model.LibraryAgentPreset:
    """
    Creates a new AgentPreset from an AgentGraphExecution.

    Params:
        user_id: The ID of the user creating the preset.
        create_request: The data used for creation.

    Returns:
        The newly created LibraryAgentPreset.

    Raises:
        DatabaseError: If there's a database error in creating the preset.
    """
    graph_exec_id = create_request.graph_execution_id
    graph_execution = await get_graph_execution(user_id, graph_exec_id)
    if not graph_execution:
        raise NotFoundError(f"Graph execution #{graph_exec_id} not found")

    # Sanity check: credential inputs must be available if required for this preset
    if graph_execution.credential_inputs is None:
        graph = await graph_db.get_graph(
            graph_id=graph_execution.graph_id,
            version=graph_execution.graph_version,
            user_id=graph_execution.user_id,
            include_subgraphs=True,
        )
        if not graph:
            raise NotFoundError(
                f"Graph #{graph_execution.graph_id} not found or accessible"
            )
        elif len(graph.aggregate_credentials_inputs()) > 0:
            raise ValueError(
                f"Graph execution #{graph_exec_id} can't be turned into a preset "
                "because it was run before this feature existed "
                "and so the input credentials were not saved."
            )

    logger.debug(
        f"Creating preset for user #{user_id} from graph execution #{graph_exec_id}",
    )
    return await create_preset(
        user_id=user_id,
        preset=library_model.LibraryAgentPresetCreatable(
            inputs=graph_execution.inputs,
            credentials=graph_execution.credential_inputs or {},
            graph_id=graph_execution.graph_id,
            graph_version=graph_execution.graph_version,
            name=create_request.name,
            description=create_request.description,
            is_active=create_request.is_active,
        ),
    )


async def update_preset(
    user_id: str,
    preset_id: str,
    inputs: Optional[GraphInput] = None,
    credentials: Optional[dict[str, CredentialsMetaInput]] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    is_active: Optional[bool] = None,
) -> library_model.LibraryAgentPreset:
    """
    Updates an existing AgentPreset for a user.

    Args:
        user_id: The ID of the user updating the preset.
        preset_id: The ID of the preset to update.
        inputs: New inputs object to set on the preset.
        credentials: New credentials to set on the preset.
        name: New name for the preset.
        description: New description for the preset.
        is_active: New active status for the preset.

    Returns:
        The updated LibraryAgentPreset.

    Raises:
        DatabaseError: If there's a database error in updating the preset.
        NotFoundError: If attempting to update a non-existent preset.
    """
    current = await get_preset(user_id, preset_id)  # assert ownership
    if not current:
        raise NotFoundError(f"Preset #{preset_id} not found for user #{user_id}")
    logger.debug(
        f"Updating preset #{preset_id} ({repr(current.name)}) for user #{user_id}",
    )
    async with transaction() as tx:
        update_data: prisma.types.AgentPresetUpdateInput = {}
        if name:
            update_data["name"] = name
        if description:
            update_data["description"] = description
        if is_active is not None:
            update_data["isActive"] = is_active
        if inputs or credentials:
            if not (inputs and credentials):
                raise ValueError(
                    "Preset inputs and credentials must be provided together"
                )
            update_data["InputPresets"] = {
                "create": [
                    prisma.types.AgentNodeExecutionInputOutputCreateWithoutRelationsInput(  # noqa
                        name=name, data=SafeJson(data)
                    )
                    for name, data in {
                        **inputs,
                        **{
                            key: creds_meta.model_dump(exclude_none=True)
                            for key, creds_meta in credentials.items()
                        },
                    }.items()
                ],
            }
            # Existing InputPresets must be deleted, in a separate query
            await prisma.models.AgentNodeExecutionInputOutput.prisma(tx).delete_many(
                where={"agentPresetId": preset_id}
            )

        updated = await prisma.models.AgentPreset.prisma(tx).update(
            where={"id": preset_id},
            data=update_data,
            include=AGENT_PRESET_INCLUDE,
        )
    if not updated:
        raise RuntimeError(f"AgentPreset #{preset_id} vanished while updating")
    return library_model.LibraryAgentPreset.from_db(updated)


async def set_preset_webhook(
    user_id: str, preset_id: str, webhook_id: str | None
) -> library_model.LibraryAgentPreset:
    current = await prisma.models.AgentPreset.prisma().find_unique(
        where={"id": preset_id},
        include=AGENT_PRESET_INCLUDE,
    )
    if not current or current.userId != user_id:
        raise NotFoundError(f"Preset #{preset_id} not found")

    updated = await prisma.models.AgentPreset.prisma().update(
        where={"id": preset_id},
        data=(
            {"Webhook": {"connect": {"id": webhook_id}}}
            if webhook_id
            else {"Webhook": {"disconnect": True}}
        ),
        include=AGENT_PRESET_INCLUDE,
    )
    if not updated:
        raise RuntimeError(f"AgentPreset #{preset_id} vanished while updating")
    return library_model.LibraryAgentPreset.from_db(updated)


async def delete_preset(user_id: str, preset_id: str) -> None:
    """
    Soft-deletes a preset by marking it as isDeleted = True.

    Args:
        user_id: The user that owns the preset.
        preset_id: The ID of the preset to delete.

    Raises:
        DatabaseError: If there's a database error during deletion.
    """
    logger.debug(f"Setting preset #{preset_id} for user #{user_id} to deleted")
    await prisma.models.AgentPreset.prisma().update_many(
        where={"id": preset_id, "userId": user_id},
        data={"isDeleted": True},
    )


async def fork_library_agent(
    library_agent_id: str, user_id: str
) -> library_model.LibraryAgent:
    """
    Clones a library agent and its underyling graph and nodes (with new ids) for the given user.

    Args:
        library_agent_id: The ID of the library agent to fork.
        user_id: The ID of the user who owns the library agent.

    Returns:
        The forked parent (if it has sub-graphs) LibraryAgent.

    Raises:
        DatabaseError: If there's an error during the forking process.
    """
    logger.debug(f"Forking library agent {library_agent_id} for user {user_id}")

    # Fetch the original agent
    original_agent = await get_library_agent(library_agent_id, user_id)

    # Check if user owns the library agent
    # TODO: once we have open/closed sourced agents this needs to be enabled ~kcze
    # + update library/agents/[id]/page.tsx agent actions
    # if not original_agent.can_access_graph:
    #     raise DatabaseError(
    #         f"User {user_id} cannot access library agent graph {library_agent_id}"
    #     )

    # Fork the underlying graph and nodes
    new_graph = await graph_db.fork_graph(
        original_agent.graph_id, original_agent.graph_version, user_id
    )
    new_graph = await on_graph_activate(new_graph, user_id=user_id)

    # Create a library agent for the new graph, preserving safe mode settings
    return (
        await create_library_agent(
            new_graph,
            user_id,
            hitl_safe_mode=original_agent.settings.human_in_the_loop_safe_mode,
            sensitive_action_safe_mode=original_agent.settings.sensitive_action_safe_mode,
        )
    )[0]
