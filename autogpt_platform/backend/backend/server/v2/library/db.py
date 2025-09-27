import asyncio
import logging
from typing import Literal, Optional

import fastapi
import prisma.errors
import prisma.fields
import prisma.models
import prisma.types

import backend.data.graph as graph_db
import backend.server.v2.library.model as library_model
import backend.server.v2.store.exceptions as store_exceptions
import backend.server.v2.store.image_gen as store_image_gen
import backend.server.v2.store.media as store_media
from backend.data.block import BlockInput
from backend.data.db import transaction
from backend.data.execution import get_graph_execution
from backend.data.includes import AGENT_PRESET_INCLUDE, library_agent_include
from backend.data.model import CredentialsMetaInput
from backend.integrations.creds_manager import IntegrationCredentialsManager
from backend.integrations.webhooks.graph_lifecycle_hooks import on_graph_activate
from backend.util.exceptions import NotFoundError
from backend.util.json import SafeJson
from backend.util.models import Pagination
from backend.util.settings import Config

logger = logging.getLogger(__name__)
config = Config()
integration_creds_manager = IntegrationCredentialsManager()


async def list_library_agents(
    user_id: str,
    search_term: Optional[str] = None,
    sort_by: library_model.LibraryAgentSort = library_model.LibraryAgentSort.UPDATED_AT,
    page: int = 1,
    page_size: int = 50,
) -> library_model.LibraryAgentResponse:
    """
    Retrieves a paginated list of LibraryAgent records for a given user.

    Args:
        user_id: The ID of the user whose LibraryAgents we want to retrieve.
        search_term: Optional string to filter agents by name/description.
        sort_by: Sorting field (createdAt, updatedAt, isFavorite, isCreatedByUser).
        page: Current page (1-indexed).
        page_size: Number of items per page.

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
        raise store_exceptions.DatabaseError("Invalid pagination input")

    if search_term and len(search_term.strip()) > 100:
        logger.warning(f"Search term too long: {repr(search_term)}")
        raise store_exceptions.DatabaseError("Search term is too long")

    where_clause: prisma.types.LibraryAgentWhereInput = {
        "userId": user_id,
        "isDeleted": False,
        "isArchived": False,
    }

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

    # Determine sorting
    order_by: prisma.types.LibraryAgentOrderByInput | None = None

    if sort_by == library_model.LibraryAgentSort.CREATED_AT:
        order_by = {"createdAt": "asc"}
    elif sort_by == library_model.LibraryAgentSort.UPDATED_AT:
        order_by = {"updatedAt": "desc"}

    try:
        library_agents = await prisma.models.LibraryAgent.prisma().find_many(
            where=where_clause,
            include=library_agent_include(
                user_id, include_nodes=False, include_executions=False
            ),
            order=order_by,
            skip=(page - 1) * page_size,
            take=page_size,
        )
        agent_count = await prisma.models.LibraryAgent.prisma().count(
            where=where_clause
        )

        logger.debug(
            f"Retrieved {len(library_agents)} library agents for user #{user_id}"
        )

        # Only pass valid agents to the response
        valid_library_agents: list[library_model.LibraryAgent] = []

        for agent in library_agents:
            try:
                library_agent = library_model.LibraryAgent.from_db(agent)
                valid_library_agents.append(library_agent)
            except Exception as e:
                # Skip this agent if there was an error
                logger.error(
                    f"Error parsing LibraryAgent #{agent.id} from DB item: {e}"
                )
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

    except prisma.errors.PrismaError as e:
        logger.error(f"Database error fetching library agents: {e}")
        raise store_exceptions.DatabaseError("Failed to fetch library agents") from e


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
        raise store_exceptions.DatabaseError("Invalid pagination input")

    where_clause: prisma.types.LibraryAgentWhereInput = {
        "userId": user_id,
        "isDeleted": False,
        "isArchived": False,
        "isFavorite": True,  # Only fetch favorites
    }

    # Sort favorites by updated date descending
    order_by: prisma.types.LibraryAgentOrderByInput = {"updatedAt": "desc"}

    try:
        library_agents = await prisma.models.LibraryAgent.prisma().find_many(
            where=where_clause,
            include=library_agent_include(
                user_id, include_nodes=False, include_executions=False
            ),
            order=order_by,
            skip=(page - 1) * page_size,
            take=page_size,
        )
        agent_count = await prisma.models.LibraryAgent.prisma().count(
            where=where_clause
        )

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
                logger.error(
                    f"Error parsing LibraryAgent #{agent.id} from DB item: {e}"
                )
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

    except prisma.errors.PrismaError as e:
        logger.error(f"Database error fetching favorite library agents: {e}")
        raise store_exceptions.DatabaseError(
            "Failed to fetch favorite library agents"
        ) from e


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
    try:
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

        return library_model.LibraryAgent.from_db(
            library_agent,
            sub_graphs=(
                await graph_db.get_sub_graphs(library_agent.AgentGraph)
                if library_agent.AgentGraph
                else None
            ),
        )

    except prisma.errors.PrismaError as e:
        logger.error(f"Database error fetching library agent: {e}")
        raise store_exceptions.DatabaseError("Failed to fetch library agent") from e


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
    try:
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
    except prisma.errors.PrismaError as e:
        logger.error(f"Database error fetching library agent by graph ID: {e}")
        raise store_exceptions.DatabaseError("Failed to fetch library agent") from e


async def add_generated_agent_image(
    graph: graph_db.BaseGraph,
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
    create_library_agents_for_sub_graphs: bool = True,
) -> list[library_model.LibraryAgent]:
    """
    Adds an agent to the user's library (LibraryAgent table).

    Args:
        agent: The agent/Graph to add to the library.
        user_id: The user to whom the agent will be added.
        create_library_agents_for_sub_graphs: If True, creates LibraryAgent records for sub-graphs as well.

    Returns:
        The newly created LibraryAgent records.
        If the graph has sub-graphs, the parent graph will always be the first entry in the list.

    Raises:
        AgentNotFoundError: If the specified agent does not exist.
        DatabaseError: If there's an error during creation or if image generation fails.
    """
    logger.info(
        f"Creating library agent for graph #{graph.id} v{graph.version}; "
        f"user #{user_id}"
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
) -> None:
    """
    Updates the agent version in the library if useGraphIsActiveVersion is True.

    Args:
        user_id: Owner of the LibraryAgent.
        agent_graph_id: The agent graph's ID to update.
        agent_graph_version: The new version of the agent graph.

    Raises:
        DatabaseError: If there's an error with the update.
    """
    logger.debug(
        f"Updating agent version in library for user #{user_id}, "
        f"agent #{agent_graph_id} v{agent_graph_version}"
    )
    try:
        library_agent = await prisma.models.LibraryAgent.prisma().find_first_or_raise(
            where={
                "userId": user_id,
                "agentGraphId": agent_graph_id,
                "useGraphIsActiveVersion": True,
            },
        )
        await prisma.models.LibraryAgent.prisma().update(
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
        )
    except prisma.errors.PrismaError as e:
        logger.error(f"Database error updating agent version in library: {e}")
        raise store_exceptions.DatabaseError(
            "Failed to update agent version in library"
        ) from e


async def update_library_agent(
    library_agent_id: str,
    user_id: str,
    auto_update_version: Optional[bool] = None,
    is_favorite: Optional[bool] = None,
    is_archived: Optional[bool] = None,
    is_deleted: Optional[Literal[False]] = None,
) -> library_model.LibraryAgent:
    """
    Updates the specified LibraryAgent record.

    Args:
        library_agent_id: The ID of the LibraryAgent to update.
        user_id: The owner of this LibraryAgent.
        auto_update_version: Whether the agent should auto-update to active version.
        is_favorite: Whether this agent is marked as a favorite.
        is_archived: Whether this agent is archived.

    Returns:
        The updated LibraryAgent.

    Raises:
        NotFoundError: If the specified LibraryAgent does not exist.
        DatabaseError: If there's an error in the update operation.
    """
    logger.debug(
        f"Updating library agent {library_agent_id} for user {user_id} with "
        f"auto_update_version={auto_update_version}, is_favorite={is_favorite}, "
        f"is_archived={is_archived}"
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
    if not update_fields:
        raise ValueError("No values were passed to update")

    try:
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
    except prisma.errors.PrismaError as e:
        logger.error(f"Database error updating library agent: {str(e)}")
        raise store_exceptions.DatabaseError("Failed to update library agent") from e


async def delete_library_agent(
    library_agent_id: str, user_id: str, soft_delete: bool = True
) -> None:
    if soft_delete:
        deleted_count = await prisma.models.LibraryAgent.prisma().update_many(
            where={"id": library_agent_id, "userId": user_id}, data={"isDeleted": True}
        )
    else:
        deleted_count = await prisma.models.LibraryAgent.prisma().delete_many(
            where={"id": library_agent_id, "userId": user_id}
        )
    if deleted_count < 1:
        raise NotFoundError(f"Library agent #{library_agent_id} not found")


async def delete_library_agent_by_graph_id(graph_id: str, user_id: str) -> None:
    """
    Deletes a library agent for the given user
    """
    try:
        await prisma.models.LibraryAgent.prisma().delete_many(
            where={"agentGraphId": graph_id, "userId": user_id}
        )
    except prisma.errors.PrismaError as e:
        logger.error(f"Database error deleting library agent: {e}")
        raise store_exceptions.DatabaseError("Failed to delete library agent") from e


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

    try:
        store_listing_version = (
            await prisma.models.StoreListingVersion.prisma().find_unique(
                where={"id": store_listing_version_id}, include={"AgentGraph": True}
            )
        )
        if not store_listing_version or not store_listing_version.AgentGraph:
            logger.warning(
                f"Store listing version not found: {store_listing_version_id}"
            )
            raise store_exceptions.AgentNotFoundError(
                f"Store listing version {store_listing_version_id} not found or invalid"
            )

        graph = store_listing_version.AgentGraph

        # Check if user already has this agent
        existing_library_agent = await prisma.models.LibraryAgent.prisma().find_unique(
            where={
                "userId_agentGraphId_agentGraphVersion": {
                    "userId": user_id,
                    "agentGraphId": graph.id,
                    "agentGraphVersion": graph.version,
                }
            },
            include={"AgentGraph": True},
        )
        if existing_library_agent:
            if existing_library_agent.isDeleted:
                # Even if agent exists it needs to be marked as not deleted
                await update_library_agent(
                    existing_library_agent.id, user_id, is_deleted=False
                )
            else:
                logger.debug(
                    f"User #{user_id} already has graph #{graph.id} "
                    f"v{graph.version} in their library"
                )
            return library_model.LibraryAgent.from_db(existing_library_agent)

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
    except store_exceptions.AgentNotFoundError:
        # Reraise for external handling.
        raise
    except prisma.errors.PrismaError as e:
        logger.error(f"Database error adding agent to library: {e}")
        raise store_exceptions.DatabaseError("Failed to add agent to library") from e


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
        raise store_exceptions.DatabaseError("Invalid pagination parameters")

    query_filter: prisma.types.AgentPresetWhereInput = {
        "userId": user_id,
        "isDeleted": False,
    }
    if graph_id:
        query_filter["agentGraphId"] = graph_id

    try:
        presets_records = await prisma.models.AgentPreset.prisma().find_many(
            where=query_filter,
            skip=(page - 1) * page_size,
            take=page_size,
            include=AGENT_PRESET_INCLUDE,
        )
        total_items = await prisma.models.AgentPreset.prisma().count(where=query_filter)
        total_pages = (total_items + page_size - 1) // page_size

        presets = [
            library_model.LibraryAgentPreset.from_db(preset)
            for preset in presets_records
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

    except prisma.errors.PrismaError as e:
        logger.error(f"Database error getting presets: {e}")
        raise store_exceptions.DatabaseError("Failed to fetch presets") from e


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
    try:
        preset = await prisma.models.AgentPreset.prisma().find_unique(
            where={"id": preset_id},
            include=AGENT_PRESET_INCLUDE,
        )
        if not preset or preset.userId != user_id or preset.isDeleted:
            return None
        return library_model.LibraryAgentPreset.from_db(preset)
    except prisma.errors.PrismaError as e:
        logger.error(f"Database error getting preset: {e}")
        raise store_exceptions.DatabaseError("Failed to fetch preset") from e


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
    try:
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
    except prisma.errors.PrismaError as e:
        logger.error(f"Database error creating preset: {e}")
        raise store_exceptions.DatabaseError("Failed to create preset") from e


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
    inputs: Optional[BlockInput] = None,
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
    try:
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
                await prisma.models.AgentNodeExecutionInputOutput.prisma(
                    tx
                ).delete_many(where={"agentPresetId": preset_id})

            updated = await prisma.models.AgentPreset.prisma(tx).update(
                where={"id": preset_id},
                data=update_data,
                include=AGENT_PRESET_INCLUDE,
            )
        if not updated:
            raise RuntimeError(f"AgentPreset #{preset_id} vanished while updating")
        return library_model.LibraryAgentPreset.from_db(updated)
    except prisma.errors.PrismaError as e:
        logger.error(f"Database error updating preset: {e}")
        raise store_exceptions.DatabaseError("Failed to update preset") from e


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
    try:
        await prisma.models.AgentPreset.prisma().update_many(
            where={"id": preset_id, "userId": user_id},
            data={"isDeleted": True},
        )
    except prisma.errors.PrismaError as e:
        logger.error(f"Database error deleting preset: {e}")
        raise store_exceptions.DatabaseError("Failed to delete preset") from e


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
    try:
        # Fetch the original agent
        original_agent = await get_library_agent(library_agent_id, user_id)

        # Check if user owns the library agent
        # TODO: once we have open/closed sourced agents this needs to be enabled ~kcze
        # + update library/agents/[id]/page.tsx agent actions
        # if not original_agent.can_access_graph:
        #     raise store_exceptions.DatabaseError(
        #         f"User {user_id} cannot access library agent graph {library_agent_id}"
        #     )

        # Fork the underlying graph and nodes
        new_graph = await graph_db.fork_graph(
            original_agent.graph_id, original_agent.graph_version, user_id
        )
        new_graph = await on_graph_activate(new_graph, user_id=user_id)

        # Create a library agent for the new graph
        return (await create_library_agent(new_graph, user_id))[0]
    except prisma.errors.PrismaError as e:
        logger.error(f"Database error cloning library agent: {e}")
        raise store_exceptions.DatabaseError("Failed to fork library agent") from e
