import logging
from typing import Optional

import fastapi
import prisma.errors
import prisma.fields
import prisma.models
import prisma.types

import backend.data.graph
import backend.server.model
import backend.server.v2.library.model as library_model
import backend.server.v2.store.exceptions as store_exceptions
import backend.server.v2.store.image_gen as store_image_gen
import backend.server.v2.store.media as store_media
from backend.data.db import locked_transaction
from backend.data.includes import library_agent_include
from backend.util.settings import Config

logger = logging.getLogger(__name__)
config = Config()


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
                "Agent": {
                    "is": {"name": {"contains": search_term, "mode": "insensitive"}}
                }
            },
            {
                "Agent": {
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
            include=library_agent_include(user_id),
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
                    f"Error parsing LibraryAgent when getting library agents from db: {e}"
                )
                continue

        # Return the response with only valid agents
        return library_model.LibraryAgentResponse(
            agents=valid_library_agents,
            pagination=backend.server.model.Pagination(
                total_items=agent_count,
                total_pages=(agent_count + page_size - 1) // page_size,
                current_page=page,
                page_size=page_size,
            ),
        )

    except prisma.errors.PrismaError as e:
        logger.error(f"Database error fetching library agents: {e}")
        raise store_exceptions.DatabaseError("Failed to fetch library agents") from e


async def get_library_agent(id: str, user_id: str) -> library_model.LibraryAgent:
    """
    Get a specific agent from the user's library.

    Args:
        library_agent_id: ID of the library agent to retrieve.
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
            raise store_exceptions.AgentNotFoundError(f"Library agent #{id} not found")

        return library_model.LibraryAgent.from_db(library_agent)

    except prisma.errors.PrismaError as e:
        logger.error(f"Database error fetching library agent: {e}")
        raise store_exceptions.DatabaseError("Failed to fetch library agent") from e


async def add_generated_agent_image(
    graph: backend.data.graph.GraphModel,
    library_agent_id: str,
) -> Optional[prisma.models.LibraryAgent]:
    """
    Generates an image for the specified LibraryAgent and updates its record.
    """
    user_id = graph.user_id
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
    graph: backend.data.graph.GraphModel,
    user_id: str,
) -> prisma.models.LibraryAgent:
    """
    Adds an agent to the user's library (LibraryAgent table).

    Args:
        agent: The agent/Graph to add to the library.
        user_id: The user to whom the agent will be added.

    Returns:
        The newly created LibraryAgent record.

    Raises:
        AgentNotFoundError: If the specified agent does not exist.
        DatabaseError: If there's an error during creation or if image generation fails.
    """
    logger.info(
        f"Creating library agent for graph #{graph.id} v{graph.version}; "
        f"user #{user_id}"
    )

    try:
        return await prisma.models.LibraryAgent.prisma().create(
            data={
                "isCreatedByUser": (user_id == graph.user_id),
                "useGraphIsActiveVersion": True,
                "User": {"connect": {"id": user_id}},
                "Agent": {
                    "connect": {
                        "graphVersionId": {"id": graph.id, "version": graph.version}
                    }
                },
            }
        )
    except prisma.errors.PrismaError as e:
        logger.error(f"Database error creating agent in library: {e}")
        raise store_exceptions.DatabaseError("Failed to create agent in library") from e


async def update_agent_version_in_library(
    user_id: str,
    agent_id: str,
    agent_version: int,
) -> None:
    """
    Updates the agent version in the library if useGraphIsActiveVersion is True.

    Args:
        user_id: Owner of the LibraryAgent.
        agent_id: The agent's ID to update.
        agent_version: The new version of the agent.

    Raises:
        DatabaseError: If there's an error with the update.
    """
    logger.debug(
        f"Updating agent version in library for user #{user_id}, "
        f"agent #{agent_id} v{agent_version}"
    )
    try:
        library_agent = await prisma.models.LibraryAgent.prisma().find_first_or_raise(
            where={
                "userId": user_id,
                "agentId": agent_id,
                "useGraphIsActiveVersion": True,
            },
        )
        await prisma.models.LibraryAgent.prisma().update(
            where={"id": library_agent.id},
            data={
                "Agent": {
                    "connect": {
                        "graphVersionId": {"id": agent_id, "version": agent_version}
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
    is_deleted: Optional[bool] = None,
) -> None:
    """
    Updates the specified LibraryAgent record.

    Args:
        library_agent_id: The ID of the LibraryAgent to update.
        user_id: The owner of this LibraryAgent.
        auto_update_version: Whether the agent should auto-update to active version.
        is_favorite: Whether this agent is marked as a favorite.
        is_archived: Whether this agent is archived.
        is_deleted: Whether this agent is deleted.

    Raises:
        DatabaseError: If there's an error in the update operation.
    """
    logger.debug(
        f"Updating library agent {library_agent_id} for user {user_id} with "
        f"auto_update_version={auto_update_version}, is_favorite={is_favorite}, "
        f"is_archived={is_archived}, is_deleted={is_deleted}"
    )
    update_fields: prisma.types.LibraryAgentUpdateManyMutationInput = {}
    if auto_update_version is not None:
        update_fields["useGraphIsActiveVersion"] = auto_update_version
    if is_favorite is not None:
        update_fields["isFavorite"] = is_favorite
    if is_archived is not None:
        update_fields["isArchived"] = is_archived
    if is_deleted is not None:
        update_fields["isDeleted"] = is_deleted

    try:
        await prisma.models.LibraryAgent.prisma().update_many(
            where={"id": library_agent_id, "userId": user_id}, data=update_fields
        )
    except prisma.errors.PrismaError as e:
        logger.error(f"Database error updating library agent: {str(e)}")
        raise store_exceptions.DatabaseError("Failed to update library agent") from e


async def delete_library_agent_by_graph_id(graph_id: str, user_id: str) -> None:
    """
    Deletes a library agent for the given user
    """
    try:
        await prisma.models.LibraryAgent.prisma().delete_many(
            where={"agentId": graph_id, "userId": user_id}
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
        async with locked_transaction(f"user_trx_{user_id}"):
            store_listing_version = (
                await prisma.models.StoreListingVersion.prisma().find_unique(
                    where={"id": store_listing_version_id}, include={"Agent": True}
                )
            )
            if not store_listing_version or not store_listing_version.Agent:
                logger.warning(
                    f"Store listing version not found: {store_listing_version_id}"
                )
                raise store_exceptions.AgentNotFoundError(
                    f"Store listing version {store_listing_version_id} not found or invalid"
                )

            graph = store_listing_version.Agent
            if graph.userId == user_id:
                logger.warning(
                    f"User #{user_id} attempted to add their own agent to their library"
                )
                raise store_exceptions.DatabaseError("Cannot add own agent to library")

            # Check if user already has this agent
            existing_library_agent = (
                await prisma.models.LibraryAgent.prisma().find_first(
                    where={
                        "userId": user_id,
                        "agentId": graph.id,
                        "agentVersion": graph.version,
                    },
                    include=library_agent_include(user_id),
                )
            )
            if existing_library_agent:
                if existing_library_agent.isDeleted:
                    # Even if agent exists it needs to be marked as not deleted
                    await set_is_deleted_for_library_agent(
                        user_id, graph.id, graph.version, False
                    )
                else:
                    logger.debug(
                        f"User #{user_id} already has graph #{graph.id} "
                        "in their library"
                    )
                return library_model.LibraryAgent.from_db(existing_library_agent)

            # Create LibraryAgent entry
            added_agent = await prisma.models.LibraryAgent.prisma().create(
                data={
                    "userId": user_id,
                    "agentId": graph.id,
                    "agentVersion": graph.version,
                    "isCreatedByUser": False,
                },
                include=library_agent_include(user_id),
            )
            logger.debug(
                f"Added graph  #{graph.id} "
                f"for store listing #{store_listing_version.id} "
                f"to library for user #{user_id}"
            )
            return library_model.LibraryAgent.from_db(added_agent)

    except store_exceptions.AgentNotFoundError:
        # Reraise for external handling.
        raise
    except prisma.errors.PrismaError as e:
        logger.error(f"Database error adding agent to library: {e}")
        raise store_exceptions.DatabaseError("Failed to add agent to library") from e


async def set_is_deleted_for_library_agent(
    user_id: str, agent_id: str, agent_version: int, is_deleted: bool
) -> None:
    """
    Changes the isDeleted flag for a library agent.

    Args:
        user_id: The user's library from which the agent is being removed.
        agent_id: The ID of the agent to remove.
        agent_version: The version of the agent to remove.
        is_deleted: Whether the agent is being marked as deleted.

    Raises:
        DatabaseError: If there's an issue updating the Library
    """
    logger.debug(
        f"Setting isDeleted={is_deleted} for agent {agent_id} v{agent_version} "
        f"in library for user {user_id}"
    )
    try:
        logger.warning(
            f"Setting isDeleted={is_deleted} for agent {agent_id} v{agent_version} in library for user {user_id}"
        )
        count = await prisma.models.LibraryAgent.prisma().update_many(
            where={
                "userId": user_id,
                "agentId": agent_id,
                "agentVersion": agent_version,
            },
            data={"isDeleted": is_deleted},
        )
        logger.warning(f"Updated {count} isDeleted library agents")
    except prisma.errors.PrismaError as e:
        logger.error(f"Database error setting agent isDeleted: {e}")
        raise store_exceptions.DatabaseError(
            "Failed to set agent isDeleted in library"
        ) from e


##############################################
########### Presets DB Functions #############
##############################################


async def get_presets(
    user_id: str, page: int, page_size: int
) -> library_model.LibraryAgentPresetResponse:
    """
    Retrieves a paginated list of AgentPresets for the specified user.

    Args:
        user_id: The user ID whose presets are being retrieved.
        page: The current page index (0-based or 1-based, clarify in your domain).
        page_size: Number of items to retrieve per page.

    Returns:
        A LibraryAgentPresetResponse containing a list of presets and pagination info.

    Raises:
        DatabaseError: If there's a database error during the operation.
    """
    logger.debug(
        f"Fetching presets for user #{user_id}, page={page}, page_size={page_size}"
    )

    if page < 0 or page_size < 1:
        logger.warning(
            "Invalid pagination input: page=%d, page_size=%d", page, page_size
        )
        raise store_exceptions.DatabaseError("Invalid pagination parameters")

    try:
        presets_records = await prisma.models.AgentPreset.prisma().find_many(
            where={"userId": user_id},
            skip=page * page_size,
            take=page_size,
        )
        total_items = await prisma.models.AgentPreset.prisma().count(
            where={"userId": user_id}
        )
        total_pages = (total_items + page_size - 1) // page_size

        presets = [
            library_model.LibraryAgentPreset.from_db(preset)
            for preset in presets_records
        ]

        return library_model.LibraryAgentPresetResponse(
            presets=presets,
            pagination=backend.server.model.Pagination(
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
            include={"InputPresets": True},
        )
        if not preset or preset.userId != user_id:
            return None
        return library_model.LibraryAgentPreset.from_db(preset)
    except prisma.errors.PrismaError as e:
        logger.error(f"Database error getting preset: {e}")
        raise store_exceptions.DatabaseError("Failed to fetch preset") from e


async def upsert_preset(
    user_id: str,
    preset: library_model.CreateLibraryAgentPresetRequest,
    preset_id: Optional[str] = None,
) -> library_model.LibraryAgentPreset:
    """
    Creates or updates an AgentPreset for a user.

    Args:
        user_id: The ID of the user creating/updating the preset.
        preset: The preset data used for creation or update.
        preset_id: An optional preset ID to update; if None, a new preset is created.

    Returns:
        The newly created or updated LibraryAgentPreset.

    Raises:
        DatabaseError: If there's a database error in creating or updating the preset.
        ValueError: If attempting to update a non-existent preset.
    """
    logger.debug(
        f"Upserting preset #{preset_id} ({repr(preset.name)}) for user #{user_id}",
    )
    try:
        if preset_id:
            # Update existing preset
            updated = await prisma.models.AgentPreset.prisma().update(
                where={"id": preset_id},
                data={
                    "name": preset.name,
                    "description": preset.description,
                    "isActive": preset.is_active,
                    "InputPresets": {
                        "create": [
                            {"name": name, "data": prisma.fields.Json(data)}
                            for name, data in preset.inputs.items()
                        ]
                    },
                },
                include={"InputPresets": True},
            )
            if not updated:
                raise ValueError(f"AgentPreset #{preset_id} not found")
            return library_model.LibraryAgentPreset.from_db(updated)
        else:
            # Create new preset
            new_preset = await prisma.models.AgentPreset.prisma().create(
                data={
                    "userId": user_id,
                    "name": preset.name,
                    "description": preset.description,
                    "agentId": preset.agent_id,
                    "agentVersion": preset.agent_version,
                    "isActive": preset.is_active,
                    "InputPresets": {
                        "create": [
                            {"name": name, "data": prisma.fields.Json(data)}
                            for name, data in preset.inputs.items()
                        ]
                    },
                },
                include={"InputPresets": True},
            )
        return library_model.LibraryAgentPreset.from_db(new_preset)
    except prisma.errors.PrismaError as e:
        logger.error(f"Database error upserting preset: {e}")
        raise store_exceptions.DatabaseError("Failed to create preset") from e


async def delete_preset(user_id: str, preset_id: str) -> None:
    """
    Soft-deletes a preset by marking it as isDeleted = True.

    Args:
        user_id: The user that owns the preset.
        preset_id: The ID of the preset to delete.

    Raises:
        DatabaseError: If there's a database error during deletion.
    """
    logger.info(f"Deleting preset {preset_id} for user {user_id}")
    try:
        await prisma.models.AgentPreset.prisma().update_many(
            where={"id": preset_id, "userId": user_id},
            data={"isDeleted": True},
        )
    except prisma.errors.PrismaError as e:
        logger.error(f"Database error deleting preset: {e}")
        raise store_exceptions.DatabaseError("Failed to delete preset") from e
