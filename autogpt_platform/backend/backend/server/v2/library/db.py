import json
import logging

import fastapi
import prisma.errors
import prisma.models
import prisma.types

import backend.data.includes
import backend.server.model
import backend.server.v2.library.model
import backend.server.v2.store.exceptions
import backend.server.v2.store.image_gen
import backend.server.v2.store.media

logger = logging.getLogger(__name__)


async def get_library_agents(
    user_id: str,
    search_term: str | None = None,
    sort_by: (
        backend.server.v2.library.model.LibraryAgentFilter | None
    ) = backend.server.v2.library.model.LibraryAgentFilter.UPDATED_AT,
    page: int = 1,
    page_size: int = 50,
) -> backend.server.v2.library.model.LibraryAgentResponse:
    logger.debug(
        "Fetching library agents for user_id=%s search_term=%s sort_by=%s page=%d",
        user_id,
        search_term,
        sort_by,
        page,
    )

    if search_term and len(search_term.strip()) > 100:
        logger.warning("Search term too long: %s", search_term)
        raise backend.server.v2.store.exceptions.DatabaseError(
            "Search term is too long."
        )

    prisma.types.AgentGraphRelationFilter

    where_clause = prisma.types.LibraryAgentWhereInput(
        userId=user_id,
        isDeleted=False,
        isArchived=False,
    )

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

    try:
        order_by: prisma.types.LibraryAgentOrderByInput = {"updatedAt": "desc"}
        if sort_by:
            if sort_by == backend.server.v2.library.model.LibraryAgentFilter.CREATED_AT:
                order_by = {"createdAt": "desc"}
            elif (
                sort_by == backend.server.v2.library.model.LibraryAgentFilter.UPDATED_AT
            ):
                order_by = {"updatedAt": "desc"}
            elif (
                sort_by
                == backend.server.v2.library.model.LibraryAgentFilter.IS_FAVOURITE
            ):
                order_by = {"isFavorite": "desc"}
            elif (
                sort_by
                == backend.server.v2.library.model.LibraryAgentFilter.CAN_ACCESS_GRAPH
            ):
                order_by = {"isCreatedByUser": "desc"}

        library_agents = await prisma.models.LibraryAgent.prisma().find_many(
            where=where_clause,
            include={
                "Agent": {
                    "include": {
                        **backend.data.includes.AGENT_GRAPH_INCLUDE,
                        "AgentGraphExecution": {"where": {"userId": user_id}},
                    },
                },
                "Creator": True,
            },
            order=order_by,
            skip=(page - 1) * page_size,
            take=page_size,
        )
        logger.info("Retrieved %s agents for user_id=%s.", len(library_agents), user_id)

        agent_count = await prisma.models.LibraryAgent.prisma().count(
            where=where_clause
        )

        return backend.server.v2.library.model.LibraryAgentResponse(
            agents=[
                backend.server.v2.library.model.LibraryAgent.from_db(agent)
                for agent in library_agents
            ],
            pagination=backend.server.model.Pagination(
                total_items=agent_count,
                total_pages=(agent_count // page_size) + 1,
                current_page=page,
                page_size=page_size,
            ),
        )
    except prisma.errors.PrismaError as e:
        logger.error("Database error fetching library agents: %s", e)
        raise backend.server.v2.store.exceptions.DatabaseError(
            "Unable to fetch library agents."
        )


async def create_library_agent(
    agent_id: str, agent_version: int, user_id: str
) -> prisma.models.LibraryAgent:
    """
    Adds an agent to the user's library (LibraryAgent table)
    """
    logger.info(
        "Creating library agent for agent_id=%s agent_version=%s user_id=%s",
        agent_id,
        agent_version,
        user_id,
    )
    try:
        # Find the agent using the compound primary key

        agent = await prisma.models.AgentGraph.prisma().find_first(
            where={
                "id": agent_id,
                "version": agent_version,
            }
        )
    except prisma.errors.PrismaError as e:
        logger.error("Database error finding agent: %s", e)
        raise backend.server.v2.store.exceptions.DatabaseError(
            "Failed to find agent"
        ) from e

    try:
        if not agent:
            raise backend.server.v2.store.exceptions.AgentNotFoundError(
                f"Agent {agent_id} version {agent_version} not found"
            )
        try:
            # Use .jpeg here since we are generating JPEG images
            filename = f"agent_{agent_id}.jpeg"

            image_url = await backend.server.v2.store.media.check_media_exists(
                user_id, filename
            )

            if not image_url:
                # Generate agent image as JPEG
                image = await backend.server.v2.store.image_gen.generate_agent_image(
                    agent=agent
                )

                # Create UploadFile with the correct filename and content_type
                image_file = fastapi.UploadFile(
                    file=image,
                    filename=filename,
                )

                image_url = await backend.server.v2.store.media.upload_media(
                    user_id=user_id, file=image_file, use_file_name=True
                )
        except Exception as e:
            logger.error("Error generating agent image: %s", e)
            raise backend.server.v2.store.exceptions.DatabaseError(
                "Failed to generate agent image"
            ) from e
        # Ensure that we have the necessary data before proceeding.
        assert agent is not None, "Agent data is missing"
        assert image_url, "Image URL is required"
        assert user_id, "User ID is required"
        assert (
            hasattr(agent, "userId") and agent.userId
        ), "Agent must have a valid userId"

        library_agent = await prisma.models.LibraryAgent.prisma().create(
            data={
                "image_url": image_url,
                "isCreatedByUser": (user_id == agent.userId),
                "useGraphIsActiveVersion": True,
                "isDeleted": False,
                "isArchived": False,
                "User": {"connect": {"id": agent.userId}},
                "Agent": {
                    "connect": {
                        "graphVersionId": {"id": agent_id, "version": agent_version}
                    }
                },
                # "Creator": {"connect": {"userId": agent.userId}},
            }
        )
        return library_agent
    except prisma.errors.PrismaError as e:
        logger.error("Database error creating agent to library: %s", e)
        raise backend.server.v2.store.exceptions.DatabaseError(
            "Failed to create agent to library"
        ) from e


async def update_agent_version_in_library(
    user_id: str, agent_id: str, agent_version: int
) -> None:
    """
    Updates the agent version in the library
    """
    try:
        await prisma.models.LibraryAgent.prisma().update_many(
            where={
                "userId": user_id,
                "agentId": agent_id,
                "useGraphIsActiveVersion": True,
            },
            data=prisma.types.LibraryAgentUpdateInput(
                Agent=prisma.types.AgentGraphUpdateOneWithoutRelationsInput(
                    connect=prisma.types._AgentGraphCompoundPrimaryKey(
                        graphVersionId=prisma.types._AgentGraphCompoundPrimaryKeyInner(
                            id=agent_id,
                            version=agent_version,
                        )
                    ),
                ),
            ),
        )
    except prisma.errors.PrismaError as e:
        logger.error("Database error updating agent version in library: %s", e)
        raise backend.server.v2.store.exceptions.DatabaseError(
            "Failed to update agent version in library"
        ) from e


async def update_library_agent(
    library_agent_id: str,
    user_id: str,
    auto_update_version: bool = False,
    is_favorite: bool = False,
    is_archived: bool = False,
    is_deleted: bool = False,
) -> None:
    """
    Updates the library agent with the given fields
    """
    try:
        await prisma.models.LibraryAgent.prisma().update_many(
            where={"id": library_agent_id, "userId": user_id},
            data=prisma.types.LibraryAgentUpdateInput(
                useGraphIsActiveVersion=auto_update_version,
                isFavorite=is_favorite,
                isArchived=is_archived,
                isDeleted=is_deleted,
            ),
        )
    except prisma.errors.PrismaError as e:
        logger.error("Database error updating library agent: %s", e)
        raise backend.server.v2.store.exceptions.DatabaseError(
            "Failed to update library agent"
        ) from e


async def add_store_agent_to_library(
    store_listing_version_id: str, user_id: str
) -> None:
    """
    Finds the agent from the store listing version and adds it to the user's library (LibraryAgent table)
    if they don't already have it
    """
    logger.debug(
        "Adding agent from store listing version %s to library for user %s",
        store_listing_version_id,
        user_id,
    )

    try:
        # Get store listing version to find agent
        store_listing_version = (
            await prisma.models.StoreListingVersion.prisma().find_unique(
                where={"id": store_listing_version_id}, include={"Agent": True}
            )
        )

        if not store_listing_version or not store_listing_version.Agent:
            logger.warning(
                "Store listing version not found: %s", store_listing_version_id
            )
            raise backend.server.v2.store.exceptions.AgentNotFoundError(
                f"Store listing version {store_listing_version_id} not found"
            )

        # We need the agent object to be able to check if
        # the user_id is the same as the agent's user_id
        agent = store_listing_version.Agent

        if agent.userId == user_id:
            logger.warning(
                "User %s cannot add their own agent to their library", user_id
            )
            raise backend.server.v2.store.exceptions.DatabaseError(
                "Cannot add own agent to library"
            )

        # Check if user already has this agent
        existing_user_agent = await prisma.models.LibraryAgent.prisma().find_first(
            where={
                "userId": user_id,
                "agentId": agent.id,
                "agentVersion": agent.version,
            }
        )

        if existing_user_agent:
            logger.debug(
                "User %s already has agent %s in their library", user_id, agent.id
            )
            return

        # Create LibraryAgent entry
        await prisma.models.LibraryAgent.prisma().create(
            data=prisma.types.LibraryAgentCreateInput(
                userId=user_id,
                agentId=agent.id,
                agentVersion=agent.version,
                isCreatedByUser=False,
            )
        )
        logger.debug("Added agent %s to library for user %s", agent.id, user_id)

    except backend.server.v2.store.exceptions.AgentNotFoundError:
        raise
    except prisma.errors.PrismaError as e:
        logger.error("Database error adding agent to library: %s", e)
        raise backend.server.v2.store.exceptions.DatabaseError(
            "Failed to add agent to library"
        ) from e


##############################################
########### Presets DB Functions #############
##############################################


async def get_presets(
    user_id: str, page: int, page_size: int
) -> backend.server.v2.library.model.LibraryAgentPresetResponse:
    try:
        presets = await prisma.models.AgentPreset.prisma().find_many(
            where={"userId": user_id},
            skip=page * page_size,
            take=page_size,
        )

        total_items = await prisma.models.AgentPreset.prisma().count(
            where={"userId": user_id},
        )
        total_pages = (total_items + page_size - 1) // page_size

        presets = [
            backend.server.v2.library.model.LibraryAgentPreset.from_db(preset)
            for preset in presets
        ]

        return backend.server.v2.library.model.LibraryAgentPresetResponse(
            presets=presets,
            pagination=backend.server.model.Pagination(
                total_items=total_items,
                total_pages=total_pages,
                current_page=page,
                page_size=page_size,
            ),
        )

    except prisma.errors.PrismaError as e:
        logger.error("Database error getting presets: %s", e)
        raise backend.server.v2.store.exceptions.DatabaseError(
            "Failed to fetch presets"
        ) from e


async def get_preset(
    user_id: str, preset_id: str
) -> backend.server.v2.library.model.LibraryAgentPreset | None:
    try:
        preset = await prisma.models.AgentPreset.prisma().find_unique(
            where={"id": preset_id}, include={"InputPresets": True}
        )
        if not preset or preset.userId != user_id:
            return None
        return backend.server.v2.library.model.LibraryAgentPreset.from_db(preset)
    except prisma.errors.PrismaError as e:
        logger.error("Database error getting preset: %s", e)
        raise backend.server.v2.store.exceptions.DatabaseError(
            "Failed to fetch preset"
        ) from e


async def create_or_update_preset(
    user_id: str,
    preset: backend.server.v2.library.model.CreateLibraryAgentPresetRequest,
    preset_id: str | None = None,
) -> backend.server.v2.library.model.LibraryAgentPreset:
    try:
        if preset_id:
            # Update existing preset
            new_preset = await prisma.models.AgentPreset.prisma().update(
                where={"id": preset_id},
                data={
                    "name": preset.name,
                    "description": preset.description,
                    "isActive": preset.is_active,
                    "InputPresets": {
                        "create": [
                            {"name": name, "data": json.dumps(data)}
                            for name, data in preset.inputs.items()
                        ]
                    },
                },
                include={"InputPresets": True},
            )
            if not new_preset:
                raise ValueError(f"AgentPreset #{preset_id} not found")
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
                            {"name": name, "data": json.dumps(data)}
                            for name, data in preset.inputs.items()
                        ]
                    },
                },
                include={"InputPresets": True},
            )
        return backend.server.v2.library.model.LibraryAgentPreset.from_db(new_preset)
    except prisma.errors.PrismaError as e:
        logger.error("Database error creating preset: %s", e)
        raise backend.server.v2.store.exceptions.DatabaseError(
            "Failed to create preset"
        ) from e


async def delete_preset(user_id: str, preset_id: str) -> None:
    try:
        await prisma.models.AgentPreset.prisma().update_many(
            where={"id": preset_id, "userId": user_id},
            data={"isDeleted": True},
        )
    except prisma.errors.PrismaError as e:
        logger.error("Database error deleting preset: %s", e)
        raise backend.server.v2.store.exceptions.DatabaseError(
            "Failed to delete preset"
        ) from e


async def main():
    import time

    import backend.data.db

    await backend.data.db.connect()

    try:

        time.sleep(2)
        library_agents = await get_library_agents(
            "658bc98d-e647-419d-a9c9-c78e3fdbbaf2"
        )
        print(library_agents)
    finally:
        await backend.data.db.disconnect()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
