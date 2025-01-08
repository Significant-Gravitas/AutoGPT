import json
import logging
from typing import List

import prisma.errors
import prisma.models
import prisma.types

import backend.data.graph
import backend.data.includes
import backend.server.model
import backend.server.v2.library.model
import backend.server.v2.store.exceptions

logger = logging.getLogger(__name__)


async def get_library_agents(
    user_id: str,
    search_query: str | None = None,
) -> List[backend.server.v2.library.model.LibraryAgent]:
    """
    Returns all agents (AgentGraph) that belong to the user and all agents in their library (LibraryAgent table)
    """
    logger.debug(
        f"Getting library agents for user {user_id} with search query: {search_query}"
    )

    try:
        # Sanitize and validate search query by escaping special characters
        # Build where clause with sanitized inputs
        where_clause = prisma.types.LibraryAgentWhereInput(
            userId=user_id,
            isDeleted=False,
            isArchived=False,
            **(
                {
                    "OR": [
                        {
                            "Agent": {
                                "name": {
                                    "contains": search_query,
                                    "mode": "insensitive",
                                }
                            }
                        },
                        {
                            "Agent": {
                                "description": {
                                    "contains": search_query,
                                    "mode": "insensitive",
                                }
                            }
                        },
                    ]
                }
                if search_query
                else {}
            ),
        )

        # Get agents in user's library with nodes and links
        library_agents = await prisma.models.LibraryAgent.prisma().find_many(
            where=where_clause,
            include={
                "Agent": {
                    "include": {
                        "AgentNodes": {
                            "include": {
                                "Input": True,
                                "Output": True,
                            }
                        }
                    }
                },
                "AgentPreset": {"include": {"InputPresets": True}},
            },
            order=[{"updatedAt": "desc"}],
        )

        # Convert to Graph models first
        graphs = []
        # Add library agents
        for agent in library_agents:
            if agent.Agent:
                try:
                    graphs.append(backend.data.graph.GraphModel.from_db(agent.Agent))
                except Exception as e:
                    logger.error(f"Error processing library agent {agent.agentId}: {e}")
                    continue

        result = [
            backend.server.v2.library.model.LibraryAgent.from_db(agent)
            for agent in library_agents
        ]

        logger.debug(f"Found {len(result)} library agents")
        return result

    except prisma.errors.PrismaError as e:
        logger.error(f"Database error getting library agents: {str(e)}")
        raise backend.server.v2.store.exceptions.DatabaseError(
            "Failed to fetch library agents"
        ) from e


async def add_agent_to_library(store_listing_version_id: str, user_id: str) -> None:
    """
    Finds the agent from the store listing version and adds it to the user's library (LibraryAgent table)
    if they don't already have it
    """
    logger.debug(
        f"Adding agent from store listing version {store_listing_version_id} to library for user {user_id}"
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
                f"Store listing version not found: {store_listing_version_id}"
            )
            raise backend.server.v2.store.exceptions.AgentNotFoundError(
                f"Store listing version {store_listing_version_id} not found"
            )

        agent = store_listing_version.Agent

        if agent.userId == user_id:
            logger.warning(
                f"User {user_id} cannot add their own agent to their library"
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
                f"User {user_id} already has agent {agent.id} in their library"
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
        logger.debug(f"Added agent {agent.id} to library for user {user_id}")

    except backend.server.v2.store.exceptions.AgentNotFoundError:
        raise
    except prisma.errors.PrismaError as e:
        logger.error(f"Database error adding agent to library: {str(e)}")
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
        logger.error(f"Database error getting presets: {str(e)}")
        raise backend.server.v2.store.exceptions.DatabaseError(
            "Failed to fetch presets"
        ) from e


async def get_preset(
    user_id: str, preset_id: str
) -> backend.server.v2.library.model.LibraryAgentPreset | None:
    try:
        preset = await prisma.models.AgentPreset.prisma().find_unique(
            where={"id": preset_id, "userId": user_id}
        )
        if not preset:
            return None
        return backend.server.v2.library.model.LibraryAgentPreset.from_db(preset)
    except prisma.errors.PrismaError as e:
        logger.error(f"Database error getting preset: {str(e)}")
        raise backend.server.v2.store.exceptions.DatabaseError(
            "Failed to fetch preset"
        ) from e


async def create_or_update_preset(
    user_id: str,
    preset: backend.server.v2.library.model.CreateLibraryAgentPresetRequest,
    preset_id: str | None = None,
) -> backend.server.v2.library.model.LibraryAgentPreset:
    try:
        new_preset = await prisma.models.AgentPreset.prisma().upsert(
            where={
                "id": preset_id if preset_id else "",
            },
            data={
                "create": prisma.types.AgentPresetCreateInput(
                    userId=user_id,
                    name=preset.name,
                    description=preset.description,
                    agentId=preset.agent_id,
                    agentVersion=preset.agent_version,
                    Agent=prisma.types.AgentGraphUpdateOneWithoutRelationsInput(
                        connect=prisma.types.AgentGraphWhereUniqueInput(
                            id=preset.agent_id,
                            version=preset.agent_version,
                        ),
                    ),
                    isActive=preset.is_active,
                    InputPresets={
                        "create": [
                            {"name": name, "data": json.dumps(data)}
                            for name, data in preset.inputs.items()
                        ]
                    },
                ),
                "update": prisma.types.AgentPresetUpdateInput(
                    name=preset.name,
                    description=preset.description,
                    isActive=preset.is_active,
                ),
            },
        )
        return backend.server.v2.library.model.LibraryAgentPreset.from_db(new_preset)
    except prisma.errors.PrismaError as e:
        logger.error(f"Database error creating preset: {str(e)}")
        raise backend.server.v2.store.exceptions.DatabaseError(
            "Failed to create preset"
        ) from e


async def delete_preset(user_id: str, preset_id: str) -> None:
    try:
        await prisma.models.AgentPreset.prisma().update(
            where={"id": preset_id, "userId": user_id},
            data={"isDeleted": True},
        )
    except prisma.errors.PrismaError as e:
        logger.error(f"Database error deleting preset: {str(e)}")
        raise backend.server.v2.store.exceptions.DatabaseError(
            "Failed to delete preset"
        ) from e
