import logging
from typing import List
import typing

import prisma.errors
import prisma.models
import prisma.types

import backend.data.graph
import backend.data.includes
import backend.server.v2.library.model
import backend.server.v2.store.exceptions

logger = logging.getLogger(__name__)


async def get_library_agents(
    user_id: str,
    limit: int = 20,
    offset: int = 0,
) -> List[backend.server.v2.library.model.LibraryAgent]:
    """
    Returns paginated agents (AgentGraph) that belong to the user and agents in their library (UserAgent table)
    """
    logger.debug(f"Getting library agents for user {user_id} with limit {limit} offset {offset}")

    try:
        # Get agents created by user with nodes and links
        user_created = await prisma.models.AgentGraph.prisma().find_many(
            where=prisma.types.AgentGraphWhereInput(userId=user_id, isActive=True),
            include=backend.data.includes.AGENT_GRAPH_INCLUDE,
            skip=offset,
            take=limit,
        )

        # Get agents in user's library with nodes and links
        library_agents = await prisma.models.UserAgent.prisma().find_many(
            where=prisma.types.UserAgentWhereInput(
                userId=user_id, isDeleted=False, isArchived=False
            ),
            include={
                "Agent": {
                    "include": {
                        "AgentNodes": {
                            "include": {
                                "Input": True,
                                "Output": True,
                                "Webhook": True,
                                "AgentBlock": True,
                            }
                        }
                    }
                }
            },
            skip=offset,
            take=limit,
        )

        # Convert to Graph models first
        graphs = []

        # Add user created agents
        for agent in user_created:
            try:
                graphs.append(backend.data.graph.GraphModel.from_db(agent))
            except Exception as e:
                logger.error(f"Error processing user created agent {agent.id}: {e}")
                continue

        # Add library agents
        for agent in library_agents:
            if agent.Agent:
                try:
                    graphs.append(backend.data.graph.GraphModel.from_db(agent.Agent))
                except Exception as e:
                    logger.error(f"Error processing library agent {agent.agentId}: {e}")
                    continue

        # Convert Graph models to LibraryAgent models
        result = []
        for graph in graphs:
            result.append(
                backend.server.v2.library.model.LibraryAgent(
                    id=graph.id,
                    version=graph.version,
                    is_active=graph.is_active,
                    name=graph.name,
                    description=graph.description,
                    isCreatedByUser=any(a.id == graph.id for a in user_created),
                    input_schema=graph.input_schema,
                    output_schema=graph.output_schema,
                )
            )

        logger.debug(f"Found {len(result)} library agents")
        return result

    except prisma.errors.PrismaError as e:
        logger.error(f"Database error getting library agents: {str(e)}")
        raise backend.server.v2.store.exceptions.DatabaseError(
            "Failed to fetch library agents"
        ) from e

async def search_library_agents(
    user_id: str,
    search_term: str,
    sort_by: typing.Literal["most_recent", "highest_runtime", "most_runs", "alphabetical", "last_modified"],
    limit: int = 20,
    offset: int = 0,
) -> List[backend.server.v2.library.model.LibraryAgent]:
    """
    Searches paginated agents (AgentGraph) that belong to the user and agents in their library (UserAgent table)
    based on name or description containing the search term
    """
    logger.debug(f"Searching library agents for user {user_id} with term '{search_term}', limit {limit} offset {offset}")

    try:
        # Get sort field
        sort_order = "desc" if sort_by in ["most_recent", "highest_runtime", "most_runs", "last_modified"] else "asc"

        sort_field = {
            "most_recent": "createdAt",
            "last_modified": "updatedAt",
            "highest_runtime": "totalRuntime",
            "most_runs": "runCount",
            "alphabetical": "name"
        }.get(sort_by, "updatedAt")

        # Get user created agents matching search
        user_created = await prisma.models.AgentGraph.prisma().find_many(
            where=prisma.types.AgentGraphWhereInput(
                userId=user_id,
                isActive=True,
                OR=[
                    {"name": {"contains": search_term, "mode": "insensitive"}},
                    {"description": {"contains": search_term, "mode": "insensitive"}}
                ]
            ),
            include=backend.data.includes.AGENT_GRAPH_INCLUDE,
            order={sort_field: sort_order},
            skip=offset,
            take=limit,
        )

        # Get library agents matching search
        library_agents = await prisma.models.UserAgent.prisma().find_many(
            where=prisma.types.UserAgentWhereInput(
                userId=user_id,
                isDeleted=False,
                isArchived=False,
                Agent={
                    "is": {
                        "OR": [
                            {"name": {"contains": search_term, "mode": "insensitive"}},
                            {"description": {"contains": search_term, "mode": "insensitive"}}
                        ]
                    }
                }
            ),
            include={
                "Agent": {
                    "include": {
                        "AgentNodes": {
                            "include": {
                                "Input": True,
                                "Output": True,
                                "Webhook": True,
                                "AgentBlock": True,
                            }
                        }
                    }
                }
            },
            skip=offset,
            take=limit,
        )

        # Convert to Graph models
        graphs = []

        for agent in user_created:
            try:
                graphs.append(backend.data.graph.GraphModel.from_db(agent))
            except Exception as e:
                logger.error(f"Error processing searched user agent {agent.id}: {e}")
                continue

        for agent in library_agents:
            if agent.Agent:
                try:
                    graphs.append(backend.data.graph.GraphModel.from_db(agent.Agent))
                except Exception as e:
                    logger.error(f"Error processing searched library agent {agent.agentId}: {e}")
                    continue

        # Convert to LibraryAgent models
        result = []
        for graph in graphs:
            result.append(
                backend.server.v2.library.model.LibraryAgent(
                    id=graph.id,
                    version=graph.version,
                    is_active=graph.is_active,
                    name=graph.name,
                    description=graph.description,
                    isCreatedByUser=any(a.id == graph.id for a in user_created),
                    input_schema=graph.input_schema,
                    output_schema=graph.output_schema,
                )
            )

        logger.debug(f"Found {len(result)} library agents matching search")
        return result

    except prisma.errors.PrismaError as e:
        logger.error(f"Database error searching library agents: {str(e)}")
        raise backend.server.v2.store.exceptions.DatabaseError(
            "Failed to search library agents"
        ) from e

async def add_agent_to_library(store_listing_version_id: str, user_id: str) -> None:
    """
    Finds the agent from the store listing version and adds it to the user's library (UserAgent table)
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
        existing_user_agent = await prisma.models.UserAgent.prisma().find_first(
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

        # Create UserAgent entry
        await prisma.models.UserAgent.prisma().create(
            data=prisma.types.UserAgentCreateInput(
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
