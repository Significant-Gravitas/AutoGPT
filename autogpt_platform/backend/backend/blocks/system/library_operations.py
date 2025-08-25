import logging
from typing import Any

from pydantic import BaseModel

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import SchemaField
from backend.util.clients import get_database_manager_async_client

logger = logging.getLogger(__name__)


# Duplicate pydantic models for store data so we don't accidently change the data shape in the blocks unintentionally when editing the backend
class LibraryAgent(BaseModel):
    """Model representing an agent in the user's library."""

    library_agent_id: str = ""
    agent_id: str = ""
    agent_version: int = 0
    agent_name: str = ""
    description: str = ""
    creator: str = ""
    is_archived: bool = False
    categories: list[str] = []


class AddToLibraryFromStoreBlock(Block):
    """
    Block that adds an agent from the store to the user's library.
    This enables users to easily import agents from the marketplace into their personal collection.
    """

    class Input(BlockSchema):
        store_listing_version_id: str = SchemaField(
            description="The ID of the store listing version to add to library"
        )
        agent_name: str | None = SchemaField(
            description="Optional custom name for the agent in your library",
            default=None,
        )

    class Output(BlockSchema):
        success: bool = SchemaField(
            description="Whether the agent was successfully added to library"
        )
        library_agent_id: str = SchemaField(
            description="The ID of the library agent entry"
        )
        agent_id: str = SchemaField(description="The ID of the agent graph")
        agent_version: int = SchemaField(
            description="The version number of the agent graph"
        )
        agent_name: str = SchemaField(description="The name of the agent")
        message: str = SchemaField(description="Success or error message")

    def __init__(self):
        super().__init__(
            id="2602a7b1-3f4d-4e5f-9c8b-1a2b3c4d5e6f",
            description="Add an agent from the store to your personal library",
            categories={BlockCategory.BASIC},
            input_schema=AddToLibraryFromStoreBlock.Input,
            output_schema=AddToLibraryFromStoreBlock.Output,
            test_input={
                "store_listing_version_id": "test-listing-id",
                "agent_name": "My Custom Agent",
            },
            test_output=[
                ("success", True),
                ("library_agent_id", "test-library-id"),
                ("agent_id", "test-agent-id"),
                ("agent_version", 1),
                ("agent_name", "Test Agent"),
                ("message", "Agent successfully added to library"),
            ],
            test_mock={
                "_add_to_library": lambda *_, **__: LibraryAgent(
                    library_agent_id="test-library-id",
                    agent_id="test-agent-id",
                    agent_version=1,
                    agent_name="Test Agent",
                )
            },
        )

    async def run(
        self,
        input_data: Input,
        *,
        user_id: str,
        **kwargs,
    ) -> BlockOutput:
        library_agent = await self._add_to_library(
            user_id=user_id,
            store_listing_version_id=input_data.store_listing_version_id,
            custom_name=input_data.agent_name,
        )

        yield "success", True
        yield "library_agent_id", library_agent.library_agent_id
        yield "agent_id", library_agent.agent_id
        yield "agent_version", library_agent.agent_version
        yield "agent_name", library_agent.agent_name
        yield "message", "Agent successfully added to library"

    async def _add_to_library(
        self,
        user_id: str,
        store_listing_version_id: str,
        custom_name: str | None = None,
    ) -> LibraryAgent:
        """
        Add a store agent to the user's library using the existing library database function.
        """
        library_agent = (
            await get_database_manager_async_client().add_store_agent_to_library(
                store_listing_version_id=store_listing_version_id, user_id=user_id
            )
        )

        # If custom name is provided, we could update the library agent name here
        # For now, we'll just return the agent info
        agent_name = custom_name if custom_name else library_agent.name

        return LibraryAgent(
            library_agent_id=library_agent.id,
            agent_id=library_agent.graph_id,
            agent_version=library_agent.graph_version,
            agent_name=agent_name,
        )


class ListLibraryAgentsBlock(Block):
    """
    Block that lists all agents in the user's library.
    """

    class Input(BlockSchema):
        search_query: str | None = SchemaField(
            description="Optional search query to filter agents", default=None
        )
        limit: int = SchemaField(
            description="Maximum number of agents to return", default=50, ge=1, le=100
        )
        page: int = SchemaField(
            description="Page number for pagination", default=1, ge=1
        )

    class Output(BlockSchema):
        agents: list[LibraryAgent] = SchemaField(
            description="List of agents in the library",
            default_factory=list,
        )
        agent: LibraryAgent = SchemaField(
            description="Individual library agent (yielded for each agent)"
        )
        total_count: int = SchemaField(
            description="Total number of agents in library", default=0
        )
        page: int = SchemaField(description="Current page number", default=1)
        total_pages: int = SchemaField(description="Total number of pages", default=1)

    def __init__(self):
        super().__init__(
            id="082602d3-a74d-4600-9e9c-15b3af7eae98",
            description="List all agents in your personal library",
            categories={BlockCategory.BASIC, BlockCategory.DATA},
            input_schema=ListLibraryAgentsBlock.Input,
            output_schema=ListLibraryAgentsBlock.Output,
            test_input={
                "search_query": None,
                "limit": 10,
                "page": 1,
            },
            test_output=[
                (
                    "agents",
                    [
                        LibraryAgent(
                            library_agent_id="test-lib-id",
                            agent_id="test-agent-id",
                            agent_version=1,
                            agent_name="Test Library Agent",
                            description="A test agent in library",
                            creator="Test User",
                        ),
                    ],
                ),
                ("total_count", 1),
                ("page", 1),
                ("total_pages", 1),
                (
                    "agent",
                    LibraryAgent(
                        library_agent_id="test-lib-id",
                        agent_id="test-agent-id",
                        agent_version=1,
                        agent_name="Test Library Agent",
                        description="A test agent in library",
                        creator="Test User",
                    ),
                ),
            ],
            test_mock={
                "_list_library_agents": lambda *_, **__: {
                    "agents": [
                        LibraryAgent(
                            library_agent_id="test-lib-id",
                            agent_id="test-agent-id",
                            agent_version=1,
                            agent_name="Test Library Agent",
                            description="A test agent in library",
                            creator="Test User",
                        )
                    ],
                    "total": 1,
                    "page": 1,
                    "total_pages": 1,
                }
            },
        )

    async def run(
        self,
        input_data: Input,
        *,
        user_id: str,
        **kwargs,
    ) -> BlockOutput:
        result = await self._list_library_agents(
            user_id=user_id,
            search_query=input_data.search_query,
            limit=input_data.limit,
            page=input_data.page,
        )

        agents = result["agents"]

        yield "agents", agents
        yield "total_count", result["total"]
        yield "page", result["page"]
        yield "total_pages", result["total_pages"]

        # Yield each agent individually for better graph connectivity
        for agent in agents:
            yield "agent", agent

    async def _list_library_agents(
        self,
        user_id: str,
        search_query: str | None = None,
        limit: int = 50,
        page: int = 1,
    ) -> dict[str, Any]:
        """
        List agents in the user's library using the database client.
        """
        result = await get_database_manager_async_client().list_library_agents(
            user_id=user_id,
            search_term=search_query,
            page=page,
            page_size=limit,
        )

        agents = [
            LibraryAgent(
                library_agent_id=agent.id,
                agent_id=agent.graph_id,
                agent_version=agent.graph_version,
                agent_name=agent.name,
                description=getattr(agent, "description", ""),
                creator=getattr(agent, "creator", ""),
                is_archived=getattr(agent, "is_archived", False),
                categories=getattr(agent, "categories", []),
            )
            for agent in result.agents
        ]

        return {
            "agents": agents,
            "total": result.pagination.total_items,
            "page": result.pagination.current_page,
            "total_pages": result.pagination.total_pages,
        }
