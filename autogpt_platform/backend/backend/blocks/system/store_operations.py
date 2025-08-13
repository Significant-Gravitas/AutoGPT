import logging
from typing import Any, Literal, Optional

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import SchemaField
from backend.server.v2.library import db as library_db
from backend.server.v2.store import db as store_db
from backend.server.v2.store import exceptions as store_exceptions

logger = logging.getLogger(__name__)

StoreOperation = Literal[
    "add_to_library", "get_agent_details", "list_store_agents", "search_store"
]


class AddToLibraryFromStoreBlock(Block):
    """
    Block that adds an agent from the store to the user's library.
    This enables users to easily import agents from the marketplace into their personal collection.
    """

    class Input(BlockSchema):
        store_listing_version_id: str = SchemaField(
            description="The ID of the store listing version to add to library"
        )
        agent_name: Optional[str] = SchemaField(
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
                "_add_to_library": lambda *_, **__: {
                    "library_agent_id": "test-library-id",
                    "agent_id": "test-agent-id",
                    "agent_version": 1,
                    "agent_name": "Test Agent",
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
        try:
            library_agent = await self._add_to_library(
                user_id=user_id,
                store_listing_version_id=input_data.store_listing_version_id,
                custom_name=input_data.agent_name,
            )

            yield "success", True
            yield "library_agent_id", library_agent["library_agent_id"]
            yield "agent_id", library_agent["agent_id"]
            yield "agent_version", library_agent["agent_version"]
            yield "agent_name", library_agent["agent_name"]
            yield "message", "Agent successfully added to library"

        except store_exceptions.AgentNotFoundError as e:
            logger.warning(f"Agent not found: {str(e)}")
            yield "success", False
            yield "library_agent_id", ""
            yield "agent_id", ""
            yield "agent_version", 0
            yield "agent_name", ""
            yield "message", f"Agent not found: {str(e)}"
        except Exception as e:
            logger.error(f"Failed to add agent to library: {str(e)}")
            yield "success", False
            yield "library_agent_id", ""
            yield "agent_id", ""
            yield "agent_version", 0
            yield "agent_name", ""
            yield "message", f"Failed to add agent to library: {str(e)}"

    async def _add_to_library(
        self,
        user_id: str,
        store_listing_version_id: str,
        custom_name: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Add a store agent to the user's library using the existing library database function.
        """
        library_agent = await library_db.add_store_agent_to_library(
            store_listing_version_id=store_listing_version_id, user_id=user_id
        )

        # If custom name is provided, we could update the library agent name here
        # For now, we'll just return the agent info
        agent_name = custom_name if custom_name else library_agent.name

        return {
            "library_agent_id": library_agent.id,
            "agent_id": library_agent.graph_id,
            "agent_version": library_agent.graph_version,
            "agent_name": agent_name,
        }


class GetStoreAgentDetailsBlock(Block):
    """
    Block that retrieves detailed information about an agent from the store.
    """

    class Input(BlockSchema):
        slug: str = SchemaField(description="The slug identifier of the store agent")
        version: Optional[str] = SchemaField(
            description="Specific version to retrieve (optional, defaults to latest)",
            default=None,
        )

    class Output(BlockSchema):
        found: bool = SchemaField(
            description="Whether the agent was found in the store"
        )
        store_listing_version_id: str = SchemaField(
            description="The store listing version ID"
        )
        agent_name: str = SchemaField(description="Name of the agent")
        description: str = SchemaField(description="Description of the agent")
        creator: str = SchemaField(description="Creator of the agent")
        categories: list[str] = SchemaField(
            description="Categories the agent belongs to", default_factory=list
        )
        runs: int = SchemaField(
            description="Number of times the agent has been run", default=0
        )
        rating: float = SchemaField(
            description="Average rating of the agent", default=0.0
        )

    def __init__(self):
        super().__init__(
            id="3712b8c2-4f5e-5e6f-ad9c-2b3c4d5e6f7a",
            description="Get detailed information about an agent from the store",
            categories={BlockCategory.BASIC, BlockCategory.DATA},
            input_schema=GetStoreAgentDetailsBlock.Input,
            output_schema=GetStoreAgentDetailsBlock.Output,
            test_input={"slug": "test-agent-slug", "version": None},
            test_output=[
                ("found", True),
                ("store_listing_version_id", "test-listing-id"),
                ("agent_name", "Test Agent"),
                ("description", "A test agent"),
                ("creator", "Test Creator"),
                ("categories", ["productivity", "automation"]),
                ("runs", 100),
                ("rating", 4.5),
            ],
            test_mock={
                "_get_agent_details": lambda *_, **__: {
                    "found": True,
                    "store_listing_version_id": "test-listing-id",
                    "agent_name": "Test Agent",
                    "description": "A test agent",
                    "creator": "Test Creator",
                    "categories": ["productivity", "automation"],
                    "runs": 100,
                    "rating": 4.5,
                }
            },
            static_output=True,
        )

    async def run(
        self,
        input_data: Input,
        **kwargs,
    ) -> BlockOutput:
        try:
            details = await self._get_agent_details(
                slug=input_data.slug, version=input_data.version
            )

            yield "found", details["found"]
            yield "store_listing_version_id", details.get(
                "store_listing_version_id", ""
            )
            yield "agent_name", details.get("agent_name", "")
            yield "description", details.get("description", "")
            yield "creator", details.get("creator", "")
            yield "categories", details.get("categories", [])
            yield "runs", details.get("runs", 0)
            yield "rating", details.get("rating", 0.0)

        except Exception as e:
            logger.error(f"Failed to get agent details: {str(e)}")
            yield "found", False
            yield "store_listing_version_id", ""
            yield "agent_name", ""
            yield "description", ""
            yield "creator", ""
            yield "categories", []
            yield "runs", 0
            yield "rating", 0.0

    async def _get_agent_details(
        self, slug: str, version: Optional[str] = None
    ) -> dict[str, Any]:
        """
        Retrieve detailed information about a store agent.
        """
        try:
            if version:
                # Get by specific version ID
                agent_details = await store_db.get_store_agent_by_version_id(version)
            else:
                # Parse slug to get username and agent name
                # Slug format is typically "username/agent-name"
                parts = slug.split("/")
                if len(parts) != 2:
                    raise ValueError(
                        f"Invalid slug format: {slug}. Expected format: username/agent-name"
                    )

                username, agent_name = parts
                agent_details = await store_db.get_store_agent_details(
                    username, agent_name
                )

            return {
                "found": True,
                "store_listing_version_id": agent_details.store_listing_version_id,
                "agent_name": agent_details.agent_name,
                "description": agent_details.description,
                "creator": agent_details.creator,
                "categories": (
                    agent_details.categories
                    if hasattr(agent_details, "categories")
                    else []
                ),
                "runs": agent_details.runs,
                "rating": agent_details.rating,
            }
        except store_exceptions.AgentNotFoundError:
            return {
                "found": False,
                "store_listing_version_id": "",
                "agent_name": "",
                "description": "",
                "creator": "",
                "categories": [],
                "runs": 0,
                "rating": 0.0,
            }


class SearchStoreAgentsBlock(Block):
    """
    Block that searches for agents in the store based on various criteria.
    """

    class Input(BlockSchema):
        query: Optional[str] = SchemaField(
            description="Search query to find agents", default=None
        )
        category: Optional[str] = SchemaField(
            description="Filter by category", default=None
        )
        sort_by: Literal["rating", "runs", "name", "recent"] = SchemaField(
            description="How to sort the results", default="rating"
        )
        limit: int = SchemaField(
            description="Maximum number of results to return", default=10, ge=1, le=100
        )

    class Output(BlockSchema):
        agents: list[dict[str, Any]] = SchemaField(
            description="List of agents matching the search criteria",
            default_factory=list,
        )
        total_count: int = SchemaField(
            description="Total number of agents found", default=0
        )

    def __init__(self):
        super().__init__(
            id="4823c9d3-5f6e-6e7f-be9d-3c4d5e6f7a8b",
            description="Search for agents in the store",
            categories={BlockCategory.BASIC, BlockCategory.DATA},
            input_schema=SearchStoreAgentsBlock.Input,
            output_schema=SearchStoreAgentsBlock.Output,
            test_input={
                "query": "productivity",
                "category": None,
                "sort_by": "rating",
                "limit": 10,
            },
            test_output=[
                (
                    "agents",
                    [
                        {
                            "slug": "test-agent",
                            "name": "Test Agent",
                            "description": "A test agent",
                            "creator": "Test Creator",
                            "rating": 4.5,
                            "runs": 100,
                        }
                    ],
                ),
                ("total_count", 1),
            ],
            test_mock={
                "_search_agents": lambda *_, **__: {
                    "agents": [
                        {
                            "slug": "test-agent",
                            "name": "Test Agent",
                            "description": "A test agent",
                            "creator": "Test Creator",
                            "rating": 4.5,
                            "runs": 100,
                        }
                    ],
                    "total_count": 1,
                }
            },
            static_output=True,
        )

    async def run(
        self,
        input_data: Input,
        **kwargs,
    ) -> BlockOutput:
        try:
            results = await self._search_agents(
                query=input_data.query,
                category=input_data.category,
                sort_by=input_data.sort_by,
                limit=input_data.limit,
            )

            yield "agents", results.get("agents", [])
            yield "total_count", results.get("total_count", 0)

        except Exception as e:
            logger.error(f"Failed to search store agents: {str(e)}")
            yield "agents", []
            yield "total_count", 0

    async def _search_agents(
        self,
        query: Optional[str] = None,
        category: Optional[str] = None,
        sort_by: str = "rating",
        limit: int = 10,
    ) -> dict[str, Any]:
        """
        Search for agents in the store using the existing store database function.
        """
        # Map our sort_by to the store's sorted_by parameter
        sorted_by_map = {
            "rating": "most_popular",
            "runs": "most_runs",
            "name": "alphabetical",
            "recent": "recently_updated",
        }

        result = await store_db.get_store_agents(
            featured=False,
            creator=None,
            sorted_by=sorted_by_map.get(sort_by, "most_popular"),
            search_query=query,
            category=category,
            page=1,
            page_size=limit,
        )

        agents = []
        for agent in result.agents:
            agents.append(
                {
                    "slug": agent.slug,
                    "name": agent.agent_name,
                    "description": agent.description,
                    "creator": agent.creator,
                    "rating": agent.rating,
                    "runs": agent.runs,
                }
            )

        return {"agents": agents, "total_count": result.pagination.total_items}
