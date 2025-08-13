import logging
from typing import Literal, Optional

from pydantic import BaseModel

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import SchemaField
from backend.server.v2.store import exceptions as store_exceptions
from backend.util.clients import get_database_manager_async_client

logger = logging.getLogger(__name__)


# Pydantic models for store data
class StoreAgent(BaseModel):
    """Model representing a store agent."""

    slug: str
    name: str
    description: str
    creator: str
    rating: float = 0.0
    runs: int = 0
    categories: list[str] = []


class StoreAgentDetails(BaseModel):
    """Detailed information about a store agent."""

    found: bool
    store_listing_version_id: str = ""
    agent_name: str = ""
    description: str = ""
    creator: str = ""
    categories: list[str] = []
    runs: int = 0
    rating: float = 0.0


class GetStoreAgentDetailsBlock(Block):
    """
    Block that retrieves detailed information about an agent from the store.
    """

    class Input(BlockSchema):
        creator: str = SchemaField(description="The username of the agent creator")
        slug: str = SchemaField(description="The name of the agent")

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
                creator=input_data.creator, slug=input_data.slug
            )
            yield "found", details.found
            yield "store_listing_version_id", details.store_listing_version_id
            yield "agent_name", details.agent_name
            yield "description", details.description
            yield "creator", details.creator
            yield "categories", details.categories
            yield "runs", details.runs
            yield "rating", details.rating

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

    async def _get_agent_details(self, creator: str, slug: str) -> StoreAgentDetails:
        """
        Retrieve detailed information about a store agent.
        """
        try:
            # Get by specific version ID
            agent_details = (
                await get_database_manager_async_client().get_store_agent_details(
                    username=creator, agent_name=slug
                )
            )

            return StoreAgentDetails(
                found=True,
                store_listing_version_id=agent_details.store_listing_version_id,
                agent_name=agent_details.agent_name,
                description=agent_details.description,
                creator=agent_details.creator,
                categories=(
                    agent_details.categories
                    if hasattr(agent_details, "categories")
                    else []
                ),
                runs=agent_details.runs,
                rating=agent_details.rating,
            )
        except store_exceptions.AgentNotFoundError:
            return StoreAgentDetails(
                found=False,
                store_listing_version_id="",
                agent_name="",
                description="",
                creator="",
                categories=[],
                runs=0,
                rating=0.0,
            )


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
        agents: list[StoreAgent] = SchemaField(
            description="List of agents matching the search criteria",
            default_factory=list,
        )
        agent: StoreAgent = SchemaField(description="Basic information of the agent")

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
                            "store_listing_version_id": "test-version-id",
                        }
                    ],
                    "total_count": 1,
                }
            },
        )

    async def run(
        self,
        input_data: Input,
        **kwargs,
    ) -> BlockOutput:
        try:
            agents = await self._search_agents(
                query=input_data.query,
                category=input_data.category,
                sort_by=input_data.sort_by,
                limit=input_data.limit,
            )

            yield "agents", agents

            for agent in agents:
                yield "agent", agent

        except Exception as e:
            logger.error(f"Failed to search store agents: {str(e)}")
            yield "error", str(e)

    async def _search_agents(
        self,
        query: Optional[str] = None,
        category: Optional[str] = None,
        sort_by: str = "rating",
        limit: int = 10,
    ) -> list[StoreAgent]:
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

        result = await get_database_manager_async_client().get_store_agents(
            featured=False,
            creator=None,
            sorted_by=sorted_by_map.get(sort_by, "most_popular"),
            search_query=query,
            category=category,
            page=1,
            page_size=limit,
        )

        agents: list[StoreAgent] = []
        for agent in result.agents:
            agents.append(
                StoreAgent(
                    slug=agent.slug,
                    name=agent.agent_name,
                    description=agent.description,
                    creator=agent.creator,
                    rating=agent.rating,
                    runs=agent.runs,
                )
            )

        return agents
