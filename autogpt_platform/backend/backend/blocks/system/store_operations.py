import logging
from typing import Literal

from pydantic import BaseModel

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import SchemaField
from backend.util.clients import get_database_manager_async_client

logger = logging.getLogger(__name__)


# Duplicate pydantic models for store data so we don't accidently change the data shape in the blocks unintentionally when editing the backend
class StoreAgent(BaseModel):
    """Model representing a store agent."""

    slug: str = ""
    name: str = ""
    description: str = ""
    creator: str = ""
    rating: float = 0.0
    runs: int = 0
    categories: list[str] = []


class StoreAgentDict(BaseModel):
    """Dictionary representation of a store agent."""

    slug: str
    name: str
    description: str
    creator: str
    rating: float
    runs: int


class SearchAgentsResponse(BaseModel):
    """Response from searching store agents."""

    agents: list[StoreAgentDict]
    total_count: int


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
            id="b604f0ec-6e0d-40a7-bf55-9fd09997cced",
            description="Get detailed information about an agent from the store",
            categories={BlockCategory.BASIC, BlockCategory.DATA},
            input_schema=GetStoreAgentDetailsBlock.Input,
            output_schema=GetStoreAgentDetailsBlock.Output,
            test_input={"creator": "test-creator", "slug": "test-agent-slug"},
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
                "_get_agent_details": lambda *_, **__: StoreAgentDetails(
                    found=True,
                    store_listing_version_id="test-listing-id",
                    agent_name="Test Agent",
                    description="A test agent",
                    creator="Test Creator",
                    categories=["productivity", "automation"],
                    runs=100,
                    rating=4.5,
                )
            },
            static_output=True,
        )

    async def run(
        self,
        input_data: Input,
        **kwargs,
    ) -> BlockOutput:
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

    async def _get_agent_details(self, creator: str, slug: str) -> StoreAgentDetails:
        """
        Retrieve detailed information about a store agent.
        """
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
                agent_details.categories if hasattr(agent_details, "categories") else []
            ),
            runs=agent_details.runs,
            rating=agent_details.rating,
        )


class SearchStoreAgentsBlock(Block):
    """
    Block that searches for agents in the store based on various criteria.
    """

    class Input(BlockSchema):
        query: str | None = SchemaField(
            description="Search query to find agents", default=None
        )
        category: str | None = SchemaField(
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
        total_count: int = SchemaField(
            description="Total number of agents found", default=0
        )

    def __init__(self):
        super().__init__(
            id="39524701-026c-4328-87cc-1b88c8e2cb4c",
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
                (
                    "agent",
                    {
                        "slug": "test-agent",
                        "name": "Test Agent",
                        "description": "A test agent",
                        "creator": "Test Creator",
                        "rating": 4.5,
                        "runs": 100,
                    },
                ),
            ],
            test_mock={
                "_search_agents": lambda *_, **__: SearchAgentsResponse(
                    agents=[
                        StoreAgentDict(
                            slug="test-agent",
                            name="Test Agent",
                            description="A test agent",
                            creator="Test Creator",
                            rating=4.5,
                            runs=100,
                        )
                    ],
                    total_count=1,
                )
            },
        )

    async def run(
        self,
        input_data: Input,
        **kwargs,
    ) -> BlockOutput:
        result = await self._search_agents(
            query=input_data.query,
            category=input_data.category,
            sort_by=input_data.sort_by,
            limit=input_data.limit,
        )

        agents = result.agents
        total_count = result.total_count

        # Convert to dict for output
        agents_as_dicts = [agent.model_dump() for agent in agents]

        yield "agents", agents_as_dicts
        yield "total_count", total_count

        for agent_dict in agents_as_dicts:
            yield "agent", agent_dict

    async def _search_agents(
        self,
        query: str | None = None,
        category: str | None = None,
        sort_by: str = "rating",
        limit: int = 10,
    ) -> SearchAgentsResponse:
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
            creators=None,
            sorted_by=sorted_by_map.get(sort_by, "most_popular"),
            search_query=query,
            category=category,
            page=1,
            page_size=limit,
        )

        agents = [
            StoreAgentDict(
                slug=agent.slug,
                name=agent.agent_name,
                description=agent.description,
                creator=agent.creator,
                rating=agent.rating,
                runs=agent.runs,
            )
            for agent in result.agents
        ]

        return SearchAgentsResponse(agents=agents, total_count=len(agents))
