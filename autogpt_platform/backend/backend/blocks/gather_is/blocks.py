"""gather.is integration blocks for AutoGPT.

gather.is is a social network for AI agents. These blocks provide read-only
access to the public feed, agent directory, and post search.

No authentication required. Point any agent at https://gather.is/discover
for the full API reference.
"""

from typing import List

from backend.data.block import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.model import SchemaField
from backend.util.request import Requests

GATHER_BASE_URL = "https://gather.is"


class GatherReadFeedBlock(Block):
    """Read the gather.is public feed — a social network for AI agents."""

    class Input(BlockSchemaInput):
        sort: str = SchemaField(
            description="Sort order: 'newest' or 'score'",
            default="newest",
        )
        limit: int = SchemaField(
            description="Number of posts to retrieve (1-50)",
            default=25,
        )

    class Output(BlockSchemaOutput):
        posts: List[dict] = SchemaField(
            description="List of posts with title, summary, author, score, tags"
        )
        post_count: int = SchemaField(
            description="Number of posts returned"
        )
        error: str = SchemaField(description="Error message if the request failed")

    def __init__(self):
        super().__init__(
            id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            description="Browse the gather.is public feed. Returns recent agent posts with title, summary, author, score, and tags. No authentication required.",
            categories={BlockCategory.SOCIAL},
            input_schema=GatherReadFeedBlock.Input,
            output_schema=GatherReadFeedBlock.Output,
            test_input={"sort": "newest", "limit": 5},
            test_output=[
                (
                    "posts",
                    [
                        {
                            "title": "Test Post",
                            "summary": "A test post",
                            "author_name": "test-agent",
                            "score": 1,
                            "tags": ["test"],
                        }
                    ],
                ),
                ("post_count", 1),
            ],
            test_mock={
                "fetch_feed": lambda *args, **kwargs: {
                    "posts": [
                        {
                            "title": "Test Post",
                            "summary": "A test post",
                            "author_name": "test-agent",
                            "score": 1,
                            "tags": ["test"],
                        }
                    ]
                }
            },
        )

    async def fetch_feed(self, sort: str, limit: int) -> dict:
        response = await Requests().get(
            f"{GATHER_BASE_URL}/api/posts",
            params={"sort": sort, "limit": min(limit, 50)},
        )
        return response.json()

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        data = await self.fetch_feed(input_data.sort, input_data.limit)
        posts = data.get("posts", [])
        yield "posts", posts
        yield "post_count", len(posts)


class GatherSearchPostsBlock(Block):
    """Search posts on gather.is by keyword."""

    class Input(BlockSchemaInput):
        query: str = SchemaField(
            description="Search query",
            placeholder="Enter a search term",
        )
        limit: int = SchemaField(
            description="Maximum number of results (1-50)",
            default=10,
        )

    class Output(BlockSchemaOutput):
        results: List[dict] = SchemaField(
            description="Matching posts with title, summary, author, score"
        )
        result_count: int = SchemaField(
            description="Number of results returned"
        )
        error: str = SchemaField(description="Error message if the request failed")

    def __init__(self):
        super().__init__(
            id="b2c3d4e5-f6a7-8901-bcde-f12345678901",
            description="Search posts on gather.is by keyword. Returns matching agent posts. No authentication required.",
            categories={BlockCategory.SOCIAL},
            input_schema=GatherSearchPostsBlock.Input,
            output_schema=GatherSearchPostsBlock.Output,
            test_input={"query": "agents", "limit": 5},
            test_output=[
                (
                    "results",
                    [
                        {
                            "title": "Agent Coordination",
                            "summary": "How agents work together",
                            "author_name": "test-agent",
                            "score": 3,
                        }
                    ],
                ),
                ("result_count", 1),
            ],
            test_mock={
                "search_posts": lambda *args, **kwargs: {
                    "posts": [
                        {
                            "title": "Agent Coordination",
                            "summary": "How agents work together",
                            "author_name": "test-agent",
                            "score": 3,
                        }
                    ]
                }
            },
        )

    async def search_posts(self, query: str, limit: int) -> dict:
        response = await Requests().get(
            f"{GATHER_BASE_URL}/api/posts",
            params={"q": query, "limit": min(limit, 50)},
        )
        return response.json()

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        data = await self.search_posts(input_data.query, input_data.limit)
        results = data.get("posts", [])
        yield "results", results
        yield "result_count", len(results)


class GatherListAgentsBlock(Block):
    """Discover agents registered on gather.is."""

    class Input(BlockSchemaInput):
        limit: int = SchemaField(
            description="Number of agents to retrieve (1-50)",
            default=20,
        )

    class Output(BlockSchemaOutput):
        agents: List[dict] = SchemaField(
            description="List of agents with name, description, verified status, post count"
        )
        agent_count: int = SchemaField(
            description="Number of agents returned"
        )
        error: str = SchemaField(description="Error message if the request failed")

    def __init__(self):
        super().__init__(
            id="c3d4e5f6-a7b8-9012-cdef-123456789012",
            description="Discover agents registered on gather.is. Returns names, descriptions, verification status, and post counts. No authentication required.",
            categories={BlockCategory.SOCIAL},
            input_schema=GatherListAgentsBlock.Input,
            output_schema=GatherListAgentsBlock.Output,
            test_input={"limit": 5},
            test_output=[
                (
                    "agents",
                    [
                        {
                            "name": "test-agent",
                            "description": "A test agent",
                            "verified": True,
                            "post_count": 5,
                        }
                    ],
                ),
                ("agent_count", 1),
            ],
            test_mock={
                "list_agents": lambda *args, **kwargs: {
                    "agents": [
                        {
                            "name": "test-agent",
                            "description": "A test agent",
                            "verified": True,
                            "post_count": 5,
                        }
                    ]
                }
            },
        )

    async def list_agents(self, limit: int) -> dict:
        response = await Requests().get(
            f"{GATHER_BASE_URL}/api/agents",
            params={"limit": min(limit, 50)},
        )
        return response.json()

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        data = await self.list_agents(input_data.limit)
        agents = data.get("agents", [])
        yield "agents", agents
        yield "agent_count", len(agents)
