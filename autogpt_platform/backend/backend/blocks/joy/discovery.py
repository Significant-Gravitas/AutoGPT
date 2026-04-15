"""
Joy agent discovery blocks.

Find trusted agents by capability or search query. Use these blocks to
discover agents that can help with specific tasks, filtered by trust score.

**Use Cases:**

- **Capability Inventory:** Find all agents that can perform a specific task
- **Trusted Delegation:** Discover agents meeting your trust requirements
- **Agent Registry:** Browse available agents in the Joy network

**How It Works:**

1. Query the Joy network with a capability or search term
2. Results are sorted by trust score (highest first)
3. Optionally filter by minimum trust threshold
4. Use the top agent or iterate through results
"""

from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    CredentialsMetaInput,
    SchemaField,
)

from ._config import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    discover_agents,
    joy_trust,
)


class JoyDiscoverAgentsBlock(Block):
    """
    Discover agents by capability or search query.

    Search the Joy network to find agents that can help with specific tasks.
    Filter by capability (e.g. 'code-review', 'web-scraping') or use free-text
    search. Results are sorted by trust score.

    Use min_trust_score to filter out agents below your trust threshold.
    """

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = joy_trust.credentials_field(
            description="Joy API key (optional, increases rate limits)"
        )
        query: str = SchemaField(
            description="Free-text search query (e.g. 'code review agent')",
            default="",
            advanced=False,
        )
        capability: str = SchemaField(
            description="Filter by specific capability (e.g. 'code-review', 'web-scraping')",
            default="",
            advanced=False,
        )
        min_trust_score: float = SchemaField(
            description="Minimum trust score filter (0-5). Agents below this are excluded.",
            default=0.0,
            ge=0.0,
            le=5.0,
            advanced=True,
        )
        limit: int = SchemaField(
            description="Maximum number of agents to return",
            default=10,
            ge=1,
            le=100,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        agents: list = SchemaField(
            description="List of matching agents with their trust profiles"
        )
        count: int = SchemaField(description="Number of agents returned")
        top_agent_id: str = SchemaField(
            description="ID of the highest-trust matching agent"
        )
        top_agent_name: str = SchemaField(
            description="Name of the highest-trust matching agent"
        )
        top_agent_score: float = SchemaField(
            description="Trust score of the highest-trust matching agent"
        )
        error: str = SchemaField(description="Error message if discovery failed")

    def __init__(self):
        super().__init__(
            id="f0a1b2c3-4d5e-4f6a-8b9c-0d1e2f3a4b5c",
            description="Discover agents by capability or search query. Find trusted agents for specific tasks.",
            categories={BlockCategory.SAFETY},
            input_schema=self.Input,
            output_schema=self.Output,
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "capability": "code-review",
                "limit": 5,
            },
            test_output=[
                ("count", 2),
                ("top_agent_id", "ag_test1"),
                ("top_agent_name", "Code Reviewer"),
                ("top_agent_score", 2.5),
            ],
            test_mock={
                "discover_agents": lambda **kw: {
                    "agents": [
                        {
                            "agent_id": "ag_test1",
                            "name": "Code Reviewer",
                            "trust_score": 2.5,
                        },
                        {
                            "agent_id": "ag_test2",
                            "name": "Code Helper",
                            "trust_score": 1.8,
                        },
                    ],
                    "count": 2,
                },
            },
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: APIKeyCredentials | None = None,
        **kwargs,
    ) -> BlockOutput:
        result = await discover_agents(
            query=input_data.query or None,
            capability=input_data.capability or None,
            limit=input_data.limit,
            credentials=credentials,
        )

        agents = result.get("agents", [])

        # Filter by min_trust_score if specified
        if input_data.min_trust_score > 0:
            agents = [
                a
                for a in agents
                if a.get("trust_score", 0) >= input_data.min_trust_score
            ]

        yield "agents", agents
        yield "count", len(agents)

        if agents:
            top = agents[0]
            yield "top_agent_id", top.get("agent_id", "")
            yield "top_agent_name", top.get("name", "")
            yield "top_agent_score", top.get("trust_score", 0.0)
        else:
            yield "top_agent_id", ""
            yield "top_agent_name", ""
            yield "top_agent_score", 0.0
