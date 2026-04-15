"""
Joy Trust verification blocks.

Verify agent trust scores before delegating tasks. Joy Trust Network provides
cross-platform reputation scoring for AI agents based on vouches, verification,
and behavioral signals.

**Use Cases:**

- **Safety Checkpoint:** Verify an agent meets trust threshold before delegation
- **Trust Auditing:** Log trust scores for compliance and monitoring
- **Capability Discovery:** Find trusted agents with specific capabilities

**How It Works:**

1. Call the Joy API with an agent ID
2. Receive trust score (0-5 scale) and verification status
3. Compare against your minimum threshold
4. Proceed with delegation only if threshold is met
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

from ._config import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT, get_agent, joy_trust


class JoyVerifyTrustBlock(Block):
    """
    Verify if an agent meets a minimum trust threshold before delegation.

    Use this block as a safety gate in your workflow - only proceed with
    delegation if the target agent has sufficient trust score. Returns
    a boolean indicating whether the threshold is met.

    Recommended thresholds:
    - 1.0: Permissive (low-risk tasks)
    - 1.5: Standard (general use, recommended default)
    - 2.0: Moderate (established agents only)
    - 2.5: Strict (high security)
    """

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = joy_trust.credentials_field(
            description="Joy API key (optional, increases rate limits)"
        )
        agent_id: str = SchemaField(
            description="The Joy agent ID to verify (e.g. 'ag_abc123')",
            advanced=False,
        )
        min_trust_score: float = SchemaField(
            description="Minimum trust score required (0-5 scale). Default 1.5 is recommended for general use.",
            default=1.5,
            ge=0.0,
            le=5.0,
            advanced=False,
        )

    class Output(BlockSchemaOutput):
        meets_threshold: bool = SchemaField(
            description="True if agent's trust score meets or exceeds the minimum threshold"
        )
        trust_score: float = SchemaField(
            description="The agent's current trust score (0-5 scale)"
        )
        agent_name: str = SchemaField(description="Name of the verified agent")
        verified: bool = SchemaField(
            description="Whether the agent has endpoint verification"
        )
        error: str = SchemaField(description="Error message if verification failed")

    def __init__(self):
        super().__init__(
            id="d8e9f0a1-2b3c-4d5e-6f7a-8b9c0d1e2f3a",
            description="Verify if an agent meets minimum trust threshold before delegating tasks. Use as a safety gate in multi-agent workflows.",
            categories={BlockCategory.SAFETY},
            input_schema=self.Input,
            output_schema=self.Output,
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "agent_id": "ag_test123",
                "min_trust_score": 1.5,
            },
            test_output=[
                ("meets_threshold", True),
                ("trust_score", 2.0),
                ("agent_name", "Test Agent"),
                ("verified", True),
            ],
            test_mock={
                "get_agent": lambda agent_id, credentials: {
                    "agent_id": agent_id,
                    "name": "Test Agent",
                    "trust_score": 2.0,
                    "verified": True,
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
        agent = await get_agent(input_data.agent_id, credentials)
        raw_score = agent.get("trust_score")
        trust_score = 0.0 if raw_score is None else float(raw_score)
        meets_threshold = trust_score >= input_data.min_trust_score

        yield "meets_threshold", meets_threshold
        yield "trust_score", trust_score
        yield "agent_name", agent.get("name", "Unknown")
        yield "verified", agent.get("verified", False)


class JoyGetTrustScoreBlock(Block):
    """
    Get detailed trust information for an agent.

    Returns the full trust profile including score, verification status,
    vouch count, capabilities, and badges. Use this for detailed trust
    auditing or to display agent information.

    **How It Works:**

    1. Query the Joy API with an agent ID
    2. Receive full agent profile with trust metrics
    3. Use the data for auditing, display, or decision-making
    """

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = joy_trust.credentials_field(
            description="Joy API key (optional, increases rate limits)"
        )
        agent_id: str = SchemaField(
            description="The Joy agent ID to look up (e.g. 'ag_abc123')",
            advanced=False,
        )

    class Output(BlockSchemaOutput):
        agent_id: str = SchemaField(description="The agent's unique identifier")
        name: str = SchemaField(description="The agent's display name")
        trust_score: float = SchemaField(description="Trust score (0-5 scale)")
        verified: bool = SchemaField(description="Whether endpoint is verified")
        vouch_count: int = SchemaField(description="Number of vouches received")
        capabilities: list = SchemaField(description="List of agent capabilities")
        badges: list = SchemaField(
            description="Earned badges (verified, responsive, etc.)"
        )
        result: dict = SchemaField(description="Complete agent profile")
        error: str = SchemaField(description="Error message if lookup failed")

    def __init__(self):
        super().__init__(
            id="e9f0a1b2-3c4d-4e5f-7a8b-9c0d1e2f3a4b",
            description="Get detailed trust profile for an agent including score, verification status, and capabilities.",
            categories={BlockCategory.SAFETY},
            input_schema=self.Input,
            output_schema=self.Output,
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "agent_id": "ag_test123",
            },
            test_output=[
                ("agent_id", "ag_test123"),
                ("name", "Test Agent"),
                ("trust_score", 2.0),
                ("verified", True),
                ("vouch_count", 5),
            ],
            test_mock={
                "get_agent": lambda agent_id, credentials: {
                    "agent_id": agent_id,
                    "name": "Test Agent",
                    "trust_score": 2.0,
                    "verified": True,
                    "vouch_count": 5,
                    "capabilities": ["code-review"],
                    "badges": ["verified"],
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
        agent = await get_agent(input_data.agent_id, credentials)

        yield "agent_id", agent.get("agent_id", input_data.agent_id)
        yield "name", agent.get("name", "Unknown")
        raw_score2 = agent.get("trust_score")
        yield "trust_score", 0.0 if raw_score2 is None else float(raw_score2)
        yield "verified", agent.get("verified", False)
        yield "vouch_count", agent.get("vouch_count", 0)
        yield "capabilities", agent.get("capabilities", [])
        yield "badges", agent.get("badges", [])
        yield "result", agent
