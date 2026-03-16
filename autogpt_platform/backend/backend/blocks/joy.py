"""Joy trust network integration blocks for AutoGPT.

This module provides blocks for integrating with the Joy trust network, enabling
AI agents to verify trustworthiness before delegating tasks or sharing data.

Joy is a decentralized trust network where agents build reputation through:
- Vouches from other agents after successful collaborations
- Verification of agent identity and capabilities
- Trust scores calculated from network activity

Available blocks:
    JoyTrustVerifyBlock: Verify an agent's trust score and status.
    JoyDiscoverAgentsBlock: Find agents by capability and trust level.
    JoyShouldTrustBlock: Simple boolean trust gate for workflows.

Example usage:
    Use JoyTrustVerifyBlock before delegating sensitive tasks to check
    if an external agent meets your minimum trust requirements.

Learn more: https://choosejoy.com.au
"""

import logging
import re
from typing import Any
from urllib.parse import quote

import httpx

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.execution import ExecutionContext
from backend.data.model import SchemaField

logger = logging.getLogger(__name__)

JOY_API_URL = "https://joy-connect.fly.dev"

# Joy agent IDs follow pattern: ag_ followed by 24 hex characters
AGENT_ID_PATTERN = re.compile(r"^ag_[a-f0-9]{24}$")


def _validate_agent_id(agent_id: str) -> str:
    """Validate and sanitize agent ID to prevent injection attacks.

    Args:
        agent_id: The agent ID to validate

    Returns:
        The validated agent ID

    Raises:
        ValueError: If the agent ID format is invalid
    """
    if not agent_id or not isinstance(agent_id, str):
        raise ValueError("Agent ID must be a non-empty string")

    agent_id = agent_id.strip().lower()

    if not AGENT_ID_PATTERN.match(agent_id):
        raise ValueError(
            f"Invalid agent ID format: {agent_id!r}. "
            "Expected format: ag_ followed by 24 hex characters (e.g., 'ag_229e507d7d87f35cc2bc17ea')"
        )

    return agent_id


class JoyTrustVerifyBlock(Block):
    """Verify an agent's trustworthiness using the Joy trust network.

    This block queries the Joy platform to retrieve an agent's trust metrics
    including trust score, verification status, and vouch count. Use this
    before delegating tasks to external agents to ensure reliability.

    The Joy trust network builds reputation through agent-to-agent vouches
    after successful collaborations. Higher trust scores indicate more
    reliable agents with proven track records.

    Attributes:
        Input: Configuration for trust verification including agent_id,
            minimum trust threshold, and verification requirements.
        Output: Trust metrics including is_trusted boolean, trust_score,
            vouch_count, verified status, and capabilities list.

    Example:
        Verify an agent before delegating a code review task::

            input_data = {
                "agent_id": "ag_229e507d7d87f35cc2bc17ea",
                "min_trust_score": 0.5,
                "require_verified": True
            }
    """

    class Input(BlockSchemaInput):
        agent_id: str = SchemaField(description="Joy agent ID to verify (e.g., 'ag_xxx')")
        min_trust_score: float = SchemaField(
            description="Minimum trust score required (0.0-2.0)",
            default=0.5,
        )
        require_verified: bool = SchemaField(
            description="Only trust agents with verified badge",
            default=False,
        )

    class Output(BlockSchemaOutput):
        is_trusted: bool = SchemaField(description="Whether the agent meets trust criteria")
        trust_score: float = SchemaField(description="Agent's trust score (0.0-2.0)")
        vouch_count: int = SchemaField(description="Number of vouches the agent has received")
        verified: bool = SchemaField(description="Whether the agent has a verified badge")
        capabilities: list = SchemaField(description="List of agent capabilities")
        error: str = SchemaField(description="Error message if verification failed")

    def __init__(self):
        super().__init__(
            id="a7b3c8d9-e0f1-4a2b-8c3d-9e0f1a2b3c4d",
            description=(
                "Verify an agent's trustworthiness using the Joy trust network. "
                "Returns trust score, vouch count, and whether the agent meets "
                "your minimum trust threshold. Use before delegating tasks."
            ),
            categories={BlockCategory.AGENT, BlockCategory.SAFETY},
            input_schema=JoyTrustVerifyBlock.Input,
            output_schema=JoyTrustVerifyBlock.Output,
            test_input={
                "agent_id": "ag_229e507d7d87f35cc2bc17ea",
                "min_trust_score": 0.5,
                "require_verified": False,
            },
            test_output=[
                ("is_trusted", True),
                ("trust_score", 1.5),
                ("vouch_count", 10),
                ("verified", True),
                ("capabilities", ["github", "code"]),
                ("error", ""),
            ],
            test_mock={
                "_fetch_agent": lambda *args, **kwargs: {
                    "id": "ag_229e507d7d87f35cc2bc17ea",
                    "name": "Test Agent",
                    "trust_score": 1.5,
                    "vouch_count": 10,
                    "verified": True,
                    "capabilities": ["github", "code"],
                }
            },
        )

    async def _fetch_agent(self, agent_id: str) -> dict[str, Any]:
        """Fetch agent data from the Joy API.

        Retrieves trust metrics for a specific agent including trust score,
        verification status, vouch count, and registered capabilities.

        Args:
            agent_id: The Joy agent identifier to look up. Must match the
                pattern 'ag_' followed by 24 hexadecimal characters.

        Returns:
            Dictionary containing agent data with keys:
                - id: The agent's unique identifier
                - name: Display name of the agent
                - trust_score: Float from 0.0 to 2.0+
                - vouch_count: Number of vouches received
                - verified: Boolean verification status
                - capabilities: List of capability strings

        Raises:
            ValueError: If agent_id format is invalid.
            httpx.HTTPStatusError: If the API request fails.
        """
        validated_id = _validate_agent_id(agent_id)
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{JOY_API_URL}/agents/{quote(validated_id, safe='')}",
                headers={"User-Agent": "autogpt-joy/1.0.0"},
            )
            response.raise_for_status()
            return response.json()

    async def run(
        self,
        input_data: Input,
        *,
        execution_context: ExecutionContext,
        **kwargs,
    ) -> BlockOutput:
        """Execute trust verification against the Joy platform.

        Fetches agent data and evaluates against the configured trust
        criteria to determine if the agent should be trusted.

        Args:
            input_data: Input parameters including agent_id and trust criteria.
            execution_context: AutoGPT execution context.
            **kwargs: Additional execution parameters.

        Yields:
            Tuple of (output_name, value) for each output field:
                - is_trusted: Boolean indicating if criteria are met
                - trust_score: The agent's trust score
                - vouch_count: Number of vouches received
                - verified: Verification status
                - capabilities: List of capabilities
                - error: Error message if verification failed
        """
        try:
            data = await self._fetch_agent(input_data.agent_id)

            trust_score = float(data.get("trust_score", 0))
            vouch_count = int(data.get("vouch_count", 0))
            verified = bool(data.get("verified", False))
            capabilities = data.get("capabilities", [])

            # Determine if trusted based on criteria
            is_trusted = trust_score >= input_data.min_trust_score
            if input_data.require_verified and not verified:
                is_trusted = False

            yield "is_trusted", is_trusted
            yield "trust_score", trust_score
            yield "vouch_count", vouch_count
            yield "verified", verified
            yield "capabilities", capabilities
            yield "error", ""

        except Exception as e:
            logger.error(f"Joy trust verification failed: {e}")
            yield "is_trusted", False
            yield "trust_score", 0.0
            yield "vouch_count", 0
            yield "verified", False
            yield "capabilities", []
            yield "error", str(e)


class JoyDiscoverAgentsBlock(Block):
    """Discover trusted agents from the Joy network by capability.

    This block searches the Joy platform for agents matching specific
    capabilities and trust criteria. Use this to find reliable agents
    for task delegation in multi-agent workflows.

    The discovery system filters agents by:
        - Capability tags (e.g., 'github', 'email', 'code-generation')
        - Free-text search queries
        - Minimum trust score thresholds

    Attributes:
        Input: Search parameters including capability filter, query string,
            minimum trust score, and result limit.
        Output: List of matching agents with their trust metrics, plus
            count and any error messages.

    Example:
        Find trusted agents capable of GitHub operations::

            input_data = {
                "capability": "github",
                "min_trust_score": 0.7,
                "limit": 5
            }
    """

    class Input(BlockSchemaInput):
        capability: str = SchemaField(
            description="Capability to search for (e.g., 'github', 'email', 'code')",
            default="",
        )
        query: str = SchemaField(description="Free text search query", default="")
        min_trust_score: float = SchemaField(
            description="Minimum trust score required (0.0-2.0)",
            default=0.5,
        )
        limit: int = SchemaField(description="Maximum number of agents to return", default=10)

    class Output(BlockSchemaOutput):
        agents: list = SchemaField(description="List of trusted agents matching criteria")
        count: int = SchemaField(description="Number of agents found")
        error: str = SchemaField(description="Error message if discovery failed")

    def __init__(self):
        super().__init__(
            id="b8c4d9e0-f1a2-4b3c-9d4e-0f1a2b3c4d5e",
            description=(
                "Discover trusted agents from the Joy network. "
                "Search by capability or query to find agents that "
                "meet your minimum trust threshold."
            ),
            categories={BlockCategory.AGENT, BlockCategory.SEARCH},
            input_schema=JoyDiscoverAgentsBlock.Input,
            output_schema=JoyDiscoverAgentsBlock.Output,
            test_input={
                "capability": "github",
                "min_trust_score": 0.5,
                "limit": 5,
            },
            test_output=[
                ("agents", lambda x: len(x) == 2),
                ("count", 2),
                ("error", ""),
            ],
            test_mock={
                "_discover_agents": lambda *args, **kwargs: {
                    "agents": [
                        {
                            "id": "ag_test1",
                            "name": "Test Agent 1",
                            "description": "A test agent",
                            "trust_score": 1.5,
                            "vouch_count": 10,
                            "verified": True,
                            "capabilities": ["github", "code"],
                        },
                        {
                            "id": "ag_test2",
                            "name": "Test Agent 2",
                            "description": "Another test agent",
                            "trust_score": 1.2,
                            "vouch_count": 5,
                            "verified": False,
                            "capabilities": ["github"],
                        },
                    ],
                    "count": 2,
                }
            },
        )

    async def _discover_agents(
        self, capability: str, query: str, limit: int
    ) -> dict[str, Any]:
        """Discover agents from the Joy API with filtering.

        Searches the Joy network for agents matching the specified criteria.
        Results are ordered by trust score descending.

        Args:
            capability: Optional capability tag to filter by (e.g., 'github').
            query: Optional free-text search query.
            limit: Maximum number of agents to return.

        Returns:
            Dictionary containing:
                - agents: List of agent dictionaries with trust metrics
                - count: Total number of matching agents

        Raises:
            httpx.HTTPStatusError: If the API request fails.
        """
        params: dict[str, Any] = {"limit": limit}
        if capability:
            params["capability"] = capability
        if query:
            params["query"] = query

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{JOY_API_URL}/agents/discover",
                params=params,
                headers={"User-Agent": "autogpt-joy/1.0.0"},
            )
            response.raise_for_status()
            return response.json()

    async def run(
        self,
        input_data: Input,
        *,
        execution_context: ExecutionContext,
        **kwargs,
    ) -> BlockOutput:
        """Execute agent discovery against the Joy platform.

        Searches for agents matching the specified criteria and filters
        results by the minimum trust score threshold.

        Args:
            input_data: Search parameters including capability, query,
                min_trust_score, and limit.
            execution_context: AutoGPT execution context.
            **kwargs: Additional execution parameters.

        Yields:
            Tuple of (output_name, value) for each output field:
                - agents: List of matching agent dictionaries
                - count: Number of agents returned
                - error: Error message if discovery failed
        """
        try:
            data = await self._discover_agents(
                input_data.capability, input_data.query, input_data.limit
            )

            agents = data.get("agents", [])

            # Filter by trust score
            trusted_agents = [
                {
                    "id": agent.get("id"),
                    "name": agent.get("name"),
                    "description": agent.get("description"),
                    "trust_score": float(agent.get("trust_score", 0)),
                    "vouch_count": int(agent.get("vouch_count", 0)),
                    "capabilities": agent.get("capabilities", []),
                    "verified": bool(agent.get("verified", False)),
                }
                for agent in agents
                if float(agent.get("trust_score", 0)) >= input_data.min_trust_score
            ]

            yield "agents", trusted_agents
            yield "count", len(trusted_agents)
            yield "error", ""

        except Exception as e:
            logger.error(f"Joy agent discovery failed: {e}")
            yield "agents", []
            yield "count", 0
            yield "error", str(e)


class JoyShouldTrustBlock(Block):
    """Simple trust gate for conditional workflow decisions.

    This block provides a straightforward boolean trust check for use in
    conditional workflow branches. It returns whether an agent meets the
    minimum trust threshold along with a human-readable reason.

    Use this block as a gate before delegating sensitive tasks to external
    agents. The simple true/false output integrates easily with AutoGPT's
    conditional logic blocks.

    Attributes:
        Input: Agent ID to check and minimum trust score threshold.
        Output: Boolean trusted status and explanation string.

    Example:
        Gate a task delegation based on trust::

            input_data = {
                "agent_id": "ag_229e507d7d87f35cc2bc17ea",
                "min_trust_score": 0.5
            }
            # Output: {"trusted": True, "reason": "Trust score 1.50 meets threshold 0.5"}
    """

    class Input(BlockSchemaInput):
        agent_id: str = SchemaField(description="Joy agent ID to check (e.g., 'ag_xxx')")
        min_trust_score: float = SchemaField(
            description="Minimum trust score required (0.0-2.0)",
            default=0.5,
        )

    class Output(BlockSchemaOutput):
        trusted: bool = SchemaField(description="Whether the agent should be trusted")
        reason: str = SchemaField(description="Reason for the trust decision")

    def __init__(self):
        super().__init__(
            id="c9d5e0f1-a2b3-4c4d-ae5f-1a2b3c4d5e6f",
            description=(
                "Simple trust gate for agent verification. "
                "Returns true/false for use in conditional workflows. "
                "Use before delegating tasks to external agents."
            ),
            categories={BlockCategory.AGENT, BlockCategory.LOGIC},
            input_schema=JoyShouldTrustBlock.Input,
            output_schema=JoyShouldTrustBlock.Output,
            test_input={
                "agent_id": "ag_229e507d7d87f35cc2bc17ea",
                "min_trust_score": 0.5,
            },
            test_output=[
                ("trusted", True),
                ("reason", lambda x: "meets threshold" in x),
            ],
            test_mock={
                "_fetch_agent": lambda *args, **kwargs: {
                    "id": "ag_229e507d7d87f35cc2bc17ea",
                    "name": "Test Agent",
                    "trust_score": 1.5,
                    "vouch_count": 10,
                    "verified": True,
                    "capabilities": ["github", "code"],
                }
            },
        )

    async def _fetch_agent(self, agent_id: str) -> dict[str, Any]:
        """Fetch agent data from the Joy API.

        Retrieves the agent's trust score for threshold comparison.

        Args:
            agent_id: The Joy agent identifier to look up. Must match the
                pattern 'ag_' followed by 24 hexadecimal characters.

        Returns:
            Dictionary containing agent data including trust_score.

        Raises:
            ValueError: If agent_id format is invalid.
            httpx.HTTPStatusError: If the API request fails.
        """
        validated_id = _validate_agent_id(agent_id)
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{JOY_API_URL}/agents/{quote(validated_id, safe='')}",
                headers={"User-Agent": "autogpt-joy/1.0.0"},
            )
            response.raise_for_status()
            return response.json()

    async def run(
        self,
        input_data: Input,
        *,
        execution_context: ExecutionContext,
        **kwargs,
    ) -> BlockOutput:
        """Execute trust gate check against the Joy platform.

        Fetches the agent's trust score and compares against the minimum
        threshold to produce a simple boolean decision.

        Args:
            input_data: Input parameters including agent_id and min_trust_score.
            execution_context: AutoGPT execution context.
            **kwargs: Additional execution parameters.

        Yields:
            Tuple of (output_name, value) for each output field:
                - trusted: Boolean indicating if threshold is met
                - reason: Human-readable explanation of the decision
        """
        try:
            data = await self._fetch_agent(input_data.agent_id)

            trust_score = float(data.get("trust_score", 0))

            if trust_score >= input_data.min_trust_score:
                yield "trusted", True
                yield "reason", (
                    f"Trust score {trust_score:.2f} meets threshold "
                    f"{input_data.min_trust_score}"
                )
            else:
                yield "trusted", False
                yield "reason", (
                    f"Trust score {trust_score:.2f} below threshold "
                    f"{input_data.min_trust_score}"
                )

        except Exception as e:
            yield "trusted", False
            yield "reason", f"Verification failed: {e}"
