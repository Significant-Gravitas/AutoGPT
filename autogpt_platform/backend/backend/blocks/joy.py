# Joy Trust Network Blocks - AutoGPT Compliant Documentation
# Based on AutoGPT's documented patterns from basic.py, http.py, and agent.py

import logging
from enum import Enum
from typing import Any, Dict

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


# Module-level helper to avoid code duplication
async def _fetch_agent(
    agent_id: str, 
    api_key: str | None = None
) -> dict[str, Any]:
    """
    Fetch agent data from the Joy API with proper error handling.
    
    Args:
        agent_id: The Joy agent identifier
        api_key: Optional API key for authenticated requests
        
    Returns:
        Dictionary containing agent data from Joy API
        
    Raises:
        httpx.HTTPError: When API request fails
        ValueError: When agent ID is invalid
    """
    if not isinstance(agent_id, str) or not agent_id.startswith('ag_') or len(agent_id) <= 10:
        raise ValueError(f"Invalid agent ID format: {agent_id}")
        
    headers = {}
    if api_key:
        headers['x-api-key'] = api_key
        
    url = f"https://joy-connect.fly.dev/agents/{agent_id}"
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        return response.json()


class JoyTrustVerifyBlock(Block):
    """
    Verifies the trust score and status of an AI agent using the Joy trust network.
    
    This block queries the Joy platform to check if an agent can be trusted for delegation
    based on community vouches and verification status. Use this before delegating tasks
    to unknown agents to ensure reliability and capability.
    
    The Joy trust network allows agents to vouch for each other after successful
    collaborations, building a decentralized reputation system for the AI agent ecosystem.
    """

    class Input(BlockSchemaInput):
        agent_id: str = SchemaField(
            description="The unique identifier of the agent to verify. Must be a valid Joy agent ID starting with 'ag_'.",
            placeholder="ag_5a549ffa2bb51349ec5b3ee7",
            title="Agent ID"
        )
        min_trust_score: float = SchemaField(
            description="Minimum trust score required (0.0 to 3.0). Agents below this threshold will be marked as not trusted.",
            default=0.5,
            title="Minimum Trust Score"
        )
        require_verified: bool = SchemaField(
            description="Whether to require agent verification status. Verified agents have passed additional identity checks.",
            default=False,
            title="Require Verification",
            advanced=True
        )
        api_key: str | None = SchemaField(
            description="Joy API key for authenticated requests. Optional for basic trust checks.",
            default=None,
            title="API Key",
            advanced=True
        )

    class Output(BlockSchemaOutput):
        trust_score: float = SchemaField(
            description="The agent's trust score from 0.0 to 3.0, based on community vouches and activity."
        )
        is_trusted: bool = SchemaField(
            description="Whether the agent meets the minimum trust criteria for safe delegation."
        )
        verified: bool = SchemaField(
            description="Whether the agent has passed Joy's verification process."
        )
        vouch_count: int = SchemaField(
            description="Number of vouches the agent has received from other agents."
        )
        agent_name: str = SchemaField(
            description="The display name of the agent as registered in Joy."
        )
        capabilities: list[str] = SchemaField(
            description="List of capabilities the agent has registered."
        )
        error: str = SchemaField(
            description="Error message if verification failed or agent not found."
        )

    def __init__(self):
        super().__init__(
            id="joy-trust-verify-001",
            description="Verify an AI agent's trustworthiness using the Joy trust network before delegation.",
            categories={BlockCategory.AGENT, BlockCategory.OUTPUT},
            input_schema=JoyTrustVerifyBlock.Input,
            output_schema=JoyTrustVerifyBlock.Output,
            test_input=[
                {
                    "agent_id": "ag_5a549ffa2bb51349ec5b3ee7",
                    "min_trust_score": 0.5,
                    "require_verified": False
                },
                {
                    "agent_id": "ag_invalid_agent_id",
                    "min_trust_score": 1.0,
                    "require_verified": True,
                    "api_key": "test_key"
                }
            ],
            test_output=[
                ("trust_score", 1.7),
                ("is_trusted", True),
                ("verified", True),
                ("vouch_count", 15),
                ("agent_name", "Jenkins"),
                ("capabilities", ["automation", "communication"]),
                ("error", "")
            ]
        )

    @property
    def test_mock(self) -> dict[str, Any]:
        """
        Mock data for testing that simulates Joy API responses.
        
        Returns:
            Dictionary mapping test inputs to expected outputs for unit tests
        """
        return {
            "ag_5a549ffa2bb51349ec5b3ee7": {
                "id": "ag_5a549ffa2bb51349ec5b3ee7",
                "name": "Jenkins",
                "trust_score": 1.7,
                "verified": True,
                "vouch_count": 15,
                "capabilities": ["automation", "communication"]
            },
            "ag_invalid_agent_id": None  # Simulates agent not found
        }

    async def run(
        self,
        input_data: Input,
        *,
        execution_context: ExecutionContext,
        **kwargs
    ) -> BlockOutput:
        """
        Execute agent trust verification against the Joy platform.
        
        Args:
            input_data: Input parameters including agent_id and trust criteria
            execution_context: AutoGPT execution context
            **kwargs: Additional execution parameters
            
        Yields:
            Tuple of (output_name, output_value) for each verification result
        """
        try:
            # Fetch agent data from Joy API using shared helper
            agent_data = await _fetch_agent(
                input_data.agent_id, 
                input_data.api_key
            )
            
            if not agent_data:
                yield "error", f"Agent not found: {input_data.agent_id}"
                yield "is_trusted", False
                yield "trust_score", 0.0
                yield "verified", False
                yield "vouch_count", 0
                yield "agent_name", ""
                yield "capabilities", []
                return
            
            trust_score = agent_data.get('trust_score', 0.0)
            verified = agent_data.get('verified', False)
            
            # Apply trust criteria
            meets_trust_threshold = trust_score >= input_data.min_trust_score
            meets_verification = not input_data.require_verified or verified
            is_trusted = meets_trust_threshold and meets_verification
            
            # Yield all verification results
            yield "trust_score", trust_score
            yield "is_trusted", is_trusted
            yield "verified", verified
            yield "vouch_count", agent_data.get('vouch_count', 0)
            yield "agent_name", agent_data.get('name', 'Unknown')
            yield "capabilities", agent_data.get('capabilities', [])
            yield "error", ""
            
            logger.info(
                f"Agent {input_data.agent_id} verification complete: "
                f"trust={trust_score}, trusted={is_trusted}, verified={verified}"
            )
            
        except httpx.HTTPError as e:
            error_msg = f"Joy API error: {str(e)}"
            logger.error(error_msg)
            yield "error", error_msg
            yield "is_trusted", False
            yield "trust_score", 0.0
            yield "verified", False
            yield "vouch_count", 0
            yield "agent_name", ""
            yield "capabilities", []
        except Exception as e:
            error_msg = f"Verification failed: {str(e)}"
            logger.error(error_msg)
            yield "error", error_msg
            yield "is_trusted", False
            yield "trust_score", 0.0
            yield "verified", False
            yield "vouch_count", 0
            yield "agent_name", ""
            yield "capabilities", []


class JoyDiscoverAgentsBlock(Block):
    """
    Discover and filter AI agents by capability and trust level using the Joy network.
    
    This block searches the Joy platform for agents with specific capabilities and 
    minimum trust thresholds. Use this to find reliable agents for task delegation
    in multi-agent workflows.
    
    The discovery system allows filtering by capability tags (e.g., "data-analysis", 
    "code-generation") and trust scores to ensure you find suitable agents for your needs.
    """

    class Input(BlockSchemaInput):
        capability: str | None = SchemaField(
            description="Specific capability to search for (e.g., 'data-analysis', 'code-generation'). Leave empty to search all agents.",
            default=None,
            placeholder="data-analysis",
            title="Required Capability"
        )
        min_trust_score: float = SchemaField(
            description="Minimum trust score for discovered agents (0.0 to 3.0).",
            default=0.7,
            title="Minimum Trust Score"
        )
        limit: int = SchemaField(
            description="Maximum number of agents to return in the results.",
            default=10,
            title="Result Limit"
        )
        require_verified: bool = SchemaField(
            description="Only return agents that have passed Joy's verification process.",
            default=False,
            title="Verified Only",
            advanced=True
        )
        api_key: str | None = SchemaField(
            description="Joy API key for authenticated discovery requests.",
            default=None,
            title="API Key",
            advanced=True
        )

    class Output(BlockSchemaOutput):
        agents: list[dict[str, Any]] = SchemaField(
            description="List of discovered agents matching the search criteria."
        )
        total_found: int = SchemaField(
            description="Total number of agents found matching the criteria."
        )
        search_capability: str | None = SchemaField(
            description="The capability that was searched for."
        )
        error: str = SchemaField(
            description="Error message if discovery request failed."
        )

    def __init__(self):
        super().__init__(
            id="joy-discover-agents-001", 
            description="Discover AI agents by capability and trust level from the Joy network for task delegation.",
            categories={BlockCategory.AGENT, BlockCategory.SEARCH},
            input_schema=JoyDiscoverAgentsBlock.Input,
            output_schema=JoyDiscoverAgentsBlock.Output,
            test_input=[
                {
                    "capability": "data-analysis",
                    "min_trust_score": 0.8,
                    "limit": 5,
                    "require_verified": True
                },
                {
                    "capability": None,
                    "min_trust_score": 0.5,
                    "limit": 20
                }
            ],
            test_output=[
                ("agents", [{"id": "ag_test", "name": "Test Agent", "trust_score": 2.1}]),
                ("total_found", 3),
                ("search_capability", "data-analysis"),
                ("error", "")
            ]
        )

    async def _discover_agents(
        self,
        capability: str | None = None,
        min_trust_score: float = 0.0,
        limit: int = 10,
        require_verified: bool = False,
        api_key: str | None = None
    ) -> dict[str, Any]:
        """
        Discover agents from Joy API with specified filters.
        
        Args:
            capability: Optional capability filter
            min_trust_score: Minimum trust threshold
            limit: Maximum results to return
            require_verified: Whether to require verification
            api_key: Optional API key for authenticated requests
            
        Returns:
            Dictionary containing discovery results from Joy API
            
        Raises:
            httpx.HTTPError: When API request fails
        """
        headers = {}
        if api_key:
            headers['x-api-key'] = api_key
        
        params = {
            'limit': limit
        }
        if capability:
            params['capability'] = capability
        if min_trust_score > 0:
            params['min_trust'] = min_trust_score
        if require_verified:
            params['verified'] = 'true'
            
        url = "https://joy-connect.fly.dev/agents/discover"
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers, params=params)
            response.raise_for_status()
            return response.json()

    @property
    def test_mock(self) -> dict[str, Any]:
        """
        Mock data for testing agent discovery responses.
        
        Returns:
            Dictionary mapping test scenarios to expected API responses
        """
        return {
            "data-analysis": {
                "agents": [
                    {
                        "id": "ag_data_analyst_001",
                        "name": "Data Analyst Pro",
                        "trust_score": 2.1,
                        "verified": True,
                        "capabilities": ["data-analysis", "visualization"]
                    },
                    {
                        "id": "ag_python_expert_001", 
                        "name": "Python Expert",
                        "trust_score": 1.8,
                        "verified": True,
                        "capabilities": ["data-analysis", "code-generation"]
                    }
                ],
                "count": 3,
                "total": 3
            },
            "general": {
                "agents": [],
                "count": 0,
                "total": 0
            }
        }

    async def run(
        self,
        input_data: Input,
        *,
        execution_context: ExecutionContext,
        **kwargs
    ) -> BlockOutput:
        """
        Execute agent discovery against the Joy platform.
        
        Args:
            input_data: Discovery parameters including capability and trust criteria
            execution_context: AutoGPT execution context
            **kwargs: Additional execution parameters
            
        Yields:
            Tuple of (output_name, output_value) for discovery results
        """
        try:
            # Perform discovery request to Joy API
            discovery_result = await self._discover_agents(
                capability=input_data.capability,
                min_trust_score=input_data.min_trust_score,
                limit=input_data.limit,
                require_verified=input_data.require_verified,
                api_key=input_data.api_key
            )
            
            agents = discovery_result.get('agents', [])
            total_found = discovery_result.get('count', len(agents))
            
            # Yield discovery results
            yield "agents", agents
            yield "total_found", total_found
            yield "search_capability", input_data.capability
            yield "error", ""
            
            logger.info(
                f"Agent discovery complete: found {total_found} agents "
                f"for capability '{input_data.capability}' with min_trust={input_data.min_trust_score}"
            )
            
        except httpx.HTTPError as e:
            error_msg = f"Joy discovery API error: {str(e)}"
            logger.error(error_msg)
            yield "error", error_msg
            yield "agents", []
            yield "total_found", 0
            yield "search_capability", input_data.capability
        except Exception as e:
            error_msg = f"Agent discovery failed: {str(e)}"
            logger.error(error_msg)
            yield "error", error_msg
            yield "agents", []
            yield "total_found", 0
            yield "search_capability", input_data.capability


class TrustDecision(Enum):
    """
    Enumeration of possible trust decisions for agent delegation.
    """
    TRUST = "trust"
    REJECT = "reject"
    ESCALATE = "escalate"


class JoyShouldTrustBlock(Block):
    """
    Make an intelligent trust decision for agent delegation using Joy network data.
    
    This block combines trust scores, verification status, capability matching, and
    risk assessment to provide a recommendation on whether to trust an agent for
    a specific task. It implements business logic for trust decisions rather than
    just returning raw scores.
    
    Use this block when you need a clear trust/reject decision rather than manual
    interpretation of trust metrics.
    """

    class Input(BlockSchemaInput):
        agent_id: str = SchemaField(
            description="The unique identifier of the agent to evaluate for trust.",
            placeholder="ag_5a549ffa2bb51349ec5b3ee7",
            title="Agent ID"
        )
        task_description: str = SchemaField(
            description="Description of the task you want to delegate to the agent.",
            placeholder="Analyze sales data and generate monthly report",
            title="Task Description"
        )
        risk_tolerance: float = SchemaField(
            description="Your risk tolerance level (0.0 = very conservative, 1.0 = high risk).",
            default=0.5,
            title="Risk Tolerance"
        )
        require_capability_match: bool = SchemaField(
            description="Whether the agent must have capabilities matching the task description.",
            default=True,
            title="Require Capability Match"
        )
        api_key: str | None = SchemaField(
            description="Joy API key for authenticated trust evaluation.",
            default=None,
            title="API Key",
            advanced=True
        )

    class Output(BlockSchemaOutput):
        decision: TrustDecision = SchemaField(
            description="The recommended trust decision: 'trust', 'reject', or 'escalate'."
        )
        confidence: float = SchemaField(
            description="Confidence level in the decision (0.0 to 1.0)."
        )
        reasoning: str = SchemaField(
            description="Human-readable explanation of the trust decision."
        )
        trust_score: float = SchemaField(
            description="The agent's trust score from Joy network."
        )
        risk_assessment: str = SchemaField(
            description="Risk level assessment: 'low', 'medium', or 'high'."
        )
        error: str = SchemaField(
            description="Error message if trust evaluation failed."
        )

    def __init__(self):
        super().__init__(
            id="joy-should-trust-001",
            description="Make intelligent trust decisions for agent delegation using Joy network data and risk assessment.",
            categories={BlockCategory.AGENT, BlockCategory.LOGIC},
            input_schema=JoyShouldTrustBlock.Input,
            output_schema=JoyShouldTrustBlock.Output,
            test_input=[
                {
                    "agent_id": "ag_5a549ffa2bb51349ec5b3ee7",
                    "task_description": "Data analysis and reporting",
                    "risk_tolerance": 0.6,
                    "require_capability_match": True
                },
                {
                    "agent_id": "ag_unknown_agent",
                    "task_description": "Critical financial transaction",
                    "risk_tolerance": 0.1,
                    "require_capability_match": True
                }
            ],
            test_output=[
                ("decision", TrustDecision.TRUST),
                ("confidence", 0.85),
                ("reasoning", "High trust score and verified status"),
                ("trust_score", 1.7),
                ("risk_assessment", "low"),
                ("error", "")
            ]
        )

    def _assess_risk_level(self, trust_score: float, verified: bool, vouch_count: int) -> str:
        """
        Assess the risk level based on agent trust metrics.
        
        Args:
            trust_score: Agent's trust score
            verified: Whether agent is verified
            vouch_count: Number of vouches received
            
        Returns:
            Risk level: 'low', 'medium', or 'high'
        """
        if trust_score >= 2.0 and verified and vouch_count >= 10:
            return 'low'
        elif trust_score >= 1.0 and vouch_count >= 5:
            return 'medium'
        else:
            return 'high'

    def _make_trust_decision(
        self,
        trust_score: float,
        verified: bool,
        vouch_count: int,
        risk_tolerance: float,
        capability_match: bool,
        require_capability_match: bool
    ) -> tuple[TrustDecision, float, str]:
        """
        Make a trust decision based on multiple factors.
        
        Args:
            trust_score: Agent's trust score
            verified: Whether agent is verified
            vouch_count: Number of vouches
            risk_tolerance: User's risk tolerance
            capability_match: Whether capabilities match task
            require_capability_match: Whether capability match is required
            
        Returns:
            Tuple of (decision, confidence, reasoning)
        """
        risk_level = self._assess_risk_level(trust_score, verified, vouch_count)
        
        # Check capability requirement
        if require_capability_match and not capability_match:
            return (
                TrustDecision.REJECT,
                0.9,
                "Agent lacks required capabilities for the task"
            )
        
        # Conservative approach for high-risk agents
        if risk_level == 'high' and risk_tolerance < 0.3:
            return (
                TrustDecision.REJECT,
                0.8,
                f"High risk agent (trust={trust_score}) exceeds conservative risk tolerance"
            )
        
        # Escalate borderline cases
        if risk_level == 'medium' and trust_score < 1.5 and risk_tolerance < 0.5:
            return (
                TrustDecision.ESCALATE,
                0.6,
                "Moderate risk agent requires human review"
            )
        
        # Trust high-quality agents
        if trust_score >= 1.5 and verified:
            return (
                TrustDecision.TRUST,
                0.9,
                f"High trust score ({trust_score}) and verified status"
            )
        
        # Trust based on risk tolerance
        if trust_score >= (2.0 - risk_tolerance):
            confidence = min(0.8, trust_score / 2.5 + risk_tolerance / 2)
            return (
                TrustDecision.TRUST,
                confidence,
                f"Trust score ({trust_score}) meets risk tolerance threshold"
            )
        
        # Default to rejection for low trust
        return (
            TrustDecision.REJECT,
            0.7,
            f"Trust score ({trust_score}) below acceptable threshold"
        )

    @property
    def test_mock(self) -> dict[str, Any]:
        """
        Mock data for testing trust decision scenarios.
        
        Returns:
            Dictionary mapping agent IDs to mock trust data
        """
        return {
            "ag_5a549ffa2bb51349ec5b3ee7": {
                "id": "ag_5a549ffa2bb51349ec5b3ee7",
                "name": "Jenkins",
                "trust_score": 1.7,
                "verified": True,
                "vouch_count": 15,
                "capabilities": ["automation", "data-analysis"]
            },
            "ag_unknown_agent": {
                "id": "ag_unknown_agent", 
                "name": "Unknown Agent",
                "trust_score": 0.3,
                "verified": False,
                "vouch_count": 2,
                "capabilities": ["basic"]
            }
        }

    async def run(
        self,
        input_data: Input,
        *,
        execution_context: ExecutionContext,
        **kwargs
    ) -> BlockOutput:
        """
        Execute intelligent trust decision making for agent delegation.
        
        Args:
            input_data: Trust evaluation parameters
            execution_context: AutoGPT execution context
            **kwargs: Additional execution parameters
            
        Yields:
            Tuple of (output_name, output_value) for trust decision results
        """
        try:
            # Fetch agent data using shared helper
            agent_data = await _fetch_agent(
                input_data.agent_id, 
                input_data.api_key
            )
            
            if not agent_data:
                yield "error", f"Agent not found: {input_data.agent_id}"
                yield "decision", TrustDecision.REJECT
                yield "confidence", 0.0
                yield "reasoning", "Agent not found"
                yield "trust_score", 0.0
                yield "risk_assessment", "high"
                return
            
            # Extract agent metrics
            trust_score = agent_data.get('trust_score', 0.0)
            verified = agent_data.get('verified', False)
            vouch_count = agent_data.get('vouch_count', 0)
            capabilities = agent_data.get('capabilities', [])
            
            # Assess capability match (simple keyword matching)
            capability_match = any(
                cap.lower() in input_data.task_description.lower()
                for cap in capabilities
            ) if capabilities else False
            
            # Make trust decision
            decision, confidence, reasoning = self._make_trust_decision(
                trust_score=trust_score,
                verified=verified, 
                vouch_count=vouch_count,
                risk_tolerance=input_data.risk_tolerance,
                capability_match=capability_match,
                require_capability_match=input_data.require_capability_match
            )
            
            risk_assessment = self._assess_risk_level(trust_score, verified, vouch_count)
            
            # Yield trust decision results
            yield "decision", decision
            yield "confidence", confidence
            yield "reasoning", reasoning
            yield "trust_score", trust_score
            yield "risk_assessment", risk_assessment
            yield "error", ""
            
            logger.info(
                f"Trust decision for {input_data.agent_id}: {decision.value} "
                f"(confidence={confidence:.2f}, risk={risk_assessment})"
            )
            
        except Exception as e:
            error_msg = f"Trust decision failed: {str(e)}"
            logger.error(error_msg)
            yield "error", error_msg
            yield "decision", TrustDecision.REJECT
            yield "confidence", 0.0
            yield "reasoning", "Error occurred during trust evaluation"
            yield "trust_score", 0.0
            yield "risk_assessment", "high"