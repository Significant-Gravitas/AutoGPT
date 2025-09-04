"""Tests for get_agent_details tool."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.server.v2.chat.tools.get_agent_details import GetAgentDetailsTool
from backend.server.v2.chat.tools.models import (
    AgentDetailsNeedLoginResponse,
    AgentDetailsResponse,
    ErrorResponse,
)


@pytest.fixture
def get_agent_details_tool():
    """Create a GetAgentDetailsTool instance."""
    return GetAgentDetailsTool()


@pytest.fixture
def mock_graph():
    """Create a mock graph object."""
    return MagicMock(
        id="test-agent-id",
        name="Test Agent",
        description="A test agent for unit tests",
        version=1,
        is_latest=True,
        input_schema={
            "type": "object",
            "properties": {
                "input1": {
                    "type": "string",
                    "description": "First input",
                    "default": "default_value",
                },
                "input2": {
                    "type": "number",
                    "description": "Second input",
                },
            },
            "required": ["input2"],
        },
        credentials_input_schema={
            "openai": {
                "provider": "openai",
                "type": "api_key",
                "description": "OpenAI API key",
            },
            "github": {
                "provider": "github",
                "type": "oauth",
                "scopes": ["repo", "user"],
            },
        },
        webhook_input_node=MagicMock(
            block=MagicMock(name="WebhookTrigger"),
        ),
        has_external_trigger=True,
        trigger_setup_info={
            "type": "webhook",
            "method": "POST",
            "headers": ["X-Webhook-Secret"],
        },
        executions=[
            MagicMock(status="SUCCESS"),
            MagicMock(status="SUCCESS"),
            MagicMock(status="FAILED"),
        ],
    )


@pytest.fixture
def mock_store_listing():
    """Create a mock store listing."""
    return MagicMock(
        id="store-123",
        graph_id="test-agent-id",
        rating=4.5,
        reviews_count=10,
        runs=100,
    )


@pytest.mark.asyncio
async def test_get_agent_details_no_agent_id(get_agent_details_tool) -> None:
    """Test error when no agent ID provided."""
    result = await get_agent_details_tool.execute(
        user_id="user-123",
        session_id="test-session",
    )

    assert isinstance(result, ErrorResponse)
    assert "provide an agent ID" in result.message.lower()


@pytest.mark.asyncio
async def test_get_agent_details_agent_not_found(get_agent_details_tool) -> None:
    """Test error when agent not found."""
    with patch("backend.server.v2.chat.tools.get_agent_details.graph_db") as mock_db:
        mock_db.get_graph = AsyncMock(return_value=None)

        result = await get_agent_details_tool.execute(
            user_id="user-123",
            session_id="test-session",
            agent_id="nonexistent-agent",
        )

        assert isinstance(result, ErrorResponse)
        assert "not found" in result.message.lower()


@pytest.mark.asyncio
async def test_get_agent_details_authenticated_user(
    get_agent_details_tool, mock_graph
) -> None:
    """Test getting agent details for authenticated user."""
    with patch(
        "backend.server.v2.chat.tools.get_agent_details.graph_db"
    ) as mock_db, patch(
        "backend.server.v2.chat.tools.get_agent_details.library_db"
    ) as mock_lib:

        mock_db.get_graph = AsyncMock(return_value=mock_graph)
        mock_lib.get_library_agent = AsyncMock(
            return_value=MagicMock(graph_id="test-agent-id"),
        )

        result = await get_agent_details_tool.execute(
            user_id="user-123",
            session_id="test-session",
            agent_id="test-agent-id",
        )

        assert isinstance(result, AgentDetailsResponse)
        assert result.user_authenticated is True
        assert result.agent.id == "test-agent-id"
        assert result.agent.name == "Test Agent"
        assert result.agent.in_library is True
        assert len(result.agent.inputs) == 2
        assert len(result.agent.credentials) == 2
        assert result.agent.execution_options.webhook is True


@pytest.mark.asyncio
async def test_get_agent_details_anonymous_user(
    get_agent_details_tool, mock_graph
) -> None:
    """Test getting agent details for anonymous user."""
    with patch("backend.server.v2.chat.tools.get_agent_details.graph_db") as mock_db:
        mock_db.get_graph = AsyncMock(return_value=mock_graph)

        result = await get_agent_details_tool.execute(
            user_id=None,
            session_id="test-session",
            agent_id="test-agent-id",
        )

        assert isinstance(result, AgentDetailsNeedLoginResponse)
        assert result.agent.id == "test-agent-id"
        assert result.agent.in_library is False


@pytest.mark.asyncio
async def test_get_agent_details_with_version(
    get_agent_details_tool, mock_graph
) -> None:
    """Test getting specific version of agent."""
    mock_graph.version = 3
    mock_graph.is_latest = False

    with patch("backend.server.v2.chat.tools.get_agent_details.graph_db") as mock_db:
        mock_db.get_graph = AsyncMock(return_value=mock_graph)

        result = await get_agent_details_tool.execute(
            user_id="user-123",
            session_id="test-session",
            agent_id="test-agent-id",
            agent_version=3,
        )

        assert isinstance(result, AgentDetailsResponse)
        assert result.agent.version == 3
        assert result.agent.is_latest is False


@pytest.mark.asyncio
async def test_get_agent_details_marketplace_stats(
    get_agent_details_tool,
    mock_graph,
    mock_store_listing,
) -> None:
    """Test agent details include marketplace stats."""
    with patch(
        "backend.server.v2.chat.tools.get_agent_details.graph_db"
    ) as mock_db, patch(
        "backend.server.v2.chat.tools.get_agent_details.store_db"
    ) as mock_store:

        mock_db.get_graph = AsyncMock(return_value=mock_graph)
        mock_store.get_store_listing_by_graph_id = AsyncMock(
            return_value=mock_store_listing,
        )

        result = await get_agent_details_tool.execute(
            user_id="user-123",
            session_id="test-session",
            agent_id="test-agent-id",
        )

        assert isinstance(result, AgentDetailsResponse)
        assert result.agent.is_marketplace is True
        assert result.agent.stats is not None
        assert result.agent.stats["rating"] == 4.5
        assert result.agent.stats["reviews"] == 10
        assert result.agent.stats["runs"] == 100


@pytest.mark.asyncio
async def test_get_agent_details_no_webhook_support(get_agent_details_tool) -> None:
    """Test agent without webhook support."""
    mock_graph = MagicMock(
        id="test-agent-id",
        name="Test Agent",
        description="A test agent",
        version=1,
        webhook_input_node=None,
        has_external_trigger=False,
        input_schema={},
        credentials_input_schema={},
    )

    with patch("backend.server.v2.chat.tools.get_agent_details.graph_db") as mock_db:
        mock_db.get_graph = AsyncMock(return_value=mock_graph)

        result = await get_agent_details_tool.execute(
            user_id="user-123",
            session_id="test-session",
            agent_id="test-agent-id",
        )

        assert isinstance(result, AgentDetailsResponse)
        assert result.agent.execution_options.webhook is False


@pytest.mark.asyncio
async def test_get_agent_details_complex_input_schema(get_agent_details_tool) -> None:
    """Test agent with complex input schema including enums and formats."""
    mock_graph = MagicMock(
        id="test-agent-id",
        name="Test Agent",
        description="A test agent",
        version=1,
        input_schema={
            "type": "object",
            "properties": {
                "email": {
                    "type": "string",
                    "format": "email",
                    "description": "User email",
                },
                "priority": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "description": "Task priority",
                    "default": "medium",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tags for the task",
                },
            },
            "required": ["email"],
        },
        credentials_input_schema={},
        webhook_input_node=None,
    )

    with patch("backend.server.v2.chat.tools.get_agent_details.graph_db") as mock_db:
        mock_db.get_graph = AsyncMock(return_value=mock_graph)

        result = await get_agent_details_tool.execute(
            user_id="user-123",
            session_id="test-session",
            agent_id="test-agent-id",
        )

        assert isinstance(result, AgentDetailsResponse)

        # Check email input
        email_input = result.agent.inputs["email"]
        assert email_input["type"] == "string"
        assert email_input["format"] == "email"
        assert email_input["required"] is True

        # Check priority input with enum
        priority_input = result.agent.inputs["priority"]
        assert priority_input["type"] == "string"
        assert priority_input["options"] == ["low", "medium", "high"]
        assert priority_input["default"] == "medium"
        assert priority_input["required"] is False

        # Check array input
        tags_input = result.agent.inputs["tags"]
        assert tags_input["type"] == "array"


@pytest.mark.asyncio
async def test_get_agent_details_error_handling(get_agent_details_tool) -> None:
    """Test error handling in get_agent_details."""
    with patch("backend.server.v2.chat.tools.get_agent_details.graph_db") as mock_db:
        mock_db.get_graph = AsyncMock(side_effect=Exception("Database error"))

        result = await get_agent_details_tool.execute(
            user_id="user-123",
            session_id="test-session",
            agent_id="test-agent-id",
        )

        assert isinstance(result, ErrorResponse)
        assert "failed to get agent details" in result.message.lower()


@pytest.mark.asyncio
async def test_get_agent_details_public_agent_fallback(
    get_agent_details_tool,
    mock_graph,
) -> None:
    """Test fallback to public/marketplace agent when not in user library."""
    with patch("backend.server.v2.chat.tools.get_agent_details.graph_db") as mock_db:
        # First call returns None (not in user's library)
        # Second call returns the public agent
        mock_db.get_graph = AsyncMock(side_effect=[None, mock_graph])

        result = await get_agent_details_tool.execute(
            user_id="user-123",
            session_id="test-session",
            agent_id="test-agent-id",
        )

        assert isinstance(result, AgentDetailsResponse)
        assert result.agent.id == "test-agent-id"
        assert mock_db.get_graph.call_count == 2

        # First call with user_id
        assert mock_db.get_graph.call_args_list[0][1]["user_id"] == "user-123"
        # Second call without user_id (public)
        assert mock_db.get_graph.call_args_list[1][1]["user_id"] is None


@pytest.mark.asyncio
async def test_get_agent_details_anon_user_prefix(
    get_agent_details_tool, mock_graph
) -> None:
    """Test that anon_ prefixed users are treated as anonymous."""
    with patch("backend.server.v2.chat.tools.get_agent_details.graph_db") as mock_db:
        mock_db.get_graph = AsyncMock(return_value=mock_graph)

        result = await get_agent_details_tool.execute(
            user_id="anon_abc123",  # Anonymous user
            session_id="test-session",
            agent_id="test-agent-id",
        )

        assert isinstance(result, AgentDetailsNeedLoginResponse)
        assert "sign in" in result.message.lower()
