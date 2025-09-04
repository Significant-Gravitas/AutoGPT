"""Tests for get_required_setup_info tool."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.server.v2.chat.tools.get_required_setup_info import (
    GetRequiredSetupInfoTool,
)
from backend.server.v2.chat.tools.models import (
    ErrorResponse,
    NeedLoginResponse,
    SetupRequirementsResponse,
)


@pytest.fixture
def setup_info_tool():
    """Create a GetRequiredSetupInfoTool instance."""
    return GetRequiredSetupInfoTool()


@pytest.fixture
def mock_graph_with_requirements():
    """Create a mock graph with various requirements."""
    return MagicMock(
        id="test-agent-id",
        name="Test Agent",
        version=1,
        input_schema={
            "type": "object",
            "properties": {
                "api_endpoint": {
                    "type": "string",
                    "description": "API endpoint URL",
                    "format": "url",
                },
                "max_retries": {
                    "type": "integer",
                    "description": "Maximum number of retries",
                    "default": 3,
                },
                "mode": {
                    "type": "string",
                    "enum": ["fast", "accurate", "balanced"],
                    "description": "Processing mode",
                    "default": "balanced",
                },
            },
            "required": ["api_endpoint"],
        },
        credentials_input_schema={
            "openai": {
                "provider": "openai",
                "type": "api_key",
                "description": "OpenAI API key for GPT access",
            },
            "github": {
                "provider": "github",
                "type": "oauth",
                "scopes": ["repo", "user"],
                "description": "GitHub OAuth for repository access",
            },
        },
        webhook_input_node=MagicMock(
            block=MagicMock(name="WebhookTrigger"),
        ),
        has_external_trigger=True,
        trigger_setup_info={
            "type": "webhook",
            "method": "POST",
        },
    )


@pytest.fixture
def mock_user_credentials():
    """Mock user credentials."""
    return [
        MagicMock(
            id="cred-1",
            provider="openai",
            type="api_key",
        ),
        # Note: User doesn't have GitHub credentials
    ]


@pytest.mark.asyncio
async def test_setup_info_requires_authentication(setup_info_tool) -> None:
    """Test that tool requires authentication."""
    result = await setup_info_tool.execute(
        user_id=None,
        session_id="test-session",
        agent_id="test-agent",
    )

    assert isinstance(result, NeedLoginResponse)
    assert "sign in" in result.message.lower()


@pytest.mark.asyncio
async def test_setup_info_anonymous_user(setup_info_tool) -> None:
    """Test that anonymous users get login prompt."""
    result = await setup_info_tool.execute(
        user_id="anon_123",
        session_id="test-session",
        agent_id="test-agent",
    )

    assert isinstance(result, NeedLoginResponse)


@pytest.mark.asyncio
async def test_setup_info_no_agent_id(setup_info_tool) -> None:
    """Test error when no agent ID provided."""
    result = await setup_info_tool.execute(
        user_id="user-123",
        session_id="test-session",
    )

    assert isinstance(result, ErrorResponse)
    assert "provide an agent ID" in result.message.lower()


@pytest.mark.asyncio
async def test_setup_info_agent_not_found(setup_info_tool) -> None:
    """Test error when agent not found."""
    with patch(
        "backend.server.v2.chat.tools.get_required_setup_info.graph_db"
    ) as mock_db:
        mock_db.get_graph = AsyncMock(return_value=None)

        result = await setup_info_tool.execute(
            user_id="user-123",
            session_id="test-session",
            agent_id="nonexistent",
        )

        assert isinstance(result, ErrorResponse)
        assert "not found" in result.message.lower()


@pytest.mark.asyncio
async def test_setup_info_complete_requirements(
    setup_info_tool,
    mock_graph_with_requirements,
    mock_user_credentials,
) -> None:
    """Test getting complete setup requirements."""
    with patch(
        "backend.server.v2.chat.tools.get_required_setup_info.graph_db"
    ) as mock_db, patch(
        "backend.server.v2.chat.tools.get_required_setup_info.IntegrationCredentialsManager"
    ) as mock_creds:

        mock_db.get_graph = AsyncMock(return_value=mock_graph_with_requirements)
        mock_creds_instance = MagicMock()
        mock_creds_instance.list_credentials = AsyncMock(
            return_value=mock_user_credentials
        )
        mock_creds.return_value = mock_creds_instance

        result = await setup_info_tool.execute(
            user_id="user-123",
            session_id="test-session",
            agent_id="test-agent-id",
        )

        assert isinstance(result, SetupRequirementsResponse)
        setup_info = result.setup_info

        # Check basic info
        assert setup_info.agent_id == "test-agent-id"
        assert setup_info.agent_name == "Test Agent"
        assert setup_info.version == 1

        # Check inputs
        inputs = setup_info.requirements["inputs"]
        assert len(inputs) == 3

        # Check required input
        api_input = next(i for i in inputs if i.name == "api_endpoint")
        assert api_input.required is True
        assert api_input.type == "string"
        assert api_input.format == "url"

        # Check optional input with default
        retries_input = next(i for i in inputs if i.name == "max_retries")
        assert retries_input.required is False
        assert retries_input.default == 3

        # Check enum input
        mode_input = next(i for i in inputs if i.name == "mode")
        assert mode_input.options == ["fast", "accurate", "balanced"]

        # Check credentials
        creds = setup_info.requirements["credentials"]
        assert len(creds) == 2

        # Check user has OpenAI
        openai_cred = next(c for c in creds if c.provider == "openai")
        assert openai_cred.user_has is True
        assert openai_cred.credential_id == "cred-1"

        # Check user doesn't have GitHub
        github_cred = next(c for c in creds if c.provider == "github")
        assert github_cred.user_has is False
        assert github_cred.scopes == ["repo", "user"]

        # Check readiness
        assert setup_info.user_readiness.has_all_credentials is False
        assert "github" in setup_info.user_readiness.missing_credentials
        assert setup_info.user_readiness.ready_to_run is False

        # Check setup instructions
        assert len(setup_info.setup_instructions) > 0
        assert "github" in setup_info.setup_instructions[0].lower()


@pytest.mark.asyncio
async def test_setup_info_user_ready(
    setup_info_tool, mock_graph_with_requirements
) -> None:
    """Test when user has all required credentials."""
    all_creds = [
        MagicMock(id="cred-1", provider="openai", type="api_key"),
        MagicMock(id="cred-2", provider="github", type="oauth"),
    ]

    with patch(
        "backend.server.v2.chat.tools.get_required_setup_info.graph_db"
    ) as mock_db, patch(
        "backend.server.v2.chat.tools.get_required_setup_info.IntegrationCredentialsManager"
    ) as mock_creds:

        mock_db.get_graph = AsyncMock(return_value=mock_graph_with_requirements)
        mock_creds_instance = MagicMock()
        mock_creds_instance.list_credentials = AsyncMock(return_value=all_creds)
        mock_creds.return_value = mock_creds_instance

        result = await setup_info_tool.execute(
            user_id="user-123",
            session_id="test-session",
            agent_id="test-agent-id",
        )

        assert isinstance(result, SetupRequirementsResponse)
        setup_info = result.setup_info

        # User should be ready
        assert setup_info.user_readiness.has_all_credentials is True
        assert len(setup_info.user_readiness.missing_credentials) == 0
        assert setup_info.user_readiness.ready_to_run is True
        assert "ready" in setup_info.setup_instructions[0].lower()


@pytest.mark.asyncio
async def test_setup_info_execution_modes(
    setup_info_tool, mock_graph_with_requirements
) -> None:
    """Test execution mode information."""
    with patch(
        "backend.server.v2.chat.tools.get_required_setup_info.graph_db"
    ) as mock_db, patch(
        "backend.server.v2.chat.tools.get_required_setup_info.IntegrationCredentialsManager"
    ) as mock_creds:

        mock_db.get_graph = AsyncMock(return_value=mock_graph_with_requirements)
        mock_creds_instance = MagicMock()
        mock_creds_instance.list_credentials = AsyncMock(return_value=[])
        mock_creds.return_value = mock_creds_instance

        result = await setup_info_tool.execute(
            user_id="user-123",
            session_id="test-session",
            agent_id="test-agent-id",
        )

        assert isinstance(result, SetupRequirementsResponse)
        modes = result.setup_info.requirements["execution_modes"]

        # Check manual mode (always supported)
        manual_mode = next(m for m in modes if m.type == "manual")
        assert manual_mode.supported is True

        # Check scheduled mode
        scheduled_mode = next(m for m in modes if m.type == "scheduled")
        assert scheduled_mode.supported is True
        assert "cron" in scheduled_mode.config_required

        # Check webhook mode (supported for this agent)
        webhook_mode = next(m for m in modes if m.type == "webhook")
        assert webhook_mode.supported is True
        assert webhook_mode.trigger_info is not None


@pytest.mark.asyncio
async def test_setup_info_no_webhook_support(setup_info_tool) -> None:
    """Test agent without webhook support."""
    mock_graph = MagicMock(
        id="test-agent",
        name="Test Agent",
        version=1,
        input_schema={},
        credentials_input_schema={},
        webhook_input_node=None,
        has_external_trigger=False,
    )

    with patch(
        "backend.server.v2.chat.tools.get_required_setup_info.graph_db"
    ) as mock_db, patch(
        "backend.server.v2.chat.tools.get_required_setup_info.IntegrationCredentialsManager"
    ) as mock_creds:

        mock_db.get_graph = AsyncMock(return_value=mock_graph)
        mock_creds_instance = MagicMock()
        mock_creds_instance.list_credentials = AsyncMock(return_value=[])
        mock_creds.return_value = mock_creds_instance

        result = await setup_info_tool.execute(
            user_id="user-123",
            session_id="test-session",
            agent_id="test-agent",
        )

        assert isinstance(result, SetupRequirementsResponse)
        modes = result.setup_info.requirements["execution_modes"]

        webhook_mode = next(m for m in modes if m.type == "webhook")
        assert webhook_mode.supported is False


@pytest.mark.asyncio
async def test_setup_info_no_requirements(setup_info_tool) -> None:
    """Test agent with no input or credential requirements."""
    mock_graph = MagicMock(
        id="simple-agent",
        name="Simple Agent",
        version=1,
        input_schema=None,  # No inputs
        credentials_input_schema=None,  # No credentials
        webhook_input_node=None,
    )

    with patch(
        "backend.server.v2.chat.tools.get_required_setup_info.graph_db"
    ) as mock_db, patch(
        "backend.server.v2.chat.tools.get_required_setup_info.IntegrationCredentialsManager"
    ) as mock_creds:

        mock_db.get_graph = AsyncMock(return_value=mock_graph)
        mock_creds_instance = MagicMock()
        mock_creds_instance.list_credentials = AsyncMock(return_value=[])
        mock_creds.return_value = mock_creds_instance

        result = await setup_info_tool.execute(
            user_id="user-123",
            session_id="test-session",
            agent_id="simple-agent",
        )

        assert isinstance(result, SetupRequirementsResponse)
        setup_info = result.setup_info

        # No requirements
        assert len(setup_info.requirements["inputs"]) == 0
        assert len(setup_info.requirements["credentials"]) == 0

        # Should be ready to run
        assert setup_info.user_readiness.ready_to_run is True


@pytest.mark.asyncio
async def test_setup_info_fallback_to_marketplace(
    setup_info_tool, mock_graph_with_requirements
) -> None:
    """Test fallback to marketplace agent when not in user library."""
    with patch(
        "backend.server.v2.chat.tools.get_required_setup_info.graph_db"
    ) as mock_db, patch(
        "backend.server.v2.chat.tools.get_required_setup_info.IntegrationCredentialsManager"
    ) as mock_creds:

        # First call returns None, second returns marketplace agent
        mock_db.get_graph = AsyncMock(side_effect=[None, mock_graph_with_requirements])
        mock_creds_instance = MagicMock()
        mock_creds_instance.list_credentials = AsyncMock(return_value=[])
        mock_creds.return_value = mock_creds_instance

        result = await setup_info_tool.execute(
            user_id="user-123",
            session_id="test-session",
            agent_id="test-agent-id",
        )

        assert isinstance(result, SetupRequirementsResponse)
        assert mock_db.get_graph.call_count == 2


@pytest.mark.asyncio
async def test_setup_info_with_version(
    setup_info_tool, mock_graph_with_requirements
) -> None:
    """Test getting setup info for specific version."""
    mock_graph_with_requirements.version = 5

    with patch(
        "backend.server.v2.chat.tools.get_required_setup_info.graph_db"
    ) as mock_db, patch(
        "backend.server.v2.chat.tools.get_required_setup_info.IntegrationCredentialsManager"
    ) as mock_creds:

        mock_db.get_graph = AsyncMock(return_value=mock_graph_with_requirements)
        mock_creds_instance = MagicMock()
        mock_creds_instance.list_credentials = AsyncMock(return_value=[])
        mock_creds.return_value = mock_creds_instance

        result = await setup_info_tool.execute(
            user_id="user-123",
            session_id="test-session",
            agent_id="test-agent-id",
            agent_version=5,
        )

        assert isinstance(result, SetupRequirementsResponse)
        assert result.setup_info.version == 5

        # Verify version was passed to get_graph
        mock_db.get_graph.assert_called_with(
            graph_id="test-agent-id",
            version=5,
            user_id="user-123",
            include_subgraphs=True,
        )


@pytest.mark.asyncio
async def test_setup_info_error_handling(setup_info_tool) -> None:
    """Test error handling in setup info."""
    with patch(
        "backend.server.v2.chat.tools.get_required_setup_info.graph_db"
    ) as mock_db:
        mock_db.get_graph = AsyncMock(side_effect=Exception("Database error"))

        result = await setup_info_tool.execute(
            user_id="user-123",
            session_id="test-session",
            agent_id="test-agent",
        )

        assert isinstance(result, ErrorResponse)
        assert "failed to get setup requirements" in result.message.lower()


@pytest.mark.asyncio
async def test_setup_info_credentials_error_handling(
    setup_info_tool,
    mock_graph_with_requirements,
) -> None:
    """Test handling of credential manager errors."""
    with patch(
        "backend.server.v2.chat.tools.get_required_setup_info.graph_db"
    ) as mock_db, patch(
        "backend.server.v2.chat.tools.get_required_setup_info.IntegrationCredentialsManager"
    ) as mock_creds:

        mock_db.get_graph = AsyncMock(return_value=mock_graph_with_requirements)
        mock_creds_instance = MagicMock()
        # Credential listing fails
        mock_creds_instance.list_credentials = AsyncMock(
            side_effect=Exception("Credentials service error"),
        )
        mock_creds.return_value = mock_creds_instance

        result = await setup_info_tool.execute(
            user_id="user-123",
            session_id="test-session",
            agent_id="test-agent-id",
        )

        # Should still return requirements but without user credential status
        assert isinstance(result, SetupRequirementsResponse)
        # All credentials should be marked as not having
        for cred in result.setup_info.requirements["credentials"]:
            assert cred.user_has is False
