"""Tests for setup_agent tool."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.server.v2.chat.tools.models import (
    ErrorResponse,
    NeedLoginResponse,
    PresetCreatedResponse,
    ScheduleCreatedResponse,
    WebhookCreatedResponse,
)
from backend.server.v2.chat.tools.setup_agent import SetupAgentTool


@pytest.fixture
def setup_agent_tool():
    """Create a SetupAgentTool instance."""
    return SetupAgentTool()


@pytest.fixture
def mock_graph():
    """Create a mock graph."""
    return MagicMock(
        id="test-agent-id",
        name="Test Agent",
        version=1,
        webhook_input_node=MagicMock(
            block=MagicMock(name="WebhookTrigger"),
        ),
    )


@pytest.fixture
def mock_schedule_info():
    """Mock schedule information."""
    return MagicMock(
        id="schedule-123",
        next_run_time="2024-01-01T10:00:00Z",
    )


@pytest.fixture
def mock_webhook():
    """Mock webhook object."""
    return MagicMock(
        id="webhook-123",
        webhook_url="https://api.example.com/webhook/123",
    )


@pytest.fixture
def mock_preset():
    """Mock preset object."""
    return MagicMock(
        id="preset-123",
    )


@pytest.mark.asyncio
async def test_setup_agent_requires_authentication(setup_agent_tool) -> None:
    """Test that tool requires authentication."""
    result = await setup_agent_tool.execute(
        user_id=None,
        session_id="test-session",
        agent_id="test-agent",
        setup_type="schedule",
    )

    assert isinstance(result, NeedLoginResponse)


@pytest.mark.asyncio
async def test_setup_agent_no_agent_id(setup_agent_tool) -> None:
    """Test error when no agent ID provided."""
    result = await setup_agent_tool.execute(
        user_id="user-123",
        session_id="test-session",
        setup_type="schedule",
    )

    assert isinstance(result, ErrorResponse)
    assert "provide an agent ID" in result.message.lower()


@pytest.mark.asyncio
async def test_setup_agent_no_setup_type(setup_agent_tool) -> None:
    """Test error when no setup type provided."""
    result = await setup_agent_tool.execute(
        user_id="user-123",
        session_id="test-session",
        agent_id="test-agent",
    )

    assert isinstance(result, ErrorResponse)
    assert "specify setup type" in result.message.lower()


@pytest.mark.asyncio
async def test_setup_agent_invalid_setup_type(setup_agent_tool, mock_graph) -> None:
    """Test error with invalid setup type."""
    with patch("backend.server.v2.chat.tools.setup_agent.graph_db") as mock_db:
        mock_db.get_graph = AsyncMock(return_value=mock_graph)

        result = await setup_agent_tool.execute(
            user_id="user-123",
            session_id="test-session",
            agent_id="test-agent",
            setup_type="invalid_type",
        )

        assert isinstance(result, ErrorResponse)
        assert "unknown setup type" in result.message.lower()


@pytest.mark.asyncio
async def test_setup_agent_schedule_success(
    setup_agent_tool,
    mock_graph,
    mock_schedule_info,
) -> None:
    """Test successful schedule creation."""
    with patch("backend.server.v2.chat.tools.setup_agent.graph_db") as mock_db, patch(
        "backend.server.v2.chat.tools.setup_agent.SchedulerClient"
    ) as mock_scheduler, patch("backend.server.v2.chat.tools.setup_agent.CronTrigger"):

        mock_db.get_graph = AsyncMock(return_value=mock_graph)
        scheduler_instance = MagicMock()
        scheduler_instance.add_execution_schedule = AsyncMock(
            return_value=mock_schedule_info
        )
        mock_scheduler.return_value = scheduler_instance

        result = await setup_agent_tool.execute(
            user_id="user-123",
            session_id="test-session",
            agent_id="test-agent-id",
            setup_type="schedule",
            name="Daily Report",
            cron="0 9 * * *",
            timezone="America/New_York",
            inputs={"report_type": "daily"},
            credentials={"openai": "cred-123"},
        )

        assert isinstance(result, ScheduleCreatedResponse)
        assert result.schedule_id == "schedule-123"
        assert result.name == "Daily Report"
        assert result.cron == "0 9 * * *"
        assert result.timezone == "America/New_York"
        assert result.next_run == "2024-01-01T10:00:00Z"
        assert result.graph_id == "test-agent-id"
        assert "created successfully" in result.message


@pytest.mark.asyncio
async def test_setup_agent_schedule_missing_cron(setup_agent_tool, mock_graph) -> None:
    """Test error when cron expression missing for schedule."""
    with patch("backend.server.v2.chat.tools.setup_agent.graph_db") as mock_db:
        mock_db.get_graph = AsyncMock(return_value=mock_graph)

        result = await setup_agent_tool.execute(
            user_id="user-123",
            session_id="test-session",
            agent_id="test-agent-id",
            setup_type="schedule",
            name="Daily Report",
        )

        assert isinstance(result, ErrorResponse)
        assert "cron expression is required" in result.message.lower()


@pytest.mark.asyncio
async def test_setup_agent_schedule_invalid_cron(setup_agent_tool, mock_graph) -> None:
    """Test error with invalid cron expression."""
    with patch("backend.server.v2.chat.tools.setup_agent.graph_db") as mock_db, patch(
        "backend.server.v2.chat.tools.setup_agent.CronTrigger"
    ) as mock_cron:

        mock_db.get_graph = AsyncMock(return_value=mock_graph)
        mock_cron.from_crontab.side_effect = Exception("Invalid cron")

        result = await setup_agent_tool.execute(
            user_id="user-123",
            session_id="test-session",
            agent_id="test-agent-id",
            setup_type="schedule",
            name="Daily Report",
            cron="invalid cron",
        )

        assert isinstance(result, ErrorResponse)
        assert "invalid cron expression" in result.message.lower()


@pytest.mark.asyncio
async def test_setup_agent_schedule_invalid_timezone(
    setup_agent_tool, mock_graph
) -> None:
    """Test error with invalid timezone."""
    with patch("backend.server.v2.chat.tools.setup_agent.graph_db") as mock_db, patch(
        "backend.server.v2.chat.tools.setup_agent.CronTrigger"
    ):

        mock_db.get_graph = AsyncMock(return_value=mock_graph)

        result = await setup_agent_tool.execute(
            user_id="user-123",
            session_id="test-session",
            agent_id="test-agent-id",
            setup_type="schedule",
            name="Daily Report",
            cron="0 9 * * *",
            timezone="Invalid/Timezone",
        )

        assert isinstance(result, ErrorResponse)
        assert "invalid timezone" in result.message.lower()


@pytest.mark.asyncio
async def test_setup_agent_webhook_success(
    setup_agent_tool,
    mock_graph,
    mock_webhook,
    mock_preset,
) -> None:
    """Test successful webhook setup."""
    with patch("backend.server.v2.chat.tools.setup_agent.graph_db") as mock_db, patch(
        "backend.server.v2.chat.tools.setup_agent.setup_webhook_for_block"
    ) as mock_setup_webhook, patch(
        "backend.server.v2.chat.tools.setup_agent.library_db"
    ) as mock_lib:

        mock_db.get_graph = AsyncMock(return_value=mock_graph)
        mock_setup_webhook.return_value = (mock_webhook, "Webhook created")
        mock_lib.create_preset = AsyncMock(return_value=mock_preset)

        result = await setup_agent_tool.execute(
            user_id="user-123",
            session_id="test-session",
            agent_id="test-agent-id",
            setup_type="webhook",
            name="GitHub Webhook",
            description="Trigger on GitHub events",
            webhook_config={"secret": "webhook_secret"},
            inputs={"repo": "my-repo"},
            credentials={"github": "cred-456"},
        )

        assert isinstance(result, WebhookCreatedResponse)
        assert result.webhook_id == "webhook-123"
        assert result.webhook_url == "https://api.example.com/webhook/123"
        assert result.preset_id == "preset-123"
        assert result.name == "GitHub Webhook"
        assert "created successfully" in result.message


@pytest.mark.asyncio
async def test_setup_agent_webhook_no_support(setup_agent_tool) -> None:
    """Test error when agent doesn't support webhooks."""
    mock_graph = MagicMock(
        id="test-agent",
        name="Test Agent",
        version=1,
        webhook_input_node=None,  # No webhook support
    )

    with patch("backend.server.v2.chat.tools.setup_agent.graph_db") as mock_db:
        mock_db.get_graph = AsyncMock(return_value=mock_graph)

        result = await setup_agent_tool.execute(
            user_id="user-123",
            session_id="test-session",
            agent_id="test-agent",
            setup_type="webhook",
            name="Webhook Setup",
        )

        assert isinstance(result, ErrorResponse)
        assert "does not support webhook" in result.message.lower()


@pytest.mark.asyncio
async def test_setup_agent_webhook_creation_failed(
    setup_agent_tool, mock_graph
) -> None:
    """Test error when webhook creation fails."""
    with patch("backend.server.v2.chat.tools.setup_agent.graph_db") as mock_db, patch(
        "backend.server.v2.chat.tools.setup_agent.setup_webhook_for_block"
    ) as mock_setup:

        mock_db.get_graph = AsyncMock(return_value=mock_graph)
        mock_setup.return_value = (None, "Invalid configuration")

        result = await setup_agent_tool.execute(
            user_id="user-123",
            session_id="test-session",
            agent_id="test-agent-id",
            setup_type="webhook",
            name="Failed Webhook",
        )

        assert isinstance(result, ErrorResponse)
        assert "failed to create webhook" in result.message.lower()
        assert "Invalid configuration" in result.message


@pytest.mark.asyncio
async def test_setup_agent_preset_success(
    setup_agent_tool, mock_graph, mock_preset
) -> None:
    """Test successful preset creation."""
    with patch("backend.server.v2.chat.tools.setup_agent.graph_db") as mock_db, patch(
        "backend.server.v2.chat.tools.setup_agent.library_db"
    ) as mock_lib:

        mock_db.get_graph = AsyncMock(return_value=mock_graph)
        mock_lib.create_preset = AsyncMock(return_value=mock_preset)

        result = await setup_agent_tool.execute(
            user_id="user-123",
            session_id="test-session",
            agent_id="test-agent-id",
            setup_type="preset",
            name="My Preset",
            description="Preset for quick execution",
            inputs={"mode": "fast"},
            credentials={"api_key": "key-123"},
        )

        assert isinstance(result, PresetCreatedResponse)
        assert result.preset_id == "preset-123"
        assert result.name == "My Preset"
        assert result.graph_id == "test-agent-id"
        assert "created successfully" in result.message


@pytest.mark.asyncio
async def test_setup_agent_marketplace_agent_added_to_library(
    setup_agent_tool,
    mock_graph,
    mock_preset,
) -> None:
    """Test that marketplace agents are added to library."""
    with patch("backend.server.v2.chat.tools.setup_agent.graph_db") as mock_db, patch(
        "backend.server.v2.chat.tools.setup_agent.library_db"
    ) as mock_lib:

        # First call returns None (not in library), second returns marketplace agent
        mock_db.get_graph = AsyncMock(side_effect=[None, mock_graph])
        mock_lib.create_library_agent = AsyncMock()
        mock_lib.create_preset = AsyncMock(return_value=mock_preset)

        result = await setup_agent_tool.execute(
            user_id="user-123",
            session_id="test-session",
            agent_id="test-agent-id",
            setup_type="preset",
            name="Marketplace Preset",
        )

        assert isinstance(result, PresetCreatedResponse)
        # Verify agent was added to library
        mock_lib.create_library_agent.assert_called_once()


@pytest.mark.asyncio
async def test_setup_agent_not_found(setup_agent_tool) -> None:
    """Test error when agent not found."""
    with patch("backend.server.v2.chat.tools.setup_agent.graph_db") as mock_db:
        mock_db.get_graph = AsyncMock(return_value=None)

        result = await setup_agent_tool.execute(
            user_id="user-123",
            session_id="test-session",
            agent_id="nonexistent",
            setup_type="preset",
        )

        assert isinstance(result, ErrorResponse)
        assert "not found" in result.message.lower()


@pytest.mark.asyncio
async def test_setup_agent_credential_conversion(
    setup_agent_tool,
    mock_graph,
    mock_preset,
) -> None:
    """Test credential format conversion."""
    with patch("backend.server.v2.chat.tools.setup_agent.graph_db") as mock_db, patch(
        "backend.server.v2.chat.tools.setup_agent.library_db"
    ) as mock_lib:

        mock_db.get_graph = AsyncMock(return_value=mock_graph)
        mock_lib.create_preset = AsyncMock(return_value=mock_preset)

        result = await setup_agent_tool.execute(
            user_id="user-123",
            session_id="test-session",
            agent_id="test-agent-id",
            setup_type="preset",
            name="Cred Test",
            credentials={
                "api_key": "simple-string-id",  # String format
                "oauth": {  # Dict format
                    "id": "oauth-123",
                    "type": "oauth",
                    "provider": "github",
                },
            },
        )

        assert isinstance(result, PresetCreatedResponse)

        # Verify create_preset was called
        call_args = mock_lib.create_preset.call_args[1]
        preset_data = call_args["preset"]

        # Check credential conversion
        assert "api_key" in preset_data["credentials"]
        assert "oauth" in preset_data["credentials"]


@pytest.mark.asyncio
async def test_setup_agent_error_handling(setup_agent_tool) -> None:
    """Test general error handling."""
    with patch("backend.server.v2.chat.tools.setup_agent.graph_db") as mock_db:
        mock_db.get_graph = AsyncMock(side_effect=Exception("Database error"))

        result = await setup_agent_tool.execute(
            user_id="user-123",
            session_id="test-session",
            agent_id="test-agent",
            setup_type="preset",
        )

        assert isinstance(result, ErrorResponse)
        assert "failed to set up agent" in result.message.lower()
