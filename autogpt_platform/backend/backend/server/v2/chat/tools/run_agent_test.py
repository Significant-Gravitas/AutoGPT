"""Tests for run_agent tool."""

from unittest.mock import AsyncMock, MagicMock, patch

import prisma.enums
import pytest

from backend.server.v2.chat.tools.models import (
    ErrorResponse,
    ExecutionStartedResponse,
    InsufficientCreditsResponse,
    NeedLoginResponse,
    ValidationErrorResponse,
)
from backend.server.v2.chat.tools.run_agent import RunAgentTool


@pytest.fixture
def run_agent_tool():
    """Create a RunAgentTool instance."""
    return RunAgentTool()


@pytest.fixture
def mock_graph():
    """Create a mock graph."""
    return MagicMock(
        id="test-agent-id",
        name="Test Agent",
        version=1,
    )


@pytest.fixture
def mock_execution():
    """Create a mock execution."""
    return MagicMock(
        id="exec-123",
        graph_id="test-agent-id",
    )


@pytest.fixture
def mock_execution_status_completed():
    """Mock completed execution status."""
    return MagicMock(
        status=prisma.enums.AgentExecutionStatus.COMPLETED,
        ended_at=MagicMock(isoformat=lambda: "2024-01-01T10:00:00Z"),
        stats=MagicMock(error=None),
    )


@pytest.fixture
def mock_execution_status_failed():
    """Mock failed execution status."""
    return MagicMock(
        status=prisma.enums.AgentExecutionStatus.FAILED,
        ended_at=MagicMock(isoformat=lambda: "2024-01-01T10:00:00Z"),
        stats=MagicMock(error="Task failed: Invalid input"),
    )


@pytest.fixture
def mock_full_execution():
    """Mock full execution with outputs."""
    return MagicMock(
        id="exec-123",
        output_data={"result": "success", "data": [1, 2, 3]},
    )


@pytest.mark.asyncio
async def test_run_agent_requires_authentication(run_agent_tool) -> None:
    """Test that tool requires authentication."""
    result = await run_agent_tool.execute(
        user_id=None,
        session_id="test-session",
        agent_id="test-agent",
    )

    assert isinstance(result, NeedLoginResponse)


@pytest.mark.asyncio
async def test_run_agent_no_agent_id(run_agent_tool) -> None:
    """Test error when no agent ID provided."""
    result = await run_agent_tool.execute(
        user_id="user-123",
        session_id="test-session",
    )

    assert isinstance(result, ErrorResponse)
    assert "provide an agent ID" in result.message.lower()


@pytest.mark.asyncio
async def test_run_agent_insufficient_credits(run_agent_tool) -> None:
    """Test insufficient credits error."""
    with patch(
        "backend.server.v2.chat.tools.run_agent.get_user_credit_model"
    ) as mock_credit:
        credit_model = MagicMock()
        credit_model.get_credits = AsyncMock(return_value=0)
        mock_credit.return_value = credit_model

        result = await run_agent_tool.execute(
            user_id="user-123",
            session_id="test-session",
            agent_id="test-agent",
        )

        assert isinstance(result, InsufficientCreditsResponse)
        assert result.balance == 0
        assert "top up" in result.message.lower()


@pytest.mark.asyncio
async def test_run_agent_not_found(run_agent_tool) -> None:
    """Test error when agent not found."""
    with patch(
        "backend.server.v2.chat.tools.run_agent.get_user_credit_model"
    ) as mock_credit, patch(
        "backend.server.v2.chat.tools.run_agent.graph_db"
    ) as mock_db:

        credit_model = MagicMock()
        credit_model.get_credits = AsyncMock(return_value=100)
        mock_credit.return_value = credit_model

        mock_db.get_graph = AsyncMock(return_value=None)

        result = await run_agent_tool.execute(
            user_id="user-123",
            session_id="test-session",
            agent_id="nonexistent",
        )

        assert isinstance(result, ErrorResponse)
        assert "not found" in result.message.lower()


@pytest.mark.asyncio
async def test_run_agent_immediate_execution(
    run_agent_tool, mock_graph, mock_execution
) -> None:
    """Test immediate agent execution without waiting."""
    with patch(
        "backend.server.v2.chat.tools.run_agent.get_user_credit_model"
    ) as mock_credit, patch(
        "backend.server.v2.chat.tools.run_agent.graph_db"
    ) as mock_db, patch(
        "backend.server.v2.chat.tools.run_agent.execution_utils"
    ) as mock_exec:

        credit_model = MagicMock()
        credit_model.get_credits = AsyncMock(return_value=100)
        mock_credit.return_value = credit_model

        mock_db.get_graph = AsyncMock(return_value=mock_graph)
        mock_exec.add_graph_execution = AsyncMock(return_value=mock_execution)

        result = await run_agent_tool.execute(
            user_id="user-123",
            session_id="test-session",
            agent_id="test-agent-id",
            inputs={"input1": "value1"},
            credentials={"api_key": "key-123"},
            wait_for_result=False,
        )

        assert isinstance(result, ExecutionStartedResponse)
        assert result.execution_id == "exec-123"
        assert result.graph_id == "test-agent-id"
        assert result.graph_name == "Test Agent"
        assert result.status == "QUEUED"
        assert "started" in result.message.lower()


@pytest.mark.asyncio
async def test_run_agent_wait_for_completion(
    run_agent_tool,
    mock_graph,
    mock_execution,
    mock_execution_status_completed,
    mock_full_execution,
) -> None:
    """Test waiting for agent execution to complete."""
    with patch(
        "backend.server.v2.chat.tools.run_agent.get_user_credit_model"
    ) as mock_credit, patch(
        "backend.server.v2.chat.tools.run_agent.graph_db"
    ) as mock_db, patch(
        "backend.server.v2.chat.tools.run_agent.execution_utils"
    ) as mock_exec, patch(
        "backend.server.v2.chat.tools.run_agent.get_graph_execution_meta"
    ) as mock_meta, patch(
        "backend.server.v2.chat.tools.run_agent.get_graph_execution"
    ) as mock_get_exec:

        credit_model = MagicMock()
        credit_model.get_credits = AsyncMock(return_value=100)
        mock_credit.return_value = credit_model

        mock_db.get_graph = AsyncMock(return_value=mock_graph)
        mock_exec.add_graph_execution = AsyncMock(return_value=mock_execution)
        mock_meta.return_value = mock_execution_status_completed
        mock_get_exec.return_value = mock_full_execution

        result = await run_agent_tool.execute(
            user_id="user-123",
            session_id="test-session",
            agent_id="test-agent-id",
            wait_for_result=True,
        )

        assert isinstance(result, ExecutionStartedResponse)
        assert result.status == "COMPLETED"
        assert result.ended_at == "2024-01-01T10:00:00Z"
        assert result.outputs == {"result": "success", "data": [1, 2, 3]}
        assert "completed successfully" in result.message.lower()


@pytest.mark.asyncio
async def test_run_agent_wait_for_failure(
    run_agent_tool,
    mock_graph,
    mock_execution,
    mock_execution_status_failed,
) -> None:
    """Test waiting for agent execution that fails."""
    with patch(
        "backend.server.v2.chat.tools.run_agent.get_user_credit_model"
    ) as mock_credit, patch(
        "backend.server.v2.chat.tools.run_agent.graph_db"
    ) as mock_db, patch(
        "backend.server.v2.chat.tools.run_agent.execution_utils"
    ) as mock_exec, patch(
        "backend.server.v2.chat.tools.run_agent.get_graph_execution_meta"
    ) as mock_meta:

        credit_model = MagicMock()
        credit_model.get_credits = AsyncMock(return_value=100)
        mock_credit.return_value = credit_model

        mock_db.get_graph = AsyncMock(return_value=mock_graph)
        mock_exec.add_graph_execution = AsyncMock(return_value=mock_execution)
        mock_meta.return_value = mock_execution_status_failed

        result = await run_agent_tool.execute(
            user_id="user-123",
            session_id="test-session",
            agent_id="test-agent-id",
            wait_for_result=True,
        )

        assert isinstance(result, ExecutionStartedResponse)
        assert result.status == "FAILED"
        assert result.error == "Task failed: Invalid input"
        assert "failed" in result.message.lower()


@pytest.mark.asyncio
async def test_run_agent_wait_timeout(
    run_agent_tool, mock_graph, mock_execution
) -> None:
    """Test timeout when waiting for execution."""
    with patch(
        "backend.server.v2.chat.tools.run_agent.get_user_credit_model"
    ) as mock_credit, patch(
        "backend.server.v2.chat.tools.run_agent.graph_db"
    ) as mock_db, patch(
        "backend.server.v2.chat.tools.run_agent.execution_utils"
    ) as mock_exec, patch(
        "backend.server.v2.chat.tools.run_agent.get_graph_execution_meta"
    ) as mock_meta, patch(
        "backend.server.v2.chat.tools.run_agent.asyncio"
    ) as mock_asyncio:

        credit_model = MagicMock()
        credit_model.get_credits = AsyncMock(return_value=100)
        mock_credit.return_value = credit_model

        mock_db.get_graph = AsyncMock(return_value=mock_graph)
        mock_exec.add_graph_execution = AsyncMock(return_value=mock_execution)

        # Always return RUNNING status
        mock_meta.return_value = MagicMock(
            status=prisma.enums.AgentExecutionStatus.RUNNING,
        )

        # Mock time to simulate timeout
        loop = MagicMock()
        start_time = 0
        loop.time = MagicMock(
            side_effect=[start_time, start_time + 31]
        )  # > 30s timeout
        mock_asyncio.get_event_loop.return_value = loop
        mock_asyncio.sleep = AsyncMock()

        result = await run_agent_tool.execute(
            user_id="user-123",
            session_id="test-session",
            agent_id="test-agent-id",
            wait_for_result=True,
        )

        assert isinstance(result, ExecutionStartedResponse)
        assert result.status == "RUNNING"
        assert result.timeout_reached is True
        assert "still running" in result.message.lower()


@pytest.mark.asyncio
async def test_run_agent_marketplace_agent_added_to_library(
    run_agent_tool,
    mock_graph,
    mock_execution,
) -> None:
    """Test that marketplace agents are added to library."""
    with patch(
        "backend.server.v2.chat.tools.run_agent.get_user_credit_model"
    ) as mock_credit, patch(
        "backend.server.v2.chat.tools.run_agent.graph_db"
    ) as mock_db, patch(
        "backend.server.v2.chat.tools.run_agent.library_db"
    ) as mock_lib, patch(
        "backend.server.v2.chat.tools.run_agent.execution_utils"
    ) as mock_exec:

        credit_model = MagicMock()
        credit_model.get_credits = AsyncMock(return_value=100)
        mock_credit.return_value = credit_model

        # First call returns None (not in library), second returns marketplace agent
        mock_db.get_graph = AsyncMock(side_effect=[None, mock_graph])
        mock_lib.create_library_agent = AsyncMock()
        mock_exec.add_graph_execution = AsyncMock(return_value=mock_execution)

        result = await run_agent_tool.execute(
            user_id="user-123",
            session_id="test-session",
            agent_id="test-agent-id",
        )

        assert isinstance(result, ExecutionStartedResponse)
        # Verify agent was added to library
        mock_lib.create_library_agent.assert_called_once()


@pytest.mark.asyncio
async def test_run_agent_validation_error(run_agent_tool, mock_graph) -> None:
    """Test validation error handling."""
    with patch(
        "backend.server.v2.chat.tools.run_agent.get_user_credit_model"
    ) as mock_credit, patch(
        "backend.server.v2.chat.tools.run_agent.graph_db"
    ) as mock_db, patch(
        "backend.server.v2.chat.tools.run_agent.execution_utils"
    ) as mock_exec:

        credit_model = MagicMock()
        credit_model.get_credits = AsyncMock(return_value=100)
        mock_credit.return_value = credit_model

        mock_db.get_graph = AsyncMock(return_value=mock_graph)
        mock_exec.add_graph_execution = AsyncMock(
            side_effect=Exception("Validation failed: Missing required field 'email'"),
        )

        result = await run_agent_tool.execute(
            user_id="user-123",
            session_id="test-session",
            agent_id="test-agent-id",
            inputs={},
        )

        assert isinstance(result, ValidationErrorResponse)
        assert "validation failed" in result.message.lower()
        assert "Missing required field" in result.error


@pytest.mark.asyncio
async def test_run_agent_general_error(run_agent_tool) -> None:
    """Test general error handling."""
    with patch(
        "backend.server.v2.chat.tools.run_agent.get_user_credit_model"
    ) as mock_credit:
        mock_credit.side_effect = Exception("Service unavailable")

        result = await run_agent_tool.execute(
            user_id="user-123",
            session_id="test-session",
            agent_id="test-agent",
        )

        assert isinstance(result, ErrorResponse)
        assert "failed to execute agent" in result.message.lower()


@pytest.mark.asyncio
async def test_run_agent_with_version(
    run_agent_tool, mock_graph, mock_execution
) -> None:
    """Test running specific version of agent."""
    mock_graph.version = 5

    with patch(
        "backend.server.v2.chat.tools.run_agent.get_user_credit_model"
    ) as mock_credit, patch(
        "backend.server.v2.chat.tools.run_agent.graph_db"
    ) as mock_db, patch(
        "backend.server.v2.chat.tools.run_agent.execution_utils"
    ) as mock_exec:

        credit_model = MagicMock()
        credit_model.get_credits = AsyncMock(return_value=100)
        mock_credit.return_value = credit_model

        mock_db.get_graph = AsyncMock(return_value=mock_graph)
        mock_exec.add_graph_execution = AsyncMock(return_value=mock_execution)

        result = await run_agent_tool.execute(
            user_id="user-123",
            session_id="test-session",
            agent_id="test-agent-id",
            agent_version=5,
        )

        assert isinstance(result, ExecutionStartedResponse)

        # Verify version was passed to get_graph
        mock_db.get_graph.assert_called_with(
            graph_id="test-agent-id",
            version=5,
            user_id="user-123",
            include_subgraphs=True,
        )

        # Verify version was passed to execution
        mock_exec.add_graph_execution.assert_called_once()
        call_kwargs = mock_exec.add_graph_execution.call_args[1]
        assert call_kwargs["graph_version"] == 5


@pytest.mark.asyncio
async def test_run_agent_credential_conversion(
    run_agent_tool,
    mock_graph,
    mock_execution,
) -> None:
    """Test credential format conversion."""
    with patch(
        "backend.server.v2.chat.tools.run_agent.get_user_credit_model"
    ) as mock_credit, patch(
        "backend.server.v2.chat.tools.run_agent.graph_db"
    ) as mock_db, patch(
        "backend.server.v2.chat.tools.run_agent.execution_utils"
    ) as mock_exec:

        credit_model = MagicMock()
        credit_model.get_credits = AsyncMock(return_value=100)
        mock_credit.return_value = credit_model

        mock_db.get_graph = AsyncMock(return_value=mock_graph)
        mock_exec.add_graph_execution = AsyncMock(return_value=mock_execution)

        result = await run_agent_tool.execute(
            user_id="user-123",
            session_id="test-session",
            agent_id="test-agent-id",
            credentials={
                "api_key": "simple-string",  # String format
                "oauth": {  # Dict format
                    "id": "oauth-123",
                    "type": "oauth",
                },
            },
        )

        assert isinstance(result, ExecutionStartedResponse)

        # Verify credentials were converted
        call_kwargs = mock_exec.add_graph_execution.call_args[1]
        creds = call_kwargs["graph_credentials_inputs"]

        assert "api_key" in creds
        assert creds["api_key"].type == "api_key"

        assert "oauth" in creds
        assert creds["oauth"].id == "oauth-123"
        assert creds["oauth"].type == "oauth"
