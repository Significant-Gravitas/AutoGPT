"""Tests for session-level dry_run flag propagation.

Verifies that when a session has dry_run=True, run_block and run_agent tool
calls are forced to use dry-run mode, regardless of what the individual
tool call specifies.  The single source of truth is ``session.dry_run``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.model import ChatSession
from backend.copilot.tools.models import ErrorResponse
from backend.copilot.tools.run_agent import RunAgentInput, RunAgentTool
from backend.copilot.tools.run_block import RunBlockTool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session(dry_run: bool = False) -> ChatSession:
    """Create a minimal ChatSession for testing."""
    session = ChatSession.new("test-user", dry_run=dry_run)
    return session


def _make_mock_block(name: str = "TestBlock"):
    """Create a minimal mock block with jsonschema() methods."""
    block = MagicMock()
    block.name = name
    block.description = "A test block"
    block.disabled = False
    block.block_type = "STANDARD"
    block.id = "test-block-id"

    block.input_schema = MagicMock()
    block.input_schema.jsonschema.return_value = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }
    block.input_schema.get_credentials_fields.return_value = {}
    block.input_schema.get_credentials_fields_info.return_value = {}

    block.output_schema = MagicMock()
    block.output_schema.jsonschema.return_value = {
        "type": "object",
        "properties": {"result": {"type": "string"}},
        "required": ["result"],
    }

    return block


# ---------------------------------------------------------------------------
# RunBlockTool tests
# ---------------------------------------------------------------------------


class TestRunBlockToolSessionDryRun:
    """Test that RunBlockTool respects session-level dry_run."""

    @pytest.mark.asyncio
    async def test_session_dry_run_forces_block_dry_run(self):
        """When session dry_run is True, run_block should force dry_run=True."""
        tool = RunBlockTool()
        session = _make_session(dry_run=True)

        mock_block = _make_mock_block()

        with (
            patch(
                "backend.copilot.tools.run_block.prepare_block_for_execution"
            ) as mock_prep,
            patch("backend.copilot.tools.run_block.execute_block") as mock_exec,
            patch(
                "backend.copilot.tools.run_block.get_current_permissions",
                return_value=None,
            ),
        ):
            # Set up prepare_block_for_execution to return a mock prep
            mock_prep_result = MagicMock()
            mock_prep_result.block = mock_block
            mock_prep_result.input_data = {"query": "test"}
            mock_prep_result.matched_credentials = {}
            mock_prep_result.synthetic_node_id = "node-1"
            mock_prep.return_value = mock_prep_result

            # Set up execute_block to return a success
            mock_exec.return_value = MagicMock(
                message="[DRY RUN] Block executed",
                success=True,
            )

            await tool._execute(
                user_id="test-user",
                session=session,
                block_id="test-block-id",
                input_data={"query": "test"},
                dry_run=False,  # User passed False, but session overrides
            )

            # Verify execute_block was called with dry_run=True
            mock_exec.assert_called_once()
            call_kwargs = mock_exec.call_args
            assert call_kwargs.kwargs.get("dry_run") is True

    @pytest.mark.asyncio
    async def test_no_session_dry_run_respects_tool_param(self):
        """When session dry_run is False, tool-level dry_run should be respected."""
        tool = RunBlockTool()
        session = _make_session(dry_run=False)

        mock_block = _make_mock_block()

        with (
            patch(
                "backend.copilot.tools.run_block.prepare_block_for_execution"
            ) as mock_prep,
            patch("backend.copilot.tools.run_block.execute_block") as mock_exec,
            patch(
                "backend.copilot.tools.run_block.get_current_permissions",
                return_value=None,
            ),
            patch("backend.copilot.tools.run_block.check_hitl_review") as mock_hitl,
        ):
            mock_prep_result = MagicMock()
            mock_prep_result.block = mock_block
            mock_prep_result.input_data = {"query": "test"}
            mock_prep_result.matched_credentials = {}
            mock_prep_result.synthetic_node_id = "node-1"
            mock_prep_result.required_non_credential_keys = {"query"}
            mock_prep_result.provided_input_keys = {"query"}
            mock_prep.return_value = mock_prep_result

            mock_hitl.return_value = ("node-exec-1", {"query": "test"})

            mock_exec.return_value = MagicMock(
                message="Block executed",
                success=True,
            )

            await tool._execute(
                user_id="test-user",
                session=session,
                block_id="test-block-id",
                input_data={"query": "test"},
                dry_run=False,
            )

            # Verify execute_block was called with dry_run=False
            mock_exec.assert_called_once()
            call_kwargs = mock_exec.call_args
            assert call_kwargs.kwargs.get("dry_run") is False


# ---------------------------------------------------------------------------
# RunAgentTool tests
# ---------------------------------------------------------------------------


class TestRunAgentToolSessionDryRun:
    """Test that RunAgentTool respects session-level dry_run."""

    @pytest.mark.asyncio
    async def test_session_dry_run_forces_agent_dry_run(self):
        """When session dry_run is True, run_agent params.dry_run should be forced True."""
        tool = RunAgentTool()
        session = _make_session(dry_run=True)

        # Mock the graph and dependencies
        mock_graph = MagicMock()
        mock_graph.id = "graph-1"
        mock_graph.name = "Test Agent"
        mock_graph.description = "A test agent"
        mock_graph.input_schema = {"properties": {}, "required": []}
        mock_graph.trigger_setup_info = None

        mock_library_agent = MagicMock()
        mock_library_agent.id = "lib-1"
        mock_library_agent.graph_id = "graph-1"
        mock_library_agent.graph_version = 1
        mock_library_agent.name = "Test Agent"

        mock_execution = MagicMock()
        mock_execution.id = "exec-1"

        with (
            patch("backend.copilot.tools.run_agent.graph_db"),
            patch("backend.copilot.tools.run_agent.library_db"),
            patch(
                "backend.copilot.tools.run_agent.fetch_graph_from_store_slug",
                return_value=(mock_graph, None),
            ),
            patch(
                "backend.copilot.tools.run_agent.match_user_credentials_to_graph",
                return_value=({}, []),
            ),
            patch(
                "backend.copilot.tools.run_agent.get_or_create_library_agent",
                return_value=mock_library_agent,
            ),
            patch("backend.copilot.tools.run_agent.execution_utils") as mock_exec_utils,
            patch("backend.copilot.tools.run_agent.track_agent_run_success"),
        ):
            mock_exec_utils.add_graph_execution = AsyncMock(return_value=mock_execution)

            await tool._execute(
                user_id="test-user",
                session=session,
                username_agent_slug="user/test-agent",
                dry_run=False,  # User passed False, but session overrides
                use_defaults=True,
            )

            # Verify add_graph_execution was called with dry_run=True
            mock_exec_utils.add_graph_execution.assert_called_once()
            call_kwargs = mock_exec_utils.add_graph_execution.call_args
            assert call_kwargs.kwargs.get("dry_run") is True

    @pytest.mark.asyncio
    async def test_session_dry_run_blocks_scheduling(self):
        """When session dry_run is True, scheduling requests should be rejected."""
        tool = RunAgentTool()
        session = _make_session(dry_run=True)

        result = await tool._execute(
            user_id="test-user",
            session=session,
            username_agent_slug="user/test-agent",
            schedule_name="daily-run",
            cron="0 9 * * *",
            dry_run=False,  # Session overrides to True
        )

        assert isinstance(result, ErrorResponse)
        assert "dry-run" in result.message.lower()
        assert (
            "scheduling" in result.message.lower()
            or "schedule" in result.message.lower()
        )


# ---------------------------------------------------------------------------
# ChatSession model tests
# ---------------------------------------------------------------------------


class TestChatSessionDryRun:
    """Test the dry_run field on ChatSession model."""

    def test_new_session_default_dry_run_false(self):
        session = ChatSession.new("test-user", dry_run=False)
        assert session.dry_run is False

    def test_new_session_dry_run_true(self):
        session = ChatSession.new("test-user", dry_run=True)
        assert session.dry_run is True

    def test_new_session_dry_run_false_explicit(self):
        session = ChatSession.new("test-user", dry_run=False)
        assert session.dry_run is False


# ---------------------------------------------------------------------------
# RunAgentInput tests
# ---------------------------------------------------------------------------


class TestRunAgentInputDryRunOverride:
    """Test that RunAgentInput.dry_run can be mutated by session-level override."""

    def test_explicit_dry_run_false(self):
        params = RunAgentInput(username_agent_slug="user/agent", dry_run=False)
        assert params.dry_run is False

    def test_session_override(self):
        params = RunAgentInput(username_agent_slug="user/agent", dry_run=False)
        # Simulate session-level override
        params.dry_run = True
        assert params.dry_run is True
