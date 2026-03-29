"""Tests for session-level dry_run flag propagation.

Verifies that when a session has dry_run=True, run_block, run_agent, and
run_mcp_tool calls are forced to use dry-run mode, regardless of what the
individual tool call specifies.  The single source of truth is
``session.dry_run``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.model import ChatSession
from backend.copilot.tools.models import ErrorResponse, MCPToolOutputResponse
from backend.copilot.tools.run_agent import RunAgentInput, RunAgentTool
from backend.copilot.tools.run_block import RunBlockTool
from backend.copilot.tools.run_mcp_tool import RunMCPToolTool

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
                message="Block 'TestBlock' executed successfully",
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


# ---------------------------------------------------------------------------
# RunMCPToolTool tests
# ---------------------------------------------------------------------------


class TestRunMCPToolToolSessionDryRun:
    """Test that RunMCPToolTool respects session-level dry_run."""

    @pytest.mark.asyncio
    async def test_session_dry_run_blocks_mcp_execution(self):
        """When session dry_run is True, MCP tool execution should be skipped."""
        tool = RunMCPToolTool()
        session = _make_session(dry_run=True)

        result = await tool._execute(
            user_id="test-user",
            session=session,
            server_url="https://mcp.example.com/sse",
            tool_name="some_tool",
            tool_arguments={"key": "value"},
        )

        assert isinstance(result, MCPToolOutputResponse)
        assert result.success is True
        assert "dry-run" in result.message
        assert result.tool_name == "some_tool"
        assert result.result is None

    @pytest.mark.asyncio
    async def test_session_dry_run_allows_discovery(self):
        """When session dry_run is True, tool discovery (no tool_name) should still work."""
        tool = RunMCPToolTool()
        session = _make_session(dry_run=True)

        # Discovery requires a network call, so we mock the client
        with (
            patch(
                "backend.copilot.tools.run_mcp_tool.auto_lookup_mcp_credential",
                return_value=None,
            ),
            patch(
                "backend.copilot.tools.run_mcp_tool.validate_url_host",
                return_value=None,
            ),
            patch("backend.copilot.tools.run_mcp_tool.MCPClient") as mock_client_cls,
        ):
            mock_client = AsyncMock()
            mock_client_cls.return_value = mock_client

            mock_tool = MagicMock()
            mock_tool.name = "test_tool"
            mock_tool.description = "A test tool"
            mock_tool.input_schema = {"type": "object", "properties": {}}
            mock_client.list_tools.return_value = [mock_tool]

            result = await tool._execute(
                user_id="test-user",
                session=session,
                server_url="https://mcp.example.com/sse",
                tool_name="",  # Discovery mode
            )

            # Discovery should proceed normally
            mock_client.initialize.assert_called_once()
            mock_client.list_tools.assert_called_once()
            assert "Discovered" in result.message

    @pytest.mark.asyncio
    async def test_no_session_dry_run_allows_execution(self):
        """When session dry_run is False, MCP tool execution should proceed."""
        tool = RunMCPToolTool()
        session = _make_session(dry_run=False)

        with (
            patch(
                "backend.copilot.tools.run_mcp_tool.auto_lookup_mcp_credential",
                return_value=None,
            ),
            patch(
                "backend.copilot.tools.run_mcp_tool.validate_url_host",
                return_value=None,
            ),
            patch("backend.copilot.tools.run_mcp_tool.MCPClient") as mock_client_cls,
        ):
            mock_client = AsyncMock()
            mock_client_cls.return_value = mock_client

            mock_result = MagicMock()
            mock_result.is_error = False
            mock_result.content = [{"type": "text", "text": "hello"}]
            mock_client.call_tool.return_value = mock_result

            result = await tool._execute(
                user_id="test-user",
                session=session,
                server_url="https://mcp.example.com/sse",
                tool_name="some_tool",
                tool_arguments={"key": "value"},
            )

            # Execution should proceed
            mock_client.initialize.assert_called_once()
            mock_client.call_tool.assert_called_once_with("some_tool", {"key": "value"})
            assert isinstance(result, MCPToolOutputResponse)
            assert result.success is True


# ---------------------------------------------------------------------------
# Backward-compatibility tests for ChatSessionMetadata deserialization
# ---------------------------------------------------------------------------


class TestChatSessionMetadataBackwardCompat:
    """Verify that sessions created before the dry_run field existed still load.

    The ``metadata`` JSON column in the DB may contain ``{}``, ``null``, or a
    dict without the ``dry_run`` key for sessions created before the flag was
    introduced.  These must deserialize without errors and default to
    ``dry_run=False``.
    """

    def test_metadata_default_construction(self):
        """ChatSessionMetadata() with no args should default dry_run=False."""
        from backend.copilot.model import ChatSessionMetadata

        meta = ChatSessionMetadata()
        assert meta.dry_run is False

    def test_metadata_from_empty_dict(self):
        """Deserializing an empty dict (old-format metadata) should succeed."""
        from backend.copilot.model import ChatSessionMetadata

        meta = ChatSessionMetadata.model_validate({})
        assert meta.dry_run is False

    def test_metadata_from_dict_without_dry_run_key(self):
        """A metadata dict with other keys but no dry_run should still work."""
        from backend.copilot.model import ChatSessionMetadata

        meta = ChatSessionMetadata.model_validate({"some_future_field": 42})
        # dry_run should fall back to default
        assert meta.dry_run is False

    def test_metadata_round_trip_with_dry_run_false(self):
        """Serialize then deserialize with dry_run=False."""
        from backend.copilot.model import ChatSessionMetadata

        original = ChatSessionMetadata(dry_run=False)
        raw = original.model_dump()
        restored = ChatSessionMetadata.model_validate(raw)
        assert restored.dry_run is False

    def test_metadata_round_trip_with_dry_run_true(self):
        """Serialize then deserialize with dry_run=True."""
        from backend.copilot.model import ChatSessionMetadata

        original = ChatSessionMetadata(dry_run=True)
        raw = original.model_dump()
        restored = ChatSessionMetadata.model_validate(raw)
        assert restored.dry_run is True

    def test_metadata_json_round_trip(self):
        """Serialize to JSON string and back, simulating Redis cache flow."""
        from backend.copilot.model import ChatSessionMetadata

        original = ChatSessionMetadata(dry_run=True)
        json_str = original.model_dump_json()
        restored = ChatSessionMetadata.model_validate_json(json_str)
        assert restored.dry_run is True

    def test_session_dry_run_property_with_default_metadata(self):
        """ChatSession.dry_run returns False when metadata has no dry_run."""
        from backend.copilot.model import ChatSessionMetadata

        # Simulate building a session with metadata deserialized from an old row
        meta = ChatSessionMetadata.model_validate({})
        session = _make_session(dry_run=False)
        session.metadata = meta
        assert session.dry_run is False

    def test_session_info_dry_run_property_with_default_metadata(self):
        """ChatSessionInfo.dry_run returns False when metadata is default."""
        from datetime import UTC, datetime

        from backend.copilot.model import ChatSessionInfo, ChatSessionMetadata

        info = ChatSessionInfo(
            session_id="old-session-id",
            user_id="test-user",
            usage=[],
            started_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
            metadata=ChatSessionMetadata.model_validate({}),
        )
        assert info.dry_run is False

    def test_session_full_json_round_trip_without_dry_run(self):
        """A full ChatSession JSON round-trip preserves dry_run default."""
        session = _make_session(dry_run=False)
        json_bytes = session.model_dump_json()
        restored = ChatSession.model_validate_json(json_bytes)
        assert restored.dry_run is False
        assert restored.metadata.dry_run is False

    def test_session_full_json_round_trip_with_dry_run(self):
        """A full ChatSession JSON round-trip preserves dry_run=True."""
        session = _make_session(dry_run=True)
        json_bytes = session.model_dump_json()
        restored = ChatSession.model_validate_json(json_bytes)
        assert restored.dry_run is True
        assert restored.metadata.dry_run is True
