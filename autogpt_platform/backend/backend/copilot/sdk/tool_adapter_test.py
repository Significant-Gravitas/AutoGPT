"""Tests for tool_adapter: truncation, stash, context vars, readOnlyHint annotations."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp.types import ToolAnnotations

from backend.copilot.context import get_sdk_cwd
from backend.copilot.response_model import StreamToolOutputAvailable
from backend.util.truncate import truncate

from .tool_adapter import (
    _MCP_MAX_CHARS,
    _READ_ONLY_E2B_TOOLS,
    _text_from_mcp_result,
    create_tool_handler,
    pop_pending_tool_output,
    reset_stash_event,
    set_execution_context,
    stash_pending_tool_output,
    wait_for_stash,
)

# ---------------------------------------------------------------------------
# _text_from_mcp_result
# ---------------------------------------------------------------------------


class TestTextFromMcpResult:
    def test_single_text_block(self):
        result = {"content": [{"type": "text", "text": "hello"}]}
        assert _text_from_mcp_result(result) == "hello"

    def test_multiple_text_blocks_concatenated(self):
        result = {
            "content": [
                {"type": "text", "text": "one"},
                {"type": "text", "text": "two"},
            ]
        }
        assert _text_from_mcp_result(result) == "onetwo"

    def test_non_text_blocks_ignored(self):
        result = {
            "content": [
                {"type": "image", "data": "..."},
                {"type": "text", "text": "only this"},
            ]
        }
        assert _text_from_mcp_result(result) == "only this"

    def test_empty_content_list(self):
        assert _text_from_mcp_result({"content": []}) == ""

    def test_missing_content_key(self):
        assert _text_from_mcp_result({}) == ""

    def test_non_list_content(self):
        assert _text_from_mcp_result({"content": "raw string"}) == ""

    def test_missing_text_field(self):
        result = {"content": [{"type": "text"}]}
        assert _text_from_mcp_result(result) == ""


# ---------------------------------------------------------------------------
# get_sdk_cwd
# ---------------------------------------------------------------------------


class TestGetSdkCwd:
    def test_returns_empty_string_by_default(self):
        set_execution_context(
            user_id="test",
            session=None,  # type: ignore[arg-type]
            sandbox=None,
        )
        assert get_sdk_cwd() == ""

    def test_returns_set_value(self):
        set_execution_context(
            user_id="test",
            session=None,  # type: ignore[arg-type]
            sandbox=None,
            sdk_cwd="/tmp/copilot-test-123",
        )
        assert get_sdk_cwd() == "/tmp/copilot-test-123"


# ---------------------------------------------------------------------------
# stash / pop round-trip (the mechanism _truncating relies on)
# ---------------------------------------------------------------------------


class TestToolOutputStash:
    @pytest.fixture(autouse=True)
    def _init_context(self):
        """Initialise the context vars that stash_pending_tool_output needs."""
        set_execution_context(
            user_id="test",
            session=None,  # type: ignore[arg-type]
            sandbox=None,
            sdk_cwd="/tmp/test",
        )

    def test_stash_and_pop(self):
        stash_pending_tool_output("my_tool", "output1")
        assert pop_pending_tool_output("my_tool") == "output1"

    def test_pop_empty_returns_none(self):
        assert pop_pending_tool_output("nonexistent") is None

    def test_fifo_order(self):
        stash_pending_tool_output("t", "first")
        stash_pending_tool_output("t", "second")
        assert pop_pending_tool_output("t") == "first"
        assert pop_pending_tool_output("t") == "second"
        assert pop_pending_tool_output("t") is None

    def test_dict_serialised_to_json(self):
        stash_pending_tool_output("t", {"key": "value"})
        assert pop_pending_tool_output("t") == '{"key": "value"}'

    def test_separate_tool_names(self):
        stash_pending_tool_output("a", "alpha")
        stash_pending_tool_output("b", "beta")
        assert pop_pending_tool_output("b") == "beta"
        assert pop_pending_tool_output("a") == "alpha"


# ---------------------------------------------------------------------------
# reset_stash_event / wait_for_stash
# ---------------------------------------------------------------------------


class TestResetStashEvent:
    """Tests for reset_stash_event — the stale-signal fix for retry attempts."""

    @pytest.fixture(autouse=True)
    def _init_context(self):
        set_execution_context(
            user_id="test",
            session=None,  # type: ignore[arg-type]
            sandbox=None,
        )

    @pytest.mark.asyncio
    async def test_reset_clears_stale_signal(self):
        """After reset, wait_for_stash does NOT return immediately (blocks until timeout)."""
        # Simulate a stale signal left by a failed attempt's PostToolUse hook.
        stash_pending_tool_output("some_tool", "stale output")
        # The stash_pending_tool_output call sets the event.
        # Now reset it — simulating start of a new retry attempt.
        reset_stash_event()
        # wait_for_stash should block and time out since the event was cleared.
        result = await wait_for_stash(timeout=0.05)
        assert result is False, (
            "wait_for_stash should have timed out after reset_stash_event, "
            "but it returned True — stale signal was not cleared"
        )

    @pytest.mark.asyncio
    async def test_wait_returns_true_when_signaled_after_reset(self):
        """After reset, a new stash signal is correctly detected."""
        reset_stash_event()

        async def _signal_after_delay():
            await asyncio.sleep(0.01)
            stash_pending_tool_output("tool", "fresh output")

        asyncio.create_task(_signal_after_delay())
        result = await wait_for_stash(timeout=1.0)
        assert result is True

    @pytest.mark.asyncio
    async def test_retry_scenario_stale_event_does_not_fire_prematurely(self):
        """Simulates: attempt 1 leaves event set → reset → attempt 2 waits correctly."""
        # Attempt 1: hook fires and sets the event
        stash_pending_tool_output("t", "attempt-1-output")
        # Pop it so the stash is empty (simulating normal consumption)
        pop_pending_tool_output("t")

        # Between attempts: reset (as service.py does before each retry)
        reset_stash_event()

        # Attempt 2: wait_for_stash should NOT return True immediately
        result = await wait_for_stash(timeout=0.05)
        assert result is False, (
            "Stale event from attempt 1 caused wait_for_stash to return "
            "prematurely in attempt 2"
        )


# ---------------------------------------------------------------------------
# _truncating wrapper (integration via create_copilot_mcp_server)
# ---------------------------------------------------------------------------


class TestTruncationAndStashIntegration:
    """Test truncation + stash behavior that _truncating relies on."""

    @pytest.fixture(autouse=True)
    def _init_context(self):
        set_execution_context(
            user_id="test",
            session=None,  # type: ignore[arg-type]
            sandbox=None,
            sdk_cwd="/tmp/test",
        )

    def test_small_output_stashed(self):
        """Non-error output is stashed for the response adapter."""
        result = {
            "content": [{"type": "text", "text": "small output"}],
            "isError": False,
        }
        truncated = truncate(result, _MCP_MAX_CHARS)
        text = _text_from_mcp_result(truncated)
        assert text == "small output"
        stash_pending_tool_output("test_tool", text)
        assert pop_pending_tool_output("test_tool") == "small output"

    def test_error_result_not_stashed(self):
        """Error results should not be stashed."""
        result = {
            "content": [{"type": "text", "text": "error msg"}],
            "isError": True,
        }
        # _truncating only stashes when not result.get("isError")
        if not result.get("isError"):
            stash_pending_tool_output("err_tool", "should not happen")
        assert pop_pending_tool_output("err_tool") is None

    def test_large_output_truncated(self):
        """Output exceeding _MCP_MAX_CHARS is truncated before stashing."""
        big_text = "x" * (_MCP_MAX_CHARS + 100_000)
        result = {"content": [{"type": "text", "text": big_text}]}
        truncated = truncate(result, _MCP_MAX_CHARS)
        text = _text_from_mcp_result(truncated)
        assert len(text) < len(big_text)
        assert len(str(truncated)) <= _MCP_MAX_CHARS


# ---------------------------------------------------------------------------
# create_tool_handler (direct execution, no pre-launch)
# ---------------------------------------------------------------------------


def _make_mock_tool(name: str, output: str = "result") -> MagicMock:
    """Return a BaseTool mock that returns a successful StreamToolOutputAvailable."""
    tool = MagicMock()
    tool.name = name
    tool.parameters = {"properties": {}, "required": []}
    tool.execute = AsyncMock(
        return_value=StreamToolOutputAvailable(
            toolCallId="test-id",
            output=output,
            toolName=name,
            success=True,
        )
    )
    return tool


def _make_mock_session() -> MagicMock:
    """Return a minimal ChatSession mock."""
    return MagicMock()


def _init_ctx(session=None):
    set_execution_context(
        user_id="user-1",
        session=session,  # type: ignore[arg-type]
        sandbox=None,
    )


class TestCreateToolHandler:
    """Tests for create_tool_handler — direct tool execution."""

    @pytest.fixture(autouse=True)
    def _init(self):
        _init_ctx(session=_make_mock_session())

    @pytest.mark.asyncio
    async def test_handler_executes_tool_directly(self):
        """Handler executes the tool and returns MCP-formatted result."""
        mock_tool = _make_mock_tool("run_block", output="direct result")

        handler = create_tool_handler(mock_tool)
        result = await handler({"block_id": "b1"})

        assert result["isError"] is False
        text = result["content"][0]["text"]
        assert "direct result" in text
        mock_tool.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_handler_returns_error_on_no_session(self):
        """When session is None, handler returns MCP error."""
        mock_tool = _make_mock_tool("run_block")
        set_execution_context(user_id="u", session=None, sandbox=None)  # type: ignore[arg-type]

        handler = create_tool_handler(mock_tool)
        result = await handler({"block_id": "b1"})

        assert result["isError"] is True
        assert "session" in result["content"][0]["text"].lower()

    @pytest.mark.asyncio
    async def test_handler_returns_error_on_exception(self):
        """Exception from tool execution is caught and returned as MCP error."""
        mock_tool = _make_mock_tool("run_block")
        mock_tool.execute = AsyncMock(side_effect=RuntimeError("block exploded"))

        handler = create_tool_handler(mock_tool)
        result = await handler({"block_id": "b1"})

        assert result["isError"] is True
        assert "Failed to execute run_block" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_handler_executes_once_per_call(self):
        """Each handler call executes the tool exactly once — no duplicate execution."""
        mock_tool = _make_mock_tool("run_block", output="single-execution")

        handler = create_tool_handler(mock_tool)
        await handler({"block_id": "b1"})
        await handler({"block_id": "b2"})

        assert mock_tool.execute.await_count == 2


# ---------------------------------------------------------------------------
# readOnlyHint annotations
# ---------------------------------------------------------------------------


class TestReadOnlyAnnotations:
    """Tests that read-only tools get readOnlyHint=True via BaseTool.read_only."""

    def test_read_only_tools_exist_in_registry(self):
        """At least some tools in TOOL_REGISTRY have read_only=True."""
        from backend.copilot.tools import TOOL_REGISTRY

        read_only_names = [name for name, t in TOOL_REGISTRY.items() if t.read_only]
        assert len(read_only_names) > 0, "No read-only tools found in registry"

    def test_known_read_only_tools_have_attribute(self):
        """Key read-only tools should have read_only=True."""
        from backend.copilot.tools import TOOL_REGISTRY

        for name in ["find_block", "search_docs", "list_workspace_files", "web_fetch"]:
            if name in TOOL_REGISTRY:
                assert TOOL_REGISTRY[name].read_only, f"{name} should be read_only"

    def test_side_effect_tools_are_not_read_only(self):
        """Key side-effect tools should have read_only=False."""
        from backend.copilot.tools import TOOL_REGISTRY

        for name in ["run_block", "bash_exec", "create_agent", "run_agent"]:
            if name in TOOL_REGISTRY:
                assert not TOOL_REGISTRY[
                    name
                ].read_only, f"{name} should not be read_only"

    def test_tool_annotations_creation(self):
        """ToolAnnotations(readOnlyHint=True) works correctly."""
        ann = ToolAnnotations(readOnlyHint=True)
        assert ann.readOnlyHint is True
        assert ann.destructiveHint is None

    def test_read_only_e2b_tools_classification(self):
        """E2B read-only tools should be read_file, glob, grep."""
        assert _READ_ONLY_E2B_TOOLS == frozenset({"read_file", "glob", "grep"})
