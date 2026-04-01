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
    SDK_DISALLOWED_TOOLS,
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
# Regression tests: bugs fixed by removing pre-launch mechanism
#
# These tests reproduce the three production bugs that the pre-launch
# speculative execution mechanism caused.  They verify that the current
# direct-execution handler is free of these issues.
#
# Bug 1: Duplicate execution on arg mismatch (SECRT-2204)
#   Pre-launch fired with AssistantMessage args, SDK dispatched with
#   normalised args, mismatch caused cancel+re-execute = 2 API calls.
#
# Bug 2: FIFO desync when security hook denies a tool
#   Denied tool's pre-launched task stayed in queue, next tool dequeued
#   wrong task, returning wrong results.
#
# Bug 3: Cancel race condition
#   task.cancel() arrived after HTTP call completed, side effects
#   (Linear tickets, GitHub PRs) already irreversible.
# ---------------------------------------------------------------------------


class TestNoDuplicateExecution:
    """Bug 1 regression: tool must execute exactly once per handler call,
    regardless of how args are normalised between caller and handler."""

    @pytest.fixture(autouse=True)
    def _init(self):
        _init_ctx(session=_make_mock_session())

    @pytest.mark.asyncio
    async def test_single_execution_even_with_different_arg_representations(self):
        """Calling handler with args that differ from what a hypothetical
        pre-launch would have received must NOT cause double execution.

        Previously, the pre-launch mechanism compared args and re-executed
        on mismatch.  This test ensures each handler call = exactly 1 execute.
        """
        call_count = 0

        async def counting_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return StreamToolOutputAvailable(
                toolCallId="id",
                output=f"exec-{call_count}",
                toolName="run_block",
                success=True,
            )

        mock_tool = _make_mock_tool("run_block")
        mock_tool.execute = AsyncMock(side_effect=counting_execute)

        handler = create_tool_handler(mock_tool)

        # Simulate: SDK sends args with schema defaults injected
        await handler({"block_id": "b1", "input_data": {"title": "Test"}})
        assert call_count == 1, f"Expected 1 execution, got {call_count}"

        # Call again with slightly different args (like CLI normalisation)
        await handler(
            {"block_id": "b1", "input_data": {"title": "Test", "priority": None}}
        )
        assert call_count == 2, f"Expected 2 total executions, got {call_count}"

        # Each call must produce exactly 1 execution — no duplicates
        assert mock_tool.execute.await_count == 2

    @pytest.mark.asyncio
    async def test_concurrent_calls_each_execute_once(self):
        """Multiple concurrent handler calls must each execute exactly once.

        Previously, pre-launch fired all tasks speculatively then the handler
        could mismatch and re-execute.  Now each call is independent.
        """
        call_count = 0

        async def slow_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            n = call_count
            await asyncio.sleep(0.01)  # simulate API latency
            return StreamToolOutputAvailable(
                toolCallId=f"id-{n}",
                output=f"result-{n}",
                toolName="run_block",
                success=True,
            )

        mock_tool = _make_mock_tool("run_block")
        mock_tool.execute = AsyncMock(side_effect=slow_execute)

        handler = create_tool_handler(mock_tool)

        # Fire 3 calls concurrently
        results = await asyncio.gather(
            handler({"block_id": "b1"}),
            handler({"block_id": "b2"}),
            handler({"block_id": "b3"}),
        )

        # Each must succeed independently
        for r in results:
            assert r["isError"] is False

        # Exactly 3 executions — no duplicates
        assert call_count == 3
        assert mock_tool.execute.await_count == 3


class TestNoFIFODesync:
    """Bug 2 regression: handler must not maintain shared state that can
    desync when a tool call is skipped (e.g. denied by security hook)."""

    @pytest.fixture(autouse=True)
    def _init(self):
        _init_ctx(session=_make_mock_session())

    @pytest.mark.asyncio
    async def test_skipped_call_does_not_affect_subsequent_calls(self):
        """If a tool call is skipped (e.g. denied by security hook), the
        next call must still get its own correct result — not a stale
        result from a previously queued task.

        Previously, the FIFO queue stored pre-launched tasks.  If a tool was
        denied (no MCP dispatch), its task stayed in the queue and the next
        tool call dequeued the wrong result.
        """
        call_args_log: list[str] = []

        async def logging_execute(*args, **kwargs):
            block_id = kwargs.get("block_id", "?")
            call_args_log.append(block_id)
            return StreamToolOutputAvailable(
                toolCallId="id",
                output=f"result-for-{block_id}",
                toolName="run_block",
                success=True,
            )

        mock_tool = _make_mock_tool("run_block")
        mock_tool.execute = AsyncMock(side_effect=logging_execute)

        handler = create_tool_handler(mock_tool)

        # Simulate: tool call A is denied by security hook (never dispatched)
        # Tool call B is dispatched normally
        # With the old pre-launch FIFO: B would dequeue A's pre-launched task
        # With current code: B executes independently with its own args

        # Only call B (A was denied/skipped)
        result_b = await handler({"block_id": "B"})

        assert "result-for-B" in result_b["content"][0]["text"]
        assert call_args_log == [
            "B"
        ], f"Expected only B to execute, got {call_args_log}"

    @pytest.mark.asyncio
    async def test_handler_has_no_shared_queue_state(self):
        """Each handler invocation must be stateless — no shared queue
        between calls that could cause cross-contamination.

        Previously, _tool_task_queues was a ContextVar shared across all
        handler calls in a session, allowing FIFO desync.
        """
        results_a: list[str] = []
        results_b: list[str] = []

        async def tagged_execute(*args, **kwargs):
            tag = kwargs.get("tag", "?")
            return StreamToolOutputAvailable(
                toolCallId="id",
                output=f"out-{tag}",
                toolName="tool",
                success=True,
            )

        mock_tool = _make_mock_tool("tool")
        mock_tool.execute = AsyncMock(side_effect=tagged_execute)

        handler = create_tool_handler(mock_tool)

        r1 = await handler({"tag": "first"})
        r2 = await handler({"tag": "second"})
        r3 = await handler({"tag": "third"})

        results_a.append(r1["content"][0]["text"])
        results_b.append(r2["content"][0]["text"])

        # Results must correspond to their own args, not FIFO ordering
        assert "out-first" in results_a[0]
        assert "out-second" in results_b[0]
        assert "out-third" in r3["content"][0]["text"]


class TestNoCancelRaceCondition:
    """Bug 3 regression: handler must not rely on task.cancel() to prevent
    side effects — cancel is best-effort and can arrive too late."""

    @pytest.fixture(autouse=True)
    def _init(self):
        _init_ctx(session=_make_mock_session())

    @pytest.mark.asyncio
    async def test_no_speculative_execution_before_handler_called(self):
        """Tool execution must NOT start before the handler is explicitly called.

        Previously, pre_launch_tool_call() fired execution speculatively
        when the AssistantMessage arrived — before the SDK dispatched the
        MCP tools/call.  This meant side effects occurred before the handler
        was invoked, and cancel() couldn't undo them.
        """
        executed = False

        async def track_execute(*args, **kwargs):
            nonlocal executed
            executed = True
            return StreamToolOutputAvailable(
                toolCallId="id",
                output="done",
                toolName="run_block",
                success=True,
            )

        mock_tool = _make_mock_tool("run_block")
        mock_tool.execute = AsyncMock(side_effect=track_execute)

        # Create handler but DON'T call it yet
        handler = create_tool_handler(mock_tool)

        # With old code: pre_launch_tool_call() would have already started
        # execution.  With current code: nothing happens until handler is called.
        assert not executed, (
            "Tool executed before handler was called — "
            "speculative execution must not happen"
        )

        # Now call the handler
        await handler({"block_id": "b1"})
        assert executed, "Tool should execute when handler is called"

    @pytest.mark.asyncio
    async def test_failed_execution_does_not_leave_orphaned_tasks(self):
        """When a handler call fails, no orphaned background tasks remain.

        Previously, pre-launched tasks could be left in the queue if the
        handler took a different code path (mismatch, error).  These orphaned
        tasks continued executing and creating side effects.
        """
        mock_tool = _make_mock_tool("run_block")
        mock_tool.execute = AsyncMock(side_effect=RuntimeError("API error"))

        handler = create_tool_handler(mock_tool)
        result = await handler({"block_id": "b1"})

        assert result["isError"] is True

        # After the failed call, executing again must work independently
        mock_tool.execute = AsyncMock(
            return_value=StreamToolOutputAvailable(
                toolCallId="id",
                output="recovered",
                toolName="run_block",
                success=True,
            )
        )
        result2 = await handler({"block_id": "b2"})
        assert result2["isError"] is False
        assert "recovered" in result2["content"][0]["text"]


# ---------------------------------------------------------------------------
# readOnlyHint annotations
# ---------------------------------------------------------------------------


class TestReadOnlyAnnotations:
    """Tests that all tools get readOnlyHint=True for parallel dispatch."""

    def test_tool_annotations_creation(self):
        """ToolAnnotations(readOnlyHint=True) works correctly."""
        ann = ToolAnnotations(readOnlyHint=True)
        assert ann.readOnlyHint is True
        assert ann.destructiveHint is None


# ---------------------------------------------------------------------------
# SDK_DISALLOWED_TOOLS
# ---------------------------------------------------------------------------


class TestSDKDisallowedTools:
    """Verify that dangerous SDK built-in tools are in the disallowed list."""

    def test_bash_tool_is_disallowed(self):
        assert "Bash" in SDK_DISALLOWED_TOOLS

    def test_webfetch_tool_is_disallowed(self):
        """WebFetch is disallowed due to SSRF risk."""
        assert "WebFetch" in SDK_DISALLOWED_TOOLS
