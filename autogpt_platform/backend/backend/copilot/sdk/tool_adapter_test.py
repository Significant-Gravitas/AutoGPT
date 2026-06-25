"""Tests for tool_adapter: truncation, stash, context vars, readOnlyHint annotations."""

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp.types import ListToolsRequest, ToolAnnotations

from backend.copilot.builder_context import BUILDER_BLOCKED_TOOLS
from backend.copilot.context import get_sdk_cwd
from backend.copilot.model import ChatSession
from backend.copilot.response_model import StreamToolOutputAvailable
from backend.copilot.tools import TOOL_REGISTRY
from backend.util.truncate import truncate

from .tool_adapter import (
    _MCP_MAX_CHARS,
    _STRIP_FROM_LLM,
    SDK_DISALLOWED_TOOLS,
    _make_truncating_wrapper,
    _strip_llm_fields,
    _text_from_mcp_result,
    create_copilot_mcp_server,
    create_tool_handler,
    pop_pending_tool_output,
    reset_pending_tool_outputs,
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
            session=None,
            sandbox=None,
        )
        assert get_sdk_cwd() == ""

    def test_returns_set_value(self):
        set_execution_context(
            user_id="test",
            session=None,
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
            session=None,
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

    def test_same_tool_different_input_not_swapped(self):
        """OPEN-3158: parallel calls to the same tool with different inputs
        keep their own output regardless of stash vs pop order."""
        # Stashed in completion order (beta first), popped in call order.
        stash_pending_tool_output("web_search", "beta-out", {"query": "beta"})
        stash_pending_tool_output("web_search", "alpha-out", {"query": "alpha"})
        assert pop_pending_tool_output("web_search", {"query": "alpha"}) == "alpha-out"
        assert pop_pending_tool_output("web_search", {"query": "beta"}) == "beta-out"

    def test_same_input_key_order_insensitive(self):
        """Dict key ordering must not change the composite key."""
        stash_pending_tool_output("t", "out", {"a": 1, "b": 2})
        assert pop_pending_tool_output("t", {"b": 2, "a": 1}) == "out"

    def test_empty_input_falls_back_to_name_key(self):
        """Falsy input uses the name-only key, so a name-only pop still finds
        it (back-compat for tools called with no meaningful args)."""
        stash_pending_tool_output("t", "out", {})
        assert pop_pending_tool_output("t") == "out"

    def test_reset_pending_tool_outputs_drops_orphans(self):
        """A retry attempt must not inherit stashed outputs from a rolled-back
        attempt — orphaned entries shift the name-keyed FIFO off-by-one and
        attach stale payloads to the new attempt's tool calls."""
        stash_pending_tool_output("run_block", "stale-from-failed-attempt")
        reset_pending_tool_outputs()
        assert pop_pending_tool_output("run_block") is None

        # A fresh stash after the reset behaves normally.
        stash_pending_tool_output("run_block", "fresh")
        assert pop_pending_tool_output("run_block") == "fresh"


# ---------------------------------------------------------------------------
# reset_stash_event / wait_for_stash
# ---------------------------------------------------------------------------


class TestResetStashEvent:
    """Tests for reset_stash_event — the stale-signal fix for retry attempts."""

    @pytest.fixture(autouse=True)
    def _init_context(self):
        set_execution_context(
            user_id="test",
            session=None,
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
            session=None,
            sandbox=None,
            sdk_cwd="/tmp/test",
        )

    @pytest.mark.asyncio
    async def test_empty_args_triggers_guard_when_required_args_present(self):
        """Tools with at least one required arg should reject empty-args calls."""
        called = False

        async def handler(_args):
            nonlocal called
            called = True
            return {"content": [{"type": "text", "text": "ok"}], "isError": False}

        wrapper = _make_truncating_wrapper(
            handler,
            "tool_with_required",
            input_schema={"type": "object", "properties": {"file_path": {}}},
            required_args=["file_path"],
        )
        result = await wrapper({})
        assert called is False
        assert result.get("isError") is True
        assert "empty arguments" in _text_from_mcp_result(result)

    @pytest.mark.asyncio
    async def test_empty_args_allowed_when_no_required_args(self):
        """Tools whose params are all optional (filters-only) accept empty args."""
        called = False

        async def handler(args):
            nonlocal called
            called = True
            assert args == {}
            return {"content": [{"type": "text", "text": "listed"}], "isError": False}

        wrapper = _make_truncating_wrapper(
            handler,
            "list_only_optional_filters",
            input_schema={"type": "object", "properties": {"graph_id": {}}},
            required_args=[],
        )
        result = await wrapper({})
        assert called is True
        assert result.get("isError") is not True
        assert _text_from_mcp_result(result) == "listed"

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


def _make_mock_tool(
    name: str,
    output: str = "result",
) -> MagicMock:
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


def _make_test_session(*, dry_run: bool = False) -> ChatSession:
    """Return a minimal real ``ChatSession`` for tool-context tests."""
    return ChatSession.new(user_id="test-user", dry_run=dry_run)


def _init_ctx(session: ChatSession | None = None):
    set_execution_context(
        user_id="user-1",
        session=session,
        sandbox=None,
    )


class TestCreateToolHandler:
    """Tests for create_tool_handler — direct tool execution."""

    @pytest.fixture(autouse=True)
    def _init(self):
        _init_ctx(session=_make_test_session())

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
        set_execution_context(user_id="u", session=None, sandbox=None)

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


class TestToolInlineExecution:
    """Tools run inline to completion — no per-handler timeout, no parking."""

    @pytest.fixture(autouse=True)
    def _init(self):
        _init_ctx(session=_make_test_session())

    @pytest.mark.asyncio
    async def test_tool_runs_to_completion_regardless_of_duration(self):
        """A tool that takes a while still runs inline; the handler does not
        park, cancel, or wrap it in a timeout. The stream-level idle timer
        (in _run_stream_attempt) is what pauses while tool calls are pending."""

        async def slow_but_completes(*_args, **_kwargs):
            await asyncio.sleep(0.1)
            return StreamToolOutputAvailable(
                toolCallId="t1",
                output="final-result",
                toolName="slow_tool",
                success=True,
            )

        mock_tool = _make_mock_tool("slow_tool")
        mock_tool.execute = AsyncMock(side_effect=slow_but_completes)

        handler = create_tool_handler(mock_tool)
        result = await handler({})

        assert result["isError"] is False
        assert "final-result" in result["content"][0]["text"]


# ---------------------------------------------------------------------------
# Regression tests: bugs fixed by removing pre-launch mechanism
#
# Each test class includes a _buggy_handler fixture that reproduces the old
# pre-launch implementation inline.  Tests run against BOTH the buggy handler
# (xfail — proves the bug exists) and the current clean handler (must pass).
# ---------------------------------------------------------------------------


def _make_execute_fn(tool_name: str = "run_block"):
    """Return (execute_fn, call_log) — execute_fn records every call."""
    call_log: list[dict] = []

    async def execute_fn(*args, **kwargs):
        call_log.append(kwargs)
        return StreamToolOutputAvailable(
            toolCallId=f"id-{len(call_log)}",
            output=f"result-{len(call_log)}",
            toolName=tool_name,
            success=True,
        )

    return execute_fn, call_log


async def _buggy_prelaunch_handler(mock_tool, pre_launch_args, dispatch_args):
    """Simulate the OLD buggy pre-launch flow.

    1. pre_launch_tool_call fires _execute_tool_sync with pre_launch_args
    2. SDK dispatches handler with dispatch_args
    3. Handler compares args — on mismatch, cancels + re-executes (BUG)

    Returns the handler result.
    """
    from backend.copilot.sdk.tool_adapter import _execute_tool_sync

    user_id, session = "user-1", _make_test_session()

    # Step 1: pre-launch fires immediately (speculative)
    task = asyncio.create_task(
        _execute_tool_sync(mock_tool, user_id, session, pre_launch_args)
    )
    await asyncio.sleep(0)  # let task start

    # Step 2: SDK dispatches with (potentially different) args
    if pre_launch_args != dispatch_args:
        # Arg mismatch path: cancel pre-launched task + re-execute
        if not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        # Fall through to direct execution (duplicate!)
        return await _execute_tool_sync(mock_tool, user_id, session, dispatch_args)
    else:
        return await task


class TestBug1DuplicateExecution:
    """Bug 1 (SECRT-2204): arg mismatch causes duplicate execution.

    Pre-launch fires with raw args, SDK dispatches with normalised args.
    Mismatch → cancel (too late) + re-execute → 2 API calls.
    """

    @pytest.fixture(autouse=True)
    def _init(self):
        _init_ctx(session=_make_test_session())

    @pytest.mark.xfail(reason="Old pre-launch code causes duplicate execution")
    @pytest.mark.asyncio
    async def test_old_code_duplicates_on_arg_mismatch(self):
        """OLD CODE: pre-launch with args A, dispatch with args B → 2 calls."""
        execute_fn, call_log = _make_execute_fn()
        mock_tool = _make_mock_tool("run_block")
        mock_tool.execute = AsyncMock(side_effect=execute_fn)

        pre_launch_args = {"block_id": "b1", "input_data": {"title": "Test"}}
        dispatch_args = {
            "block_id": "b1",
            "input_data": {"title": "Test", "priority": None},
        }

        await _buggy_prelaunch_handler(mock_tool, pre_launch_args, dispatch_args)

        # BUG: pre-launch executed once + fallback executed again = 2
        assert (
            len(call_log) == 1
        ), f"Expected 1 execution but got {len(call_log)} — duplicate execution bug!"

    @pytest.mark.asyncio
    async def test_current_code_no_duplicate(self):
        """FIXED: handler executes exactly once regardless of arg shape."""
        execute_fn, call_log = _make_execute_fn()
        mock_tool = _make_mock_tool("run_block")
        mock_tool.execute = AsyncMock(side_effect=execute_fn)

        handler = create_tool_handler(mock_tool)
        await handler({"block_id": "b1", "input_data": {"title": "Test"}})

        assert len(call_log) == 1, f"Expected 1 execution but got {len(call_log)}"


class TestBug2FIFODesync:
    """Bug 2: FIFO desync when security hook denies a tool.

    Pre-launch queues [task_A, task_B]. Tool A denied (no MCP dispatch).
    Tool B's handler dequeues task_A → returns wrong result.
    """

    @pytest.fixture(autouse=True)
    def _init(self):
        _init_ctx(session=_make_test_session())

    @pytest.mark.xfail(reason="Old FIFO queue returns wrong result on denial")
    @pytest.mark.asyncio
    async def test_old_code_fifo_desync_on_denial(self):
        """OLD CODE: denied tool's task stays in queue, next tool gets wrong result."""
        from backend.copilot.sdk.tool_adapter import _execute_tool_sync

        call_log: list[str] = []

        async def tagged_execute(*args, **kwargs):
            tag = kwargs.get("block_id", "?")
            call_log.append(tag)
            return StreamToolOutputAvailable(
                toolCallId="id",
                output=f"result-for-{tag}",
                toolName="run_block",
                success=True,
            )

        mock_tool = _make_mock_tool("run_block")
        mock_tool.execute = AsyncMock(side_effect=tagged_execute)
        user_id, session = "user-1", _make_test_session()

        # Simulate old FIFO queue
        queue: asyncio.Queue = asyncio.Queue()

        # Pre-launch for tool A and tool B
        task_a = asyncio.create_task(
            _execute_tool_sync(mock_tool, user_id, session, {"block_id": "A"})
        )
        task_b = asyncio.create_task(
            _execute_tool_sync(mock_tool, user_id, session, {"block_id": "B"})
        )
        queue.put_nowait(task_a)
        queue.put_nowait(task_b)
        await asyncio.sleep(0)  # let both tasks run

        # Tool A is DENIED by security hook — no MCP dispatch, no dequeue
        # Tool B's handler dequeues from FIFO → gets task_A!
        dequeued_task = queue.get_nowait()
        result = await dequeued_task
        result_text = result["content"][0]["text"]

        # BUG: handler for B got task_A's result
        assert "result-for-B" in result_text, (
            f"Expected result for B but got: {result_text} — "
            f"FIFO desync: B got A's result!"
        )

    @pytest.mark.asyncio
    async def test_current_code_no_fifo_desync(self):
        """FIXED: each handler call executes independently, no shared queue."""
        call_log: list[str] = []

        async def tagged_execute(*args, **kwargs):
            tag = kwargs.get("block_id", "?")
            call_log.append(tag)
            return StreamToolOutputAvailable(
                toolCallId="id",
                output=f"result-for-{tag}",
                toolName="run_block",
                success=True,
            )

        mock_tool = _make_mock_tool("run_block")
        mock_tool.execute = AsyncMock(side_effect=tagged_execute)

        handler = create_tool_handler(mock_tool)

        # Tool A denied (never called). Tool B dispatched normally.
        result_b = await handler({"block_id": "B"})

        assert "result-for-B" in result_b["content"][0]["text"]
        assert call_log == ["B"]


class TestBug3CancelRace:
    """Bug 3: cancel race — task completes before cancel arrives.

    Pre-launch fires fast HTTP call (< 1s). By the time handler detects
    mismatch and calls task.cancel(), the API call already completed.
    Side effect (Linear issue created) is irreversible.
    """

    @pytest.fixture(autouse=True)
    def _init(self):
        _init_ctx(session=_make_test_session())

    @pytest.mark.xfail(reason="Old code: cancel arrives after task completes")
    @pytest.mark.asyncio
    async def test_old_code_cancel_arrives_too_late(self):
        """OLD CODE: fast task completes before cancel, side effect persists."""
        side_effects: list[str] = []

        async def fast_execute_with_side_effect(*args, **kwargs):
            # Side effect happens immediately (like an HTTP POST to Linear)
            side_effects.append("created-issue")
            return StreamToolOutputAvailable(
                toolCallId="id",
                output="issue-created",
                toolName="run_block",
                success=True,
            )

        mock_tool = _make_mock_tool("run_block")
        mock_tool.execute = AsyncMock(side_effect=fast_execute_with_side_effect)

        # Pre-launch fires immediately
        pre_launch_args = {"block_id": "b1"}
        dispatch_args = {"block_id": "b1", "extra": "normalised"}

        await _buggy_prelaunch_handler(mock_tool, pre_launch_args, dispatch_args)

        # BUG: side effect happened TWICE (pre-launch + fallback)
        assert len(side_effects) == 1, (
            f"Expected 1 side effect but got {len(side_effects)} — "
            f"cancel race: pre-launch completed before cancel!"
        )

    @pytest.mark.asyncio
    async def test_current_code_single_side_effect(self):
        """FIXED: no speculative execution, exactly 1 side effect per call."""
        side_effects: list[str] = []

        async def execute_with_side_effect(*args, **kwargs):
            side_effects.append("created-issue")
            return StreamToolOutputAvailable(
                toolCallId="id",
                output="issue-created",
                toolName="run_block",
                success=True,
            )

        mock_tool = _make_mock_tool("run_block")
        mock_tool.execute = AsyncMock(side_effect=execute_with_side_effect)

        handler = create_tool_handler(mock_tool)
        await handler({"block_id": "b1"})

        assert len(side_effects) == 1


# ---------------------------------------------------------------------------
# readOnlyHint annotations
# ---------------------------------------------------------------------------


class TestReadOnlyAnnotations:
    """Tests that all tools get readOnlyHint=True for parallel dispatch."""

    def test_parallel_annotation_constant(self):
        """_PARALLEL_ANNOTATION is a ToolAnnotations with readOnlyHint=True."""
        from .tool_adapter import _PARALLEL_ANNOTATION

        assert isinstance(_PARALLEL_ANNOTATION, ToolAnnotations)
        assert _PARALLEL_ANNOTATION.readOnlyHint is True


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

    def test_schedule_wakeup_tool_is_disallowed(self):
        assert "ScheduleWakeup" in SDK_DISALLOWED_TOOLS


# ---------------------------------------------------------------------------
# _read_file_handler — bridge_and_annotate integration
# ---------------------------------------------------------------------------


class TestReadFileHandlerBridge:
    """Verify that _read_file_handler calls bridge_and_annotate when a sandbox is active."""

    @pytest.fixture(autouse=True)
    def _init_context(self):
        set_execution_context(
            user_id="test",
            session=None,
            sandbox=None,
            sdk_cwd="/tmp/copilot-bridge-test",
        )

    @pytest.mark.asyncio
    async def test_bridge_called_when_sandbox_active(self, tmp_path, monkeypatch):
        """When a sandbox is set, bridge_and_annotate is called and its annotation appended."""
        from backend.copilot.context import _current_sandbox

        from .tool_adapter import _read_file_handler

        test_file = tmp_path / "tool-results" / "data.json"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text('{"ok": true}\n')

        monkeypatch.setattr(
            "backend.copilot.sdk.tool_adapter.is_sdk_tool_path",
            lambda path: True,
        )

        fake_sandbox: Any = object()
        token = _current_sandbox.set(fake_sandbox)
        try:
            bridge_calls: list[tuple] = []

            async def fake_bridge_and_annotate(sandbox, file_path, offset, limit):
                bridge_calls.append((sandbox, file_path, offset, limit))
                return "\n[Sandbox copy available at /tmp/abc-data.json]"

            monkeypatch.setattr(
                "backend.copilot.sdk.tool_adapter.bridge_and_annotate",
                fake_bridge_and_annotate,
            )

            result = await _read_file_handler(
                {"file_path": str(test_file), "offset": 0, "limit": 2000}
            )

            assert result["isError"] is False
            assert len(bridge_calls) == 1
            assert bridge_calls[0][0] is fake_sandbox
            assert "/tmp/abc-data.json" in result["content"][0]["text"]
        finally:
            _current_sandbox.reset(token)

    @pytest.mark.asyncio
    async def test_bridge_skipped_when_envelope_pretty_printed(
        self, tmp_path, monkeypatch
    ):
        """Pretty-printing the MCP envelope transforms the content the
        model reads. The on-disk bytes are the raw envelope, so bridging
        them to the sandbox would point the model at content that
        doesn't match what ``read_tool_result`` just returned. Skip the
        bridge in that case — the model can re-read or pipe via
        ``@@agptfile:`` if it needs bash access."""
        from backend.copilot.context import _current_sandbox

        from .tool_adapter import _read_file_handler

        # MCP envelope with JSON inner payload — _navigable_tool_result_text
        # will pretty-print this, so navigable != raw.
        test_file = tmp_path / "tool-results" / "envelope.json"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {"execution": {"node_executions": [{"status": "OK"}]}}
        envelope = json.dumps([{"type": "text", "text": json.dumps(payload)}])
        test_file.write_text(envelope)

        monkeypatch.setattr(
            "backend.copilot.sdk.tool_adapter.is_sdk_tool_path",
            lambda path: True,
        )

        fake_sandbox: Any = object()
        token = _current_sandbox.set(fake_sandbox)
        try:
            bridge_calls: list[tuple] = []

            async def fake_bridge_and_annotate(sandbox, file_path, offset, limit):
                bridge_calls.append((sandbox, file_path, offset, limit))
                return "\n[Sandbox copy available at /tmp/abc-envelope.json]"

            monkeypatch.setattr(
                "backend.copilot.sdk.tool_adapter.bridge_and_annotate",
                fake_bridge_and_annotate,
            )

            result = await _read_file_handler(
                {"file_path": str(test_file), "offset": 0, "limit": 2000}
            )

            assert result["isError"] is False
            # The bridge MUST NOT be called: model sees pretty-printed
            # JSON but the on-disk file holds the raw envelope.
            assert bridge_calls == []
            assert "Sandbox copy" not in result["content"][0]["text"]
            # And the returned text is the pretty-printed payload, not
            # the envelope wrapper, so the slicing is useful.
            assert '"status": "OK"' in result["content"][0]["text"]
        finally:
            _current_sandbox.reset(token)

    @pytest.mark.asyncio
    async def test_bridge_skipped_when_char_offset_used(self, tmp_path, monkeypatch):
        """``char_offset`` slices the navigable content; the on-disk
        bytes don't carry that slice, so bridging would mislead bash
        operations into reading a different range than the model just
        saw. Skip the bridge regardless of whether pretty-printing
        kicked in for this file."""
        from backend.copilot.context import _current_sandbox

        from .tool_adapter import _read_file_handler

        # File whose content is NOT an envelope — navigable == raw.
        # The bridge would normally fire here; char_offset must still
        # suppress it.
        test_file = tmp_path / "tool-results" / "plain.txt"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text("the quick brown fox " * 100)

        monkeypatch.setattr(
            "backend.copilot.sdk.tool_adapter.is_sdk_tool_path",
            lambda path: True,
        )

        fake_sandbox: Any = object()
        token = _current_sandbox.set(fake_sandbox)
        try:
            bridge_calls: list[tuple] = []

            async def fake_bridge_and_annotate(sandbox, file_path, offset, limit):
                bridge_calls.append((sandbox, file_path, offset, limit))
                return "\n[Sandbox copy available at /tmp/abc-plain.txt]"

            monkeypatch.setattr(
                "backend.copilot.sdk.tool_adapter.bridge_and_annotate",
                fake_bridge_and_annotate,
            )

            result = await _read_file_handler(
                {
                    "file_path": str(test_file),
                    "char_offset": 100,
                    "char_limit": 50,
                }
            )

            assert result["isError"] is False
            assert bridge_calls == []
            assert "Sandbox copy" not in result["content"][0]["text"]
        finally:
            _current_sandbox.reset(token)

    @pytest.mark.asyncio
    async def test_bridge_not_called_without_sandbox(self, tmp_path, monkeypatch):
        """When no sandbox is set, bridge_and_annotate is not called."""
        from .tool_adapter import _read_file_handler

        test_file = tmp_path / "tool-results" / "data.json"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text('{"ok": true}\n')

        monkeypatch.setattr(
            "backend.copilot.sdk.tool_adapter.is_sdk_tool_path",
            lambda path: True,
        )

        bridge_calls: list[tuple] = []

        async def fake_bridge_and_annotate(sandbox, file_path, offset, limit):
            bridge_calls.append((sandbox, file_path, offset, limit))
            return "\n[Sandbox copy available at /tmp/abc-data.json]"

        monkeypatch.setattr(
            "backend.copilot.sdk.tool_adapter.bridge_and_annotate",
            fake_bridge_and_annotate,
        )

        result = await _read_file_handler(
            {"file_path": str(test_file), "offset": 0, "limit": 2000}
        )

        assert result["isError"] is False
        assert len(bridge_calls) == 0
        assert "Sandbox copy" not in result["content"][0]["text"]


# ---------------------------------------------------------------------------
# _STRIP_FROM_LLM / _strip_llm_fields — dry-run field stripping
# ---------------------------------------------------------------------------


class TestStripLlmFields:
    """Regression tests for _strip_llm_fields — the guard that hides dry_run
    execution mode from the LLM.

    Strip-after-stash ordering is the core correctness guarantee: the frontend
    SSE stream receives the full payload (including is_dry_run) while the LLM
    sees a clean response without it.
    """

    def test_strip_from_llm_contains_is_dry_run(self):
        """_STRIP_FROM_LLM must include is_dry_run so the guard is active."""
        assert "is_dry_run" in _STRIP_FROM_LLM

    def test_is_dry_run_removed_from_json_text_block(self):
        """is_dry_run is stripped from a JSON text block before LLM sees it."""
        result = {
            "content": [
                {
                    "type": "text",
                    "text": '{"message": "ok", "is_dry_run": true, "outputs": {}}',
                }
            ],
            "isError": False,
        }
        stripped = _strip_llm_fields(result)
        parsed = json.loads(stripped["content"][0]["text"])
        assert "is_dry_run" not in parsed
        assert parsed["message"] == "ok"
        assert parsed["outputs"] == {}

    def test_other_fields_preserved_after_strip(self):
        """Stripping is_dry_run does not affect unrelated fields."""
        result = {
            "content": [
                {
                    "type": "text",
                    "text": '{"success": true, "is_dry_run": true, "block_id": "b1"}',
                }
            ],
            "isError": False,
        }
        stripped = _strip_llm_fields(result)
        parsed = json.loads(stripped["content"][0]["text"])
        assert parsed["success"] is True
        assert parsed["block_id"] == "b1"
        assert "is_dry_run" not in parsed

    def test_error_result_not_modified(self):
        """Error results pass through unchanged — stripping only applies on success."""
        result = {
            "content": [
                {"type": "text", "text": '{"is_dry_run": true, "error": "boom"}'}
            ],
            "isError": True,
        }
        stripped = _strip_llm_fields(result)
        parsed = json.loads(stripped["content"][0]["text"])
        assert "is_dry_run" in parsed

    def test_non_json_text_block_unchanged(self):
        """Plain-text blocks that are not valid JSON are left as-is."""
        result = {
            "content": [{"type": "text", "text": "plain text, not JSON"}],
            "isError": False,
        }
        stripped = _strip_llm_fields(result)
        assert stripped["content"][0]["text"] == "plain text, not JSON"

    def test_strip_after_stash_ordering(self):
        """Stash receives full payload (with is_dry_run); LLM result does not."""
        set_execution_context(user_id="test", session=None, sandbox=None)

        full_text = '{"message": "ok", "is_dry_run": true}'
        result = {
            "content": [{"type": "text", "text": full_text}],
            "isError": False,
        }

        # Simulate the stash-before-strip ordering in _truncating:
        # 1. Stash the FULL output (before any stripping)
        text = _text_from_mcp_result(result)
        stash_pending_tool_output("tool_x", text)

        # 2. Strip for the LLM
        llm_result = _strip_llm_fields(result)

        # Stash (frontend) still has is_dry_run
        stashed = pop_pending_tool_output("tool_x")
        assert stashed is not None
        assert "is_dry_run" in json.loads(stashed)

        # LLM result does NOT have is_dry_run
        llm_parsed = json.loads(llm_result["content"][0]["text"])
        assert "is_dry_run" not in llm_parsed

    def test_multiple_text_blocks_strips_only_json_blocks(self):
        """Mixed content array: JSON block is stripped, plain-text block is untouched."""
        result = {
            "content": [
                {
                    "type": "text",
                    "text": '{"message": "ok", "is_dry_run": true}',
                },
                {
                    "type": "text",
                    "text": "plain text block — not JSON",
                },
                {
                    "type": "text",
                    "text": '{"other": "data", "is_dry_run": false}',
                },
            ],
            "isError": False,
        }
        stripped = _strip_llm_fields(result)
        # First block: JSON — is_dry_run removed
        first = json.loads(stripped["content"][0]["text"])
        assert "is_dry_run" not in first
        assert first["message"] == "ok"
        # Second block: plain text — unchanged
        assert stripped["content"][1]["text"] == "plain text block — not JSON"
        # Third block: JSON — is_dry_run removed
        third = json.loads(stripped["content"][2]["text"])
        assert "is_dry_run" not in third
        assert third["other"] == "data"

    def test_non_dict_json_value_unchanged(self):
        """A JSON array or string value is valid JSON but not a dict — left as-is."""
        result = {
            "content": [
                {
                    "type": "text",
                    "text": '["is_dry_run", true]',
                }
            ],
            "isError": False,
        }
        stripped = _strip_llm_fields(result)
        # Not a dict, so should be returned unchanged
        assert stripped["content"][0]["text"] == '["is_dry_run", true]'

    @pytest.mark.asyncio
    async def test_truncating_wrapper_stash_then_strip_ordering(self):
        """The _make_truncating_wrapper must stash BEFORE strip so the frontend
        gets is_dry_run while the LLM return value does not.

        This test calls the ACTUAL _make_truncating_wrapper so that swapping
        the stash/strip lines in production code causes this test to fail.
        Uses a session with dry_run=True so that stripping is active.
        """
        dry_run_session = _make_test_session(dry_run=True)
        set_execution_context(
            user_id="test", session=dry_run_session, sandbox=None, sdk_cwd="/tmp/test"
        )

        full_payload = '{"message": "done", "is_dry_run": true}'

        async def fake_tool_fn(_args: dict) -> dict:
            return {
                "content": [{"type": "text", "text": full_payload}],
                "isError": False,
            }

        wrapper = _make_truncating_wrapper(fake_tool_fn, "fake_tool")
        llm_result = await wrapper({})

        # Stash (frontend path) must contain is_dry_run
        stashed = pop_pending_tool_output("fake_tool")
        assert stashed is not None
        assert '"is_dry_run": true' in stashed

        # LLM return value must NOT contain is_dry_run (stripped for session dry_run)
        llm_parsed = json.loads(llm_result["content"][0]["text"])
        assert "is_dry_run" not in llm_parsed
        assert llm_parsed["message"] == "done"

    @pytest.mark.asyncio
    async def test_truncating_wrapper_normal_mode_preserves_is_dry_run_for_llm(self):
        """In normal (non-session-dry_run) mode, is_dry_run=True must reach the LLM.

        When a single tool was individually dry-run but the session is not in
        dry_run mode, the LLM should see is_dry_run=True so it knows that
        specific tool result was simulated.
        """
        normal_session = _make_test_session(dry_run=False)
        set_execution_context(
            user_id="test", session=normal_session, sandbox=None, sdk_cwd="/tmp/test"
        )

        full_payload = '{"message": "simulated", "is_dry_run": true}'

        async def fake_tool_fn(_args: dict) -> dict:
            return {
                "content": [{"type": "text", "text": full_payload}],
                "isError": False,
            }

        wrapper = _make_truncating_wrapper(fake_tool_fn, "fake_tool_normal")
        llm_result = await wrapper({})

        # LLM return value MUST contain is_dry_run in normal session mode
        llm_parsed = json.loads(llm_result["content"][0]["text"])
        assert "is_dry_run" in llm_parsed
        assert llm_parsed["is_dry_run"] is True
        assert llm_parsed["message"] == "simulated"

        # Stash also still has is_dry_run (stash is always unstripped)
        stashed = pop_pending_tool_output("fake_tool_normal")
        assert stashed is not None
        assert '"is_dry_run": true' in stashed


class TestTruncatingWrapperLeavesOutputUntouched:
    """Mid-turn drain moved to the shared ``PostToolUse`` hook path so every
    tool (MCP + built-in) is covered uniformly.  The wrapper must therefore
    forward tool output verbatim and never touch ``<user_follow_up>``."""

    @pytest.mark.asyncio
    async def test_wrapper_does_not_inject_followup(self):
        session = _make_test_session(dry_run=False)
        set_execution_context(
            user_id="u", session=session, sandbox=None, sdk_cwd="/tmp/test"
        )

        async def fake_tool_fn(_args: dict) -> dict:
            return {
                "content": [{"type": "text", "text": "CLEAN_OUTPUT"}],
                "isError": False,
            }

        wrapper = _make_truncating_wrapper(fake_tool_fn, "fake_tool_clean")
        result = await wrapper({})

        text = result["content"][0]["text"]
        assert text == "CLEAN_OUTPUT"
        assert "<user_follow_up>" not in text

    @pytest.mark.asyncio
    async def test_stash_stays_clean(self):
        """The frontend-facing stash must be a byte-for-byte copy of the
        raw tool output (needed for JSON.parse in the bash widget)."""
        session = _make_test_session(dry_run=False)
        set_execution_context(
            user_id="u", session=session, sandbox=None, sdk_cwd="/tmp/test"
        )

        clean_json = '{"stdout": "hello\\n", "exit_code": 0}'

        async def fake_tool_fn(_args: dict) -> dict:
            return {
                "content": [{"type": "text", "text": clean_json}],
                "isError": False,
            }

        wrapper = _make_truncating_wrapper(fake_tool_fn, "fake_tool_stash_pure")
        await wrapper({})

        stashed = pop_pending_tool_output("fake_tool_stash_pure")
        assert stashed == clean_json
        assert "<user_follow_up>" not in (stashed or "")


class TestCreateCopilotMcpServerHidden:
    """``hidden_tool_names`` removes tools from MCP registration entirely
    so the model never sees them — guards against the production bug
    where builder-blocked tools were still advertised, the model called
    them anyway, and the CLI returned its canned "Permission to use ...
    has been denied" string that the model then narrated as a fake
    Allow/Deny UI."""

    @pytest.mark.asyncio
    async def test_hidden_tools_not_registered(self):
        # Use a named tool (find_block is stable + load-bearing in the
        # builder flow) so the test reads as a real scenario instead of
        # "the first key in dict insertion order".
        hidden_name = "find_block"
        assert hidden_name in TOOL_REGISTRY, "fixture relies on find_block"
        server = create_copilot_mcp_server(hidden_tool_names=[hidden_name])
        registered = await self._registered_tool_names(server)
        assert hidden_name not in registered
        # Other tools still register.
        assert len(registered) >= len(TOOL_REGISTRY) - 1

    @pytest.mark.asyncio
    async def test_no_hidden_tools_registers_all(self):
        server = create_copilot_mcp_server()
        registered = await self._registered_tool_names(server)
        for short in TOOL_REGISTRY:
            assert short in registered

    @pytest.mark.asyncio
    async def test_builder_blocked_tools_hidden(self):
        server = create_copilot_mcp_server(hidden_tool_names=BUILDER_BLOCKED_TOOLS)
        registered = await self._registered_tool_names(server)
        for blocked in BUILDER_BLOCKED_TOOLS:
            assert blocked not in registered
        # edit_agent must remain so the model can populate the bound graph.
        assert "edit_agent" in registered

    @pytest.mark.asyncio
    async def test_unknown_hidden_name_is_silently_ignored(self):
        """A typo in ``hidden_tool_names`` must not phantom-hide a real
        tool or raise — the registration loop intersects with
        TOOL_REGISTRY keys, so unknown names are a no-op."""
        server = create_copilot_mcp_server(
            hidden_tool_names=["this_tool_does_not_exist"]
        )
        registered = await self._registered_tool_names(server)
        # All real tools still register.
        for short in TOOL_REGISTRY:
            assert short in registered

    @staticmethod
    async def _registered_tool_names(server) -> set[str]:
        instance = server["instance"]
        handler = instance.request_handlers[ListToolsRequest]
        result = await handler(ListToolsRequest(method="tools/list"))
        return {t.name for t in result.root.tools}


class TestNavigableToolResultText:
    """``_navigable_tool_result_text`` unwraps the CLI's MCP envelope and
    pretty-prints inner JSON so line-based offset/limit slice into the
    actual payload, not the envelope wrapper. Regression coverage for
    the bug where the model bounced off ``bash_exec | python3 -c`` to
    parse a tool result it could have read directly."""

    def test_envelope_with_json_payload_is_pretty_printed(self):
        from .tool_adapter import _navigable_tool_result_text

        payload = {"execution": {"node_executions": [{"status": "FAILED"}]}}
        envelope = json.dumps([{"type": "text", "text": json.dumps(payload)}])
        out = _navigable_tool_result_text(envelope)
        # The output should be pretty-printed JSON with multiple lines,
        # not the single minified blob that was inside the envelope.
        assert "\n" in out
        assert '"status": "FAILED"' in out
        assert json.loads(out) == payload

    def test_envelope_with_non_json_text_returns_unwrapped(self):
        from .tool_adapter import _navigable_tool_result_text

        envelope = json.dumps(
            [{"type": "text", "text": "plain shell output\nwith newlines\n"}]
        )
        out = _navigable_tool_result_text(envelope)
        assert out == "plain shell output\nwith newlines\n"

    def test_non_envelope_input_is_returned_unchanged(self):
        from .tool_adapter import _navigable_tool_result_text

        # Files that don't match the envelope shape stay as-is.
        assert _navigable_tool_result_text("not even json") == "not even json"
        assert (
            _navigable_tool_result_text('{"single": "object"}')
            == '{"single": "object"}'
        )
        # Multi-block envelope: don't unwrap (lossy).
        multi = json.dumps(
            [
                {"type": "text", "text": "a"},
                {"type": "image", "data": "..."},
            ]
        )
        assert _navigable_tool_result_text(multi) == multi
