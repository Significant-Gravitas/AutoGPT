"""Tests for tool_adapter: truncation, stash, context vars, readOnlyHint annotations."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp.types import ToolAnnotations

from backend.copilot.context import get_sdk_cwd
from backend.copilot.response_model import StreamToolOutputAvailable
from backend.util.truncate import truncate

from .tool_adapter import (
    _MCP_MAX_CHARS,
    _STRIP_FROM_LLM,
    SDK_DISALLOWED_TOOLS,
    _make_truncating_wrapper,
    _strip_llm_fields,
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

    user_id, session = "user-1", _make_mock_session()

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
        _init_ctx(session=_make_mock_session())

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
        _init_ctx(session=_make_mock_session())

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
        user_id, session = "user-1", _make_mock_session()

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
        _init_ctx(session=_make_mock_session())

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


# ---------------------------------------------------------------------------
# _read_file_handler — bridge_and_annotate integration
# ---------------------------------------------------------------------------


class TestReadFileHandlerBridge:
    """Verify that _read_file_handler calls bridge_and_annotate when a sandbox is active."""

    @pytest.fixture(autouse=True)
    def _init_context(self):
        set_execution_context(
            user_id="test",
            session=None,  # type: ignore[arg-type]
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
            "backend.copilot.sdk.tool_adapter.is_allowed_local_path",
            lambda path, cwd: True,
        )

        fake_sandbox = object()
        token = _current_sandbox.set(fake_sandbox)  # type: ignore[arg-type]
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
    async def test_bridge_not_called_without_sandbox(self, tmp_path, monkeypatch):
        """When no sandbox is set, bridge_and_annotate is not called."""
        from .tool_adapter import _read_file_handler

        test_file = tmp_path / "tool-results" / "data.json"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text('{"ok": true}\n')

        monkeypatch.setattr(
            "backend.copilot.sdk.tool_adapter.is_allowed_local_path",
            lambda path, cwd: True,
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
        set_execution_context(user_id="test", session=None, sandbox=None)  # type: ignore[arg-type]

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
        dry_run_session = MagicMock()
        dry_run_session.dry_run = True
        set_execution_context(user_id="test", session=dry_run_session, sandbox=None, sdk_cwd="/tmp/test")  # type: ignore[arg-type]

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
        normal_session = MagicMock()
        normal_session.dry_run = False
        set_execution_context(user_id="test", session=normal_session, sandbox=None, sdk_cwd="/tmp/test")  # type: ignore[arg-type]

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
