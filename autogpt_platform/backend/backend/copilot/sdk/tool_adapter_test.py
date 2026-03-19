"""Tests for tool_adapter helpers: truncation, stash, context vars, parallel pre-launch."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.context import get_sdk_cwd
from backend.copilot.response_model import StreamToolOutputAvailable
from backend.copilot.sdk.file_ref import FileRefExpansionError
from backend.util.truncate import truncate

from .tool_adapter import (
    _MCP_MAX_CHARS,
    _text_from_mcp_result,
    cancel_pending_tool_tasks,
    create_tool_handler,
    pop_pending_tool_output,
    pre_launch_tool_call,
    set_execution_context,
    stash_pending_tool_output,
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
# Parallel pre-launch infrastructure
# ---------------------------------------------------------------------------


def _make_mock_tool(name: str, output: str = "result") -> MagicMock:
    """Return a BaseTool mock that returns a successful StreamToolOutputAvailable."""
    tool = MagicMock()
    tool.name = name
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


class TestPreLaunchToolCall:
    """Tests for pre_launch_tool_call and the queue-based parallel dispatch."""

    @pytest.fixture(autouse=True)
    def _init(self):
        _init_ctx(session=_make_mock_session())

    @pytest.mark.asyncio
    async def test_unknown_tool_is_silently_ignored(self):
        """pre_launch_tool_call does nothing for tools not in TOOL_REGISTRY."""
        # Should not raise even if the tool name is completely unknown
        await pre_launch_tool_call("nonexistent_tool", {})

    @pytest.mark.asyncio
    async def test_mcp_prefix_stripped_before_registry_lookup(self):
        """mcp__copilot__run_block is looked up as 'run_block'."""
        mock_tool = _make_mock_tool("run_block")
        with patch(
            "backend.copilot.sdk.tool_adapter.TOOL_REGISTRY",
            {"run_block": mock_tool},
        ):
            await pre_launch_tool_call("mcp__copilot__run_block", {"block_id": "b1"})

        # The task was enqueued — mock_tool.execute should be called once
        # (may not complete immediately but should start)
        await asyncio.sleep(0)  # yield to event loop
        mock_tool.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_bare_tool_name_without_prefix(self):
        """Tool names without __ separator are looked up as-is."""
        mock_tool = _make_mock_tool("run_block")
        with patch(
            "backend.copilot.sdk.tool_adapter.TOOL_REGISTRY",
            {"run_block": mock_tool},
        ):
            await pre_launch_tool_call("run_block", {"block_id": "b1"})

        await asyncio.sleep(0)
        mock_tool.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_task_enqueued_fifo_for_same_tool(self):
        """Two pre-launched calls for the same tool name are enqueued FIFO."""
        results = []

        async def slow_execute(*args, **kwargs):
            results.append(len(results))
            return StreamToolOutputAvailable(
                toolCallId="id",
                output=str(len(results) - 1),
                toolName="t",
                success=True,
            )

        mock_tool = _make_mock_tool("t")
        mock_tool.execute = AsyncMock(side_effect=slow_execute)

        with patch(
            "backend.copilot.sdk.tool_adapter.TOOL_REGISTRY",
            {"t": mock_tool},
        ):
            await pre_launch_tool_call("t", {"n": 1})
            await pre_launch_tool_call("t", {"n": 2})
            await asyncio.sleep(0)

        assert mock_tool.execute.await_count == 2

    @pytest.mark.asyncio
    async def test_file_ref_expansion_failure_skips_pre_launch(self):
        """When @@agptfile: expansion fails, pre_launch_tool_call skips the task.

        The handler should then fall back to direct execution (which will also
        fail with a proper MCP error via _truncating's own expansion).
        """
        mock_tool = _make_mock_tool("run_block", output="should-not-execute")

        with (
            patch(
                "backend.copilot.sdk.tool_adapter.TOOL_REGISTRY",
                {"run_block": mock_tool},
            ),
            patch(
                "backend.copilot.sdk.tool_adapter.expand_file_refs_in_args",
                AsyncMock(side_effect=FileRefExpansionError("@@agptfile:missing.txt")),
            ),
        ):
            # Should not raise — expansion failure is handled gracefully
            await pre_launch_tool_call("run_block", {"text": "@@agptfile:missing.txt"})
            await asyncio.sleep(0)

        # No task was pre-launched — execute was not called
        mock_tool.execute.assert_not_awaited()


class TestCreateToolHandlerParallel:
    """Tests for create_tool_handler using pre-launched tasks."""

    @pytest.fixture(autouse=True)
    def _init(self):
        _init_ctx(session=_make_mock_session())

    @pytest.mark.asyncio
    async def test_handler_uses_prelaunched_task(self):
        """Handler pops and awaits the pre-launched task rather than re-executing."""
        mock_tool = _make_mock_tool("run_block", output="pre-launched result")

        with patch(
            "backend.copilot.sdk.tool_adapter.TOOL_REGISTRY",
            {"run_block": mock_tool},
        ):
            await pre_launch_tool_call("run_block", {"block_id": "b1"})
            await asyncio.sleep(0)  # let task start

            handler = create_tool_handler(mock_tool)
            result = await handler({"block_id": "b1"})

        assert result["isError"] is False
        text = result["content"][0]["text"]
        assert "pre-launched result" in text
        # Should only have been called once (the pre-launched task), not twice
        mock_tool.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_handler_does_not_double_stash_for_prelaunched_task(self):
        """Pre-launched task result must NOT be stashed by tool_handler directly.

        The _truncating wrapper wraps tool_handler and handles stashing after
        tool_handler returns.  If tool_handler also stashed, the output would be
        appended twice to the FIFO queue and pop_pending_tool_output would return
        a duplicate on the second call.

        This test calls tool_handler directly (without _truncating) and asserts
        that nothing was stashed — confirming stashing is deferred to _truncating.
        """
        mock_tool = _make_mock_tool("run_block", output="stash-me")

        with patch(
            "backend.copilot.sdk.tool_adapter.TOOL_REGISTRY",
            {"run_block": mock_tool},
        ):
            await pre_launch_tool_call("run_block", {"block_id": "b1"})
            await asyncio.sleep(0)

            handler = create_tool_handler(mock_tool)
            result = await handler({"block_id": "b1"})

        assert result["isError"] is False
        assert "stash-me" in result["content"][0]["text"]
        # tool_handler must NOT stash — _truncating (which wraps handler) does it.
        # Calling pop here (without going through _truncating) should return None.
        not_stashed = pop_pending_tool_output("run_block")
        assert not_stashed is None, (
            "tool_handler must not stash directly — _truncating handles stashing "
            "to prevent double-stash in the FIFO queue"
        )

    @pytest.mark.asyncio
    async def test_handler_falls_back_when_queue_empty(self):
        """When no pre-launched task exists, handler executes directly."""
        mock_tool = _make_mock_tool("run_block", output="direct result")

        # Don't call pre_launch_tool_call — queue is empty
        handler = create_tool_handler(mock_tool)
        result = await handler({"block_id": "b1"})

        assert result["isError"] is False
        text = result["content"][0]["text"]
        assert "direct result" in text
        mock_tool.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_handler_cancelled_error_propagates(self):
        """CancelledError from a pre-launched task is re-raised to preserve cancellation semantics."""
        mock_tool = _make_mock_tool("run_block")
        mock_tool.execute = AsyncMock(side_effect=asyncio.CancelledError())

        with patch(
            "backend.copilot.sdk.tool_adapter.TOOL_REGISTRY",
            {"run_block": mock_tool},
        ):
            await pre_launch_tool_call("run_block", {"block_id": "b1"})
            await asyncio.sleep(0)

            handler = create_tool_handler(mock_tool)
            with pytest.raises(asyncio.CancelledError):
                await handler({"block_id": "b1"})

    @pytest.mark.asyncio
    async def test_handler_exception_returns_mcp_error(self):
        """Exception from a pre-launched task is caught and returned as MCP error."""
        mock_tool = _make_mock_tool("run_block")
        mock_tool.execute = AsyncMock(side_effect=RuntimeError("block exploded"))

        with patch(
            "backend.copilot.sdk.tool_adapter.TOOL_REGISTRY",
            {"run_block": mock_tool},
        ):
            await pre_launch_tool_call("run_block", {"block_id": "b1"})
            await asyncio.sleep(0)

            handler = create_tool_handler(mock_tool)
            result = await handler({"block_id": "b1"})

        assert result["isError"] is True
        assert "block exploded" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_two_same_tool_calls_dispatched_in_order(self):
        """Two pre-launched tasks for the same tool are consumed in FIFO order."""
        call_order = []

        async def execute_with_tag(*args, **kwargs):
            tag = kwargs.get("block_id", "?")
            call_order.append(tag)
            return StreamToolOutputAvailable(
                toolCallId="id", output=f"out-{tag}", toolName="run_block", success=True
            )

        mock_tool = _make_mock_tool("run_block")
        mock_tool.execute = AsyncMock(side_effect=execute_with_tag)

        with patch(
            "backend.copilot.sdk.tool_adapter.TOOL_REGISTRY",
            {"run_block": mock_tool},
        ):
            await pre_launch_tool_call("run_block", {"block_id": "first"})
            await pre_launch_tool_call("run_block", {"block_id": "second"})
            await asyncio.sleep(0)

            handler = create_tool_handler(mock_tool)
            r1 = await handler({"block_id": "first"})
            r2 = await handler({"block_id": "second"})

        assert "out-first" in r1["content"][0]["text"]
        assert "out-second" in r2["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_no_session_falls_back_gracefully(self):
        """When session is None and no pre-launched task, handler returns MCP error."""
        mock_tool = _make_mock_tool("run_block")
        # session=None means get_execution_context returns (user_id, None)
        set_execution_context(user_id="u", session=None, sandbox=None)  # type: ignore[arg-type]

        handler = create_tool_handler(mock_tool)
        result = await handler({"block_id": "b1"})

        assert result["isError"] is True
        assert "session" in result["content"][0]["text"].lower()


# ---------------------------------------------------------------------------
# cancel_pending_tool_tasks
# ---------------------------------------------------------------------------


class TestCancelPendingToolTasks:
    """Tests for cancel_pending_tool_tasks — the stream-abort cleanup helper."""

    @pytest.fixture(autouse=True)
    def _init(self):
        _init_ctx(session=_make_mock_session())

    @pytest.mark.asyncio
    async def test_cancels_queued_tasks(self):
        """Queued tasks are cancelled and the queue is cleared."""
        ran = False

        async def never_run(*_args, **_kwargs):
            nonlocal ran
            await asyncio.sleep(10)  # long enough to still be pending
            ran = True

        mock_tool = _make_mock_tool("run_block")
        mock_tool.execute = AsyncMock(side_effect=never_run)

        with patch(
            "backend.copilot.sdk.tool_adapter.TOOL_REGISTRY",
            {"run_block": mock_tool},
        ):
            await pre_launch_tool_call("run_block", {"block_id": "b1"})
            await asyncio.sleep(0)  # let task start
            cancel_pending_tool_tasks()
            await asyncio.sleep(0)  # let cancellation propagate

        assert not ran, "Task should have been cancelled before completing"

    @pytest.mark.asyncio
    async def test_noop_when_no_tasks_queued(self):
        """cancel_pending_tool_tasks does not raise when queues are empty."""
        cancel_pending_tool_tasks()  # should not raise

    @pytest.mark.asyncio
    async def test_handler_does_not_find_cancelled_task(self):
        """After cancel, tool_handler falls back to direct execution."""
        mock_tool = _make_mock_tool("run_block", output="direct-fallback")

        with patch(
            "backend.copilot.sdk.tool_adapter.TOOL_REGISTRY",
            {"run_block": mock_tool},
        ):
            await pre_launch_tool_call("run_block", {"block_id": "b1"})
            await asyncio.sleep(0)
            cancel_pending_tool_tasks()

            # Queue is now empty — handler should execute directly
            handler = create_tool_handler(mock_tool)
            result = await handler({"block_id": "b1"})

        assert result["isError"] is False
        assert "direct-fallback" in result["content"][0]["text"]


# ---------------------------------------------------------------------------
# Concurrent / parallel pre-launch scenarios
# ---------------------------------------------------------------------------


class TestAllParallelToolsPrelaunchedIndependently:
    """Simulate SDK sending N separate AssistantMessages for the same tool concurrently."""

    @pytest.fixture(autouse=True)
    def _init(self):
        _init_ctx(session=_make_mock_session())

    @pytest.mark.asyncio
    async def test_all_parallel_tools_prelaunched_independently(self):
        """5 pre-launches for the same tool all enqueue independently and run concurrently.

        Each task sleeps for PER_TASK_S seconds. If they ran sequentially the total
        wall time would be ~5*PER_TASK_S. Running concurrently it should finish in
        roughly PER_TASK_S (plus scheduling overhead).
        """
        PER_TASK_S = 0.05
        N = 5
        started: list[int] = []
        finished: list[int] = []

        async def slow_execute(*args, **kwargs):
            idx = len(started)
            started.append(idx)
            await asyncio.sleep(PER_TASK_S)
            finished.append(idx)
            return StreamToolOutputAvailable(
                toolCallId=f"id-{idx}",
                output=f"result-{idx}",
                toolName="bash_exec",
                success=True,
            )

        mock_tool = _make_mock_tool("bash_exec")
        mock_tool.execute = AsyncMock(side_effect=slow_execute)

        start = asyncio.get_running_loop().time()
        with patch(
            "backend.copilot.sdk.tool_adapter.TOOL_REGISTRY",
            {"bash_exec": mock_tool},
        ):
            for i in range(N):
                await pre_launch_tool_call("bash_exec", {"cmd": f"echo {i}"})

            # Allow all tasks to run concurrently
            await asyncio.sleep(PER_TASK_S * 2)

        elapsed = asyncio.get_running_loop().time() - start

        assert mock_tool.execute.await_count == N
        assert len(finished) == N
        # Wall time should be well under N * PER_TASK_S (sequential would be ~0.25s)
        assert elapsed < N * PER_TASK_S, (
            f"Expected concurrent execution (<{N * PER_TASK_S:.2f}s) "
            f"but took {elapsed:.2f}s"
        )


class TestHandlerReturnsResultFromCorrectPrelaunchedTask:
    """Pop pre-launched tasks in order and verify each returns its own result."""

    @pytest.fixture(autouse=True)
    def _init(self):
        _init_ctx(session=_make_mock_session())

    @pytest.mark.asyncio
    async def test_handler_returns_result_from_correct_prelaunched_task(self):
        """Two pre-launches for the same tool: first handler gets first result, second gets second."""

        async def execute_with_cmd(*args, **kwargs):
            cmd = kwargs.get("cmd", args[0] if args else "?")
            return StreamToolOutputAvailable(
                toolCallId="id",
                output=f"output-for-{cmd}",
                toolName="bash_exec",
                success=True,
            )

        mock_tool = _make_mock_tool("bash_exec")
        mock_tool.execute = AsyncMock(side_effect=execute_with_cmd)

        with patch(
            "backend.copilot.sdk.tool_adapter.TOOL_REGISTRY",
            {"bash_exec": mock_tool},
        ):
            await pre_launch_tool_call("bash_exec", {"cmd": "alpha"})
            await pre_launch_tool_call("bash_exec", {"cmd": "beta"})
            await asyncio.sleep(0)  # let both tasks start

            handler = create_tool_handler(mock_tool)
            r1 = await handler({"cmd": "alpha"})
            r2 = await handler({"cmd": "beta"})

        text1 = r1["content"][0]["text"]
        text2 = r2["content"][0]["text"]
        assert "output-for-alpha" in text1, f"Expected alpha result, got: {text1}"
        assert "output-for-beta" in text2, f"Expected beta result, got: {text2}"
        assert mock_tool.execute.await_count == 2


class TestFiveConcurrentPrelaunchAllComplete:
    """Pre-launch 5 tasks; consume all 5 via handlers; assert all succeed."""

    @pytest.fixture(autouse=True)
    def _init(self):
        _init_ctx(session=_make_mock_session())

    @pytest.mark.asyncio
    async def test_five_concurrent_prelaunch_all_complete(self):
        """All 5 pre-launched tasks complete and return successful results."""
        N = 5
        call_count = 0

        async def counting_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            n = call_count
            return StreamToolOutputAvailable(
                toolCallId=f"id-{n}",
                output=f"done-{n}",
                toolName="bash_exec",
                success=True,
            )

        mock_tool = _make_mock_tool("bash_exec")
        mock_tool.execute = AsyncMock(side_effect=counting_execute)

        with patch(
            "backend.copilot.sdk.tool_adapter.TOOL_REGISTRY",
            {"bash_exec": mock_tool},
        ):
            for i in range(N):
                await pre_launch_tool_call("bash_exec", {"cmd": f"task-{i}"})

            await asyncio.sleep(0)  # let all tasks start

            handler = create_tool_handler(mock_tool)
            results = []
            for i in range(N):
                results.append(await handler({"cmd": f"task-{i}"}))

        assert (
            mock_tool.execute.await_count == N
        ), f"Expected {N} execute calls, got {mock_tool.execute.await_count}"
        for i, result in enumerate(results):
            assert result["isError"] is False, f"Result {i} should not be an error"
            text = result["content"][0]["text"]
            assert "done-" in text, f"Result {i} missing expected output: {text}"
