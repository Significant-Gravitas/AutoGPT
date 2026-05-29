"""Tests for AutoPilotBlock: recursion guard, streaming, validation, and error paths."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from backend.blocks.autopilot import (
    AUTOPILOT_BLOCK_ID,
    AutoPilotBlock,
    SubAgentRecursionError,
    _autopilot_recursion_depth,
    _autopilot_recursion_limit,
    _check_recursion,
    _reset_recursion,
)
from backend.data.execution import ExecutionContext


def _make_context(user_id: str = "test-user-123") -> ExecutionContext:
    """Helper to build an ExecutionContext for tests."""
    return ExecutionContext(
        user_id=user_id,
        graph_id="graph-1",
        graph_exec_id="gexec-1",
        graph_version=1,
        node_id="node-1",
        node_exec_id="nexec-1",
    )


# ---------------------------------------------------------------------------
# Recursion guard unit tests
# ---------------------------------------------------------------------------


class TestCheckRecursion:
    """Unit tests for _check_recursion / _reset_recursion."""

    def test_first_call_increments_depth(self):
        tokens = _check_recursion(3)
        try:
            assert _autopilot_recursion_depth.get() == 1
            assert _autopilot_recursion_limit.get() == 3
        finally:
            _reset_recursion(tokens)

    def test_reset_restores_previous_values(self):
        assert _autopilot_recursion_depth.get() == 0
        assert _autopilot_recursion_limit.get() is None
        tokens = _check_recursion(5)
        _reset_recursion(tokens)
        assert _autopilot_recursion_depth.get() == 0
        assert _autopilot_recursion_limit.get() is None

    def test_exceeding_limit_raises(self):
        t1 = _check_recursion(2)
        try:
            t2 = _check_recursion(2)
            try:
                with pytest.raises(SubAgentRecursionError):
                    _check_recursion(2)
            finally:
                _reset_recursion(t2)
        finally:
            _reset_recursion(t1)

    def test_nested_calls_respect_inherited_limit(self):
        """Inner call with higher max_depth still respects outer limit."""
        t1 = _check_recursion(2)  # sets limit=2
        try:
            t2 = _check_recursion(10)  # inner wants 10, but inherited is 2
            try:
                # depth is now 2, limit is min(10, 2) = 2 → should raise
                with pytest.raises(SubAgentRecursionError):
                    _check_recursion(10)
            finally:
                _reset_recursion(t2)
        finally:
            _reset_recursion(t1)

    def test_limit_of_one_blocks_immediately_on_second_call(self):
        t1 = _check_recursion(1)
        try:
            with pytest.raises(SubAgentRecursionError):
                _check_recursion(1)
        finally:
            _reset_recursion(t1)


# ---------------------------------------------------------------------------
# AutoPilotBlock.run() validation tests
# ---------------------------------------------------------------------------


class TestRunValidation:
    """Tests for input validation in AutoPilotBlock.run()."""

    @pytest.fixture
    def block(self):
        return AutoPilotBlock()

    @pytest.mark.asyncio
    async def test_empty_prompt_yields_error(self, block):
        block.Input  # ensure schema is accessible
        input_data = block.Input(prompt="   ", max_recursion_depth=3)
        ctx = _make_context()
        outputs = {}
        async for name, value in block.run(input_data, execution_context=ctx):
            outputs[name] = value
        assert outputs.get("error") == "Prompt cannot be empty."
        assert "response" not in outputs

    @pytest.mark.asyncio
    async def test_missing_user_id_yields_error(self, block):
        input_data = block.Input(prompt="hello", max_recursion_depth=3)
        ctx = _make_context(user_id="")
        outputs = {}
        async for name, value in block.run(input_data, execution_context=ctx):
            outputs[name] = value
        assert "authenticated user" in outputs.get("error", "")

    @pytest.mark.asyncio
    async def test_successful_run_yields_all_outputs(self, block):
        """With execute_copilot mocked, run() should yield all 5 success outputs."""
        mock_result = (
            "Hello world",
            [],
            '[{"role":"user","content":"hi"}]',
            "sess-abc",
            {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )
        block.execute_copilot = AsyncMock(return_value=mock_result)
        block.create_session = AsyncMock(return_value="sess-abc")

        input_data = block.Input(prompt="hi", max_recursion_depth=3)
        ctx = _make_context()
        outputs = {}
        async for name, value in block.run(input_data, execution_context=ctx):
            outputs[name] = value

        assert outputs["response"] == "Hello world"
        assert outputs["tool_calls"] == []
        assert outputs["session_id"] == "sess-abc"
        assert outputs["token_usage"]["total_tokens"] == 15
        assert "error" not in outputs

    @pytest.mark.asyncio
    async def test_exception_yields_error(self, block):
        """On unexpected failure, run() should yield an error output."""
        block.execute_copilot = AsyncMock(side_effect=RuntimeError("boom"))
        block.create_session = AsyncMock(return_value="sess-fail")

        input_data = block.Input(prompt="do something", max_recursion_depth=3)
        ctx = _make_context()
        outputs = {}
        async for name, value in block.run(input_data, execution_context=ctx):
            outputs[name] = value

        assert outputs["session_id"] == "sess-fail"
        assert "boom" in outputs.get("error", "")

    @pytest.mark.asyncio
    async def test_cancelled_error_yields_error_and_reraises(self, block):
        """CancelledError should yield error, then re-raise."""
        block.execute_copilot = AsyncMock(side_effect=asyncio.CancelledError())
        block.create_session = AsyncMock(return_value="sess-cancel")

        input_data = block.Input(prompt="do something", max_recursion_depth=3)
        ctx = _make_context()
        outputs = {}
        with pytest.raises(asyncio.CancelledError):
            async for name, value in block.run(input_data, execution_context=ctx):
                outputs[name] = value

        assert outputs["session_id"] == "sess-cancel"
        assert "cancelled" in outputs.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_dry_run_inherited_from_execution_context(self, block):
        """execution_context.dry_run=True must be OR-ed into create_session dry_run
        so that nested AutoPilot sessions simulate even when input_data.dry_run=False.
        """
        mock_result = (
            "ok",
            [],
            "[]",
            "sess-dry",
            {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )
        block.execute_copilot = AsyncMock(return_value=mock_result)
        block.create_session = AsyncMock(return_value="sess-dry")

        input_data = block.Input(prompt="test", max_recursion_depth=3, dry_run=False)
        ctx = _make_context()
        ctx.dry_run = True  # outer execution is dry_run
        async for _ in block.run(input_data, execution_context=ctx):
            pass

        block.create_session.assert_called_once_with(ctx.user_id, dry_run=True)

    @pytest.mark.asyncio
    async def test_existing_session_id_skips_create(self, block):
        """When session_id is provided, create_session should not be called."""
        mock_result = (
            "ok",
            [],
            "[]",
            "existing-sid",
            {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )
        block.execute_copilot = AsyncMock(return_value=mock_result)
        block.create_session = AsyncMock()

        input_data = block.Input(
            prompt="test", session_id="existing-sid", max_recursion_depth=3
        )
        ctx = _make_context()
        async for _ in block.run(input_data, execution_context=ctx):
            pass

        block.create_session.assert_not_called()


# ---------------------------------------------------------------------------
# Block registration / ID tests
# ---------------------------------------------------------------------------


class TestBlockRegistration:
    def test_block_id_matches_constant(self):
        block = AutoPilotBlock()
        assert block.id == AUTOPILOT_BLOCK_ID

    def test_max_recursion_depth_has_upper_bound(self):
        """Schema should enforce le=10."""
        schema = AutoPilotBlock.Input.model_json_schema()
        max_rec = schema["properties"]["max_recursion_depth"]
        assert (
            max_rec.get("maximum") == 10 or max_rec.get("exclusiveMaximum", 999) <= 11
        )

    def test_output_schema_has_no_duplicate_error_field(self):
        """Output should inherit error from BlockSchemaOutput, not redefine it."""
        # The field should exist (inherited) but there should be no explicit
        # redefinition. We verify by checking the class __annotations__ directly.
        assert "error" not in AutoPilotBlock.Output.__annotations__


# ---------------------------------------------------------------------------
# Recovery enqueue integration tests
# ---------------------------------------------------------------------------


class TestRecoveryEnqueue:
    """Tests that run() enqueues orphaned sessions for recovery on failure."""

    @pytest.fixture
    def block(self):
        return AutoPilotBlock()

    @pytest.mark.asyncio
    async def test_recovery_enqueued_on_transient_exception(self, block):
        """A generic exception should trigger _enqueue_for_recovery."""
        block.execute_copilot = AsyncMock(side_effect=RuntimeError("network error"))
        block.create_session = AsyncMock(return_value="sess-recover")

        input_data = block.Input(prompt="do work", max_recursion_depth=3)
        ctx = _make_context()

        with patch("backend.blocks.autopilot._enqueue_for_recovery") as mock_enqueue:
            mock_enqueue.return_value = None
            outputs = {}
            async for name, value in block.run(input_data, execution_context=ctx):
                outputs[name] = value

        assert "network error" in outputs.get("error", "")
        mock_enqueue.assert_awaited_once_with(
            "sess-recover",
            ctx.user_id,
            "do work",
            False,
        )

    @pytest.mark.asyncio
    async def test_recovery_not_enqueued_for_recursion_limit(self, block):
        """Recursion limit errors are deliberate — no recovery enqueue."""
        block.execute_copilot = AsyncMock(
            side_effect=SubAgentRecursionError(
                "AutoPilot recursion depth limit reached (3). "
                "The autopilot has called itself too many times."
            )
        )
        block.create_session = AsyncMock(return_value="sess-rec-limit")

        input_data = block.Input(prompt="recurse", max_recursion_depth=3)
        ctx = _make_context()

        with patch("backend.blocks.autopilot._enqueue_for_recovery") as mock_enqueue:
            async for _ in block.run(input_data, execution_context=ctx):
                pass

        mock_enqueue.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_recovery_not_enqueued_for_dry_run(self, block):
        """dry_run=True sessions must not be enqueued (no real consumers)."""
        block.execute_copilot = AsyncMock(side_effect=RuntimeError("transient"))
        block.create_session = AsyncMock(return_value="sess-dry-fail")

        input_data = block.Input(prompt="test", max_recursion_depth=3, dry_run=True)
        ctx = _make_context()

        with patch("backend.blocks.autopilot._enqueue_for_recovery") as mock_enqueue:
            mock_enqueue.return_value = None
            async for _ in block.run(input_data, execution_context=ctx):
                pass

        # _enqueue_for_recovery is called with dry_run=True,
        # so the inner guard returns early without publishing to the queue.
        mock_enqueue.assert_awaited_once()
        positional = mock_enqueue.call_args_list[0][0]
        assert positional[3] is True  # dry_run=True

    @pytest.mark.asyncio
    async def test_recovery_enqueue_failure_does_not_mask_original_error(self, block):
        """If _enqueue_for_recovery itself raises, the original error is still yielded."""
        block.execute_copilot = AsyncMock(side_effect=ValueError("original"))
        block.create_session = AsyncMock(return_value="sess-enq-fail")

        input_data = block.Input(prompt="hello", max_recursion_depth=3)
        ctx = _make_context()

        async def _failing_enqueue(*args, **kwargs):
            raise OSError("rabbitmq down")

        with patch(
            "backend.blocks.autopilot._enqueue_for_recovery",
            side_effect=_failing_enqueue,
        ):
            outputs = {}
            async for name, value in block.run(input_data, execution_context=ctx):
                outputs[name] = value

        # Original error must still be surfaced despite the enqueue failure
        assert outputs.get("error") == "original"
        assert outputs.get("session_id") == "sess-enq-fail"

    @pytest.mark.asyncio
    async def test_recovery_uses_dry_run_from_context(self, block):
        """execution_context.dry_run=True is OR-ed into the dry_run arg."""
        block.execute_copilot = AsyncMock(side_effect=RuntimeError("fail"))
        block.create_session = AsyncMock(return_value="sess-ctx-dry")

        input_data = block.Input(prompt="test", max_recursion_depth=3, dry_run=False)
        ctx = _make_context()
        ctx.dry_run = True  # outer execution is dry_run

        with patch("backend.blocks.autopilot._enqueue_for_recovery") as mock_enqueue:
            mock_enqueue.return_value = None
            async for _ in block.run(input_data, execution_context=ctx):
                pass

        mock_enqueue.assert_awaited_once()
        positional = mock_enqueue.call_args_list[0][0]
        assert positional[3] is True  # dry_run=True

    @pytest.mark.asyncio
    async def test_recovery_uses_effective_prompt_with_system_context(self, block):
        """When system_context is set, _enqueue_for_recovery receives the
        effective_prompt (system_context prepended) so the dedup check in
        maybe_append_user_message passes on replay."""
        block.execute_copilot = AsyncMock(side_effect=RuntimeError("e2b timeout"))
        block.create_session = AsyncMock(return_value="sess-sys-ctx")

        input_data = block.Input(
            prompt="do work",
            system_context="Be concise.",
            max_recursion_depth=3,
        )
        ctx = _make_context()

        with patch("backend.blocks.autopilot._enqueue_for_recovery") as mock_enqueue:
            mock_enqueue.return_value = None
            async for _ in block.run(input_data, execution_context=ctx):
                pass

        mock_enqueue.assert_awaited_once()
        positional = mock_enqueue.call_args_list[0][0]
        assert positional[2] == "[System Context: Be concise.]\n\ndo work"

    @pytest.mark.asyncio
    async def test_recovery_cancelled_error_still_yields_error(self, block):
        """CancelledError during _enqueue_for_recovery still yields the error output."""
        block.execute_copilot = AsyncMock(side_effect=RuntimeError("e2b stall"))
        block.create_session = AsyncMock(return_value="sess-cancel")

        async def _cancelled_enqueue(*args, **kwargs):
            raise asyncio.CancelledError

        outputs = {}
        with patch(
            "backend.blocks.autopilot._enqueue_for_recovery",
            side_effect=_cancelled_enqueue,
        ):
            with pytest.raises(asyncio.CancelledError):
                async for name, value in block.run(
                    block.Input(prompt="do work", max_recursion_depth=3),
                    execution_context=_make_context(),
                ):
                    outputs[name] = value

        # error must be yielded even when recovery raises CancelledError
        assert outputs.get("error") == "e2b stall"
        assert outputs.get("session_id") == "sess-cancel"
