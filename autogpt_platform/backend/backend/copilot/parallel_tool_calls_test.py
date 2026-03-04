"""Tests for parallel tool call execution in CoPilot.

These tests mock _yield_tool_call to avoid importing the full copilot stack
which requires Prisma, DB connections, etc.
"""

import asyncio
import time
from typing import Any, cast

import pytest


@pytest.mark.asyncio
async def test_parallel_tool_calls_run_concurrently():
    """Multiple tool calls should complete in ~max(delays), not sum(delays)."""
    from backend.copilot.response_model import (
        StreamToolInputAvailable,
        StreamToolOutputAvailable,
    )
    from backend.copilot.service import _execute_tool_calls_parallel

    n_tools = 3
    delay_per_tool = 0.2
    tool_calls = [
        {
            "id": f"call_{i}",
            "type": "function",
            "function": {"name": f"tool_{i}", "arguments": "{}"},
        }
        for i in range(n_tools)
    ]

    class FakeSession:
        session_id = "test"
        user_id = "test"

        def __init__(self):
            self.messages = []

    original_yield = None

    async def fake_yield(tc_list, idx, sess):
        yield StreamToolInputAvailable(
            toolCallId=tc_list[idx]["id"],
            toolName=tc_list[idx]["function"]["name"],
            input={},
        )
        await asyncio.sleep(delay_per_tool)
        yield StreamToolOutputAvailable(
            toolCallId=tc_list[idx]["id"],
            toolName=tc_list[idx]["function"]["name"],
            output="{}",
        )

    import backend.copilot.service as svc

    original_yield = svc._yield_tool_call
    svc._yield_tool_call = fake_yield
    try:
        start = time.monotonic()
        events = []
        async for event in _execute_tool_calls_parallel(
            tool_calls, cast(Any, FakeSession())
        ):
            events.append(event)
        elapsed = time.monotonic() - start
    finally:
        svc._yield_tool_call = original_yield

    assert len(events) == n_tools * 2
    # Parallel: should take ~delay, not ~n*delay
    assert elapsed < delay_per_tool * (
        n_tools - 0.5
    ), f"Took {elapsed:.2f}s, expected parallel (~{delay_per_tool}s)"


@pytest.mark.asyncio
async def test_single_tool_call_works():
    """Single tool call should work identically."""
    from backend.copilot.response_model import (
        StreamToolInputAvailable,
        StreamToolOutputAvailable,
    )
    from backend.copilot.service import _execute_tool_calls_parallel

    tool_calls = [
        {
            "id": "call_0",
            "type": "function",
            "function": {"name": "t", "arguments": "{}"},
        }
    ]

    class FakeSession:
        session_id = "test"
        user_id = "test"

        def __init__(self):
            self.messages = []

    async def fake_yield(tc_list, idx, sess):
        yield StreamToolInputAvailable(toolCallId="call_0", toolName="t", input={})
        yield StreamToolOutputAvailable(toolCallId="call_0", toolName="t", output="{}")

    import backend.copilot.service as svc

    orig = svc._yield_tool_call
    svc._yield_tool_call = fake_yield
    try:
        events = [
            e
            async for e in _execute_tool_calls_parallel(
                tool_calls, cast(Any, FakeSession())
            )
        ]
    finally:
        svc._yield_tool_call = orig

    assert len(events) == 2


@pytest.mark.asyncio
async def test_retryable_error_propagates():
    """Retryable errors should be raised after all tools finish."""
    from backend.copilot.response_model import StreamToolOutputAvailable
    from backend.copilot.service import _execute_tool_calls_parallel

    tool_calls = [
        {
            "id": f"call_{i}",
            "type": "function",
            "function": {"name": f"t_{i}", "arguments": "{}"},
        }
        for i in range(2)
    ]

    class FakeSession:
        session_id = "test"
        user_id = "test"

        def __init__(self):
            self.messages = []

    async def fake_yield(tc_list, idx, sess):
        if idx == 1:
            raise KeyError("bad")
        from backend.copilot.response_model import StreamToolInputAvailable

        yield StreamToolInputAvailable(
            toolCallId=tc_list[idx]["id"], toolName="t_0", input={}
        )
        await asyncio.sleep(0.05)
        yield StreamToolOutputAvailable(
            toolCallId=tc_list[idx]["id"], toolName="t_0", output="{}"
        )

    import backend.copilot.service as svc

    orig = svc._yield_tool_call
    svc._yield_tool_call = fake_yield
    try:
        events = []
        with pytest.raises(KeyError):
            async for event in _execute_tool_calls_parallel(
                tool_calls, cast(Any, FakeSession())
            ):
                events.append(event)
        # First tool's events should still be yielded
        assert any(isinstance(e, StreamToolOutputAvailable) for e in events)
    finally:
        svc._yield_tool_call = orig


@pytest.mark.asyncio
async def test_session_shared_across_parallel_tools():
    """All parallel tools should receive the same session instance."""
    from backend.copilot.response_model import (
        StreamToolInputAvailable,
        StreamToolOutputAvailable,
    )
    from backend.copilot.service import _execute_tool_calls_parallel

    tool_calls = [
        {
            "id": f"call_{i}",
            "type": "function",
            "function": {"name": f"t_{i}", "arguments": "{}"},
        }
        for i in range(3)
    ]

    class FakeSession:
        session_id = "test"
        user_id = "test"

        def __init__(self):
            self.messages = []

    observed_sessions = []

    async def fake_yield(tc_list, idx, sess):
        observed_sessions.append(sess)
        yield StreamToolInputAvailable(
            toolCallId=tc_list[idx]["id"], toolName=f"t_{idx}", input={}
        )
        yield StreamToolOutputAvailable(
            toolCallId=tc_list[idx]["id"], toolName=f"t_{idx}", output="{}"
        )

    import backend.copilot.service as svc

    orig = svc._yield_tool_call
    svc._yield_tool_call = fake_yield
    try:
        async for _ in _execute_tool_calls_parallel(
            tool_calls, cast(Any, FakeSession())
        ):
            pass
    finally:
        svc._yield_tool_call = orig

    assert len(observed_sessions) == 3
    assert observed_sessions[0] is observed_sessions[1] is observed_sessions[2]


@pytest.mark.asyncio
async def test_cancellation_cleans_up():
    """Generator close should cancel in-flight tasks."""
    from backend.copilot.response_model import StreamToolInputAvailable
    from backend.copilot.service import _execute_tool_calls_parallel

    tool_calls = [
        {
            "id": f"call_{i}",
            "type": "function",
            "function": {"name": f"t_{i}", "arguments": "{}"},
        }
        for i in range(2)
    ]

    class FakeSession:
        session_id = "test"
        user_id = "test"

        def __init__(self):
            self.messages = []

    started = asyncio.Event()

    async def fake_yield(tc_list, idx, sess):
        yield StreamToolInputAvailable(
            toolCallId=tc_list[idx]["id"], toolName=f"t_{idx}", input={}
        )
        started.set()
        await asyncio.sleep(10)  # simulate long-running

    import backend.copilot.service as svc

    orig = svc._yield_tool_call
    svc._yield_tool_call = fake_yield
    try:
        gen = _execute_tool_calls_parallel(tool_calls, cast(Any, FakeSession()))
        await gen.__anext__()  # get first event
        await started.wait()
        await gen.aclose()  # close generator
    finally:
        svc._yield_tool_call = orig
    # If we get here without hanging, cleanup worked
