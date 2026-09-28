"""A tool call cut off by Stop reads as what the tool said it would, not ``""``."""

import asyncio

import pytest
from claude_agent_sdk import AssistantMessage, ToolUseBlock

from backend.copilot.response_model import StreamToolOutputAvailable

from .cancelled_output import record_cancelled_output
from .response_adapter import SDKResponseAdapter
from .tool_adapter import (
    _make_truncating_wrapper,
    pop_cancelled_tool_output,
    set_execution_context,
)

_ARGS = {"expert_id": "expert-b", "prompt": "Draft the PRD"}
_CANCELLED = '{"status": "cancelled", "sub_session_id": "sub-1"}'


@pytest.fixture(autouse=True)
def context():
    set_execution_context(user_id="u", session=None, sandbox=None, sdk_cwd="/tmp/t")


def _waiting_tool(started: asyncio.Event):
    async def handler(_args):
        record_cancelled_output(_CANCELLED)
        started.set()
        await asyncio.sleep(3600)
        return {"content": [{"type": "text", "text": "never"}], "isError": False}

    return _make_truncating_wrapper(
        handler, "delegate_to_expert", required_args=["expert_id", "prompt"]
    )


@pytest.mark.asyncio
async def test_a_call_stopped_mid_wait_keeps_its_cancelled_result():
    started = asyncio.Event()
    task = asyncio.create_task(_waiting_tool(started)(dict(_ARGS)))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert pop_cancelled_tool_output("delegate_to_expert", _ARGS) == _CANCELLED


@pytest.mark.asyncio
async def test_a_call_that_returns_leaves_no_cancelled_result():
    async def handler(_args):
        record_cancelled_output(_CANCELLED)
        return {"content": [{"type": "text", "text": "done"}], "isError": False}

    wrapper = _make_truncating_wrapper(
        handler, "delegate_to_expert", required_args=["expert_id", "prompt"]
    )
    await wrapper(dict(_ARGS))

    assert pop_cancelled_tool_output("delegate_to_expert", _ARGS) is None


@pytest.mark.asyncio
async def test_the_stop_flush_writes_the_cancelled_result_not_an_empty_one():
    started = asyncio.Event()
    task = asyncio.create_task(_waiting_tool(started)(dict(_ARGS)))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    adapter = SDKResponseAdapter(message_id="m1", session_id="s1")
    adapter.convert_message(
        AssistantMessage(
            content=[
                ToolUseBlock(
                    id="call-1",
                    name="mcp__copilot__delegate_to_expert",
                    input=dict(_ARGS),
                )
            ],
            model="test",
        )
    )

    flushed: list = []
    adapter.flush_unresolved_tool_calls(flushed)

    outputs = [r for r in flushed if isinstance(r, StreamToolOutputAvailable)]
    assert [o.output for o in outputs] == [_CANCELLED]
