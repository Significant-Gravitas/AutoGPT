import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from langsmith.run_trees import RunTree

from backend.copilot.response_model import StreamHeartbeat

from . import conversation_output as output_module
from . import conversation_output_test as fixtures
from . import service
from .conversation_output_test import assistant, event, result

blocked = fixtures.blocked
client = fixtures.client
script = fixtures.script

pytestmark = pytest.mark.asyncio


def fail(*args, **kwargs):
    raise RuntimeError("trace failure")


@pytest.mark.parametrize("failure", ["lookup", "publish", "malformed", "untraced"])
async def test_observer_failure_preserves_messages(
    client, script, monkeypatch, failure
):
    run = RunTree(name="claude.conversation", run_type="chain")
    monkeypatch.setattr(output_module, "get_current_run_tree", lambda: run)
    if failure == "lookup":
        monkeypatch.setattr(output_module, "get_current_run_tree", fail)
    elif failure == "publish":
        monkeypatch.setattr(RunTree, "set", fail)
    elif failure == "untraced":
        monkeypatch.setattr(output_module, "get_current_run_tree", lambda: None)
    script.extend([assistant("done"), result()])
    if failure in {"malformed", "untraced"}:
        script.insert(0, event("message_start", message=None))
    assert [message async for message in client.receive_messages()] == script


async def test_trace_end_failure_preserves_transport_error(client, script, monkeypatch):
    run = RunTree(name="claude.conversation", run_type="chain")
    monkeypatch.setattr(output_module, "get_current_run_tree", lambda: run)
    monkeypatch.setattr(RunTree, "end", fail)
    script.append(ValueError("transport failure"))
    with pytest.raises(ValueError, match="transport failure"):
        await anext(client.receive_messages())


async def test_finalization_failure_preserves_result_and_usage():
    terminal = result(
        usage={"input_tokens": 7, "output_tokens": 3}, total_cost_usd=0.25
    )

    async def receive_response():
        yield terminal
        raise RuntimeError("trace finalization failed")

    sdk_client = MagicMock(receive_response=receive_response)
    assert [message async for message in service._iter_sdk_messages(sdk_client)] == [
        terminal
    ]


async def test_close_failure_preserves_cancellation(monkeypatch):
    blocked = asyncio.Event()

    async def receive_response():
        try:
            blocked.set()
            await asyncio.Event().wait()
            yield result()
        finally:
            raise RuntimeError("trace cleanup failed")

    sdk_client = MagicMock(receive_response=receive_response)

    async def consume():
        async for _ in service._iter_sdk_messages(sdk_client):
            pass

    task = asyncio.create_task(consume())
    await asyncio.wait_for(blocked.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.parametrize("exit_kind", ["break", "error", "close"])
async def test_consumer_closes_sdk_messages(monkeypatch, exit_kind):
    closed = False

    async def messages(*args, **kwargs):
        nonlocal closed
        try:
            yield None
            await asyncio.Event().wait()
        finally:
            closed = True

    monkeypatch.setattr(service, "_iter_sdk_messages", messages)
    ctx = MagicMock()
    ctx.tool_display = None
    ctx.lock.refresh = AsyncMock()
    ctx.compaction.has_pending_start = False
    if exit_kind == "error":
        ctx.lock.refresh.side_effect = ValueError("consumer failure")
    loop_state = service._SDKLoopState(
        last_real_msg_time=0, last_flush_time=0, provider_refused=True
    )
    stream = service._consume_sdk_until_done(
        MagicMock(), ctx, MagicMock(), MagicMock(), loop_state
    )
    try:
        if exit_kind == "error":
            with pytest.raises(ValueError, match="consumer failure"):
                await anext(stream)
        elif exit_kind == "close":
            await anext(stream)
            await stream.aclose()
        else:
            assert len([response async for response in stream]) == 1
        assert closed
    finally:
        await stream.aclose()


async def test_explicit_close_failure_does_not_escape():
    async def receive_response():
        try:
            yield assistant("partial")
        finally:
            raise RuntimeError("trace cleanup failed")

    sdk_client = MagicMock(receive_response=receive_response)
    stream = service._iter_sdk_messages(sdk_client)
    await anext(stream)
    await stream.aclose()


@pytest.mark.parametrize("reprompt", [False, True])
async def test_attempt_closes_consumer_before_sdk_disconnect(monkeypatch, reprompt):
    cleanup = []

    async def consume(*args):
        try:
            yield StreamHeartbeat()
        finally:
            cleanup.append("consumer")

    async def disconnect(*args):
        cleanup.append("client")

    sdk_client = MagicMock()
    sdk_client.__aenter__.return_value = sdk_client
    sdk_client.query = AsyncMock()
    monkeypatch.setattr(
        service, "ConversationOutputClient", MagicMock(return_value=sdk_client)
    )
    monkeypatch.setattr(service, "_consume_sdk_until_done", consume)
    monkeypatch.setattr(service, "_safe_close_sdk_client", disconnect)
    ctx = MagicMock()
    ctx.attachments.image_blocks = []
    ctx.file_ids = None
    ctx.session.messages = []
    state = MagicMock()
    state.generations.start_launch = AsyncMock()
    state.query_message = "hello"
    state.usage.cli_session_total_usd = 0
    state.adapter.pending_thinking_only_reprompt = reprompt
    state.thinking_only_reprompted = False
    stream = service._run_stream_attempt(ctx, state)
    try:
        await anext(stream)  # Starting status.
        await anext(stream)  # Reading status.
        assert isinstance(await anext(stream), StreamHeartbeat)
        if reprompt:
            assert isinstance(await anext(stream), StreamHeartbeat)
        await stream.aclose()
        assert cleanup == ["consumer"] * (2 if reprompt else 1) + ["client"]
    finally:
        await stream.aclose()
