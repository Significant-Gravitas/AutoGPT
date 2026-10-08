"""The whole turn: setup -> stream -> rows, usage, history upload -> finish."""

from contextlib import ExitStack
from unittest.mock import AsyncMock, patch

import pytest
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.settings import ModelSettings

from backend.copilot.response_model import (
    StreamError,
    StreamFinish,
    StreamStart,
    StreamTextDelta,
    StreamUsage,
)

from .conftest import echo_tool, make_session
from .model import PaiRoute, mark_static_prefix
from .service import stream_chat_completion_pai
from .turn_setup import PreparedTurn


def _turn() -> PreparedTurn:
    return PreparedTurn(
        session=make_session(),
        route=PaiRoute(model="m", source="env", provider="openrouter"),
        static_instructions="STATIC",
        turn_context="<turn_context>t</turn_context>",
        user_prompt="hello",
        message="hello",
        graphiti_enabled=False,
        tools=[echo_tool()],
        disabled_groups=[],
        disabled_tools=frozenset(),
        working_dir=None,
        history=[],
        held=[],
        upload_safe=True,
        opening_entries=[],
        turn_start=1,
        message_id="msg-1",
    )


def _patches(model: FunctionModel, upload: AsyncMock, record: AsyncMock):
    return (
        patch(
            "backend.copilot.pai.service.prepare_turn",
            AsyncMock(return_value=_turn()),
        ),
        patch(
            "backend.copilot.pai.service.build_model",
            return_value=(model, ModelSettings()),
        ),
        patch(
            "backend.copilot.pai.resume.resolve_answered", AsyncMock(return_value=[])
        ),
        patch(
            "backend.copilot.pai.runner.drain_pending_safe", AsyncMock(return_value=[])
        ),
        patch(
            "backend.copilot.pai.service.upsert_chat_session",
            AsyncMock(side_effect=lambda session: session),
        ),
        patch("backend.copilot.pai.service.record_usage", record),
        patch("backend.copilot.pai.service.upload_history", upload),
    )


@pytest.mark.asyncio
async def test_turn_streams_and_stores_its_history():
    async def stream(messages, info):
        assert info.instructions is not None
        # Static first, the per-turn context last.
        assert info.instructions.startswith("STATIC")
        assert info.instructions.endswith("<turn_context>t</turn_context>")
        yield "Hi there."

    upload, record = AsyncMock(), AsyncMock()
    with ExitStack() as stack:
        for p in _patches(FunctionModel(stream_function=stream), upload, record):
            stack.enter_context(p)
        events = [
            e
            async for e in stream_chat_completion_pai(
                "sess-1", "hello", user_id="user-1"
            )
        ]

    assert isinstance(events[0], StreamStart)
    assert isinstance(events[-1], StreamFinish)
    assert (
        "".join(e.delta for e in events if isinstance(e, StreamTextDelta))
        == "Hi there."
    )
    assert any(isinstance(e, StreamUsage) for e in events)
    record.assert_awaited_once()
    upload.assert_awaited_once()
    assert upload.await_args is not None
    stored = upload.await_args.args[2]
    assert [m.kind for m in stored] == ["request", "response"]


@pytest.mark.asyncio
async def test_model_failure_ends_in_the_error_envelope():
    async def stream(messages, info):
        raise RuntimeError("upstream exploded")
        yield ""  # pragma: no cover

    upload, record = AsyncMock(), AsyncMock()
    with ExitStack() as stack:
        for p in _patches(FunctionModel(stream_function=stream), upload, record):
            stack.enter_context(p)
        stack.enter_context(
            patch(
                "backend.copilot.pai.service.classify_provider_failure",
                return_value=None,
            )
        )
        events = [
            e
            async for e in stream_chat_completion_pai(
                "sess-1", "hello", user_id="user-1"
            )
        ]

    errors = [e for e in events if isinstance(e, StreamError)]
    assert len(errors) == 1 and "upstream exploded" in errors[0].errorText
    record.assert_awaited_once()


def test_cache_breakpoint_sits_on_the_static_prefix():
    system = {"role": "system", "content": "STATIC\n\n<turn_context>x</turn_context>"}
    marked = mark_static_prefix(system, "STATIC")
    blocks = marked["content"]
    assert blocks[0]["text"] == "STATIC"
    assert "cache_control" in blocks[0]
    assert blocks[1] == {"type": "text", "text": "\n\n<turn_context>x</turn_context>"}
