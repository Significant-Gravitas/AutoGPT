"""The engine folds answered held calls into the turn it is starting.

Driven through the engine's real entry point and stopped at the fold, so
removing the fold turns the test red. The held call itself runs inside
``resolve_answered``, so the test also proves the turn's execution context —
the chat whose ceiling an approval raises included — is set before it runs.
"""

import asyncio
import contextlib
import json
from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.context import get_current_envelope, set_execution_context
from backend.copilot.gate.held import HeldCall, _result_row
from backend.copilot.model import ChatSession
from backend.copilot.pending_message_helpers import (
    combine_pending_with_current,
    persist_pending_as_user_rows,
)
from backend.copilot.pending_messages import (
    PendingMessage,
    format_pending_as_user_message,
)
from backend.copilot.response_model import StreamStatus
from backend.copilot.sdk.expert_tool_gate_test import _make_patches, _make_session
from backend.copilot.sdk.service import (
    _session_messages_to_transcript,
    stream_chat_completion_sdk,
)
from backend.copilot.sdk.tool_adapter import cap_late_tool_result
from backend.copilot.tree import root_envelope

_RESULT = PendingMessage(
    content='<held_call_result tool="post_to_chat_platform">posted</held_call_result>'
)
_RUNNING = StreamStatus(message="Running the action you approved: Post a message")
_SECOND = PendingMessage(
    content='<held_call_result tool="send_email">sent</held_call_result>'
)
_RUNNING_SECOND = StreamStatus(message="Running the action you approved: Send an email")


class _StopAtFold(Exception):
    pass


@pytest.mark.asyncio
async def test_the_sdk_engine_opens_its_turn_with_the_held_result():
    order: list[str] = []
    folded: list[list[PendingMessage]] = []

    caps: list[object] = []
    chats: list[str | None] = []

    async def resolve(_user_id, _session, deliver, **kwargs):
        order.append("resolve")
        caps.append(kwargs.get("cap"))
        chats.append(getattr(get_current_envelope(), "spend_session_id", None))
        yield _RUNNING
        deliver(_RESULT)

    def set_context(*args, **kwargs):
        order.append("context")
        set_execution_context(*args, **kwargs)

    async def persist(_session, _builder, pending, **_kwargs):
        order.append("fold")
        folded.append(list(pending))
        raise _StopAtFold

    patches, mcp_server = _make_patches()
    mcp_server.side_effect = None
    session = _make_session()
    envelope = root_envelope("turn-2", session_id=session.session_id)
    with contextlib.ExitStack() as stack:
        for target, kwargs in patches:
            stack.enter_context(patch(target, **kwargs))
        context = stack.enter_context(
            patch(
                "backend.copilot.sdk.service.set_execution_context",
                side_effect=set_context,
            )
        )
        stack.enter_context(
            patch("backend.copilot.sdk.service.resolve_answered", new=resolve)
        )
        stack.enter_context(
            patch(
                "backend.copilot.sdk.service.persist_pending_as_user_rows",
                new=persist,
            )
        )
        stack.enter_context(
            patch(
                "backend.copilot.sdk.service.build_session_context",
                new=AsyncMock(return_value=""),
            )
        )
        with contextlib.suppress(_StopAtFold):
            async with asyncio.timeout(10):
                async for chunk in stream_chat_completion_sdk(
                    session_id=session.session_id,
                    message="hello",
                    is_user_message=True,
                    user_id="test-user",
                    session=session,
                    envelope=envelope,
                ):
                    if chunk is _RUNNING:
                        order.append("streamed")
        set_execution_context(None, None)

    context.assert_called()
    assert order.index("context") < order.index("resolve")
    assert order.index("resolve") < order.index("streamed") < order.index("fold")
    assert folded == [[_RESULT]]
    # A late result is cut by the same rule as a direct MCP tool result.
    assert caps == [cap_late_tool_result]
    assert chats == [session.session_id]


@pytest.mark.asyncio
async def test_the_card_copy_on_a_held_result_never_reaches_the_model():
    """The result row carries the card's arguments for the chain row, up to the
    card's clip; each engine's fold and history replay read the content alone."""
    marker = "card-copy-only-7f3a"
    call = HeldCall(
        review_id="r1", tool_name="post_to_chat_platform", tool_call_id="c1", args={}
    )
    row = _result_row(
        call,
        "posted",
        "approved",
        {"arguments": {"text": marker}, "fields": [{"key": "text", "label": marker}]},
    )
    assert marker in json.dumps(row.metadata)

    session = ChatSession.new(user_id="u1", dry_run=False)

    async def persisted(s, _prefix):
        for i, message in enumerate(s.messages):
            message.sequence = i
        return s

    with patch(
        "backend.copilot.pending_message_helpers.persist_session_safe", persisted
    ):
        assert await persist_pending_as_user_rows(session, None, [row], log_prefix="")
    assert session.messages[-1].metadata == row.metadata

    model_inputs = [
        format_pending_as_user_message(row)["content"],
        combine_pending_with_current([row], "go on", request_arrival_at=0),
        _session_messages_to_transcript(session.messages),
    ]
    assert all("posted" in text for text in model_inputs)
    assert not [text for text in model_inputs if marker in text]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stop, kept",
    [
        ("closed at the next card's status", [_RESULT]),
        ("cancelled before the fold", [_RESULT, _SECOND]),
    ],
)
async def test_a_sdk_turn_that_ends_early_keeps_the_results_of_calls_it_ran(stop, kept):
    """Their approvals are spent, so nothing would run those calls again; a
    card whose status the stream was closed at stays held for the next turn."""

    async def resolve(_user_id, _session, deliver, **_kwargs):
        yield _RUNNING
        deliver(_RESULT)
        yield _RUNNING_SECOND
        deliver(_SECOND)

    patches, mcp_server = _make_patches()
    mcp_server.side_effect = None
    session = _make_session()
    with contextlib.ExitStack() as stack:
        for target, kwargs in patches:
            stack.enter_context(patch(target, **kwargs))
        stack.enter_context(
            patch("backend.copilot.sdk.service.resolve_answered", new=resolve)
        )
        stack.enter_context(
            patch(
                "backend.copilot.sdk.service.build_session_context",
                new=AsyncMock(return_value=""),
            )
        )
        stack.enter_context(
            patch(
                "backend.copilot.sdk.service.drain_pending_safe",
                AsyncMock(side_effect=asyncio.CancelledError),
            )
        )
        stack.enter_context(
            patch(
                "backend.copilot.pending_message_helpers.persist_session_safe",
                _numbered,
            )
        )
        turn = stream_chat_completion_sdk(
            session_id=session.session_id,
            message="hello",
            is_user_message=True,
            user_id="test-user",
            session=session,
            envelope=root_envelope("turn-2", session_id=session.session_id),
        )
        try:
            async for chunk in turn:
                if chunk is _RUNNING_SECOND and stop.startswith("closed"):
                    await turn.aclose()
                    break
        except asyncio.CancelledError:
            assert stop.startswith("cancelled")
        set_execution_context(None, None)

    assert _saved_results(session) == [r.content for r in kept]


async def _numbered(session: ChatSession, _log_prefix: str) -> ChatSession:
    for i, message in enumerate(session.messages):
        message.sequence = i
    return session


def _saved_results(session: ChatSession) -> list[str]:
    return [
        m.content or ""
        for m in session.messages
        if m.sequence is not None and "<held_call_result" in (m.content or "")
    ]
