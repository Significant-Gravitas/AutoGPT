"""The engine folds answered held calls into the turn it is starting.

Driven through the engine's real entry point and stopped at the fold, so
removing the fold turns the test red. The held call itself runs inside
``resolve_answered``, so the test also proves the turn's execution context —
the chat whose ceiling an approval raises included — is set before it runs.
"""

import contextlib
import json
from unittest.mock import patch

import pytest

from backend.copilot.context import get_current_envelope, set_execution_context
from backend.copilot.model import ChatSession
from backend.copilot.pending_messages import PendingMessage
from backend.copilot.sdk.expert_tool_gate_test import _make_patches, _make_session
from backend.copilot.sdk.tool_adapter import cap_late_tool_result
from backend.copilot.tree import root_envelope

_RESULT = PendingMessage(
    content='<held_call_result tool="post_to_chat_platform">posted</held_call_result>'
)


class _StopAtFold(Exception):
    pass


@pytest.mark.asyncio
async def test_the_sdk_engine_opens_its_turn_with_the_held_result():
    from backend.copilot.sdk.service import stream_chat_completion_sdk

    order: list[str] = []
    folded: list[list[PendingMessage]] = []

    caps: list[object] = []
    chats: list[str | None] = []

    async def resolve(*_args, **kwargs):
        order.append("resolve")
        caps.append(kwargs.get("cap"))
        chats.append(getattr(get_current_envelope(), "spend_session_id", None))
        return [_RESULT]

    def set_context(*args, **kwargs):
        order.append("context")
        set_execution_context(*args, **kwargs)

    async def persist(_session, _builder, pending, **_kwargs):
        folded.append(list(pending))
        raise _StopAtFold

    patches, _, _ = _make_patches(hire_experts_enabled=False)
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
        with contextlib.suppress(_StopAtFold):
            async for _ in stream_chat_completion_sdk(
                session_id=session.session_id,
                message="hello",
                is_user_message=True,
                user_id="test-user",
                session=session,
                envelope=envelope,
            ):
                pass
        set_execution_context(None, None)

    context.assert_called()
    assert order.index("context") < order.index("resolve")
    assert folded == [[_RESULT]]
    # A late result is cut by the same rule as a direct MCP tool result.
    assert caps == [cap_late_tool_result]
    assert chats == [session.session_id]


@pytest.mark.asyncio
async def test_the_card_copy_on_a_held_result_never_reaches_the_model():
    """The result row carries the card's arguments for the chain row, up to the
    card's clip; each engine's fold and history replay read the content alone."""
    from backend.copilot.gate.held import HeldCall, _result_row
    from backend.copilot.pending_message_helpers import (
        combine_pending_with_current,
        persist_pending_as_user_rows,
    )
    from backend.copilot.pending_messages import format_pending_as_user_message
    from backend.copilot.sdk.service import _session_messages_to_transcript

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
