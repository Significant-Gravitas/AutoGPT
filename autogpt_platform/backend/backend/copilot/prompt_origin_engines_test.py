"""Each engine writes a scheduled turn's prompt row itself; these pin that the
row it writes carries the stamp the executor handed over."""

from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.baseline.service import stream_chat_completion_baseline
from backend.copilot.model import ChatMessage, ChatSession
from backend.copilot.prompt_origin import PromptOrigin, with_prompt_origin
from backend.copilot.sdk.service import stream_chat_completion_sdk

ORIGIN = PromptOrigin(kind="scheduled_followup", schedule_id="job-1")
PROMPT = "Check the inbox and summarise anything new."


class _StopAfterPromptRow(Exception):
    """Raised by the first dependency each engine touches after the append."""


def _session_mid_conversation() -> ChatSession:
    session = ChatSession.new("user-1", dry_run=False)
    session.messages = [
        ChatMessage(role="user", content="Remind me hourly.", sequence=0),
        ChatMessage(role="assistant", content="Scheduled.", sequence=1),
    ]
    return session


def _assert_stamped_prompt_row(session: ChatSession) -> None:
    row = session.messages[-1]
    assert row.role == "assistant"
    assert row.content == PROMPT
    assert row.prompt_origin == ORIGIN
    assert row.metadata is None
    assert (
        row.model_dump() == ChatMessage(role="assistant", content=PROMPT).model_dump()
    )


@pytest.mark.asyncio
async def test_sdk_engine_stamps_the_scheduled_prompt_row():
    session = _session_mid_conversation()

    with (
        patch(
            "backend.copilot.sdk.service.build_expert_identity_suffix",
            new=AsyncMock(return_value=""),
        ),
        patch(
            "backend.copilot.sdk.service.get_redis_async",
            new=AsyncMock(side_effect=_StopAfterPromptRow),
        ),
        pytest.raises(_StopAfterPromptRow),
    ):
        async for _ in stream_chat_completion_sdk(
            session_id=session.session_id,
            message=PROMPT,
            is_user_message=False,
            user_id="user-1",
            session=session,
            message_metadata=with_prompt_origin(None, ORIGIN),
        ):
            pass

    _assert_stamped_prompt_row(session)


@pytest.mark.asyncio
async def test_baseline_engine_stamps_the_scheduled_prompt_row():
    session = _session_mid_conversation()

    with (
        patch(
            "backend.copilot.baseline.service.build_expert_identity_suffix",
            new=AsyncMock(return_value=""),
        ),
        patch(
            "backend.copilot.baseline.service.drain_pending_safe",
            new=AsyncMock(side_effect=_StopAfterPromptRow),
        ),
        pytest.raises(_StopAfterPromptRow),
    ):
        async for _ in stream_chat_completion_baseline(
            session_id=session.session_id,
            message=PROMPT,
            is_user_message=False,
            user_id="user-1",
            session=session,
            message_metadata=with_prompt_origin(None, ORIGIN),
        ):
            pass

    _assert_stamped_prompt_row(session)
