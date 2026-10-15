"""The engine writes a scheduled turn's prompt row itself; these pin that the
row it writes carries the stamp the executor handed over, on a schedule's
first fire and on a later fire of the same prompt."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.model import ChatMessage, ChatSession
from backend.copilot.prompt_origin import PromptOrigin, with_prompt_origin
from backend.copilot.sdk.service import stream_chat_completion_sdk

ORIGIN = PromptOrigin(
    kind="scheduled_followup",
    schedule_id="job-1",
    fired_at=datetime(2026, 10, 6, 9, 0, tzinfo=UTC),
)
PROMPT = "Check the inbox and summarise anything new."
FIRES = pytest.mark.parametrize(
    "after_a_fire", [False, True], ids=["first-fire", "repeat-fire"]
)


class _StopAfterPromptRow(Exception):
    """Raised by the first dependency the engine touches after the append."""


def _session_mid_conversation(after_a_fire: bool) -> ChatSession:
    session = ChatSession.new("user-1", dry_run=False)
    session.messages = [
        ChatMessage(role="user", content="Remind me hourly.", sequence=0),
        ChatMessage(role="assistant", content="Scheduled.", sequence=1),
    ]
    if after_a_fire:
        # The previous fire's prompt and reply, loaded without the stamp as
        # the executor loads every stored row.
        session.messages += [
            ChatMessage(role="assistant", content=PROMPT, sequence=2),
            ChatMessage(role="assistant", content="Nothing new.", sequence=3),
        ]
    return session


def _assert_stamped_prompt_row(session: ChatSession, rows_before: int) -> None:
    assert len(session.messages) == rows_before + 1
    row = session.messages[-1]
    assert row.role == "assistant"
    assert row.content == PROMPT
    assert row.prompt_origin == ORIGIN
    assert row.metadata is None
    assert (
        row.model_dump() == ChatMessage(role="assistant", content=PROMPT).model_dump()
    )


@FIRES
@pytest.mark.asyncio
async def test_sdk_engine_stamps_the_scheduled_prompt_row(after_a_fire: bool):
    session = _session_mid_conversation(after_a_fire)
    await _run_sdk_to_the_prompt_row(session, rows_before=len(session.messages))


async def _run_sdk_to_the_prompt_row(session: ChatSession, *, rows_before: int) -> None:
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

    _assert_stamped_prompt_row(session, rows_before)
