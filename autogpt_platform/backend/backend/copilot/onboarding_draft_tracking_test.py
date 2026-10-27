"""Otto's opening prompt in an onboarding draft is not the user's message.

During onboarding Otto starts the user's first suggested tasks as chats of
their own, opening each with the task's prompt as a user row. The user picked
the task but did not send it, so neither engine may report it as a
``chat_message_sent``; the user's own replies in the same chat still count.

Each test drives the engine's real entry point with PostHog's client swapped
for a mock, and stops the turn just after its user message is tracked.
"""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from backend.copilot.baseline.service import stream_chat_completion_baseline
from backend.copilot.model import ChatSession, ChatSessionMetadata
from backend.copilot.sdk.service import stream_chat_completion_sdk
from backend.copilot.tracking import ONBOARDING_DRAFT_KICKOFF_KIND
from backend.util import posthog_client

_SDK = "backend.copilot.sdk.service"
_BASELINE = "backend.copilot.baseline.service"
_KICKOFF = {"kind": ONBOARDING_DRAFT_KICKOFF_KIND}


class _StopAfterTracking(Exception):
    pass


@pytest.fixture
def capture(monkeypatch: pytest.MonkeyPatch) -> Mock:
    client = Mock()
    monkeypatch.setattr(posthog_client, "get_posthog_client", lambda: client)
    return client.capture


def _draft_session() -> ChatSession:
    session = ChatSession.new("user-1", dry_run=False)
    session.title = "Chase unpaid invoices"
    session.metadata = ChatSessionMetadata(origin="interactive")
    return session


def _messages_sent(capture: Mock) -> list[str]:
    return [
        call.kwargs["properties"]["chat_session_id"]
        for call in capture.call_args_list
        if call.kwargs["event"] == "chat_message_sent"
    ]


async def _sdk_turn(
    session: ChatSession, message: str, metadata: dict[str, Any] | None
) -> AsyncMock:
    persist = AsyncMock(side_effect=_StopAfterTracking)
    with (
        patch(f"{_SDK}.build_expert_identity_suffix", AsyncMock(return_value="")),
        patch(f"{_SDK}.clear_pending_question", AsyncMock()),
        patch(f"{_SDK}.persist_session_safe", persist),
        pytest.raises(_StopAfterTracking),
    ):
        async with asyncio.timeout(10):
            async for _ in stream_chat_completion_sdk(
                session_id=session.session_id,
                message=message,
                is_user_message=True,
                user_id="user-1",
                session=session,
                message_metadata=metadata,
            ):
                pass
    return persist


async def _baseline_turn(
    session: ChatSession, message: str, metadata: dict[str, Any] | None
) -> None:
    with (
        patch(f"{_BASELINE}.build_expert_identity_suffix", AsyncMock(return_value="")),
        patch(f"{_BASELINE}.clear_pending_question", AsyncMock()),
        patch(
            f"{_BASELINE}.find_initiating_user_row",
            side_effect=_StopAfterTracking,
        ),
        pytest.raises(_StopAfterTracking),
    ):
        async with asyncio.timeout(10):
            async for _ in stream_chat_completion_baseline(
                session_id=session.session_id,
                message=message,
                is_user_message=True,
                user_id="user-1",
                session=session,
                message_metadata=metadata,
            ):
                pass


@pytest.mark.asyncio
async def test_the_sdk_engine_sends_no_chat_message_for_the_draft_prompt(
    capture: Mock,
):
    session = _draft_session()

    persist = await _sdk_turn(session, "Chase my invoices.", _KICKOFF)

    assert _messages_sent(capture) == []
    # The marker is on the saved user row, where the analytics queries read it.
    saved = persist.await_args.args[0].messages[-1]
    assert (saved.role, saved.metadata) == ("user", _KICKOFF)


@pytest.mark.asyncio
async def test_the_sdk_engine_still_counts_the_users_reply_in_a_draft(
    capture: Mock,
):
    session = _draft_session()
    await _sdk_turn(session, "Chase my invoices.", _KICKOFF)

    await _sdk_turn(session, "Only the ones over 30 days.", None)

    assert _messages_sent(capture) == [session.session_id]


@pytest.mark.asyncio
async def test_the_baseline_engine_sends_no_chat_message_for_the_draft_prompt(
    capture: Mock,
):
    session = _draft_session()

    await _baseline_turn(session, "Chase my invoices.", _KICKOFF)

    assert _messages_sent(capture) == []
    saved = session.messages[-1]
    assert (saved.role, saved.metadata) == ("user", _KICKOFF)


@pytest.mark.asyncio
async def test_the_baseline_engine_still_counts_the_users_reply_in_a_draft(
    capture: Mock,
):
    session = _draft_session()
    await _baseline_turn(session, "Chase my invoices.", _KICKOFF)

    await _baseline_turn(session, "Only the ones over 30 days.", None)

    assert _messages_sent(capture) == [session.session_id]
