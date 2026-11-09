"""The server-sent kickoff for hires no browser follows — no DB required."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_mock

from backend.api.features.experts import kickoff
from backend.copilot.active_turns import ConcurrentTurnLimitError, SessionNotIdle
from backend.copilot.expert_kickoff import (
    EXPERT_KICKOFF_KIND,
    EXPERT_KICKOFF_PROMPT,
    expert_kickoff_message_id,
)

_MODULE = "backend.api.features.experts.kickoff"


def _session() -> SimpleNamespace:
    return SimpleNamespace(
        session_id="session-1",
        organization_id="org-1",
        team_id="team-1",
        metadata=SimpleNamespace(
            origin=None,
            source_platform=None,
            llm_auth_provider="platform",
            llm_credential_id=None,
        ),
    )


@pytest.fixture
def mocks(mocker: pytest_mock.MockerFixture) -> SimpleNamespace:
    return SimpleNamespace(
        sessions=mocker.patch(
            f"{_MODULE}.get_user_sessions",
            new_callable=AsyncMock,
            return_value=([], 0),
        ),
        create=mocker.patch(
            f"{_MODULE}.get_or_create_expert_kickoff_session",
            new_callable=AsyncMock,
            return_value=_session(),
        ),
        has_reply=mocker.patch(
            f"{_MODULE}.chat_message_has_assistant_reply",
            new_callable=AsyncMock,
            return_value=None,
        ),
        permissions=mocker.patch(
            f"{_MODULE}.resolve_session_permissions", return_value=None
        ),
        schedule=mocker.patch(
            f"{_MODULE}.schedule_chat_turn",
            new_callable=AsyncMock,
            return_value="turn-1",
        ),
        enqueue=mocker.patch(
            f"{_MODULE}.turn_queue.try_enqueue_turn", new_callable=AsyncMock
        ),
    )


@pytest.mark.asyncio
async def test_kickoff_starts_the_hidden_turn_in_the_canonical_session(mocks):
    started = await kickoff.start_expert_kickoff(
        "user-1", "expert-1", organization_id="org-1", team_id="team-1"
    )

    assert started is True
    mocks.create.assert_awaited_once_with(
        "user-1",
        "expert-1",
        dry_run=False,
        organization_id="org-1",
        team_id="team-1",
    )
    sent = mocks.schedule.await_args.kwargs
    assert sent["session_id"] == "session-1"
    assert sent["message"] == EXPERT_KICKOFF_PROMPT
    # The same id the chat route derives, so a browser that opens the expert
    # later adopts this turn instead of sending its own.
    assert sent["message_id"] == expert_kickoff_message_id(
        "user-1", "session-1", "expert-1"
    )
    assert sent["message_metadata"] == {
        "hidden": True,
        "kind": EXPERT_KICKOFF_KIND,
        "expert_id": "expert-1",
    }
    assert sent["expert_id"] == "expert-1"
    assert sent["organization_id"] == "org-1"


@pytest.mark.asyncio
async def test_expert_with_a_thread_gets_no_kickoff(mocks):
    mocks.sessions.return_value = ([MagicMock(session_id="old")], 1)

    assert await kickoff.start_expert_kickoff("user-1", "expert-1") is False

    mocks.create.assert_not_awaited()
    mocks.schedule.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("has_reply", [True, False])
async def test_kickoff_a_browser_already_sent_is_not_sent_again(mocks, has_reply):
    mocks.has_reply.return_value = has_reply

    assert await kickoff.start_expert_kickoff("user-1", "expert-1") is False

    mocks.schedule.assert_not_awaited()


@pytest.mark.asyncio
async def test_kickoff_yields_to_a_turn_already_in_the_session(mocks):
    mocks.schedule.side_effect = SessionNotIdle("session-1")

    assert await kickoff.start_expert_kickoff("user-1", "expert-1") is False

    mocks.enqueue.assert_not_awaited()


@pytest.mark.asyncio
async def test_kickoff_at_the_running_cap_is_queued(mocks):
    mocks.schedule.side_effect = ConcurrentTurnLimitError("at cap")

    assert await kickoff.start_expert_kickoff("user-1", "expert-1") is True

    queued = mocks.enqueue.await_args.kwargs
    assert queued["session_id"] == "session-1"
    assert queued["message"] == EXPERT_KICKOFF_PROMPT
    assert queued["message_id"] == expert_kickoff_message_id(
        "user-1", "session-1", "expert-1"
    )


@pytest.mark.asyncio
async def test_kickoff_failure_never_raises(mocks):
    mocks.create.side_effect = RuntimeError("database down")

    assert await kickoff.start_expert_kickoff("user-1", "expert-1") is False
