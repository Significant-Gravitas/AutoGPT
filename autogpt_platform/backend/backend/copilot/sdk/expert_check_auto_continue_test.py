"""Auto-continue rechecks its expert and preserves messages on refusal."""

import contextlib
from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.expert_check import (
    checked_expert,
    remember_expert_check,
    sharing_expert_checks,
)
from backend.copilot.expert_check_test import CHECK
from backend.copilot.expert_context import ExpertSessionUnavailableError
from backend.copilot.model import ChatSession
from backend.copilot.pending_messages import PendingMessage
from backend.copilot.sdk.service import stream_chat_completion_sdk

from .conftest import build_test_transcript
from .retry_scenarios_test import TestStreamChatCompletionRetryIntegration as _Turns
from .retry_scenarios_test import _make_sdk_patches

_SVC = "backend.copilot.sdk.service"
_turns = _Turns()


def _expert_session() -> ChatSession:
    session = _turns._make_session()
    session.expert_id = "expert-1"
    session.organization_id = CHECK.organization_id
    session.team_id = CHECK.team_id
    # A titled chat skips title generation, a separate model call.
    session.title = "Auto-continue"
    return session


async def _turn_that_auto_continues(
    session: ChatSession,
    identity,
    pending: PendingMessage,
    *,
    requeue: AsyncMock,
    client_calls: list[str],
) -> None:
    """Run a turn, inside the share its executor opened and holding the check
    it made, whose end finds ``pending`` queued. ``client_calls`` gets one
    entry per SDK client the turn and its continuation open."""

    def client_factory(*_, **__):
        client_calls.append("called")
        return _turns._make_client_mock(result_message=_turns._make_result_message())

    # The turn's start finds nothing queued and its end finds the follow-up;
    # the continuation's own drains find nothing more.
    drains = iter([[], [pending]])
    patches = _make_sdk_patches(
        session,
        original_transcript=build_test_transcript(
            [("user", "prior question"), ("assistant", "prior answer")]
        ),
        compacted_transcript=None,
        client_side_effect=client_factory,
    )
    with contextlib.ExitStack() as stack, sharing_expert_checks():
        for target, kwargs in patches:
            stack.enter_context(patch(target, **kwargs))
        stack.enter_context(patch(f"{_SVC}.build_expert_identity_suffix", new=identity))
        for name, value in (
            ("inject_user_context", None),
            ("build_session_context", ""),
            # A first turn's expert and team block reads other services.
            ("first_message_expert_context", ""),
            ("build_matched_skills_notice", ""),
            # The continuation's rows are stored before it runs.
            ("persist_pending_as_user_rows", True),
        ):
            stack.enter_context(
                patch(f"{_SVC}.{name}", new=AsyncMock(return_value=value))
            )
        stack.enter_context(patch(f"{_SVC}.push_pending_message", new=requeue))
        stack.enter_context(
            patch(
                f"{_SVC}.drain_pending_safe",
                new=AsyncMock(side_effect=lambda *_: next(drains, [])),
            )
        )
        remember_expert_check(session.user_id, "expert-1", CHECK)
        async for _ in stream_chat_completion_sdk(
            session_id=session.session_id,
            message="hello",
            is_user_message=True,
            user_id=session.user_id,
            session=session,
        ):
            pass


@pytest.mark.asyncio
async def test_an_auto_continue_rechecks_its_expert_and_requeues_on_refusal():
    session = _expert_session()
    seen = []
    client_calls: list[str] = []
    pending = PendingMessage(content="one more thing")
    requeue = AsyncMock(return_value=1)

    async def identity(user_id, expert_id, **_):
        current = checked_expert(user_id, expert_id)
        seen.append(current)
        if current is None:
            raise ExpertSessionUnavailableError("expert unavailable")
        return ""

    with pytest.raises(ExpertSessionUnavailableError, match="expert unavailable"):
        await _turn_that_auto_continues(
            session, identity, pending, requeue=requeue, client_calls=client_calls
        )

    assert seen == [CHECK, None]
    assert client_calls == ["called"]
    requeue.assert_awaited_once_with(session.session_id, pending)


@pytest.mark.asyncio
async def test_an_auto_continue_whose_expert_still_checks_out_runs():
    session = _expert_session()
    seen = []
    client_calls: list[str] = []
    pending = PendingMessage(content="one more thing")
    requeue = AsyncMock(return_value=1)

    async def identity(user_id, expert_id, **_):
        # No remembered check means a fresh read, which finds the expert
        # unchanged here.
        seen.append(checked_expert(user_id, expert_id))
        return ""

    await _turn_that_auto_continues(
        session, identity, pending, requeue=requeue, client_calls=client_calls
    )

    assert seen == [CHECK, None]
    assert client_calls == ["called", "called"]
    requeue.assert_not_awaited()
