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
from backend.copilot.pending_messages import PendingMessage
from backend.copilot.sdk.service import stream_chat_completion_sdk

from .conftest import build_test_transcript
from .retry_scenarios_test import TestStreamChatCompletionRetryIntegration as _Turns
from .retry_scenarios_test import _make_sdk_patches

_SVC = "backend.copilot.sdk.service"
_turns = _Turns()


@pytest.mark.asyncio
async def test_an_auto_continue_rechecks_its_expert_and_requeues_on_refusal():
    session = _turns._make_session()
    session.expert_id = "expert-1"
    session.organization_id = CHECK.organization_id
    session.team_id = CHECK.team_id
    # A titled chat skips title generation, a separate model call.
    session.title = "Auto-continue"
    seen = []
    client_calls = []
    pending = PendingMessage(content="one more thing")

    async def identity(user_id, expert_id, **_):
        current = checked_expert(user_id, expert_id)
        seen.append(current)
        if current is None:
            raise ExpertSessionUnavailableError("expert unavailable")
        return ""

    def client_factory(*_, **__):
        client_calls.append("called")
        return _turns._make_client_mock(result_message=_turns._make_result_message())

    drains = iter([[], [pending]])
    requeue = AsyncMock(return_value=1)
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
        stack.enter_context(
            patch(f"{_SVC}.inject_user_context", new=AsyncMock(return_value=None))
        )
        stack.enter_context(
            patch(f"{_SVC}.build_session_context", new=AsyncMock(return_value=""))
        )
        stack.enter_context(
            patch(
                f"{_SVC}.build_matched_skills_notice",
                new=AsyncMock(return_value=""),
            )
        )
        stack.enter_context(patch(f"{_SVC}.push_pending_message", new=requeue))
        stack.enter_context(
            patch(
                f"{_SVC}.drain_pending_safe",
                new=AsyncMock(side_effect=lambda *_: next(drains)),
            )
        )
        remember_expert_check(session.user_id, "expert-1", CHECK)
        with pytest.raises(ExpertSessionUnavailableError, match="expert unavailable"):
            async for _ in stream_chat_completion_sdk(
                session_id=session.session_id,
                message="hello",
                is_user_message=True,
                user_id=session.user_id,
                session=session,
            ):
                pass

    assert seen == [CHECK, None]
    assert client_calls == ["called"]
    requeue.assert_awaited_once_with(session.session_id, pending)
