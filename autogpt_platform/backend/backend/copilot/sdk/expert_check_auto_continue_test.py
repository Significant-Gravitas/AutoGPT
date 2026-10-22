"""An auto-continue reads its expert again instead of reusing the turn's check:
the user may have archived the expert while the turn ran."""

import contextlib
from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.expert_check import (
    checked_expert,
    remember_expert_check,
    sharing_expert_checks,
)
from backend.copilot.expert_check_test import CHECK
from backend.copilot.pending_messages import PendingMessage
from backend.copilot.sdk.service import stream_chat_completion_sdk

from .conftest import build_test_transcript
from .retry_scenarios_test import TestStreamChatCompletionRetryIntegration as _Turns
from .retry_scenarios_test import _make_sdk_patches

_SVC = "backend.copilot.sdk.service"
_turns = _Turns()


@pytest.mark.asyncio
async def test_an_auto_continue_reads_its_expert_again():
    session = _turns._make_session()
    session.expert_id = "expert-1"
    session.organization_id = CHECK.organization_id
    session.team_id = CHECK.team_id
    # A titled chat skips title generation, a separate model call.
    session.title = "Auto-continue"
    seen = []

    async def identity(user_id, expert_id, **_):
        seen.append(checked_expert(user_id, expert_id))
        return ""

    # The turn-start drain, the drain that starts an auto-continue, and the
    # continuation's own two.
    drains = iter([[], [PendingMessage(content="one more thing")], [], []])
    patches = _make_sdk_patches(
        session,
        original_transcript=build_test_transcript(
            [("user", "prior question"), ("assistant", "prior answer")]
        ),
        compacted_transcript=None,
        client_side_effect=lambda *_, **__: _turns._make_client_mock(
            result_message=_turns._make_result_message()
        ),
    )
    with contextlib.ExitStack() as stack, sharing_expert_checks():
        for target, kwargs in patches:
            stack.enter_context(patch(target, **kwargs))
        stack.enter_context(patch(f"{_SVC}.build_expert_identity_suffix", new=identity))
        # The first message's context and the queued message's save need a
        # database; neither is under test.
        stack.enter_context(
            patch(f"{_SVC}.inject_user_context", new=AsyncMock(return_value=None))
        )
        stack.enter_context(
            patch(
                "backend.copilot.pending_message_helpers.upsert_chat_session",
                new=AsyncMock(side_effect=lambda s: s),
            )
        )
        stack.enter_context(
            patch(
                f"{_SVC}.drain_pending_safe",
                new=AsyncMock(side_effect=lambda *_: next(drains)),
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

    assert seen == [CHECK, None]
