"""A turn that answers a hidden row (a kickoff the server wrote) is not
remembered or learned from as the user's words, in either engine."""

import asyncio
import contextlib
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.baseline import service as baseline
from backend.copilot.baseline.service_cancellation_test import baseline_io
from backend.copilot.model import ChatMessage, ChatSession, ChatSessionMetadata
from backend.copilot.sdk import retry_scenarios_test as scenarios
from backend.copilot.sdk.conftest import build_test_transcript
from backend.copilot.sdk.service import stream_chat_completion_sdk

__all__ = ["baseline_io"]

_SDK = "backend.copilot.sdk.service"
_turns = scenarios.TestStreamChatCompletionRetryIntegration()
KICKOFF = "Introduce yourself and start."


def _session(*, hidden: bool) -> ChatSession:
    """An ordinary chat whose latest row, the one the turn answers, is the
    server's hidden kickoff or a message the user typed."""
    return ChatSession(
        session_id="sess-hidden",
        user_id="user-1",
        usage=[],
        started_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        title="Chat",
        messages=[
            ChatMessage(role="user", content="hi"),
            ChatMessage(role="assistant", content="Hello."),
            ChatMessage(
                role="user",
                content=KICKOFF,
                metadata={"hidden": True} if hidden else None,
            ),
        ],
        metadata=ChatSessionMetadata(origin="interactive"),
    )


async def _baseline_turn(
    monkeypatch: pytest.MonkeyPatch, session: ChatSession
) -> tuple[AsyncMock, AsyncMock]:
    stream = MagicMock()
    stream.__aiter__.return_value = []
    stream.close = AsyncMock()
    remember = AsyncMock()
    learn = AsyncMock()
    monkeypatch.setattr(
        baseline, "call_provider_stream", AsyncMock(return_value=stream)
    )
    monkeypatch.setattr(baseline, "download_transcript", AsyncMock(return_value=None))
    monkeypatch.setattr(baseline, "clear_pending_question", AsyncMock())
    monkeypatch.setattr(
        baseline, "build_skills_update_notice", AsyncMock(return_value="")
    )
    monkeypatch.setattr(baseline, "_enqueue_graphiti_turn", remember)
    monkeypatch.setattr(baseline, "capture_chat_turn", learn)
    async with asyncio.timeout(10):
        async for _ in baseline.stream_chat_completion_baseline(
            session.session_id,
            message=KICKOFF,
            user_id=session.user_id,
            session=session,
            is_user_message=True,
        ):
            pass
    return remember, learn


async def _sdk_turn(session: ChatSession) -> tuple[AsyncMock, AsyncMock]:
    learn = AsyncMock()
    patches = scenarios._make_sdk_patches(
        session,
        original_transcript=build_test_transcript(
            [("user", "hi"), ("assistant", "Hello.")]
        ),
        compacted_transcript=None,
        client_side_effect=lambda *_a, **_k: _turns._make_client_mock(
            result_message=_turns._make_result_message()
        ),
    )
    with contextlib.ExitStack() as stack:
        mocks = {
            target: stack.enter_context(patch(target, **kwargs))
            for target, kwargs in patches
        }
        for target in (
            "_update_title_async",
            "clear_pending_question",
            "active_mode",
            "build_session_context",
            "build_turn_budget_block",
            "persist_and_record_usage",
            "build_builder_system_prompt_suffix",
        ):
            stack.enter_context(
                patch(f"{_SDK}.{target}", new_callable=AsyncMock, return_value="")
            )
        stack.enter_context(patch(f"{_SDK}.capture_chat_turn", new=learn))
        async with asyncio.timeout(30):
            async for _ in stream_chat_completion_sdk(
                session_id=session.session_id,
                message=KICKOFF,
                is_user_message=True,
                user_id=session.user_id,
                session=session,
            ):
                pass
    return mocks[f"{_SDK}._enqueue_graphiti_turn"], learn


def _assert_memory(remember: AsyncMock, learn: AsyncMock, *, hidden: bool) -> None:
    if hidden:
        remember.assert_not_called()
        learn.assert_not_called()
        return
    # What memory is handed is the user's own message.
    assert remember.call_args.args[3] == KICKOFF
    learn.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("hidden", [True, False])
async def test_the_baseline_engine_keeps_hidden_rows_out_of_memory(
    monkeypatch: pytest.MonkeyPatch, baseline_io: list[ChatSession], hidden: bool
) -> None:
    remember, learn = await _baseline_turn(monkeypatch, _session(hidden=hidden))

    _assert_memory(remember, learn, hidden=hidden)


@pytest.mark.asyncio
@pytest.mark.parametrize("hidden", [True, False])
async def test_the_sdk_engine_keeps_hidden_rows_out_of_memory(hidden: bool) -> None:
    remember, learn = await _sdk_turn(_session(hidden=hidden))

    _assert_memory(remember, learn, hidden=hidden)
