"""A follow-up sent as a turn ends is either answered or refused, never kept.

The engine looks at the chat's queue one last time when its answer is done and
continues with anything it finds there. The turn is marked done a little
later. A follow-up the queue accepts between the two has no turn left to run
it, so it sits there unanswered until the user sends something else. These
tests run the real SDK engine over a scripted client, with its queue drains and
the stream registry on a fake Redis that runs the real Lua scripts.
"""

import contextlib
import uuid
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, patch

import pytest
from fakeredis.aioredis import FakeRedis
from fastapi import HTTPException

from backend.copilot import pending_message_helpers, pending_messages, stream_registry
from backend.copilot.model import ChatSession
from backend.copilot.pending_message_helpers import queue_pending_for_http
from backend.copilot.pending_messages import peek_pending_messages
from backend.copilot.sdk.service import stream_chat_completion_sdk

from .conftest import build_test_transcript
from .retry_scenarios_test import TestStreamChatCompletionRetryIntegration as _Turns
from .retry_scenarios_test import _make_sdk_patches

_SVC = "backend.copilot.sdk.service"
_turns = _Turns()
FOLLOW_UP = "What about tomorrow?"


@pytest.fixture
async def redis() -> AsyncIterator[FakeRedis]:
    async with FakeRedis(decode_responses=True) as fake:
        with contextlib.ExitStack() as stack:
            for module in (pending_messages, pending_message_helpers, stream_registry):
                stack.enter_context(
                    patch.object(
                        module, "get_redis_async", AsyncMock(return_value=fake)
                    )
                )
            chat_db = stack.enter_context(patch.object(stream_registry, "chat_db"))
            chat_db.return_value.set_turn_duration = AsyncMock()
            yield fake


async def _send_follow_up(session: ChatSession) -> int:
    """What ``POST /messages/pending`` answers: 200, or the refusal's status."""
    try:
        await queue_pending_for_http(
            session_id=session.session_id,
            user_id=session.user_id,
            message=FOLLOW_UP,
            context=None,
            file_ids=None,
            folder_ids=None,
            expert_id=None,
            client_message_id=str(uuid.uuid4()),
        )
    except HTTPException as refusal:
        return refusal.status_code
    return 200


async def _run_turn(session: ChatSession, queries: list[str], during_answer=None):
    """Run one turn to its end. ``queries`` collects what each SDK client was
    asked; ``during_answer`` runs while the first answer streams."""

    def client_factory(*_, **__):
        client = _turns._make_client_mock(result_message=_turns._make_result_message())
        inner = client.__aenter__.return_value
        receive = inner.receive_response

        async def record_query(prompt, *_, **__):
            queries.append(prompt if isinstance(prompt, str) else repr(prompt))

        async def answer():
            if during_answer is not None and len(queries) == 1:
                await during_answer()
            async for message in receive():
                yield message

        inner.query = AsyncMock(side_effect=record_query)
        inner.receive_response = answer
        return client

    patches = _make_sdk_patches(
        session,
        original_transcript=build_test_transcript(
            [("user", "prior question"), ("assistant", "prior answer")]
        ),
        compacted_transcript=None,
        client_side_effect=client_factory,
    )
    with contextlib.ExitStack() as stack:
        for target, kwargs in patches:
            stack.enter_context(patch(target, **kwargs))
        for name, value in (
            ("inject_user_context", None),
            ("build_session_context", ""),
            ("first_message_expert_context", ""),
            ("build_matched_skills_notice", ""),
            # The continuation's rows are stored before it runs.
            ("persist_pending_as_user_rows", True),
        ):
            stack.enter_context(
                patch(f"{_SVC}.{name}", new=AsyncMock(return_value=value))
            )
        # The real drains, over the fake Redis: the race is in what they do.
        stack.enter_context(
            patch(
                f"{_SVC}.drain_pending_safe",
                new=pending_message_helpers.drain_pending_safe,
            )
        )
        async for _ in stream_chat_completion_sdk(
            session_id=session.session_id,
            message="hello",
            is_user_message=True,
            user_id=session.user_id,
            session=session,
        ):
            pass


def _session() -> ChatSession:
    session = _turns._make_session()
    # A titled chat skips title generation, a separate model call.
    session.title = "Follow-up"
    return session


@pytest.mark.asyncio
async def test_a_follow_up_after_the_last_look_is_refused_not_kept(redis: FakeRedis):
    session = _session()
    await stream_registry.create_session(
        session.session_id, None, "chat_stream", "chat", turn_id="turn-1"
    )
    queries: list[str] = []

    await _run_turn(session, queries)
    # The answer is done and the engine has looked at the queue for the last
    # time; the executor has not marked the turn done yet.
    sent = await _send_follow_up(session)
    await stream_registry.mark_session_completed(session.session_id, turn_id="turn-1")

    # A refused follow-up goes out again as a turn of its own (the client's
    # 409 fallback); an accepted one here would never be answered.
    assert (sent, await peek_pending_messages(session.session_id)) == (409, [])
    assert len(queries) == 1


@pytest.mark.asyncio
async def test_a_follow_up_sent_during_the_answer_is_answered_by_the_turn(
    redis: FakeRedis,
):
    session = _session()
    await stream_registry.create_session(
        session.session_id, None, "chat_stream", "chat", turn_id="turn-1"
    )
    queries: list[str] = []
    during: list[int] = []

    async def follow_up_during_answer():
        during.append(await _send_follow_up(session))

    await _run_turn(session, queries, during_answer=follow_up_during_answer)
    late = await _send_follow_up(session)
    await stream_registry.mark_session_completed(session.session_id, turn_id="turn-1")

    assert during == [200]
    assert len(queries) == 2 and FOLLOW_UP in queries[1]
    # The continuation looked last; what came after it is refused, not kept.
    assert (late, await peek_pending_messages(session.session_id)) == (409, [])
