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
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from claude_agent_sdk import ResultMessage
from fakeredis.aioredis import FakeRedis
from fastapi import HTTPException

from backend.copilot import pending_message_helpers, pending_messages, stream_registry
from backend.copilot.context import set_current_turn_id
from backend.copilot.model import ChatMessage, ChatSession
from backend.copilot.pending_message_helpers import queue_pending_for_http
from backend.copilot.pending_messages import (
    drain_pending_or_close,
    peek_pending_messages,
)
from backend.copilot.sdk.service import stream_chat_completion_sdk

from .conftest import build_test_transcript
from .retry_scenarios_test import _make_sdk_patches

_SVC = "backend.copilot.sdk.service"
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


async def _run_turn(
    session: ChatSession,
    queries: list[str],
    during_answer=None,
    *,
    turn_id: str = "turn-1",
):
    """Run turn *turn_id* to its end, as the executor would. ``queries``
    collects what each SDK client was asked; ``during_answer`` runs while
    the first answer streams."""

    def client_factory(*_, **__):
        async def record_query(prompt, *_, **__):
            queries.append(prompt if isinstance(prompt, str) else repr(prompt))

        async def answer():
            if during_answer is not None and len(queries) == 1:
                await during_answer()
            yield ResultMessage(
                subtype="success",
                result="done",
                duration_ms=100,
                duration_api_ms=50,
                is_error=False,
                num_turns=1,
                session_id=session.session_id,
            )

        client = MagicMock(receive_response=answer)
        client.query = AsyncMock(side_effect=record_query)
        client._transport.write = AsyncMock()
        connection = AsyncMock()
        connection.__aenter__.return_value = client
        connection.__aexit__.return_value = None
        return connection

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
        # The executor names the turn before the engine runs.
        set_current_turn_id(turn_id)
        try:
            async for _ in stream_chat_completion_sdk(
                session_id=session.session_id,
                message="hello",
                is_user_message=True,
                user_id=session.user_id,
                session=session,
            ):
                pass
        finally:
            set_current_turn_id(None)


def _session() -> ChatSession:
    return ChatSession(
        session_id="test-session-id",
        user_id="test-user",
        usage=[],
        started_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        messages=[ChatMessage(role="user", content="hello")],
        # A titled chat skips title generation, a separate model call.
        title="Follow-up",
    )


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


@pytest.mark.asyncio
async def test_a_rerun_of_the_turn_reopens_the_queue_its_dead_attempt_closed(
    redis: FakeRedis,
):
    """RabbitMQ hands a turn whose pod died out again under the same turn id,
    and nothing registers it anew. If the dead attempt had already taken its
    last look, its marker would refuse every follow-up sent during the rerun."""
    session = _session()
    await stream_registry.create_session(
        session.session_id, None, "chat_stream", "chat", turn_id="turn-1"
    )
    await drain_pending_or_close(
        session.session_id,
        session_meta_key=stream_registry.get_session_meta_key(session.session_id),
        turn_id="turn-1",
    )
    queries: list[str] = []
    during: list[int] = []

    async def follow_up_during_answer():
        during.append(await _send_follow_up(session))

    await _run_turn(session, queries, during_answer=follow_up_during_answer)

    assert during == [200]
    assert len(queries) == 2 and FOLLOW_UP in queries[1]


@pytest.mark.asyncio
async def test_a_turn_that_outlived_its_registration_leaves_the_next_ones_queue_open(
    redis: FakeRedis,
):
    """A turn force-completed after its stop-confirm timeout can still be
    running when the chat's next turn registers; its last look must not shut
    the queue the next turn is reading."""
    session = _session()
    await stream_registry.create_session(
        session.session_id, None, "chat_stream", "chat", turn_id="turn-1"
    )
    queries: list[str] = []

    async def next_turn_registers():
        await stream_registry.create_session(
            session.session_id, None, "chat_stream", "chat", turn_id="turn-2"
        )

    await _run_turn(session, queries, during_answer=next_turn_registers)

    assert await _send_follow_up(session) == 200
    assert [m.content for m in await peek_pending_messages(session.session_id)] == [
        FOLLOW_UP
    ]
