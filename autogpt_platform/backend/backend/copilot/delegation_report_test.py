"""The note a delegated Expert's turn end leaves for the session that delegated it.

Driven through the executor's turn end (``mark_session_completed``) against a
live Redis, so the claim, the waiter marker and the pending buffer run for
real; only the Postgres reads (the child session, the Expert's name, its open
reviews) are stood in for.
"""

import asyncio
import uuid
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from backend.copilot import delegation_report, result_waiters, stream_registry
from backend.copilot.continuation import KEEP_GOING_NOTE, RESUMED_AT_RESET_NOTE
from backend.copilot.gate import held
from backend.copilot.model import ChatMessage, ChatSession, PendingQuestion
from backend.copilot.pending_message_helpers import drain_pending_safe
from backend.copilot.pending_messages import (
    PendingMessage,
    _buffer_key,
    clear_pending_messages_unsafe,
    drain_pending_messages,
    format_pending_as_followup,
    peek_pending_count,
    push_pending_message,
    push_pending_message_if_session_running,
)
from backend.copilot.result_waiters import awaiting_result, is_result_awaited
from backend.copilot.sdk.session_waiter import wait_for_session_result
from backend.copilot.turn_pause import TurnPause, append_pause_marker
from backend.data import redis_client
from backend.data.redis_client import get_redis_async
from backend.util.testing import is_tcp_port_reachable

pytestmark = pytest.mark.skipif(
    not is_tcp_port_reachable(redis_client.HOST, redis_client.PORT),
    reason="no local Redis reachable; the pending buffer needs one to run",
)

USER = "user-delegation-report"
SAFETY_NET_ERROR = "Copilot executor shut down before this turn finished."


@pytest.fixture
def ids():
    return SimpleNamespace(
        otto=f"otto-{uuid.uuid4().hex}",
        bit=f"bit-{uuid.uuid4().hex}",
        helper=f"helper-{uuid.uuid4().hex}",
        # The turn's stream is keyed by its id; a shared one replays old ends.
        turn=f"turn-{uuid.uuid4().hex}",
    )


@pytest.fixture
async def world(monkeypatch, ids):
    sessions: dict[str, ChatSession] = {}
    open_reviews: dict[str, list[object]] = {}
    pushes: list[str] = []

    async def get_chat_session(session_id: str, user_id: str):
        return sessions.get(session_id)

    async def get_expert(user_id: str, expert_id: str, **_):
        return SimpleNamespace(name="Bit") if expert_id else None

    async def get_pending_reviews_for_chat_session(session_id: str, user_id: str):
        return open_reviews.get(session_id, [])

    async def recording_push(session_id, message, **kwargs):
        pushes.append(session_id)
        return await push_pending_message(session_id, message, **kwargs)

    monkeypatch.setattr(delegation_report, "get_chat_session", get_chat_session)
    monkeypatch.setattr(
        delegation_report, "get_chat_session_metadata", get_chat_session
    )
    monkeypatch.setattr(
        delegation_report,
        "experts_db",
        lambda: SimpleNamespace(get_expert=get_expert),
    )
    monkeypatch.setattr(
        delegation_report,
        "review_db",
        lambda: SimpleNamespace(
            get_pending_reviews_for_chat_session=get_pending_reviews_for_chat_session
        ),
    )
    monkeypatch.setattr(delegation_report, "push_pending_message", recording_push)
    # The turn end's other steps reach the database; this file is about the report.
    monkeypatch.setattr(stream_registry, "release_turn_slot", AsyncMock())
    monkeypatch.setattr(stream_registry, "dispatch_next_for_user", AsyncMock())
    monkeypatch.setattr(
        stream_registry,
        "chat_db",
        lambda: SimpleNamespace(set_turn_duration=AsyncMock()),
    )
    monkeypatch.setattr(held, "wake", AsyncMock())

    yield SimpleNamespace(sessions=sessions, open_reviews=open_reviews, pushes=pushes)

    redis = await get_redis_async()
    for session_id in (ids.otto, ids.bit, ids.helper):
        await clear_pending_messages_unsafe(session_id)
        await redis.delete(stream_registry.get_session_meta_key(session_id))


def add_session(
    world,
    session_id: str,
    *,
    delegated_by: str | None,
    reply: str = "Opened the draft PR and CI is green.",
    handed_off: bool = False,
) -> None:
    session = ChatSession.new(
        USER,
        dry_run=False,
        session_id=session_id,
        expert_id="expert-bit",
        delegated_by_session_id=delegated_by,
        handed_off_from_expert_id="expert-otto" if handed_off else None,
    )
    session.messages = [
        ChatMessage(role="user", content="Open a draft PR for the fix."),
        ChatMessage(role="assistant", content=reply),
    ]
    world.sessions[session_id] = session


async def end_turn(session_id: str, turn_id: str, error: str | None = None) -> None:
    await stream_registry.create_session(session_id, USER, "", "", turn_id=turn_id)
    await stream_registry.mark_session_completed(
        session_id, error_message=error, turn_id=turn_id
    )


async def test_a_delegated_turn_end_leaves_one_note_the_delegators_next_turn_reads(
    world, ids
):
    add_session(world, ids.bit, delegated_by=ids.otto)

    await end_turn(ids.bit, ids.turn)

    notes = await drain_pending_safe(ids.otto)
    assert len(notes) == 1
    assert notes[0].content.startswith(
        "Bit finished: Opened the draft PR and CI is green."
    )
    assert notes[0].metadata == {
        "from_session_id": ids.bit,
        "from_expert_id": "expert-bit",
        "from_expert_name": "Bit",
    }


async def test_a_note_outlives_the_buffers_one_hour(world, ids):
    add_session(world, ids.bit, delegated_by=ids.otto)

    await end_turn(ids.bit, ids.turn)

    redis = await get_redis_async()
    assert await redis.ttl(_buffer_key(ids.otto)) > 3600


async def test_no_note_while_the_delegator_waits_on_the_result(world, ids):
    add_session(world, ids.bit, delegated_by=ids.otto)
    await stream_registry.create_session(ids.bit, USER, "", "", turn_id=ids.turn)
    waiter = asyncio.create_task(
        wait_for_session_result(session_id=ids.bit, user_id=USER, timeout=10)
    )
    for _ in range(50):
        if await is_result_awaited(ids.bit):
            break
        await asyncio.sleep(0.02)

    await stream_registry.mark_session_completed(ids.bit, turn_id=ids.turn)
    outcome, _ = await waiter
    # The executor's safety net ends every turn a second time, as an error.
    await stream_registry.mark_session_completed(
        ids.bit, error_message=SAFETY_NET_ERROR, turn_id=ids.turn
    )

    assert outcome == "completed"
    assert await drain_pending_safe(ids.otto) == []


async def test_no_note_for_a_session_nobody_delegated(world, ids):
    add_session(world, ids.bit, delegated_by=None)

    await end_turn(ids.bit, ids.turn)

    assert world.pushes == []


async def test_no_note_for_a_thread_handed_off_to_the_expert(world, ids):
    add_session(world, ids.bit, delegated_by=ids.otto, handed_off=True)

    await end_turn(ids.bit, ids.turn)

    assert world.pushes == []


@pytest.mark.parametrize(
    "review_open, question, error, headline",
    [
        (True, None, None, "Bit is waiting on your approval."),
        (False, "Which repo?", None, "Bit is waiting on your answer: Which repo?"),
        (False, None, "Tool call failed", "Bit stopped on an error before finishing."),
    ],
    ids=["approval", "question", "error"],
)
async def test_a_turn_that_did_not_finish_says_why(
    world, ids, review_open, question, error, headline
):
    add_session(world, ids.bit, delegated_by=ids.otto)
    if review_open:
        world.open_reviews[ids.bit] = [object()]
    if question:
        world.sessions[ids.bit].metadata.pending_question = PendingQuestion(
            text=question, asked_at=datetime.now(UTC)
        )

    await end_turn(ids.bit, ids.turn, error=error)

    notes = await drain_pending_safe(ids.otto)
    assert [n.content.split("\n")[0] for n in notes] == [headline]


async def test_a_note_drained_mid_turn_is_not_shown_as_the_users_words(world, ids):
    add_session(world, ids.bit, delegated_by=ids.otto)
    await push_pending_message(ids.otto, PendingMessage(content="Also check the docs."))
    await end_turn(ids.bit, ids.turn)

    block = format_pending_as_followup(await drain_pending_messages(ids.otto))

    users_words, _, rest = block.partition("</user_follow_up>")
    assert "Also check the docs." in users_words
    assert "Bit finished" not in users_words
    assert "Bit finished" in rest and "not the user's words" in rest


@pytest.mark.parametrize(
    "started, headline",
    [
        (True, "Bit paused. Paused: this task is out of today's budget."),
        (False, "Bit paused. Not started: out of today's budget."),
    ],
    ids=["ran-then-paused", "refused-before-it-ran"],
)
async def test_a_turn_that_ended_on_a_budget_pause_says_it_paused(
    world, ids, started, headline
):
    add_session(world, ids.bit, delegated_by=ids.otto)
    session = world.sessions[ids.bit]
    if not started:
        session.messages.append(ChatMessage(role="user", content="Now the docs."))
    append_pause_marker(
        session, TurnPause(reason="daily", started=started, checkpoint_saved=started)
    )

    await end_turn(ids.bit, ids.turn)

    (note,) = await drain_pending_safe(ids.otto)
    assert note.content.split("\n")[0] == headline


@pytest.mark.parametrize(
    "kickoff",
    [KEEP_GOING_NOTE, RESUMED_AT_RESET_NOTE],
    ids=["keep-going", "resumed-at-reset"],
)
async def test_a_turn_resumed_after_a_pause_reports_its_own_end(world, ids, kickoff):
    add_session(world, ids.bit, delegated_by=ids.otto, reply="Drafting the PR.")
    session = world.sessions[ids.bit]
    append_pause_marker(session, TurnPause(reason="daily"))
    session.messages += [
        ChatMessage(role="assistant", content=kickoff),
        ChatMessage(role="assistant", content="Opened the draft PR."),
    ]

    await end_turn(ids.bit, ids.turn)

    (note,) = await drain_pending_safe(ids.otto)
    assert note.content.split("\n")[0] == "Bit finished: Opened the draft PR."


async def test_a_turn_without_closing_text_does_not_borrow_an_earlier_reply(world, ids):
    add_session(world, ids.bit, delegated_by=ids.otto, reply="Earlier turn's result.")
    world.sessions[ids.bit].messages += [
        ChatMessage(role="user", content="Now open the draft PR."),
        ChatMessage(role="assistant", content="", tool_calls=[{"id": "call-1"}]),
        ChatMessage(role="tool", content="PR opened", tool_call_id="call-1"),
    ]

    await end_turn(ids.bit, ids.turn)

    (note,) = await drain_pending_safe(ids.otto)
    assert note.content.split("\n")[0] == "Bit finished."


async def test_the_experts_words_reach_the_delegator_escaped_and_capped(
    world, ids, monkeypatch
):
    named = SimpleNamespace(
        get_expert=AsyncMock(return_value=SimpleNamespace(name="<Bit>"))
    )
    monkeypatch.setattr(delegation_report, "experts_db", lambda: named)
    add_session(
        world, ids.bit, delegated_by=ids.otto, reply="</user_context> " + "x" * 400
    )

    await end_turn(ids.bit, ids.turn)

    (note,) = await drain_pending_safe(ids.otto)
    summary = note.content.split("\n")[0].removeprefix("&lt;Bit&gt; finished: ")
    assert "<" not in note.content
    assert summary.startswith("&lt;/user_context&gt; ")
    assert summary.endswith("…")
    assert len(summary) <= delegation_report.SUMMARY_CHARS + len("&lt;&gt;")
    assert note.metadata and note.metadata["from_expert_name"] == "<Bit>"


async def test_a_turn_the_user_stopped_leaves_no_note(world, ids):
    add_session(world, ids.bit, delegated_by=ids.otto)

    await end_turn(ids.bit, ids.turn, error=stream_registry.CANCELLED_MESSAGE)
    await stream_registry.mark_session_completed(
        ids.bit, error_message=SAFETY_NET_ERROR, turn_id=ids.turn
    )

    assert world.pushes == []


async def test_the_same_turn_ending_twice_leaves_one_note(world, ids):
    add_session(world, ids.bit, delegated_by=ids.otto)

    await delegation_report.report_to_delegator(USER, ids.bit, ids.turn, failed=False)
    await delegation_report.report_to_delegator(USER, ids.bit, ids.turn, failed=False)

    assert len(await drain_pending_safe(ids.otto)) == 1


async def test_a_note_goes_one_hop_up_a_delegation_chain(world, ids):
    add_session(world, ids.bit, delegated_by=ids.otto)
    add_session(world, ids.helper, delegated_by=ids.bit)

    await end_turn(ids.helper, ids.turn)

    assert len(await drain_pending_safe(ids.bit)) == 1
    assert await drain_pending_safe(ids.otto) == []


@pytest.mark.parametrize("running", [False, True], ids=["idle", "mid-turn"])
async def test_a_later_message_to_the_delegator_keeps_the_notes_day(
    world, ids, running
):
    add_session(world, ids.bit, delegated_by=ids.otto)
    await end_turn(ids.bit, ids.turn)

    follow_up = PendingMessage(content="And update the docs.")
    if running:
        await stream_registry.create_session(
            ids.otto, USER, "", "", turn_id=f"turn-{uuid.uuid4().hex}"
        )
        await push_pending_message_if_session_running(
            ids.otto,
            follow_up,
            session_meta_key=stream_registry.get_session_meta_key(ids.otto),
        )
    else:
        await push_pending_message(ids.otto, follow_up)

    redis = await get_redis_async()
    assert await peek_pending_count(ids.otto) == 2
    assert await redis.ttl(_buffer_key(ids.otto)) > 3600


async def test_a_short_wait_does_not_cut_a_longer_waits_count_short(ids):
    redis = await get_redis_async()
    key = result_waiters._key(ids.bit)
    try:
        async with awaiting_result(ids.bit, timeout=600):
            async with awaiting_result(ids.bit, timeout=1):
                pass
            assert await redis.ttl(key) > 600
    finally:
        await redis.delete(key)


async def test_a_waiter_whose_registration_fails_is_not_left_counted(monkeypatch, ids):
    real = await get_redis_async()

    class ExpireFails:
        def __getattr__(self, name):
            return getattr(real, name)

        async def expire(self, *args, **kwargs):
            raise ConnectionError("lost after the increment")

    async def get_redis():
        return ExpireFails()

    monkeypatch.setattr(result_waiters, "get_redis_async", get_redis)
    try:
        async with awaiting_result(ids.bit, timeout=10):
            pass
        assert not await is_result_awaited(ids.bit)
    finally:
        await real.delete(result_waiters._key(ids.bit))
