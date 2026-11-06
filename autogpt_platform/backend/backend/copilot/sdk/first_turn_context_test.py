"""First-turn replay and parallel context-read behavior."""

from __future__ import annotations

import asyncio
import gc
import inspect
import logging
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from backend.copilot.context_blocks import ContextBlock
from backend.copilot.model import ChatMessage, ChatSession
from backend.copilot.sdk.service import _build_query_message

from .first_turn_context import FirstTurnContext
from .retry_scenarios_test import TestFirstTurnWarmContextIsEphemeral as _FirstTurn
from .retry_scenarios_test import TestFollowUpWarmContextCallSite as _FollowUp


def _make_session(messages: list[ChatMessage]) -> ChatSession:
    now = datetime.now(UTC)
    return ChatSession(
        session_id="test-session",
        user_id="user-1",
        messages=messages,
        title="test",
        usage=[],
        started_at=now,
        updated_at=now,
    )


def _msgs(*pairs: tuple[str, str]) -> list[ChatMessage]:
    """Build messages with sequences 0..N-1 mirroring the production schema."""
    return [
        ChatMessage(role=r, content=c, sequence=i) for i, (r, c) in enumerate(pairs)
    ]


class TestFirstTurnWithoutACliSession:
    """The first SDK turn after N earlier turns that left no CLI session
    file.  ``_build_query_message`` must inject the complete prior session
    so the model has full context.
    """

    @pytest.mark.asyncio
    async def test_scenario_p_full_session_injected_on_the_first_turn(
        self, monkeypatch
    ):
        """Scenario P: the first SDK turn injects every earlier turn."""
        # 4 earlier messages (2 turns) followed by the first SDK turn.
        session = _make_session(
            _msgs(
                ("user", "earlier-q1"),
                ("assistant", "earlier-a1"),
                ("user", "earlier-q2"),
                ("assistant", "earlier-a2"),
                ("user", "sdk-q1"),  # current SDK turn
            )
        )

        async def _mock_compress(msgs, target_tokens=None):
            return msgs, False, None

        monkeypatch.setattr(
            "backend.copilot.sdk.service._compress_messages", _mock_compress
        )

        # transcript_msg_count=4: a transcript covers all 4 prior messages,
        # but use_resume=False (no CLI session to resume).
        result, stats = await _build_query_message(
            "sdk-q1",
            session,
            use_resume=False,
            transcript_msg_count=4,
            session_id="s",
        )

        # All earlier turns must appear — none of them can be silently dropped.
        assert "<conversation_history>" in result
        assert "earlier-q1" in result
        assert "earlier-a1" in result
        assert "earlier-q2" in result
        assert "earlier-a2" in result
        assert "Now, the user says:\nsdk-q1" in result
        assert stats is None

    @pytest.mark.asyncio
    async def test_scenario_r_single_earlier_turn_injected(self, monkeypatch):
        """Scenario R: even a single earlier turn is captured on the first turn."""
        session = _make_session(
            _msgs(
                ("user", "earlier-q1"),
                ("assistant", "earlier-a1"),
                ("user", "sdk-q1"),
            )
        )

        async def _mock_compress(msgs, target_tokens=None):
            return msgs, False, None

        monkeypatch.setattr(
            "backend.copilot.sdk.service._compress_messages", _mock_compress
        )

        result, _ = await _build_query_message(
            "sdk-q1",
            session,
            use_resume=False,
            transcript_msg_count=2,
            session_id="s",
        )

        assert "<conversation_history>" in result
        assert "earlier-q1" in result
        assert "earlier-a1" in result
        assert "Now, the user says:\nsdk-q1" in result

    @pytest.mark.asyncio
    async def test_scenario_q_the_next_turn_uses_resume(self):
        """Scenario Q: the turn after uses --resume.

        The first turn set session_id → the CLI wrote its session file →
        T2 restores the session → use_resume=True.  _build_query_message must
        return the bare message (--resume supplies context via native session).
        """
        # T2: 4 earlier messages + 1 SDK turn already recorded.
        session = _make_session(
            _msgs(
                ("user", "earlier-q1"),
                ("assistant", "earlier-a1"),
                ("user", "earlier-q2"),
                ("assistant", "earlier-a2"),
                ("user", "sdk-q1"),
                ("assistant", "sdk-a1"),
                ("user", "sdk-q2"),  # current SDK T2 message
            )
        )

        # transcript_msg_count=6 covers all prior messages → no gap.
        result, stats = await _build_query_message(
            "sdk-q2",
            session,
            use_resume=True,  # T2: --resume works after T1 set session_id
            transcript_msg_count=6,
            session_id="s",
        )

        # --resume has full context — bare message only.
        assert result == "sdk-q2"
        assert stats is None

    @pytest.mark.asyncio
    async def test_the_first_turn_compresses_every_earlier_turn(self, monkeypatch):
        """_compress_messages is called with ALL prior messages.

        There is exactly one compression call containing all 4 earlier messages
        — not just the 2 post-transcript-end messages.
        """
        session = _make_session(
            _msgs(
                ("user", "earlier-q1"),
                ("assistant", "earlier-a1"),
                ("user", "earlier-q2"),
                ("assistant", "earlier-a2"),
                ("user", "sdk-q1"),
            )
        )
        compressed_batches: list[list] = []

        async def _mock_compress(msgs, target_tokens=None):
            compressed_batches.append(list(msgs))
            return msgs, False, None

        monkeypatch.setattr(
            "backend.copilot.sdk.service._compress_messages", _mock_compress
        )

        await _build_query_message(
            "sdk-q1",
            session,
            use_resume=False,
            transcript_msg_count=4,
            session_id="s",
        )

        # Exactly one compression call, with all 4 prior messages.
        assert len(compressed_batches) == 1
        assert len(compressed_batches[0]) == 4


_SVC = "backend.copilot.sdk.service"

_EXPERT_CONTEXT = f"{_SVC}.build_expert_context"

_SESSION_BLOCK = "session_id: test-session-id; pending_followups: 1"

_SKILLS_INDEX = "- name: weekly-report — writes the weekly report"

_TEAM_BLOCK = "<team_context>Pat writes reports</team_context>\n\n"


async def _value(value: str, started: asyncio.Event | None = None) -> str:
    if started is not None:
        started.set()
    return value


def _read_marking(started: asyncio.Event, value: str):
    """A context read that marks when it started."""

    async def read(*args, **kwargs) -> str:
        started.set()
        return value

    return read


async def _fail(message: str) -> str:
    raise RuntimeError(message)


@pytest.mark.asyncio
async def test_the_reads_start_when_the_context_is_made() -> None:
    started = [asyncio.Event() for _ in range(3)]

    context = FirstTurnContext(
        _value("session", started[0]),
        _value("skills", started[1]),
        _value("expert", started[2]),
    )
    await asyncio.sleep(0)

    assert all(event.is_set() for event in started)
    assert await context.results() == ("session", "skills", "expert")


@pytest.mark.asyncio
async def test_a_turn_without_a_user_reads_no_session_block() -> None:
    context = FirstTurnContext(None, _value("skills"), _value("expert"))

    assert await context.results() == ("", "skills", "expert")


@pytest.mark.asyncio
async def test_a_failed_skills_read_leaves_the_index_empty(caplog) -> None:
    context = FirstTurnContext(
        _value("session"), _fail("registry down"), _value("expert")
    )

    with caplog.at_level(logging.ERROR):
        assert await context.results() == ("session", "", "expert")
    assert "failed to build skills_ctx" in caplog.text


@pytest.mark.asyncio
async def test_a_failed_session_read_raises_where_the_turn_reads_it() -> None:
    context = FirstTurnContext(
        _fail("scheduler down"), _value("skills"), _value("expert")
    )

    with pytest.raises(RuntimeError, match="scheduler down"):
        await context.results()
    context.close()


@pytest.mark.asyncio
async def test_closing_stops_the_reads_a_turn_never_awaited() -> None:
    cancelled: list[str] = []
    all_cancelled = asyncio.Event()

    async def hang(name: str) -> str:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.append(name)
            if len(cancelled) == 3:
                all_cancelled.set()
            raise
        return name

    context = FirstTurnContext(hang("session"), hang("skills"), hang("expert"))
    await asyncio.sleep(0)
    context.close()
    await asyncio.wait_for(all_cancelled.wait(), timeout=2)

    assert sorted(cancelled) == ["expert", "session", "skills"]


@pytest.mark.asyncio
@pytest.mark.parametrize("has_user", [True, False])
async def test_closing_before_dispatch_closes_all_reads(has_user: bool) -> None:
    started = [asyncio.Event() for _ in range(3)]
    reads = [
        _value(name, event)
        for name, event in zip(("session", "skills", "expert"), started)
    ]
    session, skills, expert = reads
    if not has_user:
        session.close()
    context = FirstTurnContext(session if has_user else None, skills, expert)
    try:
        context.close()
        checkpoint = asyncio.get_running_loop().create_future()
        asyncio.get_running_loop().call_soon(checkpoint.set_result, None)
        await checkpoint

        assert not any(event.is_set() for event in started)
        assert [inspect.getcoroutinestate(read) for read in reads] == [
            inspect.CORO_CLOSED
        ] * 3
    finally:
        for read in reads:
            read.close()


@pytest.mark.asyncio
async def test_a_failed_read_the_turn_never_awaited_is_not_reported() -> None:
    """A turn that ends early (its setup failed, say) never awaits the reads;
    one that failed on its own must not be logged later as an exception
    nobody retrieved."""
    reported: list[str] = []
    loop = asyncio.get_running_loop()
    loop.set_exception_handler(lambda _, ctx: reported.append(ctx["message"]))
    try:
        context = FirstTurnContext(
            _fail("scheduler down"), _value("skills"), _value("expert")
        )
        await asyncio.sleep(0)
        context.close()
        await asyncio.sleep(0)
        del context
        gc.collect()
    finally:
        loop.set_exception_handler(None)

    assert reported == []


@pytest.mark.asyncio
async def test_a_first_turn_reads_its_context_while_setup_runs() -> None:
    """The session block, skills index and expert block are read while the
    setup gather runs (its memory leg waits for all three reads to start),
    each once, and still reach the first message as it is stored."""
    started = {name: asyncio.Event() for name in ("session", "skills", "expert")}
    reads = {
        name: AsyncMock(side_effect=_read_marking(started[name], value))
        for name, value in (
            ("session", _SESSION_BLOCK),
            ("skills", ContextBlock(tag="available_skills", head=_SKILLS_INDEX)),
            ("expert", [ContextBlock(tag="team_context", head="Pat writes reports")]),
        )
    }
    during_setup: list[bool] = []

    async def warm_context(*args, **kwargs) -> None:
        waits = asyncio.gather(*(event.wait() for event in started.values()))
        try:
            await asyncio.wait_for(waits, timeout=2)
        except TimeoutError:
            during_setup.append(False)
        else:
            during_setup.append(True)

    db = await _FirstTurn()._run(
        _FirstTurn._first_turn("hi"),
        _FollowUp._clients([]),
        first_block=None,
        refresh=AsyncMock(return_value=None),
        extra=[
            (
                f"{_SVC}.fetch_warm_context",
                dict(new=AsyncMock(side_effect=warm_context)),
            ),
            (f"{_SVC}.build_session_context", dict(new=reads["session"])),
            (f"{_SVC}.skills_context_block", dict(new=reads["skills"])),
            (_EXPERT_CONTEXT, dict(new=reads["expert"])),
        ],
    )

    assert during_setup == [True]
    assert [read.await_count for read in reads.values()] == [1, 1, 1]
    (stored,) = _FirstTurn._stored(db)
    assert "pending_followups: 1" in stored
    assert "weekly-report" in stored
    assert "Pat writes reports" in stored
    assert stored.endswith("hi")


@pytest.mark.asyncio
async def test_a_follow_up_reads_no_first_turn_context() -> None:
    reads = [
        AsyncMock(return_value=""),
        AsyncMock(return_value=None),
        AsyncMock(return_value=[]),
    ]

    await _FirstTurn()._run(
        _FirstTurn._second_turn("first message", "and then?"),
        _FollowUp._clients([]),
        first_block=None,
        refresh=AsyncMock(return_value=None),
        extra=[
            (f"{_SVC}._write_cli_session_to_disk", dict(return_value=True)),
            (f"{_SVC}.build_session_context", dict(new=reads[0])),
            (f"{_SVC}.skills_context_block", dict(new=reads[1])),
            (_EXPERT_CONTEXT, dict(new=reads[2])),
        ],
    )

    assert [read.await_count for read in reads] == [0, 0, 0]


@pytest.mark.asyncio
async def test_a_first_turn_that_fails_in_setup_stops_its_context_reads() -> None:
    """The reads start before setup, so a turn whose setup fails never awaits
    them; its cleanup must not leave them running."""
    cancelled: list[str] = []

    def hanging(name: str):
        async def read(*args, **kwargs) -> str:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.append(name)
                raise
            return name

        return read

    with pytest.raises(RuntimeError, match="prompt store down"):
        await _FirstTurn()._run(
            _FirstTurn._first_turn("hi"),
            _FollowUp._clients([]),
            first_block=None,
            refresh=AsyncMock(return_value=None),
            extra=[
                (
                    f"{_SVC}._build_system_prompt",
                    dict(new=AsyncMock(side_effect=RuntimeError("prompt store down"))),
                ),
                (f"{_SVC}.build_session_context", dict(new=hanging("session"))),
                (f"{_SVC}.skills_context_block", dict(new=hanging("skills"))),
                (_EXPERT_CONTEXT, dict(new=hanging("expert"))),
            ],
        )
    for _ in range(3):
        await asyncio.sleep(0)

    assert sorted(cancelled) == ["expert", "session", "skills"]
