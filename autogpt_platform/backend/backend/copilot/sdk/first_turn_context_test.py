import asyncio
import logging
from unittest.mock import AsyncMock

import pytest

from .first_turn_context import FirstTurnContext
from .retry_scenarios_test import TestFirstTurnWarmContextIsEphemeral as _FirstTurn
from .retry_scenarios_test import TestFollowUpWarmContextCallSite as _FollowUp

_SVC = "backend.copilot.sdk.service"
_EXPERT_CONTEXT = "backend.copilot.service.build_expert_context"
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
    await context.close()


@pytest.mark.asyncio
async def test_closing_stops_the_reads_a_turn_never_awaited() -> None:
    cancelled: list[str] = []

    async def hang(name: str) -> str:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.append(name)
            raise
        return name

    context = FirstTurnContext(hang("session"), hang("skills"), hang("expert"))
    await asyncio.sleep(0)
    await asyncio.wait_for(context.close(), timeout=2)

    assert sorted(cancelled) == ["expert", "session", "skills"]


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
            ("skills", _SKILLS_INDEX),
            ("expert", _TEAM_BLOCK),
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
            (f"{_SVC}.build_skills_context", dict(new=reads["skills"])),
            (_EXPERT_CONTEXT, dict(new=reads["expert"])),
        ],
    )

    assert during_setup == [True]
    assert [read.await_count for read in reads.values()] == [1, 1, 1]
    (stored,) = _FirstTurn._stored(db)
    assert "pending_followups: 1" in stored
    assert "weekly-report" in stored
    assert "<team_context>Pat writes reports</team_context>" in stored
    assert stored.endswith("hi")


@pytest.mark.asyncio
async def test_a_follow_up_reads_no_first_turn_context() -> None:
    reads = [AsyncMock(return_value="") for _ in range(3)]

    await _FirstTurn()._run(
        _FirstTurn._second_turn("first message", "and then?"),
        _FollowUp._clients([]),
        first_block=None,
        refresh=AsyncMock(return_value=None),
        extra=[
            (f"{_SVC}.build_session_context", dict(new=reads[0])),
            (f"{_SVC}.build_skills_context", dict(new=reads[1])),
            (_EXPERT_CONTEXT, dict(new=reads[2])),
        ],
    )

    assert [read.await_count for read in reads] == [0, 0, 0]
