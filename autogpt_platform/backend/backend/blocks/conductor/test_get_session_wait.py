"""Tests for the in-tool wait of Get Session (`wait_until_idle`) and the
helpers behind it: the status-only wait loop, the newest-rows-after-cursor
read, and the poll interval that scales with the timeout."""

from typing import Any
from unittest import mock

import pytest

from backend.blocks.conductor._api import poll_interval_for
from backend.blocks.conductor._paging import fetch_latest_after
from backend.blocks.conductor._transcript import wait_for_reply, wait_until_idle
from backend.blocks.conductor.get_session import ConductorGetSessionBlock
from backend.blocks.conductor.test_fixtures import (
    RECEIPT,
    TEST_CREDENTIALS_INPUT,
    FakeTranscript,
    agent_row,
    claude_text,
    claude_turn,
    collect,
    user_row,
    wait_client,
    wait_clock,
)


def _turn_transcript() -> FakeTranscript:
    return FakeTranscript(
        [user_row("row-prompt", RECEIPT, 1), *claude_turn(RECEIPT, 1, "Done")]
    )


def _get_session(transcript: FakeTranscript, statuses: list[dict]):
    client = wait_client(transcript, statuses)
    client.get_session = mock.AsyncMock(return_value={"id": "s1", "deepLink": "d"})
    calls = {"status": 0}
    inner = client.session_status

    async def counting_status(session_id: str) -> dict[str, Any]:
        calls["status"] += 1
        return await inner(session_id)

    client.session_status = counting_status
    return client, calls


@pytest.mark.asyncio
async def test_wait_until_idle_polls_status_until_the_session_settles():
    transcript = _turn_transcript()
    client, calls = _get_session(
        transcript, [{"status": "working"}, {"status": "working"}, {"status": "idle"}]
    )
    sleeps: list[float] = []
    with (
        mock.patch(
            "backend.blocks.conductor.get_session.ConductorClient",
            return_value=client,
        ),
        wait_clock(sleeps=sleeps),
    ):
        outputs = await collect(
            ConductorGetSessionBlock(),
            {
                "credentials": TEST_CREDENTIALS_INPUT,
                "session_id": "s1",
                "wait_until_idle": True,
                "timeout_seconds": 600,
                "poll_interval_seconds": 15,
            },
        )
    assert calls["status"] == 3
    assert sleeps == [15, 15]
    assert outputs["status"] == "idle"
    assert outputs["timed_out"] is False
    assert outputs["latest_reply"] == "Done"
    assert outputs["next_after"] == "r-idle"


@pytest.mark.asyncio
async def test_wait_until_idle_costs_one_request_when_already_idle():
    client, calls = _get_session(_turn_transcript(), [{"status": "idle"}])
    sleeps: list[float] = []
    with (
        mock.patch(
            "backend.blocks.conductor.get_session.ConductorClient",
            return_value=client,
        ),
        wait_clock(sleeps=sleeps),
    ):
        outputs = await collect(
            ConductorGetSessionBlock(),
            {
                "credentials": TEST_CREDENTIALS_INPUT,
                "session_id": "s1",
                "wait_until_idle": True,
            },
        )
    assert calls["status"] == 1
    assert sleeps == []
    assert outputs["timed_out"] is False


@pytest.mark.asyncio
async def test_wait_until_idle_reports_a_timeout_and_keeps_the_cursor():
    transcript = FakeTranscript([user_row("row-prompt", RECEIPT, 1)])
    client, _ = _get_session(transcript, [{"status": "working"}])
    with (
        mock.patch(
            "backend.blocks.conductor.get_session.ConductorClient",
            return_value=client,
        ),
        wait_clock() as now,
    ):
        outputs = await collect(
            ConductorGetSessionBlock(),
            {
                "credentials": TEST_CREDENTIALS_INPUT,
                "session_id": "s1",
                "wait_until_idle": True,
                "timeout_seconds": 30,
                "poll_interval_seconds": 10,
                "after": "row-prompt",
            },
        )
    assert outputs["timed_out"] is True
    assert outputs["status"] == "working"
    assert outputs["messages"] == []
    assert outputs["next_after"] == "row-prompt"
    assert now[0] < 60


@pytest.mark.asyncio
async def test_without_wait_the_status_is_read_once():
    client, calls = _get_session(_turn_transcript(), [{"status": "working"}])
    with mock.patch(
        "backend.blocks.conductor.get_session.ConductorClient", return_value=client
    ):
        outputs = await collect(
            ConductorGetSessionBlock(),
            {"credentials": TEST_CREDENTIALS_INPUT, "session_id": "s1"},
        )
    assert calls["status"] == 1
    assert outputs["status"] == "working"
    assert outputs["timed_out"] is False


@pytest.mark.asyncio
async def test_wait_until_idle_stops_on_a_session_error():
    client = wait_client(FakeTranscript([]), [{"status": "error", "lastError": "boom"}])
    with wait_clock():
        status, timed_out = await wait_until_idle(client, "s1", 60, 10)
    assert timed_out is False
    assert status["lastError"] == "boom"


@pytest.mark.asyncio
async def test_wait_until_idle_makes_no_request_once_the_deadline_passed():
    client = wait_client(FakeTranscript([]), [{"status": "working"}])
    calls = {"n": 0}
    inner = client.session_status

    async def counting_status(session_id: str) -> dict[str, Any]:
        calls["n"] += 1
        return await inner(session_id)

    client.session_status = counting_status
    with wait_clock(step=100.0):
        status, timed_out = await wait_until_idle(client, "s1", 1, 10)
    assert timed_out is True
    assert calls["n"] == 0
    assert status == {}


@pytest.mark.asyncio
async def test_fetch_latest_after_keeps_the_newest_rows_after_the_cursor():
    rows = [user_row("row-prompt", RECEIPT, 1), *claude_turn(RECEIPT, 1, "Done")]
    transcript = FakeTranscript(rows, default_limit=3)
    client = wait_client(transcript, [{"status": "idle"}])
    kept, skipped = await fetch_latest_after(client, "s1", "r-sys", 2)
    assert [row["id"] for row in kept] == ["r-final", "r-idle"]
    assert skipped is True
    assert all(call["after"] for call in transcript.calls)


@pytest.mark.asyncio
async def test_fetch_latest_after_reads_to_the_end_when_everything_fits():
    rows = [
        user_row("row-prompt", RECEIPT, 1),
        agent_row("answer", RECEIPT, 2, claude_text("Done")),
    ]
    client = wait_client(FakeTranscript(rows), [{"status": "idle"}])
    kept, skipped = await fetch_latest_after(client, "s1", "row-prompt", 20)
    assert [row["id"] for row in kept] == ["answer"]
    assert skipped is False


@pytest.mark.parametrize(
    ("timeout", "explicit", "expected"),
    [
        (1800, 0, 20),
        (7200, 0, 60),
        (900, 0, 10),
        (15, 0, 7),
        (5, 0, 2),
        (2, 0, 1),
        (1800, 3, 3),
    ],
)
def test_poll_interval_scales_with_the_timeout(
    timeout: int, explicit: int, expected: int
):
    assert poll_interval_for(timeout, explicit) == expected


@pytest.mark.asyncio
async def test_a_short_automatic_wait_still_checks_the_session():
    rows = [user_row("row-prompt", RECEIPT, 1), *claude_turn(RECEIPT, 1, "Quick")]
    client = wait_client(FakeTranscript(rows), [{"status": "idle"}])
    with wait_clock(step=0.0):
        result = await wait_for_reply(client, "s1", RECEIPT, 5, poll_interval_for(5))
    assert result["timed_out"] is False
    assert result["reply"] == "Reading.\n\nQuick"


@pytest.mark.asyncio
async def test_wait_until_idle_with_a_prompt_ignores_idle_before_its_turn_starts():
    """A continued wait must not end on a session that is idle because the
    prompt is still queued or has only launched the agent."""
    prompt = user_row("row-prompt", RECEIPT, 1)
    turn = claude_turn(RECEIPT, 1, "Done")
    transcript = FakeTranscript([])
    growth = iter([[], [prompt], turn[:2], turn[2:]])
    client = wait_client(transcript, [{"status": "idle"}])
    inner = client.session_status
    polls = {"n": 0}

    async def status(session_id: str) -> dict[str, Any]:
        transcript.rows.extend(next(growth, []))
        polls["n"] += 1
        return await inner(session_id)

    client.session_status = status
    with wait_clock():
        status_read, timed_out = await wait_until_idle(
            client, "s1", 600, 10, prompt_message_id=RECEIPT
        )
    assert timed_out is False
    assert status_read["status"] == "idle"
    assert polls["n"] == 4


@pytest.mark.asyncio
async def test_wait_until_idle_with_a_prompt_times_out_while_it_stays_queued():
    transcript = FakeTranscript([user_row("row-prompt", RECEIPT, 1)])
    client = wait_client(transcript, [{"status": "idle"}])
    with wait_clock():
        _, timed_out = await wait_until_idle(
            client, "s1", 30, 10, prompt_message_id=RECEIPT
        )
    assert timed_out is True


@pytest.mark.asyncio
async def test_fetch_latest_after_falls_back_to_the_tail_past_the_scan_cap():
    rows = [user_row("row-prompt", RECEIPT, 0)]
    rows += [
        agent_row(f"r-{i}", RECEIPT, i, claude_text(f"step {i}")) for i in range(1, 31)
    ]
    transcript = FakeTranscript(rows)
    client = wait_client(transcript, [{"status": "idle"}])
    kept, skipped = await fetch_latest_after(client, "s1", "row-prompt", 2, cap=10)
    assert [row["id"] for row in kept] == ["r-29", "r-30"]
    assert skipped is True
