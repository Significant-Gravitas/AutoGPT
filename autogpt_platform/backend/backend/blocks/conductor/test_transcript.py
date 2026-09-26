"""Tests for transcript parsing and the wait-for-reply loop (R1-01..R1-04,
R2-01..R2-03).

Fixtures are sanitized copies of live Claude and Codex envelopes, see
``test_fixtures.py``.
"""

import asyncio
from typing import Any
from unittest import mock

import pytest

from backend.blocks.conductor import _paging, _transcript
from backend.blocks.conductor._api import ConductorClient
from backend.blocks.conductor._transcript import (
    is_agent_message,
    latest_reply,
    message_text,
    reply_text,
    wait_for_reply,
)
from backend.blocks.conductor.test_fixtures import (
    CLAUDE_LIFECYCLE,
    CLAUDE_SYSTEM,
    CLAUDE_TOOL_USE,
    OLD_TURN,
    RECEIPT,
    TEST_CREDENTIALS,
    FakeTranscript,
    agent_row,
    claude_result,
    claude_text,
    claude_turn,
    codex_event,
    user_row,
    wait_client,
    wait_clock,
)

# --- transcript parsing (R1-01) ---------------------------------------------


def test_user_message_envelope_is_not_an_agent_message():
    row = user_row("row-1", RECEIPT, 1)
    assert is_agent_message(row) is False
    assert message_text(row) == ""


def test_claude_rows_yield_only_visible_assistant_text():
    rows = claude_turn(RECEIPT, 1, "Done")
    texts = {r["id"]: message_text(r) for r in rows}
    assert texts == {
        "r-sys": "",
        "r-life": "",
        "r-think": "",
        "r-interim": "Reading.",
        "r-tool": "",
        "r-progress": "",
        "r-result": "",
        "r-answer": "Done",
        "r-final": "",
        "r-idle": "",
    }
    assert reply_text(rows) == "Reading.\n\nDone"
    assert latest_reply(rows) == "Done"


def test_codex_rows_yield_each_agent_message_once():
    rows = [
        user_row("row-1", RECEIPT, 1),
        agent_row("row-2", RECEIPT, 2, codex_event("thread.started")),
        agent_row("row-3", RECEIPT, 3, codex_event("turn.started")),
        agent_row(
            "row-4",
            RECEIPT,
            4,
            codex_event(
                "item.started", {"type": "userMessage", "id": "i1", "content": []}
            ),
        ),
        agent_row(
            "row-5",
            RECEIPT,
            5,
            codex_event(
                "item.started", {"type": "agentMessage", "id": "i2", "text": ""}
            ),
        ),
        agent_row(
            "row-6",
            RECEIPT,
            6,
            codex_event(
                "item.completed", {"type": "agentMessage", "id": "i2", "text": "Hi"}
            ),
        ),
        agent_row(
            "row-7",
            RECEIPT,
            7,
            codex_event(
                "item.completed",
                {"type": "reasoning", "id": "i3", "summary": ["thinking"]},
            ),
        ),
        agent_row(
            "row-8",
            RECEIPT,
            8,
            codex_event(
                "item.completed",
                {"type": "commandExecution", "id": "i4", "command": "ls"},
            ),
        ),
        agent_row(
            "row-9",
            RECEIPT,
            9,
            codex_event(
                "item.completed",
                {"type": "agentMessage", "id": "i5", "text": "All green."},
            ),
        ),
        agent_row("row-10", RECEIPT, 10, codex_event("turn.completed")),
    ]
    assert reply_text(rows) == "Hi\n\nAll green."
    assert latest_reply(rows) == "All green."


def test_result_text_is_a_fallback_when_no_assistant_text_exists():
    rows = [agent_row("r-final", RECEIPT, 1, claude_result("Only in result"))]
    assert reply_text(rows) == "Only in result"
    assert latest_reply(rows) == "Only in result"


def test_reply_text_never_serialises_unknown_shapes_as_json():
    messages = [
        {"type": "user", "content": "prompt"},
        {"type": "assistant", "content": {"kind": "tool", "name": "bash"}},
        {"type": "assistant", "content": [{"type": "text", "text": "plain"}]},
        {"type": "assistant", "content": {"text": "dict"}},
        {"type": "assistant", "content": "string"},
    ]
    assert reply_text(messages) == "plain\n\ndict\n\nstring"
    assert latest_reply(messages) == "string"


# --- waiting (R1-02, R1-03, R1-04) ------------------------------------------


@pytest.mark.asyncio
async def test_wait_for_reply_ignores_idle_before_the_prompt_starts():
    """Idle is not accepted until the prompt row and an agent row of its turn
    exist; the receipt is matched via content.id, never used as a cursor."""
    prior = [
        user_row("row-0", OLD_TURN, 0),
        *[
            agent_row(f"old-{i}", OLD_TURN, i, claude_text("earlier"))
            for i in range(1, 4)
        ],
    ]
    transcript = FakeTranscript(list(prior))
    prompt = user_row("row-prompt", RECEIPT, 4)
    turn = claude_turn(RECEIPT, 4, "Done")
    growth = iter(
        [
            [],  # poll 1: idle, prompt not delivered yet
            [prompt],  # poll 2: idle, prompt queued, nothing started
            turn[:4],  # poll 3: working
            turn[4:],  # poll 4: idle with the answer
        ]
    )
    statuses = [
        {"status": "idle"},
        {"status": "idle"},
        {"status": "working"},
        {"status": "idle"},
    ]
    polls = {"n": 0}
    client = ConductorClient(TEST_CREDENTIALS)

    async def status(session_id: str) -> dict[str, Any]:
        transcript.rows.extend(next(growth, []))
        result = statuses[min(polls["n"], len(statuses) - 1)]
        polls["n"] += 1
        return result

    client.session_status = status
    client.list_messages = transcript.list_messages
    with wait_clock() as _:
        result = await wait_for_reply(client, "s1", RECEIPT, 600, 10)

    assert polls["n"] == 4
    assert result["timed_out"] is False
    assert result["session_status"] == "idle"
    assert result["reply"] == "Reading.\n\nDone"
    assert [m["id"] for m in result["messages"]] == [
        "row-prompt",
        *[r["id"] for r in turn],
    ]
    assert result["truncated"] is False
    assert all(call["after"] != RECEIPT for call in transcript.calls)


@pytest.mark.asyncio
async def test_wait_for_reply_accepts_a_fast_turn_finished_between_polls():
    rows = [user_row("row-prompt", RECEIPT, 1), *claude_turn(RECEIPT, 1, "Quick")]
    client = wait_client(FakeTranscript(rows), [{"status": "idle"}])
    with wait_clock() as _:
        result = await wait_for_reply(client, "s1", RECEIPT, 600, 10)
    assert result["timed_out"] is False
    assert result["reply"] == "Reading.\n\nQuick"


@pytest.mark.asyncio
async def test_wait_for_reply_only_returns_rows_of_the_prompts_turn():
    """A prompt queued behind earlier work: rows after the prompt row that
    belong to the previous turn are not part of the reply."""
    rows = [
        user_row("row-0", OLD_TURN, 0),
        agent_row("old-1", OLD_TURN, 1, claude_text("earlier")),
        user_row("row-prompt", RECEIPT, 2),
        agent_row("old-2", OLD_TURN, 3, claude_text("still the old turn")),
        agent_row("old-3", OLD_TURN, 4, claude_result("still the old turn")),
        *claude_turn(RECEIPT, 4, "Ours"),
    ]
    client = wait_client(FakeTranscript(rows), [{"status": "idle"}])
    with wait_clock() as _:
        result = await wait_for_reply(client, "s1", RECEIPT, 600, 10)
    assert result["reply"] == "Reading.\n\nOurs"
    assert "old-2" not in [m["id"] for m in result["messages"]]


@pytest.mark.asyncio
async def test_wait_for_reply_pages_through_the_whole_turn():
    """The answer sits on the third page of a 250-row turn: cursors advance,
    rows are ordered and unique, and no row is fetched by receipt id."""
    filler = [
        agent_row(f"tool-{i}", RECEIPT, i + 1, CLAUDE_TOOL_USE) for i in range(240)
    ]
    rows = [user_row("row-prompt", RECEIPT, 0), *filler]
    rows += [agent_row("row-answer", RECEIPT, 250, claude_text("Found it"))]
    rows += [agent_row("row-trailing", RECEIPT, 251, CLAUDE_LIFECYCLE)]
    transcript = FakeTranscript(rows)
    client = wait_client(transcript, [{"status": "idle"}])
    with wait_clock() as _:
        result = await wait_for_reply(client, "s1", RECEIPT, 600, 10)
    ids = [m["id"] for m in result["messages"]]
    assert ids[0] == "row-prompt"
    assert ids[-2:] == ["row-answer", "row-trailing"]
    assert len(ids) == len(set(ids)) == len(rows)
    assert result["reply"] == "Found it"
    assert result["truncated"] is False
    afters = [c["after"] for c in transcript.calls if c["after"]]
    assert afters, "expected cursor-based paging"
    assert afters == sorted(
        afters, key=lambda a: rows.index(next(r for r in rows if r["id"] == a))
    )


@pytest.mark.asyncio
async def test_wait_for_reply_marks_truncation_when_the_turn_exceeds_the_cap():
    rows = [
        user_row("row-prompt", RECEIPT, 0),
        *[agent_row(f"tool-{i}", RECEIPT, i + 1, CLAUDE_TOOL_USE) for i in range(10)],
    ]
    rows.append(agent_row("row-answer", RECEIPT, 11, claude_text("Late answer")))
    client = wait_client(FakeTranscript(rows), [{"status": "idle"}])
    with mock.patch.object(_transcript, "MAX_TURN_MESSAGES", 4), wait_clock():
        result = await wait_for_reply(client, "s1", RECEIPT, 600, 10)
    assert result["truncated"] is True
    assert len(result["messages"]) == 4
    assert result["messages"][-1]["id"] == "row-answer"
    assert result["reply"] == "Late answer"


@pytest.mark.asyncio
async def test_wait_for_reply_times_out_before_the_prompt_is_delivered():
    client = wait_client(FakeTranscript([]), [{"status": "idle"}])
    with wait_clock() as now:
        result = await wait_for_reply(client, "s1", RECEIPT, 30, 10)
    assert result["timed_out"] is True
    assert result["reply"] == ""
    assert result["messages"] == []
    assert result["session_status"] == "idle"
    assert now[0] < 60


@pytest.mark.asyncio
async def test_wait_for_reply_caps_the_sleep_by_the_remaining_deadline():
    client = wait_client(FakeTranscript([]), [{"status": "working"}])
    sleeps: list[float] = []
    with wait_clock(step=0.0, sleeps=sleeps):
        result = await wait_for_reply(client, "s1", RECEIPT, 1, 300)
    assert result["timed_out"] is True
    assert sleeps and max(sleeps) <= 1


@pytest.mark.asyncio
async def test_wait_for_reply_bounds_a_hanging_status_request():
    client = ConductorClient(TEST_CREDENTIALS)
    never = asyncio.Event()

    async def hang(session_id: str) -> dict[str, Any]:
        await never.wait()
        return {}

    client.session_status = hang
    client.list_messages = FakeTranscript([]).list_messages
    result = await asyncio.wait_for(
        wait_for_reply(client, "s1", RECEIPT, 0.2, 0.05), timeout=5
    )
    assert result["timed_out"] is True
    assert result["reply"] == ""


@pytest.mark.asyncio
async def test_wait_for_reply_returns_on_session_error():
    rows = [
        user_row("row-prompt", RECEIPT, 1),
        agent_row("row-2", RECEIPT, 2, claude_text("Hi")),
    ]
    client = wait_client(
        FakeTranscript(rows), [{"status": "error", "lastError": "boom"}]
    )
    with wait_clock() as _:
        result = await wait_for_reply(client, "s1", RECEIPT, 10, 1)
    assert result["reply"] == "Hi"
    assert result["session_status"] == "error"
    assert result["error_message"] == "boom"
    assert result["timed_out"] is False


@pytest.mark.asyncio
async def test_wait_for_reply_supports_plain_rows_keyed_by_row_id():
    rows = [
        {"id": "m1", "type": "user", "content": "go"},
        {"id": "m2", "type": "assistant", "content": [{"text": "Hi"}]},
        {"id": "m3", "type": "user", "content": "ignored"},
        {"id": "m4", "type": "assistant", "content": {"text": "Done"}},
    ]
    client = wait_client(FakeTranscript(rows), [{"status": "idle"}])
    with wait_clock() as _:
        result = await wait_for_reply(client, "s1", "m1", 10, 1)
    assert result["reply"] == "Hi\n\nDone"
    assert [m["id"] for m in result["messages"]] == ["m1", "m2", "m3", "m4"]


# --- round 2: prompt outside the first tail (R2-01) --------------------------


def lifecycle_rows(turn: str, start: int, count: int) -> list[dict]:
    return [
        agent_row(f"life-{start + i}", turn, start + i, CLAUDE_LIFECYCLE)
        for i in range(count)
    ]


async def run_wait(client, timeout: float, poll: float):
    with wait_clock():
        return await wait_for_reply(client, "s1", RECEIPT, timeout, poll)


@pytest.mark.asyncio
async def test_wait_for_reply_correlates_a_turn_whose_prompt_is_outside_the_tail():
    """A 307-row turn already complete at the first poll: the prompt row is
    older than the newest 300 rows, but the tail is tagged with its turn, so
    the answer is returned and the omitted rows are reported as truncation."""
    rows = [
        user_row("row-prompt", RECEIPT, 0),
        *lifecycle_rows(RECEIPT, 1, 305),
        agent_row("row-answer", RECEIPT, 306, claude_text("Completed")),
    ]
    transcript = FakeTranscript(rows)
    client = wait_client(transcript, [{"status": "idle"}])
    result = await run_wait(client, 30, 10)
    assert result["reply"] == "Completed"
    assert result["timed_out"] is False
    assert result["truncated"] is True
    ids = [m["id"] for m in result["messages"]]
    assert ids[-1] == "row-answer" and "row-prompt" not in ids


@pytest.mark.asyncio
async def test_wait_for_reply_searches_older_pages_for_a_finished_turn():
    """Our turn finished, then another prompt produced 320 rows before the
    first poll: nothing in the tail is ours, so older pages are read until
    the prompt row is found and the whole turn is returned untruncated."""
    other = "22222222-0000-4000-8000-000000000000"
    rows = [
        user_row("row-0", OLD_TURN, 0),
        agent_row("old-1", OLD_TURN, 1, claude_text("earlier")),
        user_row("row-prompt", RECEIPT, 2),
        agent_row("row-answer", RECEIPT, 3, claude_text("Ours")),
        agent_row("row-final", RECEIPT, 4, claude_result("Ours")),
        user_row("row-other", other, 5),
        *lifecycle_rows(other, 6, 320),
    ]
    transcript = FakeTranscript(rows)
    client = wait_client(transcript, [{"status": "idle"}])
    result = await run_wait(client, 60, 10)
    assert result["reply"] == "Ours"
    assert result["timed_out"] is False
    assert result["truncated"] is False
    assert [m["id"] for m in result["messages"]] == [
        "row-prompt",
        "row-answer",
        "row-final",
    ]


@pytest.mark.asyncio
async def test_wait_for_reply_bounds_the_search_for_an_unrecorded_prompt():
    """A long session whose newest 1500 rows are older turns and whose prompt
    has not been recorded yet: the backward search stops at its bound, then
    the prompt and its turn are picked up as they arrive."""
    rows = [user_row("row-0", OLD_TURN, 0), *lifecycle_rows(OLD_TURN, 1, 1500)]
    transcript = FakeTranscript(rows)
    later = [user_row("row-prompt", RECEIPT, 1501), *claude_turn(RECEIPT, 1501, "Late")]
    statuses = iter([{"status": "idle"}, {"status": "working"}, {"status": "idle"}])
    client = ConductorClient(TEST_CREDENTIALS)

    async def status(session_id: str) -> dict[str, Any]:
        state = next(statuses, {"status": "idle"})
        if state["status"] == "working":
            transcript.rows.extend(later)
        return state

    client.session_status = status
    client.list_messages = transcript.list_messages
    result = await run_wait(client, 600, 10)
    assert result["reply"] == "Reading.\n\nLate"
    assert result["timed_out"] is False
    assert result["truncated"] is False
    offsets = [c for c in transcript.calls if c["offset"] is not None]
    assert len(offsets) <= 30, "backward search must be bounded"


# --- round 2: a real deadline (R2-02) ----------------------------------------


@pytest.mark.asyncio
async def test_wait_for_reply_makes_no_requests_once_the_deadline_passed():
    rows = [user_row("row-prompt", RECEIPT, 1), *claude_turn(RECEIPT, 1, "Done")]
    transcript = FakeTranscript(rows)
    client = wait_client(transcript, [{"status": "idle"}])
    status_calls = {"n": 0}
    inner = client.session_status

    async def counting_status(session_id: str) -> dict[str, Any]:
        status_calls["n"] += 1
        return await inner(session_id)

    client.session_status = counting_status
    result = await run_wait(client, 1, 10)
    assert result["timed_out"] is True
    assert result["reply"] == ""
    assert transcript.calls == []
    assert status_calls["n"] == 0


@pytest.mark.asyncio
async def test_bounded_refuses_to_start_a_request_after_the_deadline():
    ran = {"started": False}

    async def request() -> dict[str, Any]:
        ran["started"] = True
        return {}

    with pytest.raises(TimeoutError):
        await _paging.bounded(request(), lambda: 0.0)
    assert ran["started"] is False


@pytest.mark.asyncio
async def test_tail_probes_stop_at_the_deadline_without_a_grace_period():
    rows = [user_row("row-0", OLD_TURN, 0), *lifecycle_rows(OLD_TURN, 1, 5000)]
    transcript = FakeTranscript(rows)
    client = wait_client(transcript, [{"status": "working"}])
    budget = iter([5.0, 5.0, 0.0])

    def remaining() -> float:
        return next(budget, 0.0)

    with pytest.raises(TimeoutError):
        await _paging.fetch_tail(client, "s1", 300, remaining)
    assert len(transcript.calls) == 2


@pytest.mark.asyncio
async def test_wait_for_reply_keeps_collected_rows_when_a_page_request_times_out():
    rows = [
        user_row("row-prompt", RECEIPT, 1),
        agent_row("row-interim", RECEIPT, 2, claude_text("Reading.")),
    ]
    transcript = FakeTranscript(rows)
    never = asyncio.Event()
    calls = {"n": 0}

    async def list_messages(
        session_id: str,
        after: str = "",
        limit: int | None = None,
        offset: int | None = None,
    ) -> dict[str, Any]:
        calls["n"] += 1
        if calls["n"] > 1:
            await never.wait()
        return await transcript.list_messages(session_id, after, limit, offset)

    client = wait_client(transcript, [{"status": "working"}])
    client.list_messages = list_messages
    result = await asyncio.wait_for(
        wait_for_reply(client, "s1", RECEIPT, 0.3, 0.05), timeout=5
    )
    assert result["timed_out"] is True
    assert result["reply"] == "Reading."
    assert [m["id"] for m in result["messages"]] == ["row-prompt", "row-interim"]


# --- round 2: startup events are not completion (R2-03) ----------------------


class LaggingSession:
    """Status and transcript advance one tick per API call, so the status the
    loop sees can be older than the rows it has already read."""

    def __init__(self, statuses: list[str], rows_at: dict[int, list[dict]]):
        self.tick = 0
        self.statuses = statuses
        self.rows_at = rows_at
        self.transcript = FakeTranscript([])

    def _advance(self) -> int:
        tick, self.tick = self.tick, self.tick + 1
        for at, rows in self.rows_at.items():
            if at == tick:
                self.transcript.rows.extend(rows)
        return tick

    async def status(self, session_id: str) -> dict[str, Any]:
        tick = self._advance()
        return {"status": self.statuses[min(tick, len(self.statuses) - 1)]}

    async def list_messages(
        self,
        session_id: str,
        after: str = "",
        limit: int | None = None,
        offset: int | None = None,
    ) -> dict[str, Any]:
        self._advance()
        return await self.transcript.list_messages(session_id, after, limit, offset)

    def client(self) -> ConductorClient:
        client = ConductorClient(TEST_CREDENTIALS)
        client.session_status = self.status
        client.list_messages = self.list_messages
        return client


@pytest.mark.asyncio
async def test_wait_for_reply_keeps_polling_past_a_startup_only_turn():
    """The session is idle while the prompt is queued; the agent starts
    (system/init only) during the same poll, works, then answers."""
    turn = claude_turn(RECEIPT, 1, "Answer")
    session = LaggingSession(
        statuses=["idle", "idle", "working", "working", "working", "idle"],
        rows_at={0: [user_row("row-prompt", RECEIPT, 1), turn[0]], 3: turn[1:]},
    )
    result = await run_wait(session.client(), 600, 10)
    assert result["timed_out"] is False
    assert result["reply"] == "Reading.\n\nAnswer"
    assert result["messages"][-1]["id"] == "r-idle"


@pytest.mark.asyncio
async def test_wait_for_reply_does_not_complete_on_startup_events_alone():
    rows = [
        user_row("row-prompt", RECEIPT, 1),
        agent_row("r-sys", RECEIPT, 2, CLAUDE_SYSTEM),
        agent_row("r-life", RECEIPT, 3, CLAUDE_LIFECYCLE),
        agent_row("r-codex", RECEIPT, 4, codex_event("turn.started")),
    ]
    client = wait_client(FakeTranscript(rows), [{"status": "idle"}])
    result = await run_wait(client, 30, 10)
    assert result["timed_out"] is True
    assert result["reply"] == ""
    assert result["session_status"] == "idle"
    assert [m["id"] for m in result["messages"]] == [
        "row-prompt",
        "r-sys",
        "r-life",
        "r-codex",
    ]


@pytest.mark.asyncio
async def test_wait_for_reply_reads_rows_written_just_before_idle():
    """Rows that land between the transcript read and the status read of the
    same poll belong to the reply."""
    turn = claude_turn(RECEIPT, 1, "Final")
    session = LaggingSession(
        statuses=["working", "idle"],
        rows_at={0: [user_row("row-prompt", RECEIPT, 1), *turn[:4]], 1: turn[4:]},
    )
    result = await run_wait(session.client(), 600, 10)
    assert result["timed_out"] is False
    assert result["reply"] == "Reading.\n\nFinal"
    assert result["messages"][-1]["id"] == "r-idle"
