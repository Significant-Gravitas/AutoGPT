import pytest

from backend.blocks.conductor._transcript import (
    latest_reply,
    read_after,
    wait_for_reply,
    wait_until_idle,
)
from backend.blocks.conductor.test_fixtures import (
    RECEIPT,
    FakeTranscript,
    agent_row,
    claude_text,
    user_row,
    wait_client,
    wait_clock,
)


@pytest.mark.asyncio
async def test_untagged_agent_row_cannot_complete_a_tagged_turn():
    untagged = agent_row("unrelated", "", 2, claude_text("Wrong turn"))
    rows = [user_row("prompt", RECEIPT, 1), untagged]
    client = wait_client(FakeTranscript(rows), [{"status": "idle"}])
    with wait_clock():
        result = await wait_for_reply(client, "s1", RECEIPT, 60, 1)

    assert result["timed_out"] is True
    assert result["reply"] == ""
    assert [row["id"] for row in result["messages"]] == ["prompt"]


@pytest.mark.asyncio
async def test_tagged_reply_excludes_untagged_rows():
    rows = [
        user_row("prompt", RECEIPT, 1),
        agent_row("unrelated", "", 2, claude_text("Wrong turn")),
        agent_row("answer", RECEIPT, 3, claude_text("Correct answer")),
    ]
    client = wait_client(FakeTranscript(rows), [{"status": "idle"}])
    with wait_clock():
        result = await wait_for_reply(client, "s1", RECEIPT, 60, 1)

    assert result["timed_out"] is False
    assert result["reply"] == "Correct answer"
    assert [row["id"] for row in result["messages"]] == ["prompt", "answer"]


@pytest.mark.asyncio
async def test_a_continued_wait_ignores_untagged_rows_when_judging_idle():
    """Continuing a timed-out wait from `next_after` with the prompt's receipt
    keeps the turn guard: an idle session whose only new row belongs to
    another turn is not finished, one with the tagged answer is."""
    transcript = FakeTranscript([user_row("prompt", RECEIPT, 1)])
    client = wait_client(transcript, [{"status": "working"}])
    with wait_clock():
        result = await wait_for_reply(client, "s1", RECEIPT, 30, 10)
    assert result["timed_out"] is True
    assert result["prompt_row_id"] == "prompt"

    growth = iter(
        [
            [agent_row("unrelated", "", 2, claude_text("Wrong turn"))],
            [agent_row("answer", RECEIPT, 3, claude_text("Late"))],
        ]
    )
    idle = wait_client(transcript, [{"status": "idle"}])
    inner = idle.session_status

    async def status(session_id: str) -> dict:
        transcript.rows.extend(next(growth, []))
        return await inner(session_id)

    idle.session_status = status
    with wait_clock():
        _, timed_out = await wait_until_idle(
            idle, "s1", 600, 10, prompt_message_id=RECEIPT
        )
    assert timed_out is False
    rows, _, cursor = await read_after(idle, "s1", "prompt", 10, latest=True)
    assert cursor == "prompt"
    assert [row["id"] for row in rows] == ["unrelated", "answer"]
    assert latest_reply(rows) == "Late"
