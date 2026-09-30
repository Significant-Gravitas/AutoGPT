import pytest

from backend.blocks.conductor._transcript import wait_for_reply
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
