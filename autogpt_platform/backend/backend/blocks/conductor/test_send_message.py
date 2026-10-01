from unittest import mock

import pytest

from backend.blocks.conductor._api import DEFAULT_WAIT_SECONDS
from backend.blocks.conductor._mocks import WAIT_MOCK_REPLY
from backend.blocks.conductor.get_session import ConductorGetSessionBlock
from backend.blocks.conductor.send_message import ConductorSendMessageBlock
from backend.blocks.conductor.test_fixtures import (
    RECEIPT,
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    FakeTranscript,
    claude_turn,
    collect,
    mock_block,
    user_row,
    wait_client,
    wait_clock,
)
from backend.util.exceptions import BlockExecutionError


@pytest.mark.asyncio
@pytest.mark.parametrize("wait_for_reply", [True, False])
@pytest.mark.parametrize("receipt", [{}, {"messageId": None}, {"messageId": ""}])
async def test_missing_receipt_fails_before_emitting_outputs(
    receipt: dict, wait_for_reply: bool
):
    block = ConductorSendMessageBlock()
    mock_block(
        block,
        {
            "_send": lambda *args, **kwargs: receipt,
            "_wait": lambda *args, **kwargs: pytest.fail(
                "must not poll without a receipt"
            ),
        },
    )
    input_data = block.Input.model_validate(
        {
            "credentials": TEST_CREDENTIALS_INPUT,
            "session_id": "s1",
            "message": "go",
            "wait_for_reply": wait_for_reply,
        }
    )
    outputs = []

    with pytest.raises(BlockExecutionError, match="messageId"):
        async for output in block.run(input_data, credentials=TEST_CREDENTIALS):
            outputs.append(output)

    assert outputs == []


@pytest.mark.asyncio
async def test_timed_out_wait_continues_with_get_session_after_next_after():
    """A wait that runs out mid-turn hands over `next_after`; Get Session with
    `wait_until_idle` picks the turn up from there and returns its end."""
    rows = [user_row("row-prompt", RECEIPT, 1)]
    turn = claude_turn(RECEIPT, 1, "Done")
    transcript = FakeTranscript(rows + turn[:4])
    send_block = ConductorSendMessageBlock()
    mock_block(
        send_block,
        {"_send": lambda *a, **k: {"messageId": RECEIPT, "state": "sent"}},
    )
    with (
        mock.patch(
            "backend.blocks.conductor.send_message.ConductorClient",
            return_value=wait_client(transcript, [{"status": "working"}]),
        ),
        wait_clock(),
    ):
        sent = await collect(
            send_block,
            {
                "credentials": TEST_CREDENTIALS_INPUT,
                "session_id": "s1",
                "message": "go",
                "timeout_seconds": 30,
                "poll_interval_seconds": 10,
            },
        )
    assert sent["timed_out"] is True
    assert sent["reply"] == "Reading."
    assert sent["next_after"] == "r-interim"

    transcript.rows.extend(turn[4:])
    client = wait_client(transcript, [{"status": "working"}, {"status": "idle"}])
    client.get_session = mock.AsyncMock(return_value={"id": "s1", "deepLink": "d"})
    get_block = ConductorGetSessionBlock()
    sleeps: list[float] = []
    with (
        mock.patch(
            "backend.blocks.conductor.get_session.ConductorClient",
            return_value=client,
        ),
        wait_clock(sleeps=sleeps),
    ):
        continued = await collect(
            get_block,
            {
                "credentials": TEST_CREDENTIALS_INPUT,
                "session_id": "s1",
                "wait_until_idle": True,
                "after": sent["next_after"],
                "message_limit": 3,
            },
        )
    assert continued["timed_out"] is False
    assert continued["status"] == "idle"
    assert continued["latest_reply"] == "Done"
    assert [m["id"] for m in continued["messages"]] == ["r-answer", "r-final", "r-idle"]
    assert continued["has_more"] is True
    assert continued["next_after"] == "r-idle"
    assert sleeps == [20]


@pytest.mark.asyncio
async def test_send_message_scales_the_poll_interval_with_the_timeout():
    block = ConductorSendMessageBlock()
    seen: dict = {}

    def wait(_creds, _session, _after, timeout, poll):
        seen.update(timeout=timeout, poll=poll)
        return {**WAIT_MOCK_REPLY, "timed_out": True, "next_after": "row-7"}

    mock_block(
        block,
        {"_send": lambda *a, **k: {"messageId": "m1", "state": "sent"}, "_wait": wait},
    )
    outputs = await collect(
        block,
        {"credentials": TEST_CREDENTIALS_INPUT, "session_id": "s1", "message": "go"},
    )
    assert seen == {"timeout": DEFAULT_WAIT_SECONDS, "poll": 20}
    assert outputs["timed_out"] is True
    assert outputs["next_after"] == "row-7"
