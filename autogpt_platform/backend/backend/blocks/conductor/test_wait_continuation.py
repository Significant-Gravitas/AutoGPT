"""Tests for continuing a Conductor wait: the AutoPilot tool cap, a prompt
that is still queued when Get Session's wait ends, the status re-read after
a queued turn starts, and the cursor a timed-out Get Session hands back."""

from typing import Any
from unittest import mock

import pytest

from backend.blocks.conductor import _api
from backend.blocks.conductor._api import COPILOT_WAIT_SECONDS, wait_seconds_for
from backend.blocks.conductor._transcript import wait_until_idle
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
from backend.copilot.constants import COPILOT_SESSION_PREFIX, MAX_TOOL_WAIT_SECONDS
from backend.data.execution import ExecutionContext

COPILOT_CONTEXT = ExecutionContext(graph_exec_id=f"{COPILOT_SESSION_PREFIX}chat-1")


def _get_session_client(transcript: FakeTranscript, statuses: list[dict]):
    client = wait_client(transcript, statuses)
    client.get_session = mock.AsyncMock(return_value={"id": "s1", "deepLink": "d"})
    return client


def test_copilot_waits_end_before_the_tool_cap():
    assert COPILOT_WAIT_SECONDS < MAX_TOOL_WAIT_SECONDS
    with mock.patch.object(_api.time, "monotonic", return_value=100.0):
        assert wait_seconds_for(1800, COPILOT_CONTEXT, 100.0) == COPILOT_WAIT_SECONDS
        assert wait_seconds_for(60, COPILOT_CONTEXT, 100.0) == 60
        assert wait_seconds_for(1800, COPILOT_CONTEXT, 80.0) == (
            COPILOT_WAIT_SECONDS - 20
        )
        assert wait_seconds_for(1800, COPILOT_CONTEXT, -1000.0) == 1


@pytest.mark.parametrize(
    "context", [None, ExecutionContext(), ExecutionContext(graph_exec_id="exec-1")]
)
def test_graph_waits_keep_their_timeout(context: ExecutionContext | None):
    assert wait_seconds_for(1800, context, 0.0) == 1800


@pytest.mark.asyncio
async def test_send_message_from_copilot_waits_within_the_tool_cap():
    block = ConductorSendMessageBlock()
    waits: list[tuple[int, int]] = []

    def record_wait(credentials, session_id, receipt, timeout, interval):
        waits.append((timeout, interval))
        return {**_timed_out_wait(), "prompt_row_id": "row-prompt"}

    mock_block(
        block,
        {
            "_send": lambda *a, **k: {"messageId": RECEIPT, "state": "sent"},
            "_wait": record_wait,
        },
    )
    input_data = block.Input.model_validate(
        {"credentials": TEST_CREDENTIALS_INPUT, "session_id": "s1", "message": "go"}
    )
    outputs = {}
    async for name, value in block.run(
        input_data, credentials=TEST_CREDENTIALS, execution_context=COPILOT_CONTEXT
    ):
        outputs[name] = value
    assert waits[0][0] <= COPILOT_WAIT_SECONDS
    assert outputs["timed_out"] is True
    assert outputs["next_after"] == "row-prompt"


@pytest.mark.asyncio
async def test_get_session_keeps_the_receipt_while_the_prompt_is_queued():
    """The continued wait ends with the prompt still queued, so `after` (Send
    Message's receipt) has no row yet: that is a timeout, not a failure."""
    transcript = FakeTranscript([user_row("row-other", "other-receipt", 1)])
    client = _get_session_client(transcript, [{"status": "working"}])
    with (
        mock.patch(
            "backend.blocks.conductor.get_session.ConductorClient",
            return_value=client,
        ),
        wait_clock(),
    ):
        outputs = await collect(
            ConductorGetSessionBlock(),
            {
                "credentials": TEST_CREDENTIALS_INPUT,
                "session_id": "s1",
                "wait_until_idle": True,
                "after": RECEIPT,
                "prompt_message_id": RECEIPT,
                "timeout_seconds": 30,
                "poll_interval_seconds": 10,
            },
        )
    assert outputs["timed_out"] is True
    assert outputs["message_count"] == 0
    assert outputs["next_after"] == RECEIPT


@pytest.mark.asyncio
async def test_get_session_still_rejects_an_unknown_cursor():
    transcript = FakeTranscript([user_row("row-other", "other-receipt", 1)])
    client = _get_session_client(transcript, [{"status": "idle"}])
    with (
        mock.patch(
            "backend.blocks.conductor.get_session.ConductorClient",
            return_value=client,
        ),
        pytest.raises(Exception, match="neither a transcript row ID"),
    ):
        await collect(
            ConductorGetSessionBlock(),
            {
                "credentials": TEST_CREDENTIALS_INPUT,
                "session_id": "s1",
                "wait_until_idle": True,
                "after": "not-a-row",
            },
        )


@pytest.mark.asyncio
async def test_timed_out_get_session_keeps_its_cursor_for_the_reply():
    """The answer lands just before the wait gives up; continuing from the
    returned cursor must still find it once only lifecycle rows follow."""
    turn = claude_turn(RECEIPT, 1, "Done")
    transcript = FakeTranscript([user_row("row-prompt", RECEIPT, 1), *turn[:8]])
    inputs = {
        "credentials": TEST_CREDENTIALS_INPUT,
        "session_id": "s1",
        "wait_until_idle": True,
        "after": "row-prompt",
        "prompt_message_id": RECEIPT,
        "timeout_seconds": 30,
        "poll_interval_seconds": 10,
    }
    client = _get_session_client(transcript, [{"status": "working"}])
    with (
        mock.patch(
            "backend.blocks.conductor.get_session.ConductorClient",
            return_value=client,
        ),
        wait_clock(),
    ):
        first = await collect(ConductorGetSessionBlock(), inputs)
    assert first["timed_out"] is True
    assert first["latest_reply"] == "Done"
    assert first["next_after"] == "row-prompt"

    transcript.rows.append(turn[9])
    client = _get_session_client(transcript, [{"status": "idle"}])
    with (
        mock.patch(
            "backend.blocks.conductor.get_session.ConductorClient",
            return_value=client,
        ),
        wait_clock(),
    ):
        second = await collect(
            ConductorGetSessionBlock(), {**inputs, "after": first["next_after"]}
        )
    assert second["timed_out"] is False
    assert second["latest_reply"] == "Done"
    assert second["next_after"] == "r-idle"


@pytest.mark.asyncio
async def test_wait_until_idle_rereads_the_status_once_the_turn_shows_progress():
    """Idle was read while the prompt was queued; by the time the transcript
    was read its turn had started, so the second status read wins."""
    turn = claude_turn(RECEIPT, 1, "Done")
    transcript = FakeTranscript([user_row("row-prompt", RECEIPT, 1), *turn[:4]])
    statuses = iter([{"status": "idle"}, {"status": "working"}, {"status": "working"}])
    client = wait_client(transcript, [{"status": "idle"}])

    async def status(session_id: str) -> dict[str, Any]:
        return next(statuses, {"status": "idle"})

    client.session_status = status
    with wait_clock():
        status_read, timed_out = await wait_until_idle(
            client, "s1", 600, 10, prompt_message_id=RECEIPT
        )
    assert timed_out is False
    assert status_read["status"] == "idle"
    assert next(statuses, None) is None


def _timed_out_wait() -> dict[str, Any]:
    return {
        "session_status": "working",
        "error_message": "",
        "messages": [],
        "reply": "",
        "timed_out": True,
        "truncated": False,
    }
