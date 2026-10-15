"""Get Session `after` cursors and the `next_after` handed out by the
prompt-sending blocks (SECRT-2789).

The API's `after` is a transcript row id, while Send Message / Create
Session / Create Workspace return the prompt's receipt id. The blocks bridge
the two: senders resolve the prompt's row into `next_after`, and Get Session
resolves a receipt passed as `after` instead of surfacing the API's 404.
"""

from typing import Any
from unittest import mock

import pytest

from backend.blocks.conductor._api import ConductorAPIError
from backend.blocks.conductor._transcript import find_prompt_row, read_after
from backend.blocks.conductor.create_session import ConductorCreateSessionBlock
from backend.blocks.conductor.create_workspace import ConductorCreateWorkspaceBlock
from backend.blocks.conductor.get_session import ConductorGetSessionBlock
from backend.blocks.conductor.send_message import ConductorSendMessageBlock
from backend.blocks.conductor.test_fixtures import (
    CLAUDE_LIFECYCLE,
    OLD_TURN,
    RECEIPT,
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    FakeTranscript,
    agent_row,
    claude_turn,
    collect,
    mock_block,
    user_row,
    wait_client,
)
from backend.util.exceptions import BlockExecutionError

PROMPT_ROW = "row-prompt"


def _session_client(transcript: FakeTranscript):
    client = mock.Mock()
    client.get_session = mock.AsyncMock(
        return_value={"id": "s1", "deepLink": "conductor://s/1"}
    )
    client.session_status = mock.AsyncMock(return_value={"status": "idle"})
    client.list_messages = transcript.list_messages
    return client


async def _get_session(transcript: FakeTranscript, **inputs: Any) -> dict:
    block = ConductorGetSessionBlock()
    with mock.patch(
        "backend.blocks.conductor.get_session.ConductorClient",
        return_value=_session_client(transcript),
    ):
        return await collect(
            block,
            {"credentials": TEST_CREDENTIALS_INPUT, "session_id": "s1", **inputs},
        )


# --- Get Session: `after` accepts either id ---------------------------------


@pytest.mark.asyncio
async def test_initial_message_id_works_as_the_after_cursor():
    """The receipt from Create Session / Send Message, which the API rejects
    with 404, reads the prompt's turn instead of failing."""
    turn = claude_turn(RECEIPT, 0, "Done")
    transcript = FakeTranscript([user_row(PROMPT_ROW, RECEIPT, 0), *turn])

    outputs = await _get_session(transcript, after=RECEIPT, message_limit=20)

    assert [m["id"] for m in outputs["messages"]] == [r["id"] for r in turn]
    assert outputs["latest_reply"] == "Done"
    assert outputs["has_more"] is False
    assert outputs["next_after"] == "r-idle"
    assert transcript.calls[0] == {"after": RECEIPT, "limit": 20, "offset": None}
    assert transcript.calls[-1]["after"] == PROMPT_ROW


@pytest.mark.asyncio
async def test_after_receipt_with_no_reply_yet_returns_the_prompt_row_as_cursor():
    """Nothing follows a queued prompt: no rows, no error, and `next_after`
    is the resolved row id so the next poll pages directly."""
    transcript = FakeTranscript(
        [user_row("row-0", OLD_TURN, 0), user_row(PROMPT_ROW, RECEIPT, 1)]
    )

    outputs = await _get_session(transcript, after=RECEIPT)

    assert outputs["messages"] == []
    assert outputs["has_more"] is False
    assert outputs["next_after"] == PROMPT_ROW


@pytest.mark.asyncio
async def test_after_receipt_is_found_beyond_the_newest_rows():
    """The prompt sits 320 rows back: the same bounded history search as the
    wait loop locates it."""
    other = "22222222-0000-4000-8000-000000000000"
    rows = [
        user_row(PROMPT_ROW, RECEIPT, 0),
        *claude_turn(RECEIPT, 0, "Ours"),
        user_row("row-other", other, 11),
        *[agent_row(f"life-{i}", other, 12 + i, CLAUDE_LIFECYCLE) for i in range(320)],
    ]
    transcript = FakeTranscript(rows)

    outputs = await _get_session(transcript, after=RECEIPT, message_limit=3)

    assert [m["id"] for m in outputs["messages"]] == ["r-sys", "r-life", "r-think"]
    assert outputs["has_more"] is True
    assert outputs["next_after"] == "r-think"


@pytest.mark.asyncio
async def test_a_row_id_cursor_is_paged_without_a_lookup():
    rows = [user_row(PROMPT_ROW, RECEIPT, 0), *claude_turn(RECEIPT, 0, "Done")]
    transcript = FakeTranscript(rows)

    outputs = await _get_session(transcript, after=PROMPT_ROW, message_limit=2)

    assert [m["id"] for m in outputs["messages"]] == ["r-sys", "r-life"]
    assert transcript.calls == [{"after": PROMPT_ROW, "limit": 2, "offset": None}]


@pytest.mark.asyncio
async def test_an_unknown_cursor_names_both_id_spaces():
    transcript = FakeTranscript([user_row(PROMPT_ROW, RECEIPT, 0)])

    with pytest.raises(
        BlockExecutionError, match="neither a transcript row ID.*prompt"
    ):
        await _get_session(transcript, after="not-an-id")


@pytest.mark.asyncio
async def test_non_404_errors_are_not_treated_as_a_receipt():
    client = wait_client(FakeTranscript([]), [{"status": "idle"}])

    async def boom(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise ConductorAPIError("Conductor API error (HTTP 500)", 500)

    client.list_messages = boom
    with pytest.raises(ConductorAPIError, match="HTTP 500"):
        await read_after(client, "s1", RECEIPT, 10)


@pytest.mark.asyncio
@pytest.mark.parametrize("messages", [[], [{"content": "missing row ID"}]])
async def test_incremental_poll_preserves_cursor_without_a_new_row_id(messages: list):
    block = ConductorGetSessionBlock()
    mock_block(
        block,
        {"_fetch": lambda *args, **kwargs: {"messages": {"data": messages}}},
    )

    result = await collect(
        block,
        {
            "credentials": TEST_CREDENTIALS_INPUT,
            "session_id": "s1",
            "after": "last-seen-row",
        },
    )

    assert result["messages"] == messages
    assert result["next_after"] == "last-seen-row"


# --- find_prompt_row ----------------------------------------------------------


@pytest.mark.asyncio
async def test_find_prompt_row_maps_a_receipt_to_its_row():
    rows = [user_row("row-0", OLD_TURN, 0), user_row(PROMPT_ROW, RECEIPT, 1)]
    client = wait_client(FakeTranscript(rows), [{"status": "idle"}])
    assert await find_prompt_row(client, "s1", RECEIPT) == PROMPT_ROW
    assert await find_prompt_row(client, "s1", "unknown") == ""


@pytest.mark.asyncio
async def test_find_prompt_row_without_history_reads_only_the_newest_rows():
    rows = [
        user_row(PROMPT_ROW, RECEIPT, 0),
        *[
            agent_row(f"life-{i}", OLD_TURN, 1 + i, CLAUDE_LIFECYCLE)
            for i in range(320)
        ],
    ]
    transcript = FakeTranscript(rows)
    client = wait_client(transcript, [{"status": "idle"}])

    assert await find_prompt_row(client, "s1", RECEIPT, search_history=False) == ""
    # Only the tail is read: the first page, one-row probes for the end and
    # the newest 300 rows; no older page is fetched.
    assert all(
        call["limit"] == 1 or call["offset"] == 0 or call["offset"] >= len(rows) - 300
        for call in transcript.calls
        if call["after"] == ""
    )
    shallow_calls = len(transcript.calls)
    assert await find_prompt_row(client, "s1", RECEIPT) == PROMPT_ROW
    assert len(transcript.calls) > shallow_calls + 1


# --- senders hand out `next_after` --------------------------------------------


def _sender(block, send_name: str, receipt: dict, prompt_row):
    mock_block(
        block,
        {
            send_name: lambda *a, **k: receipt,
            "_prompt_row": prompt_row,
            "_wait": lambda *a, **k: pytest.fail("should not wait"),
        },
    )
    return block


@pytest.mark.asyncio
async def test_send_message_without_waiting_hands_out_the_prompt_row():
    block = _sender(
        ConductorSendMessageBlock(),
        "_send",
        {"messageId": RECEIPT, "state": "sent", "deepLink": "d"},
        lambda *a, **k: PROMPT_ROW,
    )
    outputs = await collect(
        block,
        {
            "credentials": TEST_CREDENTIALS_INPUT,
            "session_id": "s1",
            "message": "go",
            "wait_for_reply": False,
        },
    )
    assert outputs["message_id"] == RECEIPT
    assert outputs["next_after"] == PROMPT_ROW


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "prompt_row",
    [lambda *a, **k: "", mock.Mock(side_effect=ValueError("HTTP 503"))],
    ids=["not recorded yet", "lookup failed"],
)
async def test_create_session_falls_back_to_the_receipt_as_cursor(prompt_row):
    """The prompt may still be queued when the block returns, and the session
    was already created, so a missing or failed lookup is not an error: the
    receipt is handed out instead and Get Session resolves it later."""
    block = _sender(
        ConductorCreateSessionBlock(),
        "_create",
        {
            "id": "sess_1",
            "deepLink": "conductor://s/1",
            "initialMessage": {"messageId": RECEIPT, "state": "queued"},
        },
        prompt_row,
    )
    outputs = await collect(
        block,
        {
            "credentials": TEST_CREDENTIALS_INPUT,
            "workspace_id": "ws_1",
            "message": "go",
            "wait_for_reply": False,
        },
    )
    assert outputs["initial_message_id"] == RECEIPT
    assert outputs["next_after"] == RECEIPT


@pytest.mark.asyncio
async def test_create_session_without_a_prompt_has_no_cursor():
    block = _sender(
        ConductorCreateSessionBlock(),
        "_create",
        {"id": "sess_1", "deepLink": "conductor://s/1"},
        lambda *a, **k: pytest.fail("nothing to resolve"),
    )
    outputs = await collect(
        block, {"credentials": TEST_CREDENTIALS_INPUT, "workspace_id": "ws_1"}
    )
    assert outputs["initial_message_id"] == ""
    assert "next_after" not in outputs


@pytest.mark.asyncio
async def test_create_workspace_waiting_reports_the_prompt_row_seen_by_the_wait():
    block = ConductorCreateWorkspaceBlock()
    mock_block(
        block,
        {
            "_create": lambda *a, **k: {
                "workspaceId": "ws_1",
                "sessionId": "sess_1",
                "deepLink": "d",
                "initialMessage": {"messageId": RECEIPT, "state": "queued"},
            },
            "_prompt_row": lambda *a, **k: pytest.fail("the wait resolves it"),
            "_wait": lambda *a, **k: {
                "session_status": "idle",
                "error_message": "",
                "messages": [],
                "reply": "",
                "timed_out": False,
                "truncated": False,
                "prompt_row_id": PROMPT_ROW,
            },
        },
    )
    outputs = await collect(
        block,
        {
            "credentials": TEST_CREDENTIALS_INPUT,
            "project_id": "proj_1",
            "message": "go",
            "wait_for_reply": True,
        },
    )
    assert outputs["next_after"] == PROMPT_ROW
    assert outputs["session_status"] == "idle"


@pytest.mark.asyncio
async def test_sender_prompt_row_uses_a_shallow_lookup():
    """Right after sending, the prompt is among the newest rows or not
    recorded at all, so the senders never page into older history."""
    rows = [
        user_row("row-0", OLD_TURN, 0),
        *[
            agent_row(f"life-{i}", OLD_TURN, 1 + i, CLAUDE_LIFECYCLE)
            for i in range(320)
        ],
    ]
    transcript = FakeTranscript(rows)
    block = ConductorSendMessageBlock()
    with mock.patch(
        "backend.blocks.conductor.send_message.ConductorClient",
        return_value=wait_client(transcript, [{"status": "idle"}]),
    ):
        assert await block._prompt_row(TEST_CREDENTIALS, "s1", RECEIPT) == ""
    assert len(transcript.calls) < 12
