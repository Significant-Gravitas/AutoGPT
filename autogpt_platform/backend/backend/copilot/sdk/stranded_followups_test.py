"""A follow-up the model already read is saved however the turn ends.

A message drained into a tool result waits in the persist queue until the
next tool output is dispatched. A turn that stops, fails or ends before then
must still save it, or it stays in that queue unsaved though the model read
it: a card decision among them would hold its card until the queue expires.
"""

import asyncio
import contextlib
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest

from backend.copilot.pending_messages import PendingMessage
from backend.copilot.sdk import service
from backend.copilot.sdk.service import stream_chat_completion_sdk

from .codex_usage_persistence_test import (
    _client,
    _consume_turn,
    _record_accounting,
    _session_and_lease,
    _stub_sdk,
)
from .codex_usage_test import usage_gateway

_SVC = "backend.copilot.sdk.service"
_STRANDED = PendingMessage(
    content="Approved: hire Maya (confirmation_id: c-1).",
    card_owner="click-1",
)


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["cancel", "failure", "success"])
async def test_a_turn_saves_the_follow_ups_it_left_waiting_to_be_saved(ending):
    session, lease = _session_and_lease()
    waiting = asyncio.Event()
    async with usage_gateway("success" if ending == "success" else "failed") as gateway:
        client = _client(gateway, ending, waiting, session.session_id)
        with contextlib.ExitStack() as stack:
            _stub_sdk(stack, session, gateway, client)
            _record_accounting(stack)
            stack.enter_context(
                patch(
                    f"{_SVC}.drain_pending_for_persist",
                    new_callable=AsyncMock,
                    return_value=[_STRANDED],
                )
            )
            save_rows = stack.enter_context(
                patch(
                    f"{_SVC}.persist_pending_as_user_rows",
                    new_callable=AsyncMock,
                    return_value=True,
                )
            )
            stream = stream_chat_completion_sdk(
                session_id=session.session_id,
                message="hello",
                is_user_message=True,
                user_id=session.user_id,
                session=session,
                credential_lease=lease,
            )
            await _consume_turn(stream, ending, waiting)

    save_rows.assert_any_await(session, ANY, [_STRANDED], log_prefix=ANY)


@pytest.mark.asyncio
@pytest.mark.parametrize("saved", [True, False], ids=["saved", "rolled-back"])
async def test_saved_follow_ups_count_as_rows_the_cli_transcript_lacks(saved):
    """A saved follow-up is in the chat but not in the CLI's transcript, as a
    mid-turn one is: counting it steps the upload's watermark back, so the
    next turn fills it in. One the save rolled back is not counted."""
    session, _ = _session_and_lease()
    state = SimpleNamespace(transcript_builder=MagicMock(), midturn_user_rows=2)
    with (
        patch(
            f"{_SVC}.drain_pending_for_persist",
            new_callable=AsyncMock,
            return_value=[_STRANDED],
        ),
        patch(
            f"{_SVC}.persist_pending_as_user_rows",
            new_callable=AsyncMock,
            return_value=saved,
        ),
    ):
        await service._persist_stranded_followups(session, state, "[test]")

    assert state.midturn_user_rows == (3 if saved else 2)
