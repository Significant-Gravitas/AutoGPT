import asyncio
import logging
import threading
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.copilot import stream_registry
from backend.copilot.executor import processor
from backend.copilot.executor.utils import CoPilotExecutionEntry, CoPilotLogMetadata
from backend.copilot.model import ChatSession
from backend.copilot.response_model import StreamError, StreamStatus


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode",
    [
        "cooperative",
        "task_cancel",
        "shutdown_cancel",
        "provider_error",
        "failure_then_stop",
    ],
)
async def test_user_stop_suppresses_cancellation_without_hiding_failures(mocker, mode):
    cancel = threading.Event()
    closed = []
    entry = CoPilotExecutionEntry(
        session_id="session-stop",
        turn_id="turn-stop",
        user_id="user-stop",
        message="hi",
        is_user_message=True,
        request_arrival_at=0.0,
    )

    async def published_stream(**kwargs):
        try:
            if mode in ("task_cancel", "shutdown_cancel"):
                if mode == "task_cancel":
                    cancel.set()
                raise asyncio.CancelledError()
            if mode == "cooperative":
                cancel.set()
                yield StreamStatus(message="Working")
            else:
                yield StreamError(
                    errorText="provider unavailable", code="provider_down"
                )
        finally:
            if mode == "failure_then_stop":
                cancel.set()
            closed.append(True)

    @asynccontextmanager
    async def no_trial_cost(*args):
        yield

    mocker.patch.object(processor, "is_user_banned", AsyncMock(return_value=False))
    mocker.patch.object(processor, "refusal_for", AsyncMock(return_value=None))
    mocker.patch.object(
        processor, "close_browser_daemon", AsyncMock(return_value=False)
    )
    mocker.patch.object(processor, "trial_cost_context", no_trial_cost)
    mocker.patch.object(processor, "ChatConfig", return_value=MagicMock(test_mode=True))
    mocker.patch.object(processor, "stream_chat_completion_dummy", return_value=None)
    mocker.patch(
        "backend.copilot.model.get_chat_session",
        AsyncMock(return_value=ChatSession.new("user-stop", dry_run=False)),
    )
    mocker.patch.object(stream_registry, "publish_chunk", AsyncMock())
    mocker.patch.object(stream_registry, "stream_and_publish", published_stream)
    completed = mocker.patch.object(
        stream_registry, "mark_session_completed", AsyncMock()
    )
    run = processor.CoPilotProcessor()._execute_async(
        entry, cancel, MagicMock(), CoPilotLogMetadata(logging.getLogger(__name__))
    )
    if mode in ("task_cancel", "shutdown_cancel"):
        with pytest.raises(asyncio.CancelledError):
            await run
    else:
        await run

    assert closed == [True]
    completed.assert_awaited_once_with(
        entry.session_id,
        turn_id=entry.turn_id,
        error_message=(
            "provider unavailable"
            if mode in ("provider_error", "failure_then_stop")
            else stream_registry.CANCELLED_MESSAGE
        ),
        error_code=(
            "provider_down" if mode in ("provider_error", "failure_then_stop") else None
        ),
        skip_error_publish=mode in ("cooperative", "task_cancel"),
    )
