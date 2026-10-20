"""The executor hands a turn's error code to the turn's completion.

Subscribers such as the chat bots name the failure from that code, and
``mark_session_completed`` is what publishes it; the end-to-end path over
Redis is covered in ``bot/stream_error_code_test.py``.
"""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.bot.bot_backend import BotStreamError
from backend.copilot.bot.failures import (
    INTERNAL,
    PROVIDER_LIMIT,
    STOPPED,
    FailureCategory,
    categorize,
)
from backend.copilot.executor.processor import CoPilotProcessor
from backend.copilot.model import ChatSession
from backend.copilot.response_model import StreamError

from .processor_test import _make_entry, _make_log, _TrackedStream


@pytest.fixture(autouse=True)
def allowance_admits_every_turn(mocker):
    """The pickup budget gate has its own tests (processor_budget_gate_test)."""
    mocker.patch(
        "backend.copilot.executor.processor.refusal_for",
        AsyncMock(return_value=None),
    )


@pytest.fixture(autouse=True)
def no_trial_attribution(mocker):
    store = MagicMock()
    store.get_usage_activation_state = AsyncMock(
        side_effect=lambda user_id: SimpleNamespace(
            user_id=user_id, generation=None, trial_id=None, ready=True
        )
    )
    mocker.patch(
        "backend.copilot.usage_activation.pro_activation_db", return_value=store
    )


class _FailingClose(_TrackedStream):
    """The stream ends on a StreamError, then closing it fails."""

    def __init__(self, events: list, close_error: BaseException):
        super().__init__(events)
        self._close_error = close_error

    async def aclose(self) -> None:
        raise self._close_error


def _limit_error() -> StreamError:
    return StreamError(errorText="Usage limit reached.", code="usage_limit")


async def _run(published: _TrackedStream, mark_completed: AsyncMock) -> None:
    with (
        patch(
            "backend.copilot.executor.processor.ChatConfig",
            return_value=MagicMock(test_mode=True),
        ),
        patch(
            "backend.copilot.executor.processor.stream_chat_completion_dummy",
            return_value=MagicMock(),
        ),
        patch(
            "backend.copilot.executor.processor.stream_registry.stream_and_publish",
            return_value=published,
        ),
        patch(
            "backend.copilot.executor.processor.stream_registry.mark_session_completed",
            new=mark_completed,
        ),
        patch(
            "backend.copilot.model.get_chat_session",
            new=AsyncMock(return_value=ChatSession.new("user-1", dry_run=False)),
        ),
    ):
        await CoPilotProcessor()._execute_async(
            _make_entry(), threading.Event(), MagicMock(), _make_log()
        )


def _what_a_bot_user_is_told(mark_completed: AsyncMock) -> FailureCategory:
    published = mark_completed.await_args.kwargs
    return categorize(
        BotStreamError(
            "backend_stream_error",
            published["error_message"],
            code=published["error_code"],
        )
    )


@pytest.mark.asyncio
async def test_a_streamed_errors_code_reaches_the_turns_completion() -> None:
    mark_completed = AsyncMock()
    await _run(_TrackedStream(events=[_limit_error()]), mark_completed)

    mark_completed.assert_awaited_once_with(
        "sess-1",
        error_message="Usage limit reached.",
        error_code="usage_limit",
        turn_id="turn-1",
    )
    assert _what_a_bot_user_is_told(mark_completed) is PROVIDER_LIMIT


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "close_error, told",
    [
        (asyncio.CancelledError(), STOPPED),
        (RuntimeError("stream close broke"), INTERNAL),
    ],
)
async def test_a_later_failure_does_not_inherit_the_earlier_code(
    close_error: BaseException, told: FailureCategory
) -> None:
    mark_completed = AsyncMock()
    with pytest.raises(type(close_error)):
        await _run(_FailingClose([_limit_error()], close_error), mark_completed)

    assert mark_completed.await_args.kwargs["error_code"] is None
    assert _what_a_bot_user_is_told(mark_completed) is told
