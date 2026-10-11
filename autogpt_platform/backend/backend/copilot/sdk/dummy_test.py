from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.response_model import StreamFinish, StreamTextDelta
from backend.copilot.sdk import dummy


async def _run(message: str) -> tuple[list[object], list[float]]:
    """Stream the dummy reply, recording every event and every sleep."""
    timeline: list[object] = []
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        timeline.append(("sleep", seconds))

    with (
        patch.object(dummy.asyncio, "sleep", fake_sleep),
        patch.object(dummy, "get_chat_session", AsyncMock(return_value=None)),
        patch.object(dummy, "upsert_chat_session", AsyncMock()),
    ):
        async for event in dummy.stream_chat_completion_dummy(
            "session-1", message=message, user_id="user-1"
        ):
            timeline.append(event)
    return timeline, sleeps


@pytest.mark.asyncio
async def test_quiet_pause_goes_silent_before_any_text():
    timeline, sleeps = await _run("__test_quiet_pause__ count to three")

    pause = timeline.index(("sleep", dummy.QUIET_PAUSE_S))
    first_text = next(
        i for i, event in enumerate(timeline) if isinstance(event, StreamTextDelta)
    )
    assert pause < first_text
    # Short of the 10 s silence status, long enough for the 5 s heartbeat.
    assert 5 < dummy.QUIET_PAUSE_S < 10
    assert isinstance(timeline[-1], StreamFinish)
    assert sleeps.count(dummy.QUIET_PAUSE_S) == 1


@pytest.mark.asyncio
async def test_plain_message_has_no_quiet_pause():
    _, sleeps = await _run("count to three")

    assert dummy.QUIET_PAUSE_S not in sleeps
